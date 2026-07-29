// Copyright (c) 2026 DANCE Authors. Licensed under the MIT License.



#include <algorithm>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "gpu_vamana_builder.h"
#include "index.h"
#include "program_options_utils.hpp"

namespace po = boost::program_options;

static constexpr uint32_t INVALID_ID = std::numeric_limits<uint32_t>::max();

struct Args
{
    std::string data_type;
    std::string data_path;
    std::string label_file;
    std::string index_path_prefix;
    std::string universal_label;
    std::string timing_json;
    uint32_t r_small = 32;
    uint32_t l_small = 100;
    uint32_t r_out = 64;
    uint32_t c = 96;
    uint32_t steps = 64;
    uint32_t threads = 32;
    float alpha = 1.2f;
};

static double now_sec()
{
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration<double>(clock::now() - t0).count();
}

static Args parse_args(int argc, char **argv)
{
    Args args;
    po::options_description desc{
        program_options_utils::make_program_description("gpu_stitched_vamana_index",
                                                        "Build GPU StitchedVamana for single- or multi-label data.")};
    desc.add_options()("help,h", "Print information on arguments")(
        "data_type", po::value<std::string>(&args.data_type)->required(), program_options_utils::DATA_TYPE_DESCRIPTION)(
        "data_path", po::value<std::string>(&args.data_path)->required(), program_options_utils::INPUT_DATA_PATH)(
        "label_file", po::value<std::string>(&args.label_file)->required(), program_options_utils::LABEL_FILE)(
        "index_path_prefix", po::value<std::string>(&args.index_path_prefix)->required(),
        program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION)(
        "universal_label", po::value<std::string>(&args.universal_label)->default_value(""),
        program_options_utils::UNIVERSAL_LABEL)(
        "timing_json", po::value<std::string>(&args.timing_json)->default_value(""),
        "Optional path to write GPU-Stitched timing JSON")(
        "Rsmall", po::value<uint32_t>(&args.r_small)->default_value(32), "Per-label GPU Vamana degree")(
        "Lsmall", po::value<uint32_t>(&args.l_small)->default_value(100), "Per-label GPU Vamana build/search L")(
        "Rstitched", po::value<uint32_t>(&args.r_out)->default_value(64), "Output graph degree cap")(
        "C", po::value<uint32_t>(&args.c)->default_value(96), "Per-label GPU Vamana work C")(
        "steps", po::value<uint32_t>(&args.steps)->default_value(64), "Per-label GPU Vamana max expansion steps")(
        "num_threads,T", po::value<uint32_t>(&args.threads)->default_value(32), "Final-prune threads")(
        "alpha", po::value<float>(&args.alpha)->default_value(1.2f), "Final robust-prune alpha");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help"))
    {
        std::cout << desc;
        std::exit(0);
    }
    po::notify(vm);
    if (args.r_out < args.r_small)
        args.r_out = args.r_small;
    return args;
}

template <typename T> static std::vector<T> load_bin(const std::string &path, uint32_t &npts, uint32_t &dim)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("failed to open data file: " + path);
    int32_t header[2];
    in.read(reinterpret_cast<char *>(header), sizeof(header));
    npts = (uint32_t)header[0];
    dim = (uint32_t)header[1];
    std::vector<T> data((size_t)npts * dim);
    in.read(reinterpret_cast<char *>(data.data()), (std::streamsize)(data.size() * sizeof(T)));
    if (!in)
        throw std::runtime_error("failed to read data payload: " + path);
    return data;
}

static void copy_file(const std::string &src, const std::string &dst)
{
    std::ifstream in(src, std::ios::binary);
    std::ofstream out(dst, std::ios::binary);
    out << in.rdbuf();
}

static std::vector<std::vector<std::string>> parse_labels(const std::string &label_file,
                                                          const std::string &universal_label,
                                                          std::unordered_map<std::string, uint32_t> &label_to_id,
                                                          std::vector<std::string> &id_to_label,
                                                          std::vector<std::vector<uint32_t>> &label_points)
{
    std::ifstream in(label_file);
    if (!in)
        throw std::runtime_error("failed to open label file: " + label_file);
    std::vector<std::vector<std::string>> point_label_names;
    std::string line;
    while (std::getline(in, line))
    {
        line.erase(std::remove_if(line.begin(), line.end(), [](unsigned char ch) { return ch == '\r' || ch == ' '; }),
                   line.end());
        if (line.empty())
            throw std::runtime_error("empty label row is not supported");
        std::vector<std::string> names;
        size_t start = 0;
        while (start <= line.size())
        {
            size_t comma = line.find(',', start);
            std::string name = line.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
            if (name.empty())
                throw std::runtime_error("empty label token is not supported");
            if (!universal_label.empty() && name == universal_label)
                throw std::runtime_error("GPU StitchedMVP does not support universal-label rows");
            if (std::find(names.begin(), names.end(), name) == names.end())
                names.push_back(name);
            if (comma == std::string::npos)
                break;
            start = comma + 1;
        }
        const uint32_t point_id = (uint32_t)point_label_names.size();
        for (const auto &name : names)
        {
            auto it = label_to_id.find(name);
            if (it == label_to_id.end())
            {
                uint32_t id = (uint32_t)id_to_label.size() + 1;
                label_to_id.emplace(name, id);
                id_to_label.push_back(name);
                label_points.emplace_back();
            }
            label_points[label_to_id.at(name) - 1].push_back(point_id);
        }
        point_label_names.push_back(std::move(names));
    }
    return point_label_names;
}

static void write_sidecars(const Args &args,
                           const std::vector<std::vector<std::string>> &point_label_names,
                           const std::unordered_map<std::string, uint32_t> &label_to_id,
                           const std::vector<std::string> &id_to_label,
                           const std::vector<uint32_t> &label_medoids)
{
    copy_file(args.data_path, args.index_path_prefix + ".data");
    {
        std::ofstream labels(args.index_path_prefix + "_labels.txt");
        std::ofstream formatted(args.index_path_prefix + "_label_formatted.txt");
        for (const auto &names : point_label_names)
        {
            for (size_t j = 0; j < names.size(); ++j)
            {
                if (j)
                {
                    labels << ',';
                    formatted << ',';
                }
                uint32_t id = label_to_id.at(names[j]);
                labels << id;
                formatted << id;
            }
            labels << "\n";
            formatted << "\n";
        }
    }
    {
        std::ofstream map(args.index_path_prefix + "_labels_map.txt");
        for (uint32_t i = 0; i < id_to_label.size(); ++i)
            map << id_to_label[i] << "\t" << (i + 1) << "\n";
    }
    {
        std::ofstream medoids(args.index_path_prefix + "_labels_to_medoids.txt");
        for (uint32_t i = 0; i < label_medoids.size(); ++i)
            medoids << (i + 1) << ", " << label_medoids[i] << "\n";
    }
}

static uint64_t write_graph(const std::string &prefix, const std::vector<std::vector<uint32_t>> &graph)
{
    uint64_t bytes = 2 * sizeof(uint64_t) + 2 * sizeof(uint32_t);
    uint32_t max_degree = 0;
    uint64_t edges = 0;
    for (const auto &row : graph)
    {
        max_degree = std::max<uint32_t>(max_degree, (uint32_t)row.size());
        bytes += sizeof(uint32_t) + row.size() * sizeof(uint32_t);
        edges += row.size();
    }
    uint32_t start = 0;
    uint64_t frozen = 0;
    std::ofstream out(prefix, std::ios::binary);
    out.write(reinterpret_cast<char *>(&bytes), sizeof(uint64_t));
    out.write(reinterpret_cast<char *>(&max_degree), sizeof(uint32_t));
    out.write(reinterpret_cast<char *>(&start), sizeof(uint32_t));
    out.write(reinterpret_cast<char *>(&frozen), sizeof(uint64_t));
    for (const auto &row : graph)
    {
        uint32_t deg = (uint32_t)row.size();
        out.write(reinterpret_cast<char *>(&deg), sizeof(uint32_t));
        if (deg)
            out.write(reinterpret_cast<const char *>(row.data()), (std::streamsize)(deg * sizeof(uint32_t)));
    }
    std::cout << "[GPU stitched] wrote graph bytes=" << bytes << " edges=" << edges
              << " avg_degree=" << (double)edges / (double)graph.size() << " max_degree=" << max_degree << std::endl;
    return edges;
}

static double current_gpu_used_gb()
{
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess || total_bytes == 0)
        return 0.0;
    return (double)(total_bytes - free_bytes) / (1024.0 * 1024.0 * 1024.0);
}

template <typename T> static int build_gpu_stitched(const Args &args)
{
    double t0 = now_sec();
    uint32_t npts = 0, dim = 0;
    std::vector<T> data = load_bin<T>(args.data_path, npts, dim);
    std::unordered_map<std::string, uint32_t> label_to_id;
    std::vector<std::string> id_to_label;
    std::vector<std::vector<uint32_t>> label_points;
    auto point_label_names =
        parse_labels(args.label_file, args.universal_label, label_to_id, id_to_label, label_points);
    if (point_label_names.size() != npts)
        throw std::runtime_error("number of label rows does not match data points");

    std::vector<std::vector<uint32_t>> global_graph(npts);
    std::vector<uint32_t> label_medoids(label_points.size(), 0);
    double build_seconds = 0.0;
    double max_label_build_seconds = 0.0;
    double materialize_seconds = 0.0;
    double remap_seconds = 0.0;
    double write_index_seconds = 0.0;
    double final_prune_seconds = 0.0;
    double metadata_seconds = 0.0;
    double peak_gpu_mem_gb = current_gpu_used_gb();
    uint64_t processed_label_memberships = 0;
    uint64_t remap_invalid_local_ids = 0;
    uint64_t remap_self_loops = 0;
    uint64_t remap_duplicate_edges = 0;
    uint32_t min_label_npts = std::numeric_limits<uint32_t>::max();
    uint32_t max_label_npts = 0;
    uint64_t sum_label_npts = 0;

    std::cout << "[GPU stitched] enabled labels=" << label_points.size() << " N=" << npts << " dim=" << dim
              << " Rsmall=" << args.r_small << " Lsmall=" << args.l_small << " Rstitched=" << args.r_out
              << " C=" << args.c << std::endl;

    for (size_t lid = 0; lid < label_points.size(); ++lid)
    {
        const auto &ids = label_points[lid];
        if (ids.empty())
            continue;
        min_label_npts = std::min<uint32_t>(min_label_npts, (uint32_t)ids.size());
        max_label_npts = std::max<uint32_t>(max_label_npts, (uint32_t)ids.size());
        sum_label_npts += ids.size();
        processed_label_memberships += ids.size();
        std::vector<T> local_data((size_t)ids.size() * dim);
        double mt0 = now_sec();
        for (size_t i = 0; i < ids.size(); ++i)
            std::copy_n(data.data() + (size_t)ids[i] * dim, dim, local_data.data() + i * dim);
        materialize_seconds += now_sec() - mt0;
        std::vector<uint32_t> local_graph((size_t)ids.size() * args.r_small, INVALID_ID);
        std::vector<uint32_t> local_degree(ids.size(), 0);
        double lt0 = now_sec();
        int ret = -1;
        if constexpr (std::is_same<T, uint8_t>::value)
            ret = gpu_vamana_vnew2_build(local_data.data(), (uint32_t)ids.size(), dim, args.r_small, args.l_small,
                                         args.c, args.steps, local_graph.data(), local_degree.data());
        else
            ret = gpu_vamana_vnew2_build_float(local_data.data(), (uint32_t)ids.size(), dim, args.r_small,
                                               args.l_small, args.c, args.steps, local_graph.data(),
                                               local_degree.data());
        if (ret != 0)
            throw std::runtime_error("gpu_vamana_vnew2_build failed for label " + id_to_label[lid]);
        double label_build_seconds = now_sec() - lt0;
        build_seconds += label_build_seconds;
        max_label_build_seconds = std::max(max_label_build_seconds, label_build_seconds);
        peak_gpu_mem_gb = std::max(peak_gpu_mem_gb, current_gpu_used_gb());
        uint32_t medoid = std::is_same<T, uint8_t>::value ? gpu_vamana_vnew2_get_last_medoid()
                                                          : gpu_vamana_vnew2_get_last_medoid();
        if (medoid >= ids.size())
            medoid = 0;
        label_medoids[lid] = ids[medoid];
        double rt0 = now_sec();
        for (size_t i = 0; i < ids.size(); ++i)
        {
            auto &row = global_graph[ids[i]];
            uint32_t deg = std::min<uint32_t>(local_degree[i], args.r_out);
            row.reserve(deg);
            for (uint32_t j = 0; j < deg; ++j)
            {
                uint32_t nb = local_graph[i * args.r_small + j];
                if (nb == INVALID_ID || nb >= ids.size())
                {
                    ++remap_invalid_local_ids;
                    continue;
                }
                if (ids[nb] == ids[i])
                {
                    ++remap_self_loops;
                    continue;
                }
                uint32_t g = ids[nb];
                if (std::find(row.begin(), row.end(), g) == row.end())
                    row.push_back(g);
                else
                    ++remap_duplicate_edges;
            }
        }
        remap_seconds += now_sec() - rt0;
        std::cout << "[GPU stitched] label=" << id_to_label[lid] << " points=" << ids.size()
                << " medoid=" << label_medoids[lid] << " seconds=" << (now_sec() - lt0) << std::endl;
    }

    double wt0 = now_sec();
    uint64_t output_edges = write_graph(args.index_path_prefix, global_graph);
    write_index_seconds = now_sec() - wt0;
    double metat0 = now_sec();
    write_sidecars(args, point_label_names, label_to_id, id_to_label, label_medoids);
    metadata_seconds = now_sec() - metat0;
    const bool single_label_rows = std::all_of(point_label_names.begin(), point_label_names.end(),
                                               [](const auto &labels) { return labels.size() == 1; });
    const bool final_prune_is_noop = single_label_rows && args.r_small <= args.r_out;
    if (final_prune_is_noop)
    {
        std::cout << "[GPU stitched] global label-aware prune is a no-op: single-label rows and "
                     "Rsmall <= Rstitched"
                  << std::endl;
    }
    else
    {


        copy_file(args.index_path_prefix, args.index_path_prefix + "_full");
        double pt0 = now_sec();
        {
            diskann::Index<T> index(diskann::Metric::L2, dim, npts, nullptr, nullptr, 0, false, false, false, false, 0,
                                    false);
            index.load(args.index_path_prefix.c_str(), args.threads, 1);
            index.prune_all_neighbors(args.r_out, 750, args.alpha);
            index.save(args.index_path_prefix.c_str());
        }
        final_prune_seconds = now_sec() - pt0;
    }
    double total_seconds = now_sec() - t0;
    if (processed_label_memberships != sum_label_npts)
        throw std::runtime_error("not every label membership participated in a local graph");
    double mean_label_npts = label_points.empty() ? 0.0 : (double)sum_label_npts / (double)label_points.size();
    if (min_label_npts == std::numeric_limits<uint32_t>::max())
        min_label_npts = 0;
    std::cout << "[GPU stitched] total_seconds=" << total_seconds << " gpu_label_build_seconds=" << build_seconds
              << " output_edges=" << output_edges << std::endl;
    std::cout << "[GPU stitched timing]"
              << " num_labels=" << label_points.size()
              << " npts_per_label_min=" << min_label_npts
              << " npts_per_label_mean=" << mean_label_npts
              << " npts_per_label_max=" << max_label_npts
              << " total_label_memberships=" << sum_label_npts
              << " processed_label_memberships=" << processed_label_memberships
              << " remap_invalid_local_ids=" << remap_invalid_local_ids
              << " remap_self_loops=" << remap_self_loops
              << " remap_duplicate_edges=" << remap_duplicate_edges
              << " label_data_materialization_seconds=" << materialize_seconds
              << " per_label_gpu_build_sum_seconds=" << build_seconds
              << " per_label_gpu_build_max_seconds=" << max_label_build_seconds
              << " local_to_global_remap_seconds=" << remap_seconds
              << " write_index_seconds=" << write_index_seconds
              << " final_prune_seconds=" << final_prune_seconds
              << " metadata_seconds=" << metadata_seconds
              << " total_seconds=" << total_seconds
              << " peak_gpu_mem_gb=" << peak_gpu_mem_gb
              << " output_edges=" << output_edges
              << std::endl;
    if (!args.timing_json.empty())
    {
        std::ofstream json(args.timing_json);
        json << "{\n"
             << "  \"num_labels\": " << label_points.size() << ",\n"
             << "  \"npts_per_label_min\": " << min_label_npts << ",\n"
             << "  \"npts_per_label_mean\": " << mean_label_npts << ",\n"
             << "  \"npts_per_label_max\": " << max_label_npts << ",\n"
             << "  \"total_label_memberships\": " << sum_label_npts << ",\n"
             << "  \"processed_label_memberships\": " << processed_label_memberships << ",\n"
             << "  \"remap_invalid_local_ids\": " << remap_invalid_local_ids << ",\n"
             << "  \"remap_self_loops\": " << remap_self_loops << ",\n"
             << "  \"remap_duplicate_edges\": " << remap_duplicate_edges << ",\n"
             << "  \"label_data_materialization_seconds\": " << materialize_seconds << ",\n"
             << "  \"per_label_gpu_build_sum_seconds\": " << build_seconds << ",\n"
             << "  \"per_label_gpu_build_max_seconds\": " << max_label_build_seconds << ",\n"
             << "  \"local_to_global_remap_seconds\": " << remap_seconds << ",\n"
             << "  \"write_index_seconds\": " << write_index_seconds << ",\n"
             << "  \"final_prune_seconds\": " << final_prune_seconds << ",\n"
             << "  \"metadata_seconds\": " << metadata_seconds << ",\n"
             << "  \"total_seconds\": " << total_seconds << ",\n"
             << "  \"peak_gpu_mem_gb\": " << peak_gpu_mem_gb << ",\n"
             << "  \"output_edges\": " << output_edges << "\n"
             << "}\n";
    }
    return 0;
}

int main(int argc, char **argv)
{
    try
    {
        Args args = parse_args(argc, argv);
        if (args.data_type == "uint8")
            return build_gpu_stitched<uint8_t>(args);
        if (args.data_type == "float")
            return build_gpu_stitched<float>(args);
        std::cerr << "GPU StitchedMVP supports uint8 and float in this version" << std::endl;
        return 2;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[GPU stitched] failed: " << e.what() << std::endl;
        return 1;
    }
}
