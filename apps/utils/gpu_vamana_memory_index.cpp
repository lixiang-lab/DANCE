// Copyright (c) 2026 DANCE Authors. Licensed under the MIT License.
#include "gpu_vamana_builder.h"

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <limits>
#include <type_traits>
#include <vector>

namespace
{
double now_sec()
{
    using clock = std::chrono::high_resolution_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration<double>(clock::now() - t0).count();
}

std::string json_escape(const std::string &value)
{
    std::string escaped;
    escaped.reserve(value.size());
    for (const unsigned char ch : value)
    {
        switch (ch)
        {
        case '"':
            escaped += "\\\"";
            break;
        case '\\':
            escaped += "\\\\";
            break;
        case '\b':
            escaped += "\\b";
            break;
        case '\f':
            escaped += "\\f";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            if (ch < 0x20)
                throw std::runtime_error("control character in JSON path");
            escaped.push_back(static_cast<char>(ch));
        }
    }
    return escaped;
}

struct Args
{
    std::string data_path;
    std::string data_type = "uint8";
    std::string index_prefix;
    uint64_t offset = 0;
    uint32_t npts = 0;
    uint32_t R = 42;
    uint32_t L = 100;
    uint32_t C = 80;
    uint32_t STEPS = 64;
    std::string builder = "vnew2";
};

uint64_t parse_uint64(const std::string &text, const std::string &option)
{
    if (text.empty() || !std::all_of(text.begin(), text.end(), [](unsigned char ch) { return ch >= '0' && ch <= '9'; }))
        throw std::runtime_error(option + " must be an unsigned integer");
    size_t consumed = 0;
    unsigned long long value = 0;
    try
    {
        value = std::stoull(text, &consumed, 10);
    }
    catch (const std::exception &)
    {
        throw std::runtime_error(option + " must be an unsigned integer");
    }
    if (consumed != text.size())
        throw std::runtime_error(option + " must be an unsigned integer");
    return static_cast<uint64_t>(value);
}

uint32_t parse_uint32(const std::string &text, const std::string &option, bool allow_zero)
{
    const uint64_t value = parse_uint64(text, option);
    if (value > std::numeric_limits<uint32_t>::max() || (!allow_zero && value == 0))
        throw std::runtime_error(option + (allow_zero ? " is out of uint32 range" : " must be in [1, UINT32_MAX]"));
    return static_cast<uint32_t>(value);
}

void usage(const char *argv0)
{
    std::cerr << "Usage: " << argv0
              << " --data_path <bin> [--data_type uint8|float] [--offset N] [--npts N|0=all]"
              << " [--index_prefix PATH]"
              << " [--R 42] [--L 100] [--C 80] [--STEPS 64] [--builder vnew2|v8]\n";
}

Args parse_args(int argc, char **argv)
{
    Args args;
    for (int i = 1; i < argc; i++)
    {
        std::string key = argv[i];
        auto need_value = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error("missing value for " + key);
            return argv[++i];
        };
        if (key == "--data_path")
            args.data_path = need_value();
        else if (key == "--data_type")
            args.data_type = need_value();
        else if (key == "--index_prefix")
            args.index_prefix = need_value();
        else if (key == "--offset")
            args.offset = parse_uint64(need_value(), key);
        else if (key == "--npts")
            args.npts = parse_uint32(need_value(), key, true);
        else if (key == "--R")
            args.R = parse_uint32(need_value(), key, false);
        else if (key == "--L")
            args.L = parse_uint32(need_value(), key, false);
        else if (key == "--C")
            args.C = parse_uint32(need_value(), key, false);
        else if (key == "--STEPS")
            args.STEPS = parse_uint32(need_value(), key, false);
        else if (key == "--builder")
            args.builder = need_value();
        else
            throw std::runtime_error("unknown argument: " + key);
    }
    if (args.data_path.empty())
        throw std::runtime_error("--data_path is required");
    if (args.data_type != "uint8" && args.data_type != "float")
        throw std::runtime_error("--data_type must be uint8 or float");
    if (args.builder != "vnew2" && args.builder != "v8")
        throw std::runtime_error("--builder must be vnew2 or v8");
    (void)json_escape(args.data_path);
    (void)json_escape(args.index_prefix);
    return args;
}

template <typename T>
std::vector<T> load_slice(const Args &args, uint32_t &dim, uint64_t &base_npts, uint32_t &loaded_npts)
{
    std::ifstream reader(args.data_path, std::ios::binary);
    if (!reader)
        throw std::runtime_error("failed to open " + args.data_path);

    uint32_t npts32 = 0;
    uint32_t dim32 = 0;
    reader.read(reinterpret_cast<char *>(&npts32), sizeof(uint32_t));
    reader.read(reinterpret_cast<char *>(&dim32), sizeof(uint32_t));
    if (!reader)
        throw std::runtime_error("failed to read DiskANN matrix header");
    base_npts = npts32;
    dim = dim32;
    const uint64_t element_bytes = sizeof(T);
    if (dim == 0 || base_npts > (std::numeric_limits<uint64_t>::max() - 8) / ((uint64_t)dim * element_bytes))
        throw std::runtime_error("DiskANN matrix shape overflows its byte-size calculation");
    const uint64_t expected_bytes = 8 + base_npts * (uint64_t)dim * element_bytes;
    reader.seekg(0, std::ios::end);
    const std::streamoff actual_bytes = reader.tellg();
    if (actual_bytes < 0 || static_cast<uint64_t>(actual_bytes) != expected_bytes)
        throw std::runtime_error("DiskANN matrix header does not match exact file size");
    const uint64_t remaining = base_npts - std::min<uint64_t>(args.offset, base_npts);
    const uint64_t requested = args.npts == 0 ? remaining : args.npts;
    if (args.offset > base_npts || requested > remaining || requested > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("requested slice exceeds base file point count");
    if (requested == 0 || dim == 0)
        throw std::runtime_error("requested slice and dimension must be nonzero");
    loaded_npts = static_cast<uint32_t>(requested);

    std::vector<T> data((size_t)loaded_npts * dim);
    const uint64_t byte_offset = 2 * sizeof(uint32_t) + args.offset * (uint64_t)dim * sizeof(T);
    reader.seekg((std::streamoff)byte_offset, std::ios::beg);
    reader.read(reinterpret_cast<char *>(data.data()), (std::streamsize)(data.size() * sizeof(T)));
    if (!reader)
        throw std::runtime_error("failed to read requested slice");
    return data;
}

template <typename T>
void save_memory_index(const Args &args, const std::vector<T> &data, uint32_t npts, uint32_t dim,
                       const std::vector<uint32_t> &graph, const std::vector<uint32_t> &degree, uint32_t medoid)
{
    if (args.index_prefix.empty())
        return;
    const std::string data_path = args.index_prefix + ".data";
    const std::string graph_temp = args.index_prefix + ".tmp";
    const std::string data_temp = data_path + ".tmp";
    if (std::ifstream(args.index_prefix, std::ios::binary) || std::ifstream(data_path, std::ios::binary) ||
        std::ifstream(graph_temp, std::ios::binary) || std::ifstream(data_temp, std::ios::binary))
        throw std::runtime_error("refusing to overwrite existing memory-index output");
    const size_t edge_count = [&]() {
        size_t total = 0;
        for (uint32_t value : degree)
            total += std::min(value, args.R);
        return total;
    }();
    const uint32_t max_degree = *std::max_element(degree.begin(), degree.end());
    const size_t frozen_points = 0;
    const size_t header_bytes = sizeof(size_t) + 2 * sizeof(uint32_t) + sizeof(size_t);
    if ((size_t)npts > (std::numeric_limits<size_t>::max() - header_bytes) / sizeof(uint32_t) ||
        edge_count > (std::numeric_limits<size_t>::max() - header_bytes -
                      (size_t)npts * sizeof(uint32_t)) / sizeof(uint32_t))
        throw std::runtime_error("serialized graph size overflows size_t");
    const size_t graph_bytes =
        header_bytes + (size_t)npts * sizeof(uint32_t) + edge_count * sizeof(uint32_t);
    const size_t data_bytes = 2 * sizeof(uint32_t) + data.size() * sizeof(T);
    try
    {
        std::ofstream writer(graph_temp, std::ios::binary | std::ios::trunc);
        if (!writer)
            throw std::runtime_error("failed to create graph temporary output " + graph_temp);
        writer.write(reinterpret_cast<const char *>(&graph_bytes), sizeof(graph_bytes));
        writer.write(reinterpret_cast<const char *>(&max_degree), sizeof(max_degree));
        writer.write(reinterpret_cast<const char *>(&medoid), sizeof(medoid));
        writer.write(reinterpret_cast<const char *>(&frozen_points), sizeof(frozen_points));
        for (uint32_t node = 0; node < npts; ++node)
        {
            const uint32_t row_degree = std::min(degree[node], args.R);
            writer.write(reinterpret_cast<const char *>(&row_degree), sizeof(row_degree));
            writer.write(reinterpret_cast<const char *>(graph.data() + (size_t)node * args.R),
                         (size_t)row_degree * sizeof(uint32_t));
        }
        if (!writer)
            throw std::runtime_error("failed while writing graph temporary output " + graph_temp);
        writer.close();
        if (!writer)
            throw std::runtime_error("failed while closing graph temporary output " + graph_temp);
    }
    catch (...)
    {
        std::remove(graph_temp.c_str());
        throw;
    }
    try
    {
        std::ofstream writer(data_temp, std::ios::binary | std::ios::trunc);
        if (!writer)
            throw std::runtime_error("failed to create vector temporary output " + data_temp);
        writer.write(reinterpret_cast<const char *>(&npts), sizeof(npts));
        writer.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
        writer.write(reinterpret_cast<const char *>(data.data()), (std::streamsize)(data.size() * sizeof(T)));
        if (!writer)
            throw std::runtime_error("failed while writing vector temporary output " + data_temp);
        writer.close();
        if (!writer)
            throw std::runtime_error("failed while closing vector temporary output " + data_temp);
    }
    catch (...)
    {
        std::remove(graph_temp.c_str());
        std::remove(data_temp.c_str());
        throw;
    }
    std::ifstream graph_reader(graph_temp, std::ios::binary | std::ios::ate);
    std::ifstream data_reader(data_temp, std::ios::binary | std::ios::ate);
    if (!graph_reader || graph_reader.tellg() != static_cast<std::streamoff>(graph_bytes) ||
        !data_reader || data_reader.tellg() != static_cast<std::streamoff>(data_bytes))
    {
        graph_reader.close();
        data_reader.close();
        std::remove(graph_temp.c_str());
        std::remove(data_temp.c_str());
        throw std::runtime_error("serialized temporary output size does not match its DiskANN layout");
    }
    graph_reader.close();
    data_reader.close();
    if (std::rename(data_temp.c_str(), data_path.c_str()) != 0)
    {
        std::remove(graph_temp.c_str());
        std::remove(data_temp.c_str());
        throw std::runtime_error("failed to publish vector output " + data_path);
    }
    if (std::rename(graph_temp.c_str(), args.index_prefix.c_str()) != 0)
    {
        std::remove(graph_temp.c_str());
        std::remove(data_path.c_str());
        throw std::runtime_error("failed to publish graph output " + args.index_prefix);
    }
}

template <typename T>
int run_typed(const Args &args)
{
    uint32_t dim = 0;
    uint32_t npts = 0;
    uint64_t base_npts = 0;
    const double load_start = now_sec();
    std::vector<T> data = load_slice<T>(args, dim, base_npts, npts);
    const double load_seconds = now_sec() - load_start;

    if ((size_t)npts > std::numeric_limits<size_t>::max() / sizeof(uint32_t) / (size_t)args.R)
        throw std::runtime_error("N * R overflows the host graph allocation");
    std::vector<uint32_t> graph((size_t)npts * args.R);
    std::vector<uint32_t> degree(npts);

    const double gpu_start = now_sec();
    int ret = -1;
    if constexpr (std::is_same<T, uint8_t>::value)
    {
        if (args.builder == "vnew2")
            ret = gpu_vamana_vnew2_build(data.data(), npts, dim, args.R, args.L, args.C, args.STEPS,
                                         graph.data(), degree.data());
        else
            ret = gpu_vamana_build(data.data(), npts, dim, args.R, args.L, args.C, args.STEPS,
                                   graph.data(), degree.data());
    }
    else
    {
        if (args.builder == "vnew2")
            ret = gpu_vamana_vnew2_build_float(data.data(), npts, dim, args.R, args.L, args.C, args.STEPS,
                                               graph.data(), degree.data());
        else
            ret = gpu_vamana_build_float(data.data(), npts, dim, args.R, args.L, args.C, args.STEPS,
                                         graph.data(), degree.data());
    }
    const double gpu_call_seconds = now_sec() - gpu_start;

    GPUVamanaStats stats{};
    uint32_t medoid = 0;
    if (args.builder == "vnew2")
    {
        medoid = gpu_vamana_vnew2_get_last_medoid();
        gpu_vamana_vnew2_get_last_stats(&stats);
    }
    else
    {
        medoid = gpu_vamana_get_last_medoid();
        gpu_vamana_get_last_stats(&stats);
    }

    uint64_t edge_count = 0;
    uint64_t invalid_neighbor_count = 0;
    uint64_t self_loop_count = 0;
    uint64_t duplicate_edge_count = 0;
    uint64_t degree_overflow_count = 0;
    uint32_t max_degree = 0;
    std::vector<uint32_t> row;
    row.reserve(args.R);
    for (uint32_t node = 0; node < npts; ++node)
    {
        const uint32_t d = degree[node];
        edge_count += d;
        if (d > max_degree)
            max_degree = d;
        if (d > args.R)
            ++degree_overflow_count;
        row.clear();
        for (uint32_t slot = 0; slot < std::min(d, args.R); ++slot)
        {
            const uint32_t neighbor = graph[(size_t)node * args.R + slot];
            if (neighbor >= npts)
            {
                ++invalid_neighbor_count;
                continue;
            }
            if (neighbor == node)
                ++self_loop_count;
            row.push_back(neighbor);
        }
        std::sort(row.begin(), row.end());
        const auto unique_end = std::unique(row.begin(), row.end());
        duplicate_edge_count += (uint64_t)std::distance(unique_end, row.end());
    }
    const bool graph_valid = ret == 0 && invalid_neighbor_count == 0 &&
                             self_loop_count == 0 && duplicate_edge_count == 0 &&
                             degree_overflow_count == 0 && medoid < npts;
    if (graph_valid)
        save_memory_index(args, data, npts, dim, graph, degree, medoid);

    std::cout << "{\n"
              << "  \"data_path\": \"" << json_escape(args.data_path) << "\",\n"
              << "  \"data_type\": \"" << args.data_type << "\",\n"
              << "  \"base_npts\": " << base_npts << ",\n"
              << "  \"offset\": " << args.offset << ",\n"
              << "  \"npts\": " << npts << ",\n"
              << "  \"dim\": " << dim << ",\n"
              << "  \"R\": " << args.R << ",\n"
              << "  \"L\": " << args.L << ",\n"
              << "  \"C\": " << args.C << ",\n"
              << "  \"STEPS\": " << args.STEPS << ",\n"
              << "  \"builder\": \"" << args.builder << "\",\n"
              << "  \"index_prefix\": \"" << json_escape(args.index_prefix) << "\",\n"
              << "  \"memory_index_serialized\": " << (!args.index_prefix.empty() && graph_valid ? "true" : "false")
              << ",\n"
              << "  \"load_seconds\": " << load_seconds << ",\n"
              << "  \"gpu_call_seconds\": " << gpu_call_seconds << ",\n"
              << "  \"stats_gpu_build_seconds\": " << stats.gpu_build_seconds << ",\n"
              << "  \"stats_h2d_seconds\": " << stats.h2d_seconds << ",\n"
              << "  \"stats_search_online_insert_seconds\": " << stats.search_online_insert_seconds << ",\n"
              << "  \"stats_reverse_seconds\": " << stats.reverse_seconds << ",\n"
              << "  \"stats_final_prune_seconds\": " << stats.final_prune_seconds << ",\n"
              << "  \"stats_d2h_seconds\": " << stats.d2h_seconds << ",\n"
              << "  \"selective_active_nodes\": " << stats.selective_active_nodes << ",\n"
              << "  \"medoid\": " << medoid << ",\n"
              << "  \"edge_count\": " << edge_count << ",\n"
              << "  \"avg_degree\": " << (double)edge_count / (double)npts << ",\n"
              << "  \"max_degree\": " << max_degree << ",\n"
              << "  \"invalid_neighbor_count\": " << invalid_neighbor_count << ",\n"
              << "  \"self_loop_count\": " << self_loop_count << ",\n"
              << "  \"duplicate_edge_count\": " << duplicate_edge_count << ",\n"
              << "  \"degree_overflow_count\": " << degree_overflow_count << ",\n"
              << "  \"graph_valid\": " << (graph_valid ? "true" : "false") << ",\n"
              << "  \"return_code\": " << ret << "\n"
              << "}\n";
    return graph_valid ? 0 : (ret != 0 ? ret : 3);
}
}

int main(int argc, char **argv)
{
    try
    {
        Args args = parse_args(argc, argv);
        const std::string executable = argc > 0 ? argv[0] : "";
        const size_t separator = executable.find_last_of("/\\");
        const std::string executable_name =
            separator == std::string::npos ? executable : executable.substr(separator + 1);
        if (executable_name == "gpu_vamana_memory_index" && args.index_prefix.empty())
            throw std::runtime_error("gpu_vamana_memory_index requires --index_prefix");
        if (executable_name == "gpu_vamana_memory_index" && args.builder != "vnew2")
            throw std::runtime_error("gpu_vamana_memory_index requires --builder vnew2");
        if (args.data_type == "uint8")
            return run_typed<uint8_t>(args);
        return run_typed<float>(args);
    }
    catch (const std::exception &e)
    {
        usage(argv[0]);
        std::cerr << "error: " << e.what() << std::endl;
        return 2;
    }
}
