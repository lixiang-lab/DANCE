// Copyright (c) 2026 DANCE Authors. Licensed under the MIT License.



#include "gpu_merge.h"

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace diskann
{
namespace
{

constexpr size_t METADATA_SIZE = sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint64_t);
constexpr uint64_t RESERVED_GPU_BYTES = 4ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr uint64_t MIN_BUCKET_TARGET_EDGES = 1ULL << 20;
constexpr uint64_t MAX_BUCKET_TARGET_EDGES = 128ULL << 20;
constexpr uint64_t MAX_BUCKET_COUNT = 65536;
constexpr uint32_t MAX_TOPR_DEGREE = 256;

struct ShardInfo
{
    std::string graph_path;
    std::string idmap_path;
    std::vector<uint32_t> idmap;
    uint32_t width = 0;
    uint32_t medoid = 0;
    uint64_t frozen = 0;
    uint64_t edges = 0;
};

size_t file_size(const std::string &path)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
        return 0;
    return (size_t)in.tellg();
}

bool read_u32_bin_1d(const std::string &path, std::vector<uint32_t> &out)
{
    const size_t actual_size = file_size(path);
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return false;
    uint32_t n = 0, dim = 0;
    in.read(reinterpret_cast<char *>(&n), sizeof(uint32_t));
    in.read(reinterpret_cast<char *>(&dim), sizeof(uint32_t));
    if (!in || dim != 1 || actual_size != 2 * sizeof(uint32_t) + (size_t)n * sizeof(uint32_t))
        return false;
    out.resize(n);
    if (n > 0)
        in.read(reinterpret_cast<char *>(out.data()), (size_t)n * sizeof(uint32_t));
    return (bool)in;
}

bool read_graph_header(std::ifstream &in, uint32_t &width, uint32_t &medoid, uint64_t &frozen)
{
    uint64_t expected_size = 0;
    in.read(reinterpret_cast<char *>(&expected_size), sizeof(uint64_t));
    in.read(reinterpret_cast<char *>(&width), sizeof(uint32_t));
    in.read(reinterpret_cast<char *>(&medoid), sizeof(uint32_t));
    in.read(reinterpret_cast<char *>(&frozen), sizeof(uint64_t));
    return (bool)in;
}

uint64_t pack_edge(uint32_t src, uint32_t dst)
{
    return (uint64_t(src) << 32) | uint64_t(dst);
}

__device__ __host__ uint32_t hash_edge32(uint32_t src, uint32_t dst, uint64_t seed)
{
    uint64_t x = (uint64_t(src) << 32) ^ uint64_t(dst) ^ seed;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (uint32_t)(x ^ (x >> 32));
}

__global__ void build_rank_keys_kernel(const uint64_t *unique_edges, uint64_t *rank_keys, uint32_t *dst_vals,
                                       size_t n, uint64_t seed)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const uint64_t edge = unique_edges[i];
    const uint32_t src = (uint32_t)(edge >> 32);
    const uint32_t dst = (uint32_t)(edge & 0xffffffffULL);
    rank_keys[i] = (uint64_t(src) << 32) | uint64_t(hash_edge32(src, dst, seed));
    dst_vals[i] = dst;
}

__global__ void mark_src_segments_kernel(const uint64_t *rank_keys, uint32_t *segment_flags, size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const uint32_t src = (uint32_t)(rank_keys[i] >> 32);
    const uint32_t prev_src = i == 0 ? UINT32_MAX : (uint32_t)(rank_keys[i - 1] >> 32);
    segment_flags[i] = (i == 0 || src != prev_src) ? 1u : 0u;
}

__global__ void scatter_segment_info_kernel(const uint64_t *rank_keys, const uint32_t *segment_flags,
                                            const uint32_t *segment_ids, uint32_t *segment_srcs,
                                            uint32_t *segment_starts, size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || segment_flags[i] == 0)
        return;
    const uint32_t segment = segment_ids[i] - 1;
    segment_srcs[segment] = (uint32_t)(rank_keys[i] >> 32);
    segment_starts[segment] = (uint32_t)i;
}

__global__ void truncate_segments_kernel(const uint32_t *segment_srcs, const uint32_t *segment_starts,
                                         uint32_t num_segments, const uint32_t *dst_vals, uint32_t bucket_begin,
                                         uint32_t bucket_nodes, uint32_t max_degree, size_t unique_count,
                                         uint32_t *out_degrees, uint32_t *out_neighbors)
{
    const uint32_t segment = blockIdx.x;
    if (segment >= num_segments)
        return;
    const uint32_t src = segment_srcs[segment];
    if (src < bucket_begin)
        return;
    const uint32_t src_local = src - bucket_begin;
    if (src_local >= bucket_nodes)
        return;
    const uint32_t start = segment_starts[segment];
    const uint32_t end = (segment + 1 < num_segments) ? segment_starts[segment + 1] : (uint32_t)unique_count;
    const uint32_t length = end - start;
    const uint32_t keep = length < max_degree ? length : max_degree;
    if (threadIdx.x == 0)
        out_degrees[src_local] = keep;
    for (uint32_t j = threadIdx.x; j < keep; j += blockDim.x)
        out_neighbors[(size_t)src_local * max_degree + j] = dst_vals[start + j];
}

__global__ void top_r_hash_select_segments_kernel(const uint64_t *unique_edges, const uint32_t *segment_srcs,
                                                  const uint32_t *segment_starts, uint32_t num_segments,
                                                  uint32_t bucket_begin, uint32_t bucket_nodes, uint32_t max_degree,
                                                  size_t unique_count, uint64_t seed, uint32_t *out_degrees,
                                                  uint32_t *out_neighbors)
{
    const uint32_t segment = blockIdx.x;
    if (segment >= num_segments || threadIdx.x != 0)
        return;

    const uint32_t src = segment_srcs[segment];
    if (src < bucket_begin)
        return;
    const uint32_t src_local = src - bucket_begin;
    if (src_local >= bucket_nodes)
        return;

    const uint32_t start = segment_starts[segment];
    const uint32_t end = (segment + 1 < num_segments) ? segment_starts[segment + 1] : (uint32_t)unique_count;
    const uint32_t count = end - start;
    const uint32_t keep = count < max_degree ? count : max_degree;
    out_degrees[src_local] = keep;

    if (count <= max_degree)
    {
        for (uint32_t j = 0; j < keep; j++)
            out_neighbors[(size_t)src_local * max_degree + j] = (uint32_t)(unique_edges[start + j] & 0xffffffffULL);
        return;
    }

    uint32_t best_dst[MAX_TOPR_DEGREE];
    uint32_t best_hash[MAX_TOPR_DEGREE];
    uint32_t selected = 0;
    for (uint32_t i = start; i < end; i++)
    {
        const uint32_t dst = (uint32_t)(unique_edges[i] & 0xffffffffULL);
        const uint32_t h = hash_edge32(src, dst, seed);
        if (selected < max_degree)
        {
            best_dst[selected] = dst;
            best_hash[selected] = h;
            selected++;
            continue;
        }

        uint32_t worst = 0;
        for (uint32_t j = 1; j < max_degree; j++)
        {
            if (best_hash[j] > best_hash[worst] ||
                (best_hash[j] == best_hash[worst] && best_dst[j] > best_dst[worst]))
                worst = j;
        }
        if (h < best_hash[worst] || (h == best_hash[worst] && dst < best_dst[worst]))
        {
            best_hash[worst] = h;
            best_dst[worst] = dst;
        }
    }

    for (uint32_t i = 1; i < selected; i++)
    {
        uint32_t dst = best_dst[i];
        uint32_t h = best_hash[i];
        int j = (int)i - 1;
        while (j >= 0 && (best_hash[j] > h || (best_hash[j] == h && best_dst[j] > dst)))
        {
            best_hash[j + 1] = best_hash[j];
            best_dst[j + 1] = best_dst[j];
            j--;
        }
        best_hash[j + 1] = h;
        best_dst[j + 1] = dst;
    }

    for (uint32_t j = 0; j < keep; j++)
        out_neighbors[(size_t)src_local * max_degree + j] = best_dst[j];
}

std::vector<int> parse_devices()
{
    std::vector<int> devices;
    const char *env = std::getenv("DISKANN_GPU_MERGE_DEVICES");
    if (env != nullptr && env[0] != '\0')
    {
        std::stringstream ss(env);
        std::string tok;
        while (std::getline(ss, tok, ','))
        {
            if (!tok.empty())
                devices.push_back(std::atoi(tok.c_str()));
        }
    }
    if (devices.empty())
    {
        int current = 0;
        cudaGetDevice(&current);
        devices.push_back(current);
    }
    return devices;
}

uint64_t choose_bucket_target_edges(const std::vector<int> &devices)
{
    uint64_t best_target = MAX_BUCKET_TARGET_EDGES;
    bool saw_valid_device = false;
    for (int device : devices)
    {
        if (cudaSetDevice(device) != cudaSuccess)
            continue;
        size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
            continue;
        uint64_t usable = 0;
        if (free_bytes > RESERVED_GPU_BYTES)
            usable = (uint64_t)free_bytes - RESERVED_GPU_BYTES;
        uint64_t target = usable / (sizeof(uint64_t) * 4);
        target = std::max<uint64_t>(target, MIN_BUCKET_TARGET_EDGES);
        target = std::min<uint64_t>(target, MAX_BUCKET_TARGET_EDGES);
        best_target = std::min(best_target, target);
        saw_valid_device = true;
        std::cout << "[GPU merge] device=" << device << " free_gb="
                  << (double)free_bytes / (1024.0 * 1024.0 * 1024.0)
                  << " target_edges=" << target << std::endl;
    }
    if (!saw_valid_device)
        best_target = MIN_BUCKET_TARGET_EDGES;
    const char *override_env = std::getenv("DISKANN_GPU_MERGE_BUCKET_EDGES");
    if (override_env != nullptr && override_env[0] != '\0')
        best_target = std::max<uint64_t>(1, std::strtoull(override_env, nullptr, 10));
    return best_target;
}

uint64_t bucket_for_src(uint32_t src, uint64_t nnodes, uint64_t bucket_count)
{
    if (nnodes == 0)
        return 0;
    uint64_t b = (uint64_t(src) * bucket_count) / nnodes;
    return std::min<uint64_t>(b, bucket_count - 1);
}

uint32_t bucket_start(uint64_t bucket, uint64_t nnodes, uint64_t bucket_count)
{
    return (uint32_t)((bucket * nnodes + bucket_count - 1) / bucket_count);
}

uint32_t bucket_end(uint64_t bucket, uint64_t nnodes, uint64_t bucket_count)
{
    return (uint32_t)(((bucket + 1) * nnodes + bucket_count - 1) / bucket_count);
}

bool append_edges(const std::string &path, const std::vector<uint64_t> &edges)
{
    if (edges.empty())
        return true;
    std::ofstream out(path, std::ios::binary | std::ios::app);
    if (!out)
        return false;
    out.write(reinterpret_cast<const char *>(edges.data()), edges.size() * sizeof(uint64_t));
    return (bool)out;
}

bool read_bucket_edges(const std::string &path, std::vector<uint64_t> &edges)
{
    const size_t bytes = file_size(path);
    if (bytes % sizeof(uint64_t) != 0)
        return false;
    edges.resize(bytes / sizeof(uint64_t));
    if (edges.empty())
        return true;
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return false;
    in.read(reinterpret_cast<char *>(edges.data()), bytes);
    return (bool)in;
}

double seconds_since(const std::chrono::steady_clock::time_point &start,
                     const std::chrono::steady_clock::time_point &end)
{
    return std::chrono::duration<double>(end - start).count();
}

double avg(uint64_t sum, uint64_t n)
{
    return n == 0 ? 0.0 : (double)sum / (double)n;
}

}

int merge_shards_gpu(const std::string &vamana_prefix, const std::string &vamana_suffix,
                     const std::string &idmaps_prefix, const std::string &idmaps_suffix, uint64_t nshards,
                     uint32_t max_degree, const std::string &output_vamana, const std::string &medoids_file)
{
    const auto merge_start = std::chrono::steady_clock::now();
    std::cout << "[GPU merge] enabled" << std::endl;

    const std::vector<int> devices = parse_devices();
    if (devices.empty())
    {
        std::cerr << "[GPU merge] no CUDA devices selected" << std::endl;
        return -1;
    }

    std::vector<ShardInfo> shards(nshards);
    uint64_t nnodes = 0;
    uint64_t replicated_nodes = 0;
    uint64_t estimated_input_edges = 0;
    uint64_t total_input_edges = 0;
    double idmap_read_seconds = 0.0;
    double shard_header_read_seconds = 0.0;
    double medoid_write_seconds = 0.0;
    double bucket_file_prepare_seconds = 0.0;
    double pass1_edge_expand_seconds = 0.0;
    double pass1_bucket_flush_seconds = 0.0;
    double pass2_bucket_file_read_seconds = 0.0;
    double pass2_device_alloc_seconds = 0.0;
    double pass2_h2d_seconds = 0.0;
    double pass2_sort_unique_seconds = 0.0;
    double pass2_group_truncate_seconds = 0.0;
    double pass2_compact_d2h_seconds = 0.0;
    double pass2_output_pack_seconds = 0.0;
    double merged_index_header_write_seconds = 0.0;
    double merged_index_body_write_seconds = 0.0;
    double cleanup_seconds = 0.0;

    for (uint64_t shard = 0; shard < nshards; shard++)
    {
        auto &s = shards[shard];
        s.graph_path = vamana_prefix + std::to_string(shard) + vamana_suffix;
        s.idmap_path = idmaps_prefix + std::to_string(shard) + idmaps_suffix;
        const auto idmap_start = std::chrono::steady_clock::now();
        if (!read_u32_bin_1d(s.idmap_path, s.idmap))
        {
            std::cerr << "[GPU merge] failed to read idmap: " << s.idmap_path << std::endl;
            return -2;
        }
        const auto idmap_end = std::chrono::steady_clock::now();
        idmap_read_seconds += seconds_since(idmap_start, idmap_end);
        replicated_nodes += s.idmap.size();
        for (uint32_t id : s.idmap)
            nnodes = std::max<uint64_t>(nnodes, (uint64_t)id + 1);

        const auto header_start = std::chrono::steady_clock::now();
        std::ifstream in(s.graph_path, std::ios::binary);
        if (!read_graph_header(in, s.width, s.medoid, s.frozen) || s.frozen != 0)
        {
            std::cerr << "[GPU merge] invalid graph header: " << s.graph_path << std::endl;
            return -3;
        }
        if (s.medoid >= s.idmap.size())
        {
            std::cerr << "[GPU merge] shard medoid out of range: " << s.graph_path << std::endl;
            return -4;
        }
        estimated_input_edges += (uint64_t)s.idmap.size() * (uint64_t)s.width;
        const auto header_end = std::chrono::steady_clock::now();
        shard_header_read_seconds += seconds_since(header_start, header_end);
    }

    const uint64_t bucket_target_edges = choose_bucket_target_edges(devices);
    uint64_t bucket_count =
        std::max<uint64_t>(1, (estimated_input_edges + bucket_target_edges - 1) / bucket_target_edges);
    const char *bucket_count_env = std::getenv("DISKANN_GPU_MERGE_BUCKETS");
    if (bucket_count_env != nullptr && bucket_count_env[0] != '\0')
        bucket_count = std::max<uint64_t>(1, std::strtoull(bucket_count_env, nullptr, 10));
    bucket_count = std::min<uint64_t>(bucket_count, std::max<uint64_t>(1, nnodes));
    bucket_count = std::min<uint64_t>(bucket_count, MAX_BUCKET_COUNT);

    const auto bucket_prepare_start = std::chrono::steady_clock::now();
    std::filesystem::path tmp_dir = output_vamana + ".gpu_merge_buckets";
    bool memory_bucket_mode = false;
    const char *bucket_mode_env = std::getenv("DISKANN_GPU_MERGE_BUCKET_MODE");
    if (bucket_mode_env != nullptr && std::string(bucket_mode_env) == "memory")
    {
        const char *budget_env = std::getenv("DISKANN_GPU_MERGE_HOST_MEM_BUDGET_GB");
        const double budget_gb = budget_env == nullptr ? 0.0 : std::atof(budget_env);
        const double estimated_edge_gb =
            (double)estimated_input_edges * (double)sizeof(uint64_t) / (1024.0 * 1024.0 * 1024.0);
        if (budget_gb > 0.0 && estimated_edge_gb < budget_gb)
        {
            memory_bucket_mode = true;
            std::cout << "[GPU merge] bucket_mode=memory estimated_edge_gb=" << estimated_edge_gb
                      << " budget_gb=" << budget_gb << std::endl;
        }
        else
        {
            std::cout << "[GPU merge] warning: bucket_mode=memory requested but estimated_edge_gb="
                      << estimated_edge_gb << " budget_gb=" << budget_gb << "; falling back to file mode"
                      << std::endl;
        }
    }
    if (!memory_bucket_mode)
    {
        std::filesystem::remove_all(tmp_dir);
        std::filesystem::create_directories(tmp_dir);
    }

    std::vector<std::string> bucket_paths(bucket_count);
    std::vector<uint64_t> bucket_edges(bucket_count, 0);
    if (!memory_bucket_mode)
    {
        for (uint64_t b = 0; b < bucket_count; b++)
            bucket_paths[b] = (tmp_dir / ("bucket_" + std::to_string(b) + ".edges")).string();
    }
    const auto bucket_prepare_end = std::chrono::steady_clock::now();
    bucket_file_prepare_seconds += seconds_since(bucket_prepare_start, bucket_prepare_end);

    std::cout << "[GPU merge] nshards=" << nshards << " nnodes=" << nnodes
              << " replicated_nodes=" << replicated_nodes << " estimated_input_edges=" << estimated_input_edges
              << " bucket_count=" << bucket_count << " bucket_target_edges=" << bucket_target_edges << std::endl;

    const auto pass1_start = std::chrono::steady_clock::now();
    const auto pass1_expand_start = std::chrono::steady_clock::now();
    uint64_t self_loops = 0;
    constexpr size_t FLUSH_EDGES = 1 << 20;
    std::vector<std::vector<uint64_t>> buffers(bucket_count);

    for (uint64_t shard = 0; shard < nshards; shard++)
    {
        const auto &s = shards[shard];
        std::ifstream in(s.graph_path, std::ios::binary);
        uint32_t width = 0, medoid = 0;
        uint64_t frozen = 0;
        read_graph_header(in, width, medoid, frozen);

        for (uint32_t local_src = 0; local_src < s.idmap.size(); local_src++)
        {
            const uint32_t src_global = s.idmap[local_src];
            const uint64_t bucket = bucket_for_src(src_global, nnodes, bucket_count);

            uint32_t degree = 0;
            in.read(reinterpret_cast<char *>(&degree), sizeof(uint32_t));
            total_input_edges += degree;
            std::vector<uint32_t> local_nbrs(degree);
            if (degree > 0)
                in.read(reinterpret_cast<char *>(local_nbrs.data()), (size_t)degree * sizeof(uint32_t));
            if (!in)
                return -6;

            auto &buf = buffers[bucket];
            for (uint32_t local_dst : local_nbrs)
            {
                if (local_dst >= s.idmap.size())
                {
                    std::cerr << "[GPU merge] local neighbor out of range in shard " << shard << std::endl;
                    return -7;
                }
                const uint32_t dst_global = s.idmap[local_dst];
                if (dst_global == src_global)
                {
                    self_loops++;
                    continue;
                }
                buf.push_back(pack_edge(src_global, dst_global));
                bucket_edges[bucket]++;
                if (!memory_bucket_mode && buf.size() >= FLUSH_EDGES)
                {
                    const auto flush_start = std::chrono::steady_clock::now();
                    if (!append_edges(bucket_paths[bucket], buf))
                        return -8;
                    const auto flush_end = std::chrono::steady_clock::now();
                    pass1_bucket_flush_seconds += seconds_since(flush_start, flush_end);
                    buf.clear();
                }
            }
        }
    }
    const auto pass1_expand_end = std::chrono::steady_clock::now();
    pass1_edge_expand_seconds += seconds_since(pass1_expand_start, pass1_expand_end);
    for (uint64_t b = 0; b < bucket_count; b++)
    {
        if (memory_bucket_mode)
            continue;
        const auto flush_start = std::chrono::steady_clock::now();
        if (!append_edges(bucket_paths[b], buffers[b]))
            return -9;
        const auto flush_end = std::chrono::steady_clock::now();
        pass1_bucket_flush_seconds += seconds_since(flush_start, flush_end);
        buffers[b].clear();
    }
    const auto pass1_end = std::chrono::steady_clock::now();

    uint64_t max_bucket_edges = 0;
    for (uint64_t e : bucket_edges)
        max_bucket_edges = std::max(max_bucket_edges, e);
    std::cout << "[GPU merge] pass1_seconds=" << seconds_since(pass1_start, pass1_end)
              << " actual_input_edges=" << total_input_edges << " self_loops=" << self_loops
              << " max_bucket_edges=" << max_bucket_edges << std::endl;

    const auto medoid_start = std::chrono::steady_clock::now();
    std::ofstream medoid_writer(medoids_file, std::ios::binary);
    if (!medoid_writer)
        return -10;
    uint32_t nshards_u32 = (uint32_t)nshards;
    uint32_t one_val = 1;
    medoid_writer.write(reinterpret_cast<char *>(&nshards_u32), sizeof(uint32_t));
    medoid_writer.write(reinterpret_cast<char *>(&one_val), sizeof(uint32_t));
    uint32_t output_medoid = 0;
    for (uint64_t shard = 0; shard < nshards; shard++)
    {
        uint32_t medoid_global = shards[shard].idmap[shards[shard].medoid];
        medoid_writer.write(reinterpret_cast<char *>(&medoid_global), sizeof(uint32_t));
        if (shard == nshards - 1)
            output_medoid = medoid_global;
    }
    medoid_writer.close();
    const auto medoid_end = std::chrono::steady_clock::now();
    medoid_write_seconds += seconds_since(medoid_start, medoid_end);

    const auto header_write_start = std::chrono::steady_clock::now();
    std::ofstream merged_writer(output_vamana, std::ios::binary);
    if (!merged_writer)
        return -11;
    uint64_t merged_index_size = METADATA_SIZE;
    uint32_t output_width = max_degree;
    uint64_t merged_frozen = 0;
    merged_writer.write(reinterpret_cast<char *>(&merged_index_size), sizeof(uint64_t));
    merged_writer.write(reinterpret_cast<char *>(&output_width), sizeof(uint32_t));
    merged_writer.write(reinterpret_cast<char *>(&output_medoid), sizeof(uint32_t));
    merged_writer.write(reinterpret_cast<char *>(&merged_frozen), sizeof(uint64_t));
    const auto header_write_end = std::chrono::steady_clock::now();
    merged_index_header_write_seconds += seconds_since(header_write_start, header_write_end);

    const auto pass2_start = std::chrono::steady_clock::now();
    const uint64_t seed = [] {
        const char *env = std::getenv("DISKANN_GPU_MERGE_HASH_SEED");
        return env ? std::strtoull(env, nullptr, 10) : 0x9e3779b97f4a7c15ULL;
    }();

    uint64_t total_unique_edges = 0;
    uint64_t output_edges = 0;
    uint64_t truncated_edges = 0;
    std::vector<uint64_t> degree_hist(max_degree + 1, 0);
    uint32_t output_min_degree = nnodes ? std::numeric_limits<uint32_t>::max() : 0;
    uint32_t output_max_degree = 0;
    constexpr bool rank_sort_pass2 = true;
    std::cout << "[GPU merge] rank-sort GPU pass2 enabled" << std::endl;

    uint32_t max_bucket_nodes = 0;
    for (uint64_t b = 0; b < bucket_count; b++)
        max_bucket_nodes = std::max<uint32_t>(max_bucket_nodes, bucket_end(b, nnodes, bucket_count) -
                                                                    bucket_start(b, nnodes, bucket_count));

    thrust::device_vector<uint64_t> d_edges_buf;
    thrust::device_vector<uint64_t> d_rank_keys_buf;
    thrust::device_vector<uint32_t> d_dst_vals_buf;
    thrust::device_vector<uint32_t> d_segment_flags_buf;
    thrust::device_vector<uint32_t> d_segment_ids_buf;
    thrust::device_vector<uint32_t> d_segment_srcs_buf;
    thrust::device_vector<uint32_t> d_segment_starts_buf;
    thrust::device_vector<uint32_t> d_out_degrees_buf;
    thrust::device_vector<uint32_t> d_out_neighbors_buf;
        const auto alloc_start = std::chrono::steady_clock::now();
        d_edges_buf.resize(max_bucket_edges);
        d_segment_flags_buf.resize(max_bucket_edges);
        d_segment_ids_buf.resize(max_bucket_edges);
        d_segment_srcs_buf.resize(max_bucket_nodes);
        d_segment_starts_buf.resize(max_bucket_nodes);
        d_out_degrees_buf.resize(max_bucket_nodes);
        d_out_neighbors_buf.resize((size_t)max_bucket_nodes * max_degree);
        if (rank_sort_pass2)
        {
            d_rank_keys_buf.resize(max_bucket_edges);
            d_dst_vals_buf.resize(max_bucket_edges);
        }
        d_edges_buf.resize(0);
        d_segment_flags_buf.resize(0);
        d_segment_ids_buf.resize(0);
        d_segment_srcs_buf.resize(0);
        d_segment_starts_buf.resize(0);
        d_out_degrees_buf.resize(0);
        d_out_neighbors_buf.resize(0);
        if (rank_sort_pass2)
        {
            d_rank_keys_buf.resize(0);
            d_dst_vals_buf.resize(0);
        }
        cudaDeviceSynchronize();
        const auto alloc_end = std::chrono::steady_clock::now();
        pass2_device_alloc_seconds += seconds_since(alloc_start, alloc_end);

    for (uint64_t b = 0; b < bucket_count; b++)
    {
        const int device = devices[b % devices.size()];
        if (cudaSetDevice(device) != cudaSuccess)
        {
            std::cerr << "[GPU merge] cudaSetDevice failed for device " << device << std::endl;
            return -12;
        }

        std::vector<uint64_t> edges;
        if (memory_bucket_mode)
        {
            edges.swap(buffers[b]);
        }
        else
        {
            const auto bucket_read_start = std::chrono::steady_clock::now();
            if (!read_bucket_edges(bucket_paths[b], edges))
                return -13;
            const auto bucket_read_end = std::chrono::steady_clock::now();
            pass2_bucket_file_read_seconds += seconds_since(bucket_read_start, bucket_read_end);
        }
        const uint32_t start_node = bucket_start(b, nnodes, bucket_count);
        const uint32_t end_node = bucket_end(b, nnodes, bucket_count);
        const uint32_t bucket_nodes = end_node - start_node;

        std::vector<uint32_t> out_degrees(bucket_nodes, 0);
            std::vector<uint32_t> out_neighbors((size_t)bucket_nodes * max_degree, 0);
            size_t unique_count = 0;

            if (!edges.empty())
            {
                const auto h2d_start = std::chrono::steady_clock::now();
                d_edges_buf.resize(edges.size());
                thrust::copy(edges.begin(), edges.end(), d_edges_buf.begin());
                cudaDeviceSynchronize();
                const auto h2d_end = std::chrono::steady_clock::now();
                pass2_h2d_seconds += seconds_since(h2d_start, h2d_end);

                const auto sort_start = std::chrono::steady_clock::now();
                thrust::sort(d_edges_buf.begin(), d_edges_buf.end());
                auto unique_end = thrust::unique(d_edges_buf.begin(), d_edges_buf.end());
                unique_count = (size_t)(unique_end - d_edges_buf.begin());
                d_edges_buf.resize(unique_count);
                cudaDeviceSynchronize();
                const auto sort_end = std::chrono::steady_clock::now();
                pass2_sort_unique_seconds += seconds_since(sort_start, sort_end);
                total_unique_edges += unique_count;

                if (unique_count > 0)
                {
                    const auto group_start = std::chrono::steady_clock::now();
                    d_segment_flags_buf.resize(unique_count);
                    d_segment_ids_buf.resize(unique_count);
                    d_out_degrees_buf.resize(bucket_nodes);
                    d_out_neighbors_buf.resize((size_t)bucket_nodes * max_degree);
                    thrust::fill(d_out_degrees_buf.begin(), d_out_degrees_buf.end(), 0);

                    constexpr uint32_t THREADS = 256;
                    const uint32_t blocks_edges = (uint32_t)((unique_count + THREADS - 1) / THREADS);
                    if (rank_sort_pass2)
                    {
                        d_rank_keys_buf.resize(unique_count);
                        d_dst_vals_buf.resize(unique_count);
                        build_rank_keys_kernel<<<blocks_edges, THREADS>>>(thrust::raw_pointer_cast(d_edges_buf.data()),
                                                                          thrust::raw_pointer_cast(d_rank_keys_buf.data()),
                                                                          thrust::raw_pointer_cast(d_dst_vals_buf.data()),
                                                                          unique_count, seed);
                        if (cudaDeviceSynchronize() != cudaSuccess)
                            return -15;

                        thrust::sort_by_key(d_rank_keys_buf.begin(), d_rank_keys_buf.end(), d_dst_vals_buf.begin());
                        mark_src_segments_kernel<<<blocks_edges, THREADS>>>(
                            thrust::raw_pointer_cast(d_rank_keys_buf.data()),
                            thrust::raw_pointer_cast(d_segment_flags_buf.data()), unique_count);
                    }
                    else
                    {
                        mark_src_segments_kernel<<<blocks_edges, THREADS>>>(
                            thrust::raw_pointer_cast(d_edges_buf.data()),
                            thrust::raw_pointer_cast(d_segment_flags_buf.data()), unique_count);
                    }
                    if (cudaDeviceSynchronize() != cudaSuccess)
                        return -16;

                    thrust::inclusive_scan(d_segment_flags_buf.begin(), d_segment_flags_buf.end(),
                                           d_segment_ids_buf.begin());
                    uint32_t num_segments = 0;
                    thrust::copy_n(d_segment_ids_buf.begin() + (unique_count - 1), 1, &num_segments);

                    d_segment_srcs_buf.resize(num_segments);
                    d_segment_starts_buf.resize(num_segments);
                    if (rank_sort_pass2)
                    {
                        scatter_segment_info_kernel<<<blocks_edges, THREADS>>>(
                            thrust::raw_pointer_cast(d_rank_keys_buf.data()),
                            thrust::raw_pointer_cast(d_segment_flags_buf.data()),
                            thrust::raw_pointer_cast(d_segment_ids_buf.data()),
                            thrust::raw_pointer_cast(d_segment_srcs_buf.data()),
                            thrust::raw_pointer_cast(d_segment_starts_buf.data()), unique_count);
                    }
                    else
                    {
                        scatter_segment_info_kernel<<<blocks_edges, THREADS>>>(
                            thrust::raw_pointer_cast(d_edges_buf.data()),
                            thrust::raw_pointer_cast(d_segment_flags_buf.data()),
                            thrust::raw_pointer_cast(d_segment_ids_buf.data()),
                            thrust::raw_pointer_cast(d_segment_srcs_buf.data()),
                            thrust::raw_pointer_cast(d_segment_starts_buf.data()), unique_count);
                    }
                    if (cudaDeviceSynchronize() != cudaSuccess)
                        return -17;

                    if (rank_sort_pass2)
                    {
                        truncate_segments_kernel<<<num_segments, THREADS>>>(
                            thrust::raw_pointer_cast(d_segment_srcs_buf.data()),
                            thrust::raw_pointer_cast(d_segment_starts_buf.data()), num_segments,
                            thrust::raw_pointer_cast(d_dst_vals_buf.data()), start_node, bucket_nodes, max_degree,
                            unique_count, thrust::raw_pointer_cast(d_out_degrees_buf.data()),
                            thrust::raw_pointer_cast(d_out_neighbors_buf.data()));
                    }
                    else
                    {
                        top_r_hash_select_segments_kernel<<<num_segments, 1>>>(
                            thrust::raw_pointer_cast(d_edges_buf.data()),
                            thrust::raw_pointer_cast(d_segment_srcs_buf.data()),
                            thrust::raw_pointer_cast(d_segment_starts_buf.data()), num_segments, start_node,
                            bucket_nodes, max_degree, unique_count, seed,
                            thrust::raw_pointer_cast(d_out_degrees_buf.data()),
                            thrust::raw_pointer_cast(d_out_neighbors_buf.data()));
                    }
                    if (cudaDeviceSynchronize() != cudaSuccess)
                        return -18;
                    const auto group_end = std::chrono::steady_clock::now();
                    pass2_group_truncate_seconds += seconds_since(group_start, group_end);

                    const auto d2h_start = std::chrono::steady_clock::now();
                    thrust::copy(d_out_degrees_buf.begin(), d_out_degrees_buf.end(), out_degrees.begin());
                    thrust::copy(d_out_neighbors_buf.begin(), d_out_neighbors_buf.end(), out_neighbors.begin());
                    cudaDeviceSynchronize();
                    const auto d2h_end = std::chrono::steady_clock::now();
                    pass2_compact_d2h_seconds += seconds_since(d2h_start, d2h_end);
                }
            }

            const auto pack_start = std::chrono::steady_clock::now();
            std::vector<char> bucket_output;
            bucket_output.reserve((size_t)bucket_nodes * (sizeof(uint32_t) + (size_t)max_degree * sizeof(uint32_t)));
            for (uint32_t local = 0; local < bucket_nodes; local++)
            {
                uint32_t degree = out_degrees[local];
                if (degree > max_degree)
                    return -19;
                const char *degree_ptr = reinterpret_cast<const char *>(&degree);
                bucket_output.insert(bucket_output.end(), degree_ptr, degree_ptr + sizeof(uint32_t));
                if (degree > 0)
                {
                    const char *nbr_ptr =
                        reinterpret_cast<const char *>(out_neighbors.data() + (size_t)local * max_degree);
                    bucket_output.insert(bucket_output.end(), nbr_ptr, nbr_ptr + (size_t)degree * sizeof(uint32_t));
                }
                merged_index_size += sizeof(uint32_t) + (uint64_t)degree * sizeof(uint32_t);
                output_edges += degree;
                output_min_degree = std::min(output_min_degree, degree);
                output_max_degree = std::max(output_max_degree, degree);
                if (degree <= max_degree)
                    degree_hist[degree]++;
            }
            const auto pack_end = std::chrono::steady_clock::now();
            pass2_output_pack_seconds += seconds_since(pack_start, pack_end);
            merged_writer.write(bucket_output.data(), bucket_output.size());
            const auto write_end = std::chrono::steady_clock::now();
            merged_index_body_write_seconds += seconds_since(pack_end, write_end);
        if (!memory_bucket_mode)
        {
            const auto bucket_cleanup_start = std::chrono::steady_clock::now();
            std::remove(bucket_paths[b].c_str());
            const auto bucket_cleanup_end = std::chrono::steady_clock::now();
            cleanup_seconds += seconds_since(bucket_cleanup_start, bucket_cleanup_end);
        }
    }
    truncated_edges = total_unique_edges - output_edges;
    if (nnodes == 0)
        output_min_degree = 0;
    const auto pass2_end = std::chrono::steady_clock::now();

    const auto final_header_start = std::chrono::steady_clock::now();
    merged_writer.seekp(0, std::ios::beg);
    merged_writer.write(reinterpret_cast<char *>(&merged_index_size), sizeof(uint64_t));
    merged_writer.close();
    const auto final_header_end = std::chrono::steady_clock::now();
    merged_index_header_write_seconds += seconds_since(final_header_start, final_header_end);

    const auto final_cleanup_start = std::chrono::steady_clock::now();
    if (!memory_bucket_mode)
        std::filesystem::remove_all(tmp_dir);
    const auto final_cleanup_end = std::chrono::steady_clock::now();
    cleanup_seconds += seconds_since(final_cleanup_start, final_cleanup_end);

    const auto merge_end = std::chrono::steady_clock::now();
    std::cout << "[GPU merge] unique_edges=" << total_unique_edges << " output_edges=" << output_edges
              << " truncated_edges=" << truncated_edges << " self_loops=" << self_loops << std::endl;
    std::cout << "[GPU merge] degree min=" << output_min_degree << " avg=" << avg(output_edges, nnodes)
              << " max=" << output_max_degree << std::endl;
    std::cout << "[GPU merge] degree_histogram";
    for (uint32_t d = 0; d <= max_degree; d++)
        std::cout << " d" << d << "=" << degree_hist[d];
    std::cout << std::endl;
    std::cout << "[GPU merge] idmap_read_seconds=" << idmap_read_seconds << std::endl;
    std::cout << "[GPU merge] shard_header_read_seconds=" << shard_header_read_seconds << std::endl;
    std::cout << "[GPU merge] medoid_write_seconds=" << medoid_write_seconds << std::endl;
    std::cout << "[GPU merge] bucket_file_prepare_seconds=" << bucket_file_prepare_seconds << std::endl;
    std::cout << "[GPU merge] pass1_edge_expand_seconds=" << pass1_edge_expand_seconds << std::endl;
    std::cout << "[GPU merge] pass1_bucket_flush_seconds=" << pass1_bucket_flush_seconds << std::endl;
    std::cout << "[GPU merge] pass2_bucket_file_read_seconds=" << pass2_bucket_file_read_seconds << std::endl;
    std::cout << "[GPU merge] pass2_device_alloc_seconds=" << pass2_device_alloc_seconds << std::endl;
    std::cout << "[GPU merge] pass2_h2d_seconds=" << pass2_h2d_seconds << std::endl;
    std::cout << "[GPU merge] pass2_sort_unique_seconds=" << pass2_sort_unique_seconds << std::endl;
    std::cout << "[GPU merge] pass2_group_truncate_seconds=" << pass2_group_truncate_seconds << std::endl;
    std::cout << "[GPU merge] pass2_compact_d2h_seconds=" << pass2_compact_d2h_seconds << std::endl;
    std::cout << "[GPU merge] pass2_output_pack_seconds=" << pass2_output_pack_seconds << std::endl;
    std::cout << "[GPU merge] merged_index_header_write_seconds=" << merged_index_header_write_seconds << std::endl;
    std::cout << "[GPU merge] merged_index_body_write_seconds=" << merged_index_body_write_seconds << std::endl;
    std::cout << "[GPU merge] cleanup_seconds=" << cleanup_seconds << std::endl;
    std::cout << "[GPU merge] pass2_seconds=" << seconds_since(pass2_start, pass2_end)
              << " merge_seconds=" << seconds_since(merge_start, merge_end) << std::endl;
    return 0;
}

}
