// Copyright (c) 2026 DANCE Authors. Licensed under the MIT License.












#include <algorithm>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>

#include "../include/gpu_vamana_builder.h"
#include "../include/gpu_vamana_filtered.h"
#include "../include/filtered_label_coverage.cuh"

#ifndef INVALID_ID
#define INVALID_ID 0xFFFFFFFFu
#endif

#ifndef CUDA_CHECK_FILTERED
#define CUDA_CHECK_FILTERED(call)                                                                 \
    do                                                                                            \
    {                                                                                             \
        cudaError_t err__ = (call);                                                               \
        if (err__ != cudaSuccess)                                                                 \
        {                                                                                         \
            fprintf(stderr, "[gpu_vamana_filtered] CUDA error at %s:%d: %s\n", __FILE__,         \
                    __LINE__, cudaGetErrorString(err__));                                         \
            return -1;                                                                            \
        }                                                                                         \
    } while (0)
#endif

static GPUFilteredVamanaStats g_last_filtered_stats;
__device__ __constant__ uint32_t g_filtered_geometric_only_prune;
__device__ __constant__ uint32_t g_filtered_balance_per_label;
__device__ unsigned long long g_canonical_distance_counts[6];
__device__ __constant__ uint32_t g_canonical_profile;

enum
{
    GPU_FILTERED_STAGE_MINV1 = 0,
    GPU_FILTERED_STAGE_LABEL_PRUNE_ONLY = 1,
    GPU_FILTERED_STAGE_B1_TWO_POOL_ORDINARY_PRUNE = 2,
    GPU_FILTERED_STAGE_B2_TWO_POOL_RESERVE_ORDINARY_PRUNE = 3,
    GPU_FILTERED_STAGE_B3_TWO_POOL_WORK_LABEL_PRUNE = 4,
    GPU_FILTERED_STAGE_REFINE_VNEW2 = 5,
    GPU_FILTERED_STAGE_LABEL_SYNC = 6
};

static double now_sec_filtered()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void update_filtered_peak_gpu_memory()
{
    size_t free_mem = 0, total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess)
    {
        const double used_gib = (double)(total_mem - free_mem) /
                                (1024.0 * 1024.0 * 1024.0);
        g_last_filtered_stats.peak_gpu_mem_gb =
            std::max(g_last_filtered_stats.peak_gpu_mem_gb, used_gib);
    }
}

static bool canonical_host_has_label(uint32_t point, uint32_t label, const uint32_t *offsets,
                                     const uint32_t *labels)
{
    const uint32_t begin = offsets[point], end = offsets[point + 1];
    return std::binary_search(labels + begin, labels + end, label);
}

static void canonical_print_round_diagnostics(uint32_t N, uint32_t R, const uint32_t *graph,
                                              const uint32_t *degrees, const uint32_t *offsets,
                                              const uint32_t *labels, uint32_t universal_label)
{
    const double start = now_sec_filtered();
    std::vector<uint32_t> sorted_degree(degrees, degrees + N);
    std::sort(sorted_degree.begin(), sorted_degree.end());
    auto percentile = [&](double q) -> uint32_t {
        if (sorted_degree.empty())
            return 0;
        const size_t index = (size_t)(q * double(sorted_degree.size() - 1));
        return sorted_degree[index];
    };
    uint64_t degree_sum = 0, degree_at_R = 0, compatible_sum = 0, min_per_label_sum = 0;
    std::vector<uint32_t> compatible_degrees(N, 0), min_per_label_degrees(N, 0);
    for (uint32_t row = 0; row < N; ++row)
    {
        const uint32_t degree = std::min(degrees[row], R);
        degree_sum += degree;
        degree_at_R += degree == R;
        uint32_t compatible = 0;
        uint32_t min_per_label = R;
        bool has_specific_label = false;
        for (uint32_t p = offsets[row]; p < offsets[row + 1]; ++p)
        {
            const uint32_t label = labels[p];
            if (label == universal_label)
                continue;
            has_specific_label = true;
            uint32_t label_degree = 0;
            for (uint32_t i = 0; i < degree; ++i)
            {
                const uint32_t neighbor = graph[(size_t)row * R + i];
                label_degree += neighbor < N && canonical_host_has_label(neighbor, label, offsets, labels);
            }
            min_per_label = std::min(min_per_label, label_degree);
        }
        for (uint32_t i = 0; i < degree; ++i)
        {
            const uint32_t neighbor = graph[(size_t)row * R + i];
            bool shares = false;
            for (uint32_t p = offsets[row]; !shares && p < offsets[row + 1] && neighbor < N; ++p)
                shares = labels[p] == universal_label ||
                         canonical_host_has_label(neighbor, labels[p], offsets, labels);
            compatible += shares;
        }
        if (!has_specific_label)
            min_per_label = compatible;
        compatible_degrees[row] = compatible;
        min_per_label_degrees[row] = min_per_label;
        compatible_sum += compatible;
        min_per_label_sum += min_per_label;
    }
    std::sort(compatible_degrees.begin(), compatible_degrees.end());
    std::sort(min_per_label_degrees.begin(), min_per_label_degrees.end());
    auto vector_percentile = [](const std::vector<uint32_t> &values, double q) -> uint32_t {
        return values.empty() ? 0 : values[(size_t)(q * double(values.size() - 1))];
    };
    printf("[gpu_vamana_filtered_canonical_degree] avg=%.3f p50=%u p95=%u p99=%u degree_eq_R_frac=%.6f "
           "compatible_avg=%.3f compatible_p50=%u compatible_p95=%u compatible_p99=%u "
           "min_per_label_avg=%.3f min_per_label_p50=%u min_per_label_p95=%u min_per_label_p99=%u diagnostic_seconds=%.6f\n",
           N ? double(degree_sum) / N : 0.0, percentile(0.50), percentile(0.95), percentile(0.99),
           N ? double(degree_at_R) / N : 0.0, N ? double(compatible_sum) / N : 0.0,
           vector_percentile(compatible_degrees, 0.50), vector_percentile(compatible_degrees, 0.95),
           vector_percentile(compatible_degrees, 0.99), N ? double(min_per_label_sum) / N : 0.0,
           vector_percentile(min_per_label_degrees, 0.50), vector_percentile(min_per_label_degrees, 0.95),
           vector_percentile(min_per_label_degrees, 0.99), now_sec_filtered() - start);
}

static uint32_t env_u32_clamped(const char *name, uint32_t default_value, uint32_t min_value, uint32_t max_value)
{
    uint32_t value = default_value;
    const char *env = getenv(name);
    if (env && atoi(env) > 0)
        value = (uint32_t)atoi(env);
    if (value < min_value)
        value = min_value;
    if (value > max_value)
        value = max_value;
    return value;
}

static std::vector<uint32_t> parse_u32_csv_env(const char *name)
{
    std::vector<uint32_t> out;
    const char *env = getenv(name);
    if (!env || !*env)
        return out;
    const char *p = env;
    while (*p)
    {
        while (*p == ',' || *p == ' ' || *p == '\t')
            ++p;
        if (!*p)
            break;
        char *end = nullptr;
        unsigned long v = strtoul(p, &end, 10);
        if (end == p)
            break;
        out.push_back((uint32_t)v);
        p = end;
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

static double bytes_to_gib_filtered(size_t bytes)
{
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

__device__ __forceinline__ uint8_t filtered_source_to_bit(uint8_t src)
{
    if (src == 0 || src > 6)
        return 0;
    return (uint8_t)(1u << src);
}

__device__ __forceinline__ bool filtered_source_has(uint8_t src, uint8_t source_merge_mode, uint8_t source_id)
{
    if (source_merge_mode)
        return (src & filtered_source_to_bit(source_id)) != 0;
    return src == source_id;
}

__device__ __forceinline__ uint8_t filtered_source_merge(uint8_t old_src, uint8_t new_src, uint8_t source_merge_mode)
{
    if (!source_merge_mode)
        return old_src;
    return (uint8_t)(old_src | filtered_source_to_bit(new_src));
}

__device__ __forceinline__ float filtered_source_key_factor(uint8_t src, uint32_t source_priority_mode,
                                                            uint8_t source_merge_mode)
{
    if (source_priority_mode == 1)
    {
        if (filtered_source_has(src, source_merge_mode, 2))
            return 0.94f;
        if (filtered_source_has(src, source_merge_mode, 3))
            return 0.96f;
        if (filtered_source_has(src, source_merge_mode, 5))
            return 0.98f;
        if (filtered_source_has(src, source_merge_mode, 6))
            return 0.97f;
        if (filtered_source_has(src, source_merge_mode, 4))
            return 1.04f;
        return 1.0f;
    }
    if (source_priority_mode == 2)
    {
        if (filtered_source_has(src, source_merge_mode, 5) || filtered_source_has(src, source_merge_mode, 6))
            return 0.995f;
        if (filtered_source_has(src, source_merge_mode, 4))
            return 1.02f;
        return 1.0f;
    }
    if (source_priority_mode == 3)
    {
        if (filtered_source_has(src, source_merge_mode, 2) || filtered_source_has(src, source_merge_mode, 5))
            return 0.995f;
        if (filtered_source_has(src, source_merge_mode, 3))
            return 0.998f;
        if (filtered_source_has(src, source_merge_mode, 6))
            return 0.996f;
        if (filtered_source_has(src, source_merge_mode, 4))
            return 1.01f;
        return 1.0f;
    }
    if (source_priority_mode == 4)
    {
        const bool original_common = filtered_source_has(src, source_merge_mode, 2);
        const bool filtered_top = filtered_source_has(src, source_merge_mode, 3);
        const bool visited = filtered_source_has(src, source_merge_mode, 4);
        const bool reverse = filtered_source_has(src, source_merge_mode, 5);
        const bool path = filtered_source_has(src, source_merge_mode, 6);
        uint32_t strong = (original_common ? 1u : 0u) + (filtered_top ? 1u : 0u) +
                          (reverse ? 1u : 0u) + (path ? 1u : 0u);
        if (strong >= 3)
            return 0.925f;
        if (strong == 2)
            return 0.945f;
        if (filtered_top)
            return 0.970f;
        if (reverse || path)
            return 0.985f;
        if (original_common)
            return 0.990f;
        if (visited)
            return 1.020f;
        return 1.0f;
    }
    if (source_priority_mode == 5)
    {
        const bool original_common = filtered_source_has(src, source_merge_mode, 2);
        const bool filtered_top = filtered_source_has(src, source_merge_mode, 3);
        const bool visited = filtered_source_has(src, source_merge_mode, 4);
        const bool reverse = filtered_source_has(src, source_merge_mode, 5);
        const bool path = filtered_source_has(src, source_merge_mode, 6);
        const bool path_like = reverse || path;
        if (original_common && filtered_top && path_like)
            return 0.900f;
        if (original_common && filtered_top)
            return 0.920f;
        if (filtered_top && path_like)
            return 0.945f;
        if (original_common && path_like)
            return 0.965f;
        if (filtered_top)
            return 0.980f;
        if (original_common)
            return 0.995f;
        if (path_like)
            return 1.005f;
        if (visited)
            return 1.080f;
        return 1.020f;
    }
    if (source_priority_mode == 6)
    {
        const bool original_common = filtered_source_has(src, source_merge_mode, 2);
        const bool filtered_top = filtered_source_has(src, source_merge_mode, 3);
        const bool visited = filtered_source_has(src, source_merge_mode, 4);
        const bool reverse = filtered_source_has(src, source_merge_mode, 5);
        const bool path = filtered_source_has(src, source_merge_mode, 6);
        const bool path_like = reverse || path;
        if (filtered_top && original_common && path_like)
            return 0.900f;
        if (filtered_top && original_common)
            return 0.925f;
        if (filtered_top && path_like)
            return 0.940f;
        if (filtered_top)
            return 0.970f;
        if (original_common && path_like)
            return 0.980f;
        if (original_common)
            return 0.995f;
        if (path_like)
            return 1.000f;
        if (visited)
            return 1.050f;
        return 1.010f;
    }
    return 1.0f;
}

__device__ __forceinline__ uint32_t filtered_source_strong_count(uint8_t src, uint8_t source_merge_mode)
{
    return (filtered_source_has(src, source_merge_mode, 2) ? 1u : 0u) +
           (filtered_source_has(src, source_merge_mode, 3) ? 1u : 0u) +
           (filtered_source_has(src, source_merge_mode, 5) ? 1u : 0u) +
           (filtered_source_has(src, source_merge_mode, 6) ? 1u : 0u);
}

__device__ __forceinline__ void filtered_source_count_add(unsigned long long *counts,
                                                          uint8_t src,
                                                          uint8_t source_merge_mode)
{
    if (!counts)
        return;
    if (!source_merge_mode)
    {
        if (src > 7)
            src = 0;
        atomicAdd(&counts[src], 1ull);
        return;
    }
    bool any = false;
    for (uint8_t source_id = 1; source_id <= 6; ++source_id)
    {
        if (filtered_source_has(src, 1, source_id))
        {
            atomicAdd(&counts[source_id], 1ull);
            any = true;
        }
    }
    if (!any)
        atomicAdd(&counts[0], 1ull);
}

static bool host_has_common_label_filtered(uint32_t a,
                                           uint32_t b,
                                           const uint32_t *offsets,
                                           const uint32_t *labels,
                                           uint32_t universal_label)
{
    uint32_t ab = offsets[a], ae = offsets[a + 1];
    uint32_t bb = offsets[b], be = offsets[b + 1];
    uint32_t i = ab, j = bb;
    while (i < ae && j < be)
    {
        uint32_t x = labels[i], y = labels[j];
        if (x == universal_label || y == universal_label || x == y)
            return true;
        if (x < y)
            ++i;
        else
            ++j;
    }
    return false;
}

static void write_trace_minv1_diagnostic(const char *path,
                                         const uint32_t *graph,
                                         const uint32_t *degree,
                                         uint32_t N,
                                         uint32_t R,
                                         const uint32_t *offsets,
                                         const uint32_t *labels,
                                         uint32_t universal_label)
{
    FILE *fp = fopen(path, "w");
    if (!fp)
    {
        fprintf(stderr, "[gpu_vamana_filtered] failed to open trace diagnostic %s\n", path);
        return;
    }
    fprintf(fp, "node,out_degree,common_label_degree,first_label,top_neighbors\n");
    const uint32_t LIMIT = 256;
    uint32_t worst_nodes[LIMIT];
    uint32_t worst_common[LIMIT];
    uint32_t worst_degree[LIMIT];
    uint32_t count = 0;
    for (uint32_t row = 0; row < N; ++row)
    {
        uint32_t deg = degree[row] > R ? R : degree[row];
        uint32_t common = 0;
        for (uint32_t j = 0; j < deg; ++j)
        {
            uint32_t nb = graph[(size_t)row * R + j];
            if (nb != INVALID_ID && nb < N && nb != row &&
                host_has_common_label_filtered(row, nb, offsets, labels, universal_label))
                ++common;
        }
        uint32_t pos = count;
        if (count < LIMIT)
            ++count;
        else
        {
            uint32_t max_pos = 0;
            for (uint32_t i = 1; i < LIMIT; ++i)
                if (worst_common[i] > worst_common[max_pos])
                    max_pos = i;
            if (common >= worst_common[max_pos])
                continue;
            pos = max_pos;
        }
        worst_nodes[pos] = row;
        worst_common[pos] = common;
        worst_degree[pos] = deg;
    }
    for (uint32_t k = 0; k < count; ++k)
    {
        uint32_t best = k;
        for (uint32_t j = k + 1; j < count; ++j)
            if (worst_common[j] < worst_common[best])
                best = j;
        if (best != k)
        {
            std::swap(worst_nodes[k], worst_nodes[best]);
            std::swap(worst_common[k], worst_common[best]);
            std::swap(worst_degree[k], worst_degree[best]);
        }
    }
    for (uint32_t k = 0; k < count; ++k)
    {
        uint32_t row = worst_nodes[k];
        uint32_t first_label = offsets[row] < offsets[row + 1] ? labels[offsets[row]] : INVALID_ID;
        fprintf(fp, "%u,%u,%u,%u,\"", row, worst_degree[k], worst_common[k], first_label);
        uint32_t deg = degree[row] > R ? R : degree[row];
        for (uint32_t j = 0; j < deg && j < 16; ++j)
        {
            if (j)
                fprintf(fp, " ");
            fprintf(fp, "%u", graph[(size_t)row * R + j]);
        }
        fprintf(fp, "\"\n");
    }
    fclose(fp);
    printf("[gpu_vamana_filtered] trace_minv1_diagnostic=%s rows=%u note=graph_low_common_rows_not_query_visited_trace\n",
           path, count);
}

__device__ __forceinline__ bool label_contains_device(const uint32_t *labels,
                                                      uint32_t begin,
                                                      uint32_t end,
                                                      uint32_t value)
{
    for (uint32_t i = begin; i < end; ++i)
    {
        uint32_t cur = labels[i];
        if (cur == value)
            return true;
        if (cur > value)
            return false;
    }
    return false;
}

__device__ __forceinline__ bool has_common_label_device(uint32_t a,
                                                        uint32_t b,
                                                        const uint32_t *offsets,
                                                        const uint32_t *labels,
                                                        uint32_t universal_label,
                                                        unsigned long long *label_checks,
                                                        unsigned long long *universal_pass)
{
    uint32_t ab = offsets[a];
    uint32_t ae = offsets[a + 1];
    uint32_t bb = offsets[b];
    uint32_t be = offsets[b + 1];
    if (label_checks)
        atomicAdd(label_checks, 1ull);

    uint32_t i = ab;
    uint32_t j = bb;
    while (i < ae && j < be)
    {
        uint32_t x = labels[i];
        uint32_t y = labels[j];
        if (x == universal_label || y == universal_label)
        {
            if (universal_pass)
                atomicAdd(universal_pass, 1ull);
            return true;
        }
        if (x == y)
            return true;
        if (x < y)
            ++i;
        else
            ++j;
    }
    return false;
}

__device__ __forceinline__ bool point_has_target_label_device(uint32_t point,
                                                              const uint32_t *offsets,
                                                              const uint32_t *labels,
                                                              const uint8_t *target_labels,
                                                              uint32_t num_labels);

__device__ __forceinline__ bool point_has_target_label_device(uint32_t point,
                                                              const uint32_t *offsets,
                                                              const uint32_t *labels,
                                                              const uint8_t *target_labels,
                                                              uint32_t num_labels)
{
    if (!target_labels)
        return false;
    uint32_t begin = offsets[point];
    uint32_t end = offsets[point + 1];
    for (uint32_t i = begin; i < end; ++i)
    {
        uint32_t lbl = labels[i];
        if (lbl < num_labels && target_labels[lbl] != 0)
            return true;
    }
    return false;
}

__device__ __forceinline__ bool point_has_exact_label_device(uint32_t point,
                                                             uint32_t target_label,
                                                             const uint32_t *offsets,
                                                             const uint32_t *labels)
{
    uint32_t begin = offsets[point];
    uint32_t end = offsets[point + 1];
    for (uint32_t i = begin; i < end; ++i)
    {
        uint32_t value = labels[i];
        if (value == target_label)
            return true;
        if (value > target_label)
            break;
    }
    return false;
}

__device__ __forceinline__ bool point_is_label_start_device(uint32_t point,
                                                            const uint32_t *offsets,
                                                            const uint32_t *labels,
                                                            const uint32_t *label_starts,
                                                            const uint32_t *label_multi_starts,
                                                            uint32_t starts_per_label,
                                                            uint32_t num_labels)
{
    if (!label_starts || starts_per_label == 0)
        return false;
    uint32_t begin = offsets[point];
    uint32_t end = offsets[point + 1];
    for (uint32_t i = begin; i < end; ++i)
    {
        uint32_t lbl = labels[i];
        if (lbl >= num_labels)
            continue;
        if (label_starts[lbl] == point)
            return true;
        if (label_multi_starts)
        {
            const size_t base = (size_t)lbl * starts_per_label;
            for (uint32_t s = 1; s < starts_per_label; ++s)
                if (label_multi_starts[base + s] == point)
                    return true;
        }
    }
    return false;
}

__device__ __forceinline__ bool selected_covers_row_candidate_common_labels_device(uint32_t row,
                                                                                  uint32_t selected,
                                                                                  uint32_t candidate,
                                                                                  const uint32_t *offsets,
                                                                                  const uint32_t *labels,
                                                                                  uint32_t universal_label,
                                                                                  unsigned long long *label_checks,
                                                                                  unsigned long long *universal_pass)
{
    if (g_filtered_geometric_only_prune != 0)
        return true;
    uint32_t rb = offsets[row];
    uint32_t re = offsets[row + 1];
    uint32_t cb = offsets[candidate];
    uint32_t ce = offsets[candidate + 1];
    uint32_t sb = offsets[selected];
    uint32_t se = offsets[selected + 1];
    if (label_checks)
        atomicAdd(label_checks, 1ull);



    if (universal_label == UINT32_MAX)
        return filtered_selected_covers_common_labels(row, selected, candidate, offsets, labels);

    bool row_universal = label_contains_device(labels, rb, re, universal_label);
    bool cand_universal = label_contains_device(labels, cb, ce, universal_label);
    bool selected_universal = label_contains_device(labels, sb, se, universal_label);
    if (selected_universal)
    {
        if (universal_pass)
            atomicAdd(universal_pass, 1ull);
        return true;
    }
    if (row_universal || cand_universal)
    {
        if (universal_pass)
            atomicAdd(universal_pass, 1ull);
        if (row_universal && cand_universal)
            return selected_universal;
        const uint32_t xb = row_universal ? cb : rb;
        const uint32_t xe = row_universal ? ce : re;
        for (uint32_t i = xb; i < xe; ++i)
        {
            uint32_t lbl = labels[i];
            if (lbl != universal_label && !label_contains_device(labels, sb, se, lbl))
                return false;
        }
        return true;
    }

    uint32_t i = rb;
    uint32_t j = cb;
    while (i < re && j < ce)
    {
        uint32_t x = labels[i];
        uint32_t y = labels[j];
        if (x == y)
        {
            if (!label_contains_device(labels, sb, se, x))
                return false;
            ++i;
            ++j;
        }
        else if (x < y)
        {
            ++i;
        }
        else
        {
            ++j;
        }
    }
    return true;
}

template <typename DataT>
__device__ __forceinline__ float point_distance_device(const DataT *data, uint32_t dim, uint32_t a, uint32_t b)
{
    float sum = 0.0f;
    const DataT *x = data + (size_t)a * dim;
    const DataT *y = data + (size_t)b * dim;
    for (uint32_t d = 0; d < dim; ++d)
    {
        float diff = (float)x[d] - (float)y[d];
        sum += diff * diff;
    }
    return sum;
}

__device__ __forceinline__ uint32_t filtered_hash_u32(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ uint32_t filtered_deterministic_neighbor(uint32_t node, uint32_t k, uint32_t N)
{
    if (N <= 1)
        return INVALID_ID;
    uint32_t h = filtered_hash_u32(node * 0x9e3779b1u + k * 0x85ebca6bu + 17u);
    uint32_t v = h % N;
    if (v == node)
        v = (v + 1u) % N;
    return v;
}

__global__ void filtered_deterministic_random_graph_kernel(uint32_t *graph, uint32_t *degrees,
                                                           uint32_t N, uint32_t R,
                                                           uint32_t initial_degree, uint32_t seed)
{
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N)
        return;
    const uint32_t target = min(min(R, initial_degree), N > 0 ? N - 1 : 0);
    const size_t base = (size_t)row * R;
    uint32_t degree = 0;
    uint32_t attempt = 0;
    while (degree < target)
    {
        const uint32_t mixed = filtered_hash_u32(row * 0x9e3779b1u ^ seed * 0x85ebca6bu ^
                                                  attempt * 0xc2b2ae35u);
        const uint32_t candidate = mixed % N;
        ++attempt;
        if (candidate == row)
            continue;
        bool duplicate = false;
        for (uint32_t i = 0; i < degree; ++i)
            duplicate |= graph[base + i] == candidate;
        if (!duplicate)
            graph[base + degree++] = candidate;
    }
    for (uint32_t i = degree; i < R; ++i)
        graph[base + i] = INVALID_ID;
    degrees[row] = degree;
}

template <typename DataT>
__global__ void filtered_init_graph_kernel(const DataT *data,
                                           uint32_t dim,
                                           uint32_t *graph_cur,
                                           uint32_t *degree_cur,
                                           uint32_t *graph_work,
                                           uint32_t *degree_work,
                                           float *graph_dists,
                                           uint32_t N,
                                           uint32_t R,
                                           uint32_t C)
{
    uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= N)
        return;
    const size_t cur_base = (size_t)node * R;
    const size_t work_base = (size_t)node * C;
    for (uint32_t i = 0; i < C; ++i)
    {
        graph_work[work_base + i] = INVALID_ID;
        graph_dists[work_base + i] = FLT_MAX;
        if (i < R)
            graph_cur[cur_base + i] = INVALID_ID;
    }
    uint32_t deg = 0;
    for (uint32_t i = 0; i < R; ++i)
    {
        uint32_t nb = filtered_deterministic_neighbor(node, i, N);
        if (nb == INVALID_ID || nb >= N || nb == node)
            continue;
        float d = point_distance_device<DataT>(data, dim, node, nb);
        graph_cur[cur_base + deg] = nb;
        graph_work[work_base + deg] = nb;
        graph_dists[work_base + deg] = d;
        ++deg;
    }
    degree_cur[node] = deg;
    degree_work[node] = deg;
}

template <uint32_t MAXP>
__device__ __forceinline__ bool filtered_pool_has(const uint32_t (&ids)[MAXP], uint32_t count, uint32_t id)
{
    for (uint32_t i = 0; i < count; ++i)
        if (ids[i] == id)
            return true;
    return false;
}

template <uint32_t MAXP>
__device__ __forceinline__ void filtered_pool_insert(uint32_t id,
                                                     float dist,
                                                     uint32_t capacity,
                                                     uint32_t (&ids)[MAXP],
                                                     float (&dists)[MAXP],
                                                     uint8_t (&expanded)[MAXP],
                                                     uint32_t &count)
{
    if (id == INVALID_ID)
        return;
    capacity = capacity > MAXP ? MAXP : capacity;
    if (filtered_pool_has<MAXP>(ids, count, id))
        return;
    if (count < capacity)
    {
        ids[count] = id;
        dists[count] = dist;
        expanded[count] = 0;
        ++count;
        return;
    }
    uint32_t worst = 0;
    float worst_dist = dists[0];
    for (uint32_t i = 1; i < count; ++i)
    {
        if (dists[i] > worst_dist)
        {
            worst_dist = dists[i];
            worst = i;
        }
    }
    if (dist < worst_dist)
    {
        ids[worst] = id;
        dists[worst] = dist;
        expanded[worst] = 0;
    }
}

template <uint32_t MAXP>
__device__ __forceinline__ bool filtered_path_shortcut_better(float cand_dist,
                                                              uint16_t cand_hits,
                                                              float ref_dist,
                                                              uint16_t ref_hits,
                                                              uint32_t select_mode)
{
    if (select_mode == 1 && cand_hits != ref_hits)
        return cand_hits > ref_hits;
    return cand_dist < ref_dist;
}

template <uint32_t MAXP>
__device__ __forceinline__ void filtered_path_shortcut_insert(uint32_t id,
                                                              float dist,
                                                              uint32_t capacity,
                                                              uint32_t select_mode,
                                                              uint32_t (&ids)[MAXP],
                                                              float (&dists)[MAXP],
                                                              uint16_t (&hits)[MAXP],
                                                              uint32_t &count)
{
    if (id == INVALID_ID)
        return;
    capacity = capacity > MAXP ? MAXP : capacity;
    for (uint32_t i = 0; i < count; ++i)
    {
        if (ids[i] == id)
        {
            if (hits[i] < 65535)
                ++hits[i];
            if (dist < dists[i])
                dists[i] = dist;
            return;
        }
    }
    if (count < capacity)
    {
        ids[count] = id;
        dists[count] = dist;
        hits[count] = 1;
        ++count;
        return;
    }
    uint32_t worst = 0;
    for (uint32_t i = 1; i < count; ++i)
    {
        if (filtered_path_shortcut_better<MAXP>(dists[worst], hits[worst], dists[i], hits[i], select_mode))
            worst = i;
    }
    if (filtered_path_shortcut_better<MAXP>(dist, 1, dists[worst], hits[worst], select_mode))
    {
        ids[worst] = id;
        dists[worst] = dist;
        hits[worst] = 1;
    }
}

__device__ __forceinline__ uint16_t label_local_frontier_agreement_device(uint32_t row,
                                                                          uint32_t cand,
                                                                          uint32_t R,
                                                                          const uint32_t *graph_cur,
                                                                          const uint32_t *degree_cur,
                                                                          const uint32_t *offsets,
                                                                          const uint32_t *labels,
                                                                          uint32_t universal_label,
                                                                          unsigned long long *label_checks,
                                                                          unsigned long long *universal_pass)
{
    if (cand == INVALID_ID || cand == row)
        return 0;
    uint32_t cur_deg = degree_cur[row];
    uint32_t cand_deg = degree_cur[cand];
    if (cur_deg > R)
        cur_deg = R;
    if (cand_deg > R)
        cand_deg = R;
    if (cur_deg > 64)
        cur_deg = 64;
    if (cand_deg > 64)
        cand_deg = 64;
    const size_t row_base = (size_t)row * R;
    const size_t cand_base = (size_t)cand * R;
    uint16_t hits = 0;
    for (uint32_t j = 0; j < cand_deg; ++j)
    {
        uint32_t cn = graph_cur[cand_base + j];
        if (cn == INVALID_ID || cn == row || cn == cand)
            continue;
        bool in_frontier = false;
        for (uint32_t i = 0; i < cur_deg; ++i)
        {
            uint32_t rn = graph_cur[row_base + i];
            if (rn == cn)
            {
                in_frontier = true;
                break;
            }
        }
        if (in_frontier &&
            has_common_label_device(row, cn, offsets, labels, universal_label, label_checks, universal_pass))
        {
            if (hits != 65535)
                ++hits;
        }
    }
    return hits;
}

template <uint32_t MAXP>
__device__ __forceinline__ int filtered_pool_next_unexpanded(const uint32_t (&ids)[MAXP],
                                                             const float (&dists)[MAXP],
                                                             const uint8_t (&expanded)[MAXP],
                                                             uint32_t count)
{
    int best = -1;
    float best_dist = FLT_MAX;
    for (uint32_t i = 0; i < count; ++i)
    {
        if (!expanded[i] && ids[i] != INVALID_ID && dists[i] < best_dist)
        {
            best_dist = dists[i];
            best = (int)i;
        }
    }
    return best;
}

template <typename DataT>
__global__ void filtered_two_pool_inject_kernel(const DataT *data,
                                                uint32_t N,
                                                uint32_t dim,
                                                uint32_t R,
                                                uint32_t L,
                                                uint32_t filtered_L,
                                                uint32_t C,
                                                uint32_t filtered_pool_cap,
                                                uint32_t filtered_reserve,
                                                uint32_t unfiltered_reserve,
                                                uint32_t max_expand_steps,
                                                uint32_t inverted_seeds_enabled,
                                                uint32_t starts_per_label,
                                                uint32_t start_id,
                                                const uint32_t *graph_cur,
                                                const uint32_t *degree_cur,
                                                uint32_t *graph_work,
                                                uint32_t *degree_work,
                                                float *graph_dists,
                                                const uint32_t *offsets,
                                                const uint32_t *labels,
                                                const uint32_t *label_starts,
                                                const uint32_t *label_multi_starts,
                                                const uint32_t *label_point_offsets,
                                                const uint32_t *label_points,
                                                uint32_t label_seed_count,
                                                uint32_t num_labels,
                                                uint32_t universal_label,
                                                unsigned long long *filtered_distance_count,
                                                unsigned long long *unfiltered_distance_count,
                                                unsigned long long *label_checks,
                                                unsigned long long *label_rejects,
                                                unsigned long long *filtered_candidates,
                                                unsigned long long *unfiltered_candidates,
                                                unsigned long long *merged_candidates,
                                                unsigned long long *filtered_seed_counter,
                                                unsigned long long *filtered_reserved,
                                                unsigned long long *unfiltered_reserved,
                                                unsigned long long *universal_pass)
{
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N)
        return;

    const uint32_t MAX_FILTERED = 96;
    const uint32_t MAX_UNFILTERED = 160;
    uint32_t f_ids[MAX_FILTERED];
    float f_dists[MAX_FILTERED];
    uint8_t f_exp[MAX_FILTERED];
    uint32_t u_ids[MAX_UNFILTERED];
    float u_dists[MAX_UNFILTERED];
    uint8_t u_exp[MAX_UNFILTERED];
    uint32_t f_count = 0;
    uint32_t u_count = 0;

    for (uint32_t i = 0; i < MAX_FILTERED; ++i)
    {
        f_ids[i] = INVALID_ID;
        f_dists[i] = FLT_MAX;
        f_exp[i] = 0;
    }
    for (uint32_t i = 0; i < MAX_UNFILTERED; ++i)
    {
        u_ids[i] = INVALID_ID;
        u_dists[i] = FLT_MAX;
        u_exp[i] = 0;
    }

    filtered_pool_cap = filtered_pool_cap == 0 ? 1 : filtered_pool_cap;
    if (filtered_pool_cap > MAX_FILTERED)
        filtered_pool_cap = MAX_FILTERED;
    uint32_t u_cap = L == 0 ? 1 : L;
    if (u_cap > MAX_UNFILTERED)
        u_cap = MAX_UNFILTERED;

    uint32_t lb = offsets[row];
    uint32_t le = offsets[row + 1];
    for (uint32_t p = lb; p < le; ++p)
    {
        uint32_t lbl = labels[p];
        if (lbl >= num_labels)
            continue;
        uint32_t local_starts = starts_per_label == 0 ? 1 : starts_per_label;
        for (uint32_t si = 0; si < local_starts; ++si)
        {
            uint32_t st = INVALID_ID;
            if (si == 0)
                st = label_starts[lbl];
            else if (label_multi_starts != nullptr)
                st = label_multi_starts[(size_t)lbl * local_starts + si];
            if (st == INVALID_ID || st >= N || st == row)
                continue;
            if (!has_common_label_device(row, st, offsets, labels, universal_label, label_checks, universal_pass))
            {
                atomicAdd(label_rejects, 1ull);
                continue;
            }
            float d = point_distance_device<DataT>(data, dim, row, st);
            atomicAdd(filtered_distance_count, 1ull);
            atomicAdd(filtered_seed_counter, 1ull);
            filtered_pool_insert<MAX_FILTERED>(st, d, filtered_pool_cap, f_ids, f_dists, f_exp, f_count);
        }
        if (inverted_seeds_enabled != 0 && label_point_offsets != nullptr && label_points != nullptr)
        {
            uint32_t pb = label_point_offsets[lbl];
            uint32_t pe = label_point_offsets[lbl + 1];
            uint32_t pc = pe > pb ? pe - pb : 0;
            uint32_t seeds = label_seed_count;
            if (seeds > pc)
                seeds = pc;
            if (seeds > 4)
                seeds = 4;
            for (uint32_t s = 0; s < seeds; ++s)
            {
                uint32_t off = filtered_hash_u32(row * 0x9e3779b1u + lbl * 0x85ebca6bu + s * 0xc2b2ae35u) % pc;
                uint32_t seed = label_points[pb + off];
                if (seed == INVALID_ID || seed >= N || seed == row)
                    continue;
                float sd = point_distance_device<DataT>(data, dim, row, seed);
                atomicAdd(filtered_distance_count, 1ull);
                atomicAdd(filtered_seed_counter, 1ull);
                filtered_pool_insert<MAX_FILTERED>(seed, sd, filtered_pool_cap, f_ids, f_dists, f_exp, f_count);
            }
        }
    }

    uint32_t f_steps = filtered_L;
    if (f_steps > filtered_pool_cap)
        f_steps = filtered_pool_cap;
    if (max_expand_steps > 0 && f_steps > max_expand_steps)
        f_steps = max_expand_steps;
    for (uint32_t step = 0; step < f_steps; ++step)
    {
        int pos = filtered_pool_next_unexpanded<MAX_FILTERED>(f_ids, f_dists, f_exp, f_count);
        if (pos < 0)
            break;
        uint32_t expand = f_ids[pos];
        f_exp[pos] = 1;
        uint32_t deg = degree_cur[expand];
        if (deg > R)
            deg = R;
        const size_t expand_base = (size_t)expand * R;
        for (uint32_t j = 0; j < deg; ++j)
        {
            uint32_t nb = graph_cur[expand_base + j];
            if (nb == INVALID_ID || nb >= N || nb == row)
                continue;
            if (!has_common_label_device(row, nb, offsets, labels, universal_label, label_checks, universal_pass))
            {
                atomicAdd(label_rejects, 1ull);
                continue;
            }
            float d = point_distance_device<DataT>(data, dim, row, nb);
            atomicAdd(filtered_distance_count, 1ull);
            filtered_pool_insert<MAX_FILTERED>(nb, d, filtered_pool_cap, f_ids, f_dists, f_exp, f_count);
        }
    }

    if (start_id < N && start_id != row)
    {
        float d = point_distance_device<DataT>(data, dim, row, start_id);
        atomicAdd(unfiltered_distance_count, 1ull);
        filtered_pool_insert<MAX_UNFILTERED>(start_id, d, u_cap, u_ids, u_dists, u_exp, u_count);
    }
    uint32_t u_steps = L;
    if (u_steps > u_cap)
        u_steps = u_cap;
    if (max_expand_steps > 0 && u_steps > max_expand_steps)
        u_steps = max_expand_steps;
    for (uint32_t step = 0; step < u_steps; ++step)
    {
        int pos = filtered_pool_next_unexpanded<MAX_UNFILTERED>(u_ids, u_dists, u_exp, u_count);
        if (pos < 0)
            break;
        uint32_t expand = u_ids[pos];
        u_exp[pos] = 1;
        uint32_t deg = degree_cur[expand];
        if (deg > R)
            deg = R;
        const size_t expand_base = (size_t)expand * R;
        for (uint32_t j = 0; j < deg; ++j)
        {
            uint32_t nb = graph_cur[expand_base + j];
            if (nb == INVALID_ID || nb >= N || nb == row)
                continue;
            float d = point_distance_device<DataT>(data, dim, row, nb);
            atomicAdd(unfiltered_distance_count, 1ull);
            filtered_pool_insert<MAX_UNFILTERED>(nb, d, u_cap, u_ids, u_dists, u_exp, u_count);
        }
    }

    const size_t work_base = (size_t)row * C;
    for (uint32_t i = 0; i < C; ++i)
    {
        graph_work[work_base + i] = INVALID_ID;
        graph_dists[work_base + i] = FLT_MAX;
    }

    uint32_t out = 0;
    uint32_t reserved = 0;
    uint32_t u_reserved = 0;
    uint32_t max_filtered_before_unfiltered = C;
    if (max_filtered_before_unfiltered > unfiltered_reserve)
        max_filtered_before_unfiltered -= unfiltered_reserve;
    else
        max_filtered_before_unfiltered = 0;
    uint32_t reserve_cap = filtered_reserve < max_filtered_before_unfiltered ? filtered_reserve : max_filtered_before_unfiltered;
    for (uint32_t k = 0; k < reserve_cap && k < f_count; ++k)
    {
        uint32_t best = INVALID_ID;
        float best_d = FLT_MAX;
        uint32_t best_pos = INVALID_ID;
        for (uint32_t i = 0; i < f_count; ++i)
        {
            if (f_ids[i] != INVALID_ID && f_dists[i] < best_d)
            {
                best = f_ids[i];
                best_d = f_dists[i];
                best_pos = i;
            }
        }
        if (best == INVALID_ID)
            break;
        graph_work[work_base + out] = best;
        graph_dists[work_base + out] = best_d;
        f_ids[best_pos] = INVALID_ID;
        ++out;
        ++reserved;
    }
    for (uint32_t k = 0; k < u_count && out < C && u_reserved < unfiltered_reserve; ++k)
    {
        uint32_t best = INVALID_ID;
        float best_d = FLT_MAX;
        uint32_t best_pos = INVALID_ID;
        for (uint32_t i = 0; i < u_count; ++i)
        {
            if (u_ids[i] == INVALID_ID)
                continue;
            bool dup = false;
            for (uint32_t j = 0; j < out; ++j)
            {
                if (graph_work[work_base + j] == u_ids[i])
                {
                    dup = true;
                    break;
                }
            }
            if (!dup && u_dists[i] < best_d)
            {
                best = u_ids[i];
                best_d = u_dists[i];
                best_pos = i;
            }
        }
        if (best == INVALID_ID)
            break;
        graph_work[work_base + out] = best;
        graph_dists[work_base + out] = best_d;
        u_ids[best_pos] = INVALID_ID;
        ++out;
        ++u_reserved;
    }
    for (uint32_t k = 0; k < f_count && out < max_filtered_before_unfiltered; ++k)
    {
        uint32_t best = INVALID_ID;
        float best_d = FLT_MAX;
        uint32_t best_pos = INVALID_ID;
        for (uint32_t i = 0; i < f_count; ++i)
        {
            if (f_ids[i] != INVALID_ID && f_dists[i] < best_d)
            {
                best = f_ids[i];
                best_d = f_dists[i];
                best_pos = i;
            }
        }
        if (best == INVALID_ID)
            break;
        graph_work[work_base + out] = best;
        graph_dists[work_base + out] = best_d;
        f_ids[best_pos] = INVALID_ID;
        ++out;
    }
    for (uint32_t k = 0; k < u_count && out < C; ++k)
    {
        uint32_t best = INVALID_ID;
        float best_d = FLT_MAX;
        uint32_t best_pos = INVALID_ID;
        for (uint32_t i = 0; i < u_count; ++i)
        {
            if (u_ids[i] == INVALID_ID)
                continue;
            bool dup = false;
            for (uint32_t j = 0; j < out; ++j)
            {
                if (graph_work[work_base + j] == u_ids[i])
                {
                    dup = true;
                    break;
                }
            }
            if (!dup && u_dists[i] < best_d)
            {
                best = u_ids[i];
                best_d = u_dists[i];
                best_pos = i;
            }
        }
        if (best == INVALID_ID)
            break;
        graph_work[work_base + out] = best;
        graph_dists[work_base + out] = best_d;
        u_ids[best_pos] = INVALID_ID;
        ++out;
    }
    degree_work[row] = out;
    atomicAdd(filtered_candidates, (unsigned long long)f_count);
    atomicAdd(unfiltered_candidates, (unsigned long long)u_count);
    atomicAdd(merged_candidates, (unsigned long long)out);
    atomicAdd(filtered_reserved, (unsigned long long)reserved);
    atomicAdd(unfiltered_reserved, (unsigned long long)u_reserved);
}

__global__ void filtered_copy_first_from_work_kernel(uint32_t N,
                                                     uint32_t R,
                                                     uint32_t C,
                                                     const uint32_t *graph_work,
                                                     const float *graph_dists,
                                                     const uint32_t *degree_work,
                                                     uint32_t *graph_cur,
                                                     uint32_t *degree_cur)
{
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N)
        return;
    const size_t work_base = (size_t)row * C;
    const size_t cur_base = (size_t)row * R;
    uint32_t out = 0;
    uint32_t deg = degree_work[row];
    if (deg > C)
        deg = C;
    for (uint32_t i = 0; i < R; ++i)
        graph_cur[cur_base + i] = INVALID_ID;
    for (uint32_t i = 0; i < deg && out < R; ++i)
    {
        uint32_t nb = graph_work[work_base + i];
        if (nb == INVALID_ID || nb >= N || nb == row || graph_dists[work_base + i] == FLT_MAX)
            continue;
        bool dup = false;
        for (uint32_t j = 0; j < out; ++j)
            if (graph_cur[cur_base + j] == nb)
                dup = true;
        if (!dup)
            graph_cur[cur_base + out++] = nb;
    }
    degree_cur[row] = out;
}

template <typename DataT>
__global__ void init_refine_graph_dists_kernel(const DataT *data,
                                               uint32_t N,
                                               uint32_t dim,
                                               uint32_t R,
                                               uint32_t C,
                                               const uint32_t *graph_cur,
                                               const uint32_t *degree_cur,
                                               uint32_t *graph_work,
                                               uint32_t *degree_work,
                                               float *graph_dists)
{
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N)
        return;
    const size_t cur_base = (size_t)row * R;
    const size_t work_base = (size_t)row * C;
    uint32_t deg = degree_cur[row];
    if (deg > R)
        deg = R;
    for (uint32_t i = 0; i < C; ++i)
    {
        uint32_t nb = INVALID_ID;
        float d = FLT_MAX;
        if (i < deg)
        {
            nb = graph_cur[cur_base + i];
            if (nb != INVALID_ID && nb < N && nb != row)
                d = point_distance_device<DataT>(data, dim, row, nb);
            else
                nb = INVALID_ID;
        }
        graph_work[work_base + i] = nb;
        graph_dists[work_base + i] = d;
    }
    degree_work[row] = deg;
}

__global__ void mark_filtered_refine_active_rows_kernel(const uint32_t *graph_cur,
                                                        const uint32_t *degree_cur,
                                                        uint32_t N,
                                                        uint32_t R,
                                                        const uint32_t *offsets,
                                                        const uint32_t *labels,
                                                        const uint8_t *target_labels,
                                                        const uint32_t *label_starts,
                                                        const uint32_t *label_multi_starts,
                                                        const uint32_t *label_point_offsets,
                                                        uint32_t num_labels,
                                                        uint32_t target_active_labels,
                                                        uint32_t force_label_start_active,
                                                        uint32_t starts_per_label,
                                                        uint32_t universal_label,
                                                        uint32_t common_threshold,
                                                        uint32_t active_permille,
                                                        uint32_t path_active_permille,
                                                        uint32_t per_label_active,
                                                        uint8_t *active_rows,
                                                        uint8_t *deficient_label_flags,
                                                        unsigned long long *active_count,
                                                        unsigned long long *common_sum,
                                                        unsigned int *low_common_count,
                                                        unsigned long long *label_checks,
                                                        unsigned long long *universal_pass)
{
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N)
        return;
    uint32_t deg = degree_cur[row];
    if (deg > R)
        deg = R;
    uint32_t common = 0;
    const size_t base = (size_t)row * R;
    for (uint32_t i = 0; i < deg; ++i)
    {
        uint32_t nb = graph_cur[base + i];
        if (nb != INVALID_ID && nb < N && nb != row &&
            has_common_label_device(row, nb, offsets, labels, universal_label, label_checks, universal_pass))
            ++common;
    }
    atomicAdd(common_sum, (unsigned long long)common);
    bool active = false;
    const uint32_t row_label_begin = offsets[row];
    const uint32_t row_label_end = offsets[row + 1];
    if (per_label_active == 2)
    {
        active = true;
        for (uint32_t p = row_label_begin; p < row_label_end; ++p)
            deficient_label_flags[p] = 1;
    }
    else if (per_label_active == 1)
    {
        for (uint32_t p = row_label_begin; p < row_label_end; ++p)
        {
            const uint32_t label = labels[p];
            uint32_t label_degree = 0;
            for (uint32_t i = 0; i < deg; ++i)
            {
                const uint32_t nb = graph_cur[base + i];
                if (nb != INVALID_ID && nb < N && nb != row &&
                    label_contains_device(labels, offsets[nb], offsets[nb + 1], label))
                    ++label_degree;
            }
            const uint32_t label_size = label < num_labels ? label_point_offsets[label + 1] - label_point_offsets[label] : 0;
            const uint32_t tau = label_size > 0 ? min(common_threshold, label_size - 1) : 0;
            const bool deficient = label_degree < tau;
            deficient_label_flags[p] = deficient ? 1 : 0;
            active = active || deficient;
        }
    }
    else
    {
        active = common < common_threshold;
        for (uint32_t p = row_label_begin; p < row_label_end; ++p)
            deficient_label_flags[p] = active ? 1 : 0;
    }
    if (active)
        atomicAdd(low_common_count, 1u);
    uint32_t h = filtered_hash_u32(row * 0x9e3779b1u + 12345u) % 1000u;
    if (per_label_active == 0 && active && active_permille < 1000)
    {
        active = h < active_permille;
    }
    if (per_label_active == 0 && !active && path_active_permille > 0)
    {
        uint32_t h2 = filtered_hash_u32(row * 0x85ebca6bu + 0x27d4eb2du) % 1000u;
        active = h2 < path_active_permille;
    }
    if (per_label_active == 0 && !active && target_active_labels != 0)
        active = point_has_target_label_device(row, offsets, labels, target_labels, num_labels);
    if (per_label_active == 0 && !active && force_label_start_active != 0)
        active = point_is_label_start_device(row, offsets, labels, label_starts, label_multi_starts,
                                             starts_per_label, num_labels);
    active_rows[row] = active ? 1 : 0;
    if (active)
        atomicAdd(active_count, 1ull);
}

template <typename DataT>
__global__ void filtered_refine_inject_kernel(const DataT *data,
                                              uint32_t N,
                                              uint32_t dim,
                                              uint32_t R,
                                              uint32_t filtered_L,
                                              uint32_t C,
                                              uint32_t filtered_pool_cap,
                                              uint32_t filtered_reserve,
                                              uint32_t bridge_reserve,
                                              uint32_t visited_reserve,
                                              uint32_t target_visited_reserve,
                                              uint32_t path_reserve,
                                              uint32_t visited_select_mode,
                                              uint32_t visited_low_common_threshold,
                                              uint32_t max_expand_steps,
                                              uint32_t target_max_expand_steps,
                                              uint32_t label_start_max_expand_steps,
                                              uint32_t label_start_filtered_reserve,
                                              uint32_t label_start_visited_reserve,
                                              uint32_t label_start_path_reserve,
                                              uint32_t starts_per_label,
                                              uint32_t inverted_seeds_enabled,
                                              uint32_t target_shortcut_samples,
                                              uint32_t target_shortcut_keep,
                                              uint32_t label_start_portal_samples,
                                              uint32_t label_start_portal_keep,
                                              uint32_t label_start_portal_mode,
                                              uint32_t label_local_shortcut_samples,
                                              uint32_t label_local_shortcut_keep,
                                              uint32_t label_local_shortcut_mode,
                                              uint32_t label_local_shortcut_sampler,
                                              uint32_t label_path_shortcut_fanout,
                                              uint32_t label_path_shortcut_keep,
                                              uint32_t expanded_path_shortcut_fanout,
                                              uint32_t expanded_path_shortcut_keep,
                                              uint32_t twohop_path_shortcut_fanout,
                                              uint32_t twohop_path_shortcut_keep,
                                              uint32_t path_shortcut_select_mode,
                                              uint32_t label_local_shortcut_target_only,
                                              uint32_t label_local_shortcut_as_frontier,
                                              uint32_t per_label_candidate_mode,
                                              uint32_t per_label_candidate_keep,
                                              uint32_t backbone_admission_mode,
                                              uint32_t backbone_per_label_quota,
                                              uint32_t compatible_only,
                                              const uint8_t *deficient_label_flags,
                                              uint32_t deficient_labels_only,
                                              uint32_t direct_seeds_per_label,
                                              const uint8_t *active_rows,
                                              const uint32_t *active_ids,
                                              uint32_t active_id_count,
                                              const uint32_t *graph_cur,
                                              const uint32_t *degree_cur,
                                              uint32_t *graph_work,
                                              uint32_t *degree_work,
                                              float *graph_dists,
                                              uint8_t *graph_sources,
                                              const uint32_t *offsets,
                                              const uint32_t *labels,
                                              const uint32_t *label_starts,
                                              const uint32_t *label_multi_starts,
                                              const uint8_t *target_labels,
                                              const uint32_t *label_point_offsets,
                                              const uint32_t *label_points,
                                              const uint32_t *query_anchor_offsets,
                                              const uint32_t *query_anchor_ids,
                                              uint32_t query_anchor_keep,
                                              uint32_t label_seed_count,
                                              uint32_t num_labels,
                                              uint32_t universal_label,
                                              uint32_t build_trace_row,
                                              unsigned long long *filtered_distance_count,
                                              unsigned long long *bridge_distance_count,
                                              unsigned long long *label_checks,
                                              unsigned long long *label_rejects,
                                              unsigned long long *filtered_candidates,
                                              unsigned long long *filtered_visited_count,
                                              unsigned long long *original_neighbors,
                                              unsigned long long *filtered_reserved,
                                              unsigned long long *filtered_path_reserved,
                                              unsigned long long *filtered_visited_reserved,
                                              unsigned long long *filtered_top_reserved,
                                              unsigned long long *bridge_reserved,
                                              unsigned long long *filtered_seed_counter,
                                              unsigned long long *universal_pass)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = tid;
    uint32_t work_row = tid;
    if (active_ids)
    {
        if (tid >= active_id_count)
            return;
        row = active_ids[tid];
    }
    else if (row >= N || active_rows[row] == 0)
    {
        return;
    }
    if (row >= N)
        return;

    const uint32_t MAX_FILTERED = 96;
    const uint32_t MAX_VISITED = 128;
    const uint32_t MAX_PATH = 32;
    const uint32_t MAX_PORTAL = 32;
    const uint32_t MAX_LOCAL_SHORTCUT = 32;
    const uint32_t MAX_PATH_SHORTCUT = 32;
    uint32_t f_ids[MAX_FILTERED];
    float f_dists[MAX_FILTERED];
    uint8_t f_exp[MAX_FILTERED];
    uint32_t v_ids[MAX_VISITED];
    float v_dists[MAX_VISITED];
    uint8_t v_exp[MAX_VISITED];
    uint32_t path_ids[MAX_PATH];
    float path_dists[MAX_PATH];
    uint32_t portal_ids[MAX_PORTAL];
    float portal_dists[MAX_PORTAL];
    uint32_t shortcut_ids[MAX_LOCAL_SHORTCUT];
    float shortcut_dists[MAX_LOCAL_SHORTCUT];
    uint16_t shortcut_hits[MAX_LOCAL_SHORTCUT];
    uint32_t path_shortcut_ids[MAX_PATH_SHORTCUT];
    float path_shortcut_dists[MAX_PATH_SHORTCUT];
    uint16_t path_shortcut_hits[MAX_PATH_SHORTCUT];
    uint32_t f_count = 0;
    uint32_t v_count = 0;
    uint32_t path_count = 0;
    uint32_t portal_count = 0;
    uint32_t shortcut_count = 0;
    uint32_t path_shortcut_count = 0;
    for (uint32_t i = 0; i < MAX_FILTERED; ++i)
    {
        f_ids[i] = INVALID_ID;
        f_dists[i] = FLT_MAX;
        f_exp[i] = 0;
    }
    for (uint32_t i = 0; i < MAX_VISITED; ++i)
    {
        v_ids[i] = INVALID_ID;
        v_dists[i] = FLT_MAX;
        v_exp[i] = 0;
    }
    for (uint32_t i = 0; i < MAX_PATH; ++i)
    {
        path_ids[i] = INVALID_ID;
        path_dists[i] = FLT_MAX;
    }
    for (uint32_t i = 0; i < MAX_PORTAL; ++i)
    {
        portal_ids[i] = INVALID_ID;
        portal_dists[i] = FLT_MAX;
    }
    for (uint32_t i = 0; i < MAX_LOCAL_SHORTCUT; ++i)
    {
        shortcut_ids[i] = INVALID_ID;
        shortcut_dists[i] = FLT_MAX;
        shortcut_hits[i] = 0;
    }
    for (uint32_t i = 0; i < MAX_PATH_SHORTCUT; ++i)
    {
        path_shortcut_ids[i] = INVALID_ID;
        path_shortcut_dists[i] = FLT_MAX;
        path_shortcut_hits[i] = 0;
    }
    if (filtered_pool_cap == 0)
        filtered_pool_cap = 1;
    if (filtered_pool_cap > MAX_FILTERED)
        filtered_pool_cap = MAX_FILTERED;

    const size_t cur_base = (size_t)row * R;
    const size_t work_base = (size_t)work_row * C;
    uint32_t lb = offsets[row];
    uint32_t le = offsets[row + 1];
    uint32_t cur_deg = degree_cur[row];
    if (cur_deg > R)
        cur_deg = R;
    uint32_t out = 0;
    for (uint32_t i = 0; i < C; ++i)
    {
        graph_work[work_base + i] = INVALID_ID;
        graph_dists[work_base + i] = FLT_MAX;
        if (graph_sources)
            graph_sources[work_base + i] = 0;
    }




    if (per_label_candidate_mode != 0 && per_label_candidate_keep > 0)
    {
        const uint32_t MAX_PER_LABEL_POOL = 32;
        uint32_t row_label_count = le > lb ? le - lb : 1;
        uint32_t fair_keep = C / row_label_count;
        if (fair_keep == 0)
            fair_keep = 1;
        if (fair_keep > per_label_candidate_keep)
            fair_keep = per_label_candidate_keep;
        for (uint32_t p = lb; p < le && out < C; ++p)
        {
            if (deficient_labels_only != 0 && deficient_label_flags[p] == 0)
                continue;
            uint32_t lbl = labels[p];
            if (lbl >= num_labels || lbl == universal_label)
                continue;
            uint32_t local_ids[MAX_PER_LABEL_POOL];
            float local_dists[MAX_PER_LABEL_POOL];
            uint8_t local_expanded[MAX_PER_LABEL_POOL];
            uint32_t local_count = 0;
            for (uint32_t i = 0; i < MAX_PER_LABEL_POOL; ++i)
            {
                local_ids[i] = INVALID_ID;
                local_dists[i] = FLT_MAX;
                local_expanded[i] = 0;
            }
            for (uint32_t si = 0; si < starts_per_label; ++si)
            {
                uint32_t seed = si == 0 ? label_starts[lbl]
                                        : label_multi_starts[(size_t)lbl * starts_per_label + si];
                if (seed == INVALID_ID || seed >= N || seed == row ||
                    !point_has_exact_label_device(seed, lbl, offsets, labels))
                    continue;
                float d = point_distance_device<DataT>(data, dim, row, seed);
                atomicAdd(filtered_distance_count, 1ull);
                atomicAdd(filtered_seed_counter, 1ull);
                filtered_pool_insert<MAX_PER_LABEL_POOL>(seed, d, MAX_PER_LABEL_POOL, local_ids, local_dists,
                                                         local_expanded, local_count);
            }
            if (inverted_seeds_enabled && label_point_offsets && label_points)
            {
                uint32_t pb = label_point_offsets[lbl];
                uint32_t pe = label_point_offsets[lbl + 1];
                uint32_t pc = pe > pb ? pe - pb : 0;
                uint32_t seeds = label_seed_count;
                if (seeds > 16)
                    seeds = 16;
                if (seeds > pc)
                    seeds = pc;
                for (uint32_t s = 0; s < seeds; ++s)
                {
                    uint32_t seed = label_points[pb + (filtered_hash_u32(row + s * 7919u + lbl * 17u) % pc)];
                    if (seed == INVALID_ID || seed >= N || seed == row)
                        continue;
                    float d = point_distance_device<DataT>(data, dim, row, seed);
                    atomicAdd(filtered_distance_count, 1ull);
                    atomicAdd(filtered_seed_counter, 1ull);
                    filtered_pool_insert<MAX_PER_LABEL_POOL>(seed, d, MAX_PER_LABEL_POOL, local_ids, local_dists,
                                                             local_expanded, local_count);
                }
            }
            for (uint32_t i = 0; i < cur_deg; ++i)
            {
                uint32_t seed = graph_cur[cur_base + i];
                if (seed == INVALID_ID || seed >= N || seed == row ||
                    !point_has_exact_label_device(seed, lbl, offsets, labels))
                    continue;
                float d = point_distance_device<DataT>(data, dim, row, seed);
                atomicAdd(filtered_distance_count, 1ull);
                filtered_pool_insert<MAX_PER_LABEL_POOL>(seed, d, MAX_PER_LABEL_POOL, local_ids, local_dists,
                                                         local_expanded, local_count);
            }
            uint32_t local_steps = max_expand_steps == 0 ? 8 : max_expand_steps;
            if (local_steps > MAX_PER_LABEL_POOL)
                local_steps = MAX_PER_LABEL_POOL;
            for (uint32_t step = 0; step < local_steps; ++step)
            {
                int position = filtered_pool_next_unexpanded<MAX_PER_LABEL_POOL>(
                    local_ids, local_dists, local_expanded, local_count);
                if (position < 0)
                    break;
                uint32_t expand = local_ids[position];
                local_expanded[position] = 1;
                uint32_t expand_degree = degree_cur[expand];
                if (expand_degree > R)
                    expand_degree = R;
                const size_t expand_base = (size_t)expand * R;
                for (uint32_t j = 0; j < expand_degree; ++j)
                {
                    uint32_t candidate = graph_cur[expand_base + j];
                    if (candidate == INVALID_ID || candidate >= N || candidate == row ||
                        !point_has_exact_label_device(candidate, lbl, offsets, labels))
                        continue;
                    float d = point_distance_device<DataT>(data, dim, row, candidate);
                    atomicAdd(filtered_distance_count, 1ull);
                    filtered_pool_insert<MAX_PER_LABEL_POOL>(candidate, d, MAX_PER_LABEL_POOL, local_ids,
                                                             local_dists, local_expanded, local_count);
                }
            }
            for (uint32_t retained = 0; retained < fair_keep && out < C; ++retained)
            {
                uint32_t best_position = INVALID_ID;
                for (uint32_t i = 0; i < local_count; ++i)
                {
                    if (local_ids[i] == INVALID_ID)
                        continue;
                    if (best_position == INVALID_ID || local_dists[i] < local_dists[best_position] ||
                        (local_dists[i] == local_dists[best_position] && local_ids[i] < local_ids[best_position]))
                        best_position = i;
                }
                if (best_position == INVALID_ID)
                    break;
                uint32_t candidate = local_ids[best_position];
                bool duplicate = false;
                for (uint32_t i = 0; i < out; ++i)
                    duplicate = duplicate || graph_work[work_base + i] == candidate;
                if (!duplicate)
                {
                    graph_work[work_base + out] = candidate;
                    graph_dists[work_base + out] = local_dists[best_position];
                    if (graph_sources)
                        graph_sources[work_base + out] = 3;
                    ++out;
                }
                local_ids[best_position] = INVALID_ID;
            }
        }
    }
    if (row == build_trace_row)
    {
        printf("[gpu_filtered_build_trace] stage=per_label row=%u out=%u labels=", row, out);
        for (uint32_t p = lb; p < le; ++p)
        {
            uint32_t count = 0;
            for (uint32_t i = 0; i < out; ++i)
                if (point_has_exact_label_device(graph_work[work_base + i], labels[p], offsets, labels))
                    ++count;
            printf("%u:%u ", labels[p], count);
        }
        printf("\n");
    }

    uint32_t bridge_keep = bridge_reserve;
    if (bridge_keep > cur_deg)
        bridge_keep = cur_deg;
    const uint32_t MAX_ROW_LABELS = 16;
    uint32_t inherited_label_counts[MAX_ROW_LABELS] = {};
    uint32_t quota_label_count = le - lb;
    if (quota_label_count > MAX_ROW_LABELS)
        quota_label_count = MAX_ROW_LABELS;
    uint32_t effective_backbone_quota = backbone_per_label_quota;
    if (effective_backbone_quota == 0 && quota_label_count > 0)
    {
        effective_backbone_quota = R / quota_label_count;
        if (effective_backbone_quota == 0)
            effective_backbone_quota = 1;
    }
    for (uint32_t i = 0; i < cur_deg && out < C; ++i)
    {
        uint32_t nb = graph_cur[cur_base + i];
        if (nb == INVALID_ID || nb >= N || nb == row)
            continue;
        float d = point_distance_device<DataT>(data, dim, row, nb);
        atomicAdd(bridge_distance_count, 1ull);
        const bool common =
            has_common_label_device(row, nb, offsets, labels, universal_label, label_checks, universal_pass);
        if (compatible_only != 0 && !common)
            continue;
        if (backbone_admission_mode == 2)
        {
            bool repairs_quota = false;
            for (uint32_t q = 0; q < quota_label_count; ++q)
                if (inherited_label_counts[q] < effective_backbone_quota &&
                    point_has_exact_label_device(nb, labels[lb + q], offsets, labels))
                    repairs_quota = true;
            if (!repairs_quota)
                continue;
        }
        graph_work[work_base + out] = nb;
        graph_dists[work_base + out] = d;
        if (graph_sources)
        {
            graph_sources[work_base + out] = common ? 2 : 1;
        }
        if (backbone_admission_mode == 2)
            for (uint32_t q = 0; q < quota_label_count; ++q)
                if (point_has_exact_label_device(nb, labels[lb + q], offsets, labels))
                    ++inherited_label_counts[q];
        ++out;
    }
    atomicAdd(original_neighbors, (unsigned long long)out);
    atomicAdd(bridge_reserved, (unsigned long long)bridge_keep);

    bool row_targeted = false;
    if (target_labels)
    {
        for (uint32_t p = lb; p < le; ++p)
        {
            uint32_t lbl = labels[p];
            if (lbl < num_labels && target_labels[lbl] != 0)
            {
                row_targeted = true;
                break;
            }
        }
    }
    const bool row_is_label_start =
        point_is_label_start_device(row, offsets, labels, label_starts, label_multi_starts, starts_per_label, num_labels);
    for (uint32_t p = lb; p < le; ++p)
    {
        if (deficient_labels_only != 0 && deficient_label_flags[p] == 0)
            continue;
        uint32_t lbl = labels[p];
        if (lbl >= num_labels)
            continue;
        for (uint32_t si = 0; si < starts_per_label; ++si)
        {
            uint32_t st = (si == 0) ? label_starts[lbl] : label_multi_starts[(size_t)lbl * starts_per_label + si];
            if (st == INVALID_ID || st >= N || st == row)
                continue;
            if (!has_common_label_device(row, st, offsets, labels, universal_label, label_checks, universal_pass))
            {
                atomicAdd(label_rejects, 1ull);
                continue;
            }
            float d = point_distance_device<DataT>(data, dim, row, st);
            atomicAdd(filtered_distance_count, 1ull);
            atomicAdd(filtered_seed_counter, 1ull);
            filtered_pool_insert<MAX_FILTERED>(st, d, filtered_pool_cap, f_ids, f_dists, f_exp, f_count);
            filtered_pool_insert<MAX_VISITED>(st, d, MAX_VISITED, v_ids, v_dists, v_exp, v_count);
            if (si == 0 && direct_seeds_per_label > 0 && out < C)
            {
                bool duplicate = false;
                for (uint32_t existing = 0; existing < out; ++existing)
                    duplicate = duplicate || graph_work[work_base + existing] == st;
                if (!duplicate)
                {
                    graph_work[work_base + out] = st;
                    graph_dists[work_base + out] = d;
                    if (graph_sources)
                        graph_sources[work_base + out] = 3;
                    ++out;
                }
            }
        }
        if (inverted_seeds_enabled && label_point_offsets && label_points)
        {
            uint32_t pb = label_point_offsets[lbl];
            uint32_t pe = label_point_offsets[lbl + 1];
            uint32_t pc = pe > pb ? pe - pb : 0;
            uint32_t seeds = label_seed_count > 64 ? 64 : label_seed_count;
            if (seeds > pc)
                seeds = pc;
            for (uint32_t s = 0; s < seeds; ++s)
            {
                uint32_t seed = label_points[pb + (filtered_hash_u32(row + s * 7919u + lbl * 17u) % pc)];
                if (seed == INVALID_ID || seed >= N || seed == row)
                    continue;
                float d = point_distance_device<DataT>(data, dim, row, seed);
                atomicAdd(filtered_distance_count, 1ull);
                atomicAdd(filtered_seed_counter, 1ull);
                filtered_pool_insert<MAX_FILTERED>(seed, d, filtered_pool_cap, f_ids, f_dists, f_exp, f_count);
                filtered_pool_insert<MAX_VISITED>(seed, d, MAX_VISITED, v_ids, v_dists, v_exp, v_count);
                if (s + 1 < direct_seeds_per_label && out < C)
                {
                    bool duplicate = false;
                    for (uint32_t existing = 0; existing < out; ++existing)
                        duplicate = duplicate || graph_work[work_base + existing] == seed;
                    if (!duplicate)
                    {
                        graph_work[work_base + out] = seed;
                        graph_dists[work_base + out] = d;
                        if (graph_sources)
                            graph_sources[work_base + out] = 3;
                        ++out;
                    }
                }
            }
        }
        if (query_anchor_keep > 0 && query_anchor_offsets && query_anchor_ids)
        {
            uint32_t ab = query_anchor_offsets[lbl];
            uint32_t ae = query_anchor_offsets[lbl + 1];
            uint32_t ac = ae > ab ? ae - ab : 0;
            if (ac > query_anchor_keep)
                ac = query_anchor_keep;
            for (uint32_t a = 0; a < ac; ++a)
            {
                uint32_t seed = query_anchor_ids[ab + a];
                if (seed == INVALID_ID || seed >= N || seed == row)
                    continue;
                if (!has_common_label_device(row, seed, offsets, labels, universal_label, label_checks, universal_pass))
                {
                    atomicAdd(label_rejects, 1ull);
                    continue;
                }
                float d = point_distance_device<DataT>(data, dim, row, seed);
                atomicAdd(filtered_distance_count, 1ull);
                atomicAdd(filtered_seed_counter, 1ull);
                filtered_pool_insert<MAX_FILTERED>(seed, d, filtered_pool_cap, f_ids, f_dists, f_exp, f_count);
                filtered_pool_insert<MAX_VISITED>(seed, d, MAX_VISITED, v_ids, v_dists, v_exp, v_count);
            }
        }
        if (row_targeted && target_labels && target_labels[lbl] != 0 && target_shortcut_samples > 0 &&
            target_shortcut_keep > 0 && label_point_offsets && label_points)
        {
            uint32_t pb = label_point_offsets[lbl];
            uint32_t pe = label_point_offsets[lbl + 1];
            uint32_t pc = pe > pb ? pe - pb : 0;
            uint32_t samples = target_shortcut_samples > pc ? pc : target_shortcut_samples;
            uint32_t keep = target_shortcut_keep > samples ? samples : target_shortcut_keep;
            uint32_t local_ids[32];
            float local_dists[32];
            uint32_t local_count = 0;
            if (samples > 32)
                samples = 32;
            if (keep > 32)
                keep = 32;
            for (uint32_t s = 0; s < samples; ++s)
            {
                uint32_t off = 0;
                if (label_local_shortcut_sampler == 1)
                {
                    uint32_t shift = filtered_hash_u32(row * 0x9e3779b1u + lbl * 0x85ebca6bu) % pc;
                    off = (uint32_t)(((unsigned long long)s * (unsigned long long)pc) / samples);
                    off = (off + shift) % pc;
                }
                else
                {
                    off = filtered_hash_u32(row * 0x9e3779b1u + lbl * 0x85ebca6bu + s * 0x27d4eb2du) % pc;
                }
                uint32_t seed = label_points[pb + off];
                if (seed == INVALID_ID || seed >= N || seed == row)
                    continue;
                bool dup = false;
                for (uint32_t t = 0; t < local_count; ++t)
                {
                    if (local_ids[t] == seed)
                    {
                        dup = true;
                        break;
                    }
                }
                if (dup)
                    continue;
                float d = point_distance_device<DataT>(data, dim, row, seed);
                atomicAdd(filtered_distance_count, 1ull);
                uint32_t pos = local_count;
                if (local_count < 32)
                    ++local_count;
                else
                {
                    uint32_t worst = 0;
                    for (uint32_t t = 1; t < local_count; ++t)
                        if (local_dists[t] > local_dists[worst])
                            worst = t;
                    if (d >= local_dists[worst])
                        continue;
                    pos = worst;
                }
                local_ids[pos] = seed;
                local_dists[pos] = d;
            }
            for (uint32_t k = 0; k < keep && local_count > 0; ++k)
            {
                uint32_t best_pos = 0;
                for (uint32_t t = 1; t < local_count; ++t)
                    if (local_dists[t] < local_dists[best_pos])
                        best_pos = t;
                uint32_t seed = local_ids[best_pos];
                float d = local_dists[best_pos];
                filtered_pool_insert<MAX_FILTERED>(seed, d, filtered_pool_cap, f_ids, f_dists, f_exp, f_count);
                filtered_pool_insert<MAX_VISITED>(seed, d, MAX_VISITED, v_ids, v_dists, v_exp, v_count);
                atomicAdd(filtered_seed_counter, 1ull);
                local_ids[best_pos] = local_ids[local_count - 1];
                local_dists[best_pos] = local_dists[local_count - 1];
                --local_count;
            }
        }
        if (label_local_shortcut_samples > 0 && label_local_shortcut_keep > 0 && label_point_offsets && label_points &&
            (label_local_shortcut_target_only == 0 || row_targeted))
        {
            uint32_t pb = label_point_offsets[lbl];
            uint32_t pe = label_point_offsets[lbl + 1];
            uint32_t pc = pe > pb ? pe - pb : 0;
            uint32_t samples = label_local_shortcut_samples > pc ? pc : label_local_shortcut_samples;
            uint32_t keep = label_local_shortcut_keep > MAX_LOCAL_SHORTCUT ? MAX_LOCAL_SHORTCUT : label_local_shortcut_keep;
            uint32_t retain = label_local_shortcut_mode == 2 ? MAX_LOCAL_SHORTCUT : keep;
            if (samples > 64)
                samples = 64;
            for (uint32_t s = 0; s < samples; ++s)
            {
                uint32_t off = 0;
                if (label_local_shortcut_sampler == 1)
                {
                    uint32_t shift = filtered_hash_u32(row * 0x9e3779b1u + lbl * 0x85ebca6bu) % pc;
                    off = (uint32_t)(((unsigned long long)s * (unsigned long long)pc) / samples);
                    off = (off + shift) % pc;
                }
                else
                {
                    off = filtered_hash_u32(row * 0x9e3779b1u + lbl * 0x85ebca6bu + s * 0x27d4eb2du) % pc;
                }
                uint32_t seed = label_points[pb + off];
                if (seed == INVALID_ID || seed >= N || seed == row)
                    continue;
                bool dup = false;
                for (uint32_t t = 0; t < shortcut_count; ++t)
                {
                    if (shortcut_ids[t] == seed)
                    {
                        dup = true;
                        break;
                    }
                }
                if (dup)
                    continue;
                float d = point_distance_device<DataT>(data, dim, row, seed);
                atomicAdd(filtered_distance_count, 1ull);
                uint16_t agreement_hits = 0;
                if (label_local_shortcut_mode == 3)
                {
                    agreement_hits = label_local_frontier_agreement_device(
                        row, seed, R, graph_cur, degree_cur, offsets, labels, universal_label, label_checks,
                        universal_pass);
                }
                bool prefer_far = label_local_shortcut_mode == 1;
                uint32_t pos = shortcut_count;
                if (shortcut_count < retain)
                {
                    ++shortcut_count;
                }
                else
                {
                    uint32_t replace = 0;
                    for (uint32_t t = 1; t < shortcut_count; ++t)
                    {
                        if ((label_local_shortcut_mode == 3 &&
                             (shortcut_hits[t] < shortcut_hits[replace] ||
                              (shortcut_hits[t] == shortcut_hits[replace] &&
                               shortcut_dists[t] > shortcut_dists[replace]))) ||
                            (!prefer_far && label_local_shortcut_mode != 3 &&
                             shortcut_dists[t] > shortcut_dists[replace]) ||
                            (prefer_far && shortcut_dists[t] < shortcut_dists[replace]))
                            replace = t;
                    }
                    if ((label_local_shortcut_mode == 3 &&
                         (agreement_hits < shortcut_hits[replace] ||
                          (agreement_hits == shortcut_hits[replace] && d >= shortcut_dists[replace]))) ||
                        (!prefer_far && label_local_shortcut_mode != 3 && d >= shortcut_dists[replace]) ||
                        (prefer_far && d <= shortcut_dists[replace]))
                        continue;
                    pos = replace;
                }
                shortcut_ids[pos] = seed;
                shortcut_dists[pos] = d;
                shortcut_hits[pos] = agreement_hits;
                if (label_local_shortcut_as_frontier != 0)
                {
                    filtered_pool_insert<MAX_FILTERED>(seed, d, filtered_pool_cap, f_ids, f_dists, f_exp, f_count);
                    filtered_pool_insert<MAX_VISITED>(seed, d, MAX_VISITED, v_ids, v_dists, v_exp, v_count);
                }
            }
        }
    }
    if (row_is_label_start && label_start_portal_samples > 0 && label_start_portal_keep > 0 &&
        label_point_offsets && label_points)
    {
        uint32_t max_samples = label_start_portal_samples;
        if (max_samples > 64)
            max_samples = 64;
        uint32_t max_keep = label_start_portal_keep;
        if (max_keep > MAX_PORTAL)
            max_keep = MAX_PORTAL;
        for (uint32_t p = lb; p < le; ++p)
        {
            uint32_t lbl = labels[p];
            if (lbl >= num_labels)
                continue;
            bool row_start_for_label = label_starts[lbl] == row;
            if (!row_start_for_label && label_multi_starts)
            {
                const size_t mb = (size_t)lbl * starts_per_label;
                for (uint32_t s = 1; s < starts_per_label; ++s)
                    if (label_multi_starts[mb + s] == row)
                        row_start_for_label = true;
            }
            if (!row_start_for_label)
                continue;
            uint32_t pb = label_point_offsets[lbl];
            uint32_t pe = label_point_offsets[lbl + 1];
            uint32_t pc = pe > pb ? pe - pb : 0;
            if (pc == 0)
                continue;
            uint32_t samples = max_samples > pc ? pc : max_samples;
            for (uint32_t s = 0; s < samples; ++s)
            {
                uint32_t off = (uint32_t)(((unsigned long long)s * (unsigned long long)pc) / samples);
                off = (off + (filtered_hash_u32(row * 0x9e3779b1u + lbl * 0x85ebca6bu) % pc)) % pc;
                uint32_t cand = label_points[pb + off];
                if (cand == INVALID_ID || cand >= N || cand == row)
                    continue;
                bool dup = false;
                for (uint32_t t = 0; t < portal_count; ++t)
                {
                    if (portal_ids[t] == cand)
                    {
                        dup = true;
                        break;
                    }
                }
                if (dup)
                    continue;
                float d = point_distance_device<DataT>(data, dim, row, cand);
                atomicAdd(filtered_distance_count, 1ull);
                bool prefer_far = label_start_portal_mode == 1;
                uint32_t pos = portal_count;
                if (portal_count < max_keep)
                {
                    ++portal_count;
                }
                else
                {
                    uint32_t replace = 0;
                    for (uint32_t t = 1; t < portal_count; ++t)
                    {
                        if ((!prefer_far && portal_dists[t] > portal_dists[replace]) ||
                            (prefer_far && portal_dists[t] < portal_dists[replace]))
                            replace = t;
                    }
                    if ((!prefer_far && d >= portal_dists[replace]) ||
                        (prefer_far && d <= portal_dists[replace]))
                        continue;
                    pos = replace;
                }
                portal_ids[pos] = cand;
                portal_dists[pos] = d;
            }
        }
    }
    for (uint32_t i = 0; i < cur_deg; ++i)
    {
        uint32_t nb = graph_cur[cur_base + i];
        if (nb != INVALID_ID && nb < N && nb != row &&
            has_common_label_device(row, nb, offsets, labels, universal_label, label_checks, universal_pass))
        {
            float d = point_distance_device<DataT>(data, dim, row, nb);
            atomicAdd(filtered_distance_count, 1ull);
            filtered_pool_insert<MAX_FILTERED>(nb, d, filtered_pool_cap, f_ids, f_dists, f_exp, f_count);
            filtered_pool_insert<MAX_VISITED>(nb, d, MAX_VISITED, v_ids, v_dists, v_exp, v_count);
            if (label_path_shortcut_fanout > 0 && label_path_shortcut_keep > 0)
            {
                uint32_t nb_deg = degree_cur[nb];
                if (nb_deg > R)
                    nb_deg = R;
                if (nb_deg > label_path_shortcut_fanout)
                    nb_deg = label_path_shortcut_fanout;
                const size_t nb_base = (size_t)nb * R;
                uint32_t max_keep = label_path_shortcut_keep;
                if (max_keep > MAX_PATH_SHORTCUT)
                    max_keep = MAX_PATH_SHORTCUT;
                for (uint32_t jj = 0; jj < nb_deg; ++jj)
                {
                    uint32_t cand = graph_cur[nb_base + jj];
                    if (cand == INVALID_ID || cand >= N || cand == row || cand == nb)
                        continue;
                    if (!has_common_label_device(row, cand, offsets, labels, universal_label, label_checks, universal_pass))
                    {
                        atomicAdd(label_rejects, 1ull);
                        continue;
                    }
                    float pd = point_distance_device<DataT>(data, dim, row, cand);
                    atomicAdd(filtered_distance_count, 1ull);
                    filtered_path_shortcut_insert<MAX_PATH_SHORTCUT>(
                        cand, pd, max_keep, path_shortcut_select_mode, path_shortcut_ids, path_shortcut_dists,
                        path_shortcut_hits, path_shortcut_count);
                }
            }
        }
    }

    uint32_t steps = filtered_L;
    if (steps > filtered_pool_cap)
        steps = filtered_pool_cap;
    uint32_t row_max_expand_steps = max_expand_steps;
    if (row_targeted && target_max_expand_steps > row_max_expand_steps)
        row_max_expand_steps = target_max_expand_steps;
    if (row_is_label_start && label_start_max_expand_steps > row_max_expand_steps)
        row_max_expand_steps = label_start_max_expand_steps;
    if (row_max_expand_steps > 0 && steps > row_max_expand_steps)
        steps = row_max_expand_steps;
    for (uint32_t step = 0; step < steps; ++step)
    {
        int pos = filtered_pool_next_unexpanded<MAX_FILTERED>(f_ids, f_dists, f_exp, f_count);
        if (pos < 0)
            break;
        uint32_t expand = f_ids[pos];
        float expand_dist = f_dists[pos];
        f_exp[pos] = 1;
        if ((path_reserve > 0 || expanded_path_shortcut_fanout > 0 || twohop_path_shortcut_fanout > 0) &&
            path_count < MAX_PATH)
        {
            bool dup_path = false;
            for (uint32_t pi = 0; pi < path_count; ++pi)
            {
                if (path_ids[pi] == expand)
                {
                    dup_path = true;
                    break;
                }
            }
            if (!dup_path)
            {
                path_ids[path_count] = expand;
                path_dists[path_count] = expand_dist;
                ++path_count;
            }
        }
        uint32_t deg = degree_cur[expand];
        if (deg > R)
            deg = R;
        const size_t eb = (size_t)expand * R;
        for (uint32_t j = 0; j < deg; ++j)
        {
            uint32_t nb = graph_cur[eb + j];
            if (nb == INVALID_ID || nb >= N || nb == row)
                continue;
            if (!has_common_label_device(row, nb, offsets, labels, universal_label, label_checks, universal_pass))
            {
                atomicAdd(label_rejects, 1ull);
                continue;
            }
            float d = point_distance_device<DataT>(data, dim, row, nb);
            atomicAdd(filtered_distance_count, 1ull);
            filtered_pool_insert<MAX_FILTERED>(nb, d, filtered_pool_cap, f_ids, f_dists, f_exp, f_count);
            filtered_pool_insert<MAX_VISITED>(nb, d, MAX_VISITED, v_ids, v_dists, v_exp, v_count);
        }
    }
    if (expanded_path_shortcut_fanout > 0 && expanded_path_shortcut_keep > 0)
    {
        uint32_t max_keep = expanded_path_shortcut_keep;
        if (max_keep > MAX_PATH_SHORTCUT)
            max_keep = MAX_PATH_SHORTCUT;
        for (uint32_t pi = 0; pi < path_count; ++pi)
        {
            uint32_t expand = path_ids[pi];
            if (expand == INVALID_ID || expand >= N || expand == row)
                continue;
            uint32_t deg = degree_cur[expand];
            if (deg > R)
                deg = R;
            if (deg > expanded_path_shortcut_fanout)
                deg = expanded_path_shortcut_fanout;
            const size_t eb = (size_t)expand * R;
            for (uint32_t j = 0; j < deg; ++j)
            {
                uint32_t cand = graph_cur[eb + j];
                if (cand == INVALID_ID || cand >= N || cand == row || cand == expand)
                    continue;
                if (!has_common_label_device(row, cand, offsets, labels, universal_label, label_checks, universal_pass))
                {
                    atomicAdd(label_rejects, 1ull);
                    continue;
                }
                float pd = point_distance_device<DataT>(data, dim, row, cand);
                atomicAdd(filtered_distance_count, 1ull);
                filtered_path_shortcut_insert<MAX_PATH_SHORTCUT>(
                    cand, pd, max_keep, path_shortcut_select_mode, path_shortcut_ids, path_shortcut_dists,
                    path_shortcut_hits, path_shortcut_count);
            }
        }
    }
    if (twohop_path_shortcut_fanout > 0 && twohop_path_shortcut_keep > 0)
    {
        uint32_t max_keep = twohop_path_shortcut_keep;
        if (max_keep > MAX_PATH_SHORTCUT)
            max_keep = MAX_PATH_SHORTCUT;
        if (max_keep < path_shortcut_count)
            max_keep = path_shortcut_count;
        if (max_keep > MAX_PATH_SHORTCUT)
            max_keep = MAX_PATH_SHORTCUT;
        uint32_t fanout = twohop_path_shortcut_fanout;
        if (fanout > R)
            fanout = R;
        for (uint32_t pi = 0; pi < path_count; ++pi)
        {
            uint32_t expand = path_ids[pi];
            if (expand == INVALID_ID || expand >= N || expand == row)
                continue;
            uint32_t deg1 = degree_cur[expand];
            if (deg1 > fanout)
                deg1 = fanout;
            const size_t eb1 = (size_t)expand * R;
            for (uint32_t j = 0; j < deg1; ++j)
            {
                uint32_t mid = graph_cur[eb1 + j];
                if (mid == INVALID_ID || mid >= N || mid == row || mid == expand)
                    continue;
                if (!has_common_label_device(row, mid, offsets, labels, universal_label, label_checks, universal_pass))
                {
                    atomicAdd(label_rejects, 1ull);
                    continue;
                }
                uint32_t deg2 = degree_cur[mid];
                if (deg2 > fanout)
                    deg2 = fanout;
                const size_t eb2 = (size_t)mid * R;
                for (uint32_t jj = 0; jj < deg2; ++jj)
                {
                    uint32_t cand = graph_cur[eb2 + jj];
                    if (cand == INVALID_ID || cand >= N || cand == row || cand == expand || cand == mid)
                        continue;
                    if (!has_common_label_device(row, cand, offsets, labels, universal_label, label_checks, universal_pass))
                    {
                        atomicAdd(label_rejects, 1ull);
                        continue;
                    }
                    float pd = point_distance_device<DataT>(data, dim, row, cand);
                    atomicAdd(filtered_distance_count, 1ull);
                    filtered_path_shortcut_insert<MAX_PATH_SHORTCUT>(
                        cand, pd, max_keep, path_shortcut_select_mode, path_shortcut_ids, path_shortcut_dists,
                        path_shortcut_hits, path_shortcut_count);
                }
            }
        }
    }

    uint32_t filtered_added = 0;
    uint32_t portal_added = 0;
    uint32_t path_shortcut_added = 0;
    uint32_t shortcut_added = 0;
    uint32_t visited_added = 0;
    uint32_t max_filtered = filtered_reserve;
    if (row_is_label_start && label_start_filtered_reserve > max_filtered)
        max_filtered = label_start_filtered_reserve;
    if (max_filtered > filtered_pool_cap)
        max_filtered = filtered_pool_cap;
    for (uint32_t k = 0; k < portal_count && portal_added < label_start_portal_keep && out < C; ++k)
    {
        uint32_t best = INVALID_ID;
        float best_d = label_start_portal_mode == 1 ? -1.0f : FLT_MAX;
        uint32_t best_pos = INVALID_ID;
        for (uint32_t i = 0; i < portal_count; ++i)
        {
            if (portal_ids[i] == INVALID_ID)
                continue;
            bool dup = false;
            for (uint32_t j = 0; j < out; ++j)
            {
                if (graph_work[work_base + j] == portal_ids[i])
                {
                    dup = true;
                    break;
                }
            }
            if (!dup && ((label_start_portal_mode == 1 && portal_dists[i] > best_d) ||
                         (label_start_portal_mode != 1 && portal_dists[i] < best_d)))
            {
                best = portal_ids[i];
                best_d = portal_dists[i];
                best_pos = i;
            }
        }
        if (best == INVALID_ID)
            break;
        graph_work[work_base + out] = best;
        graph_dists[work_base + out] = best_d;
        if (graph_sources)
            graph_sources[work_base + out] = 3;
        portal_ids[best_pos] = INVALID_ID;
        ++out;
        ++portal_added;
    }
    uint32_t max_path_shortcut_add = label_path_shortcut_keep;
    if (expanded_path_shortcut_keep > max_path_shortcut_add)
        max_path_shortcut_add = expanded_path_shortcut_keep;
    if (twohop_path_shortcut_keep > max_path_shortcut_add)
        max_path_shortcut_add = twohop_path_shortcut_keep;
    if (max_path_shortcut_add > MAX_PATH_SHORTCUT)
        max_path_shortcut_add = MAX_PATH_SHORTCUT;
    for (uint32_t k = 0; k < path_shortcut_count && path_shortcut_added < max_path_shortcut_add && out < C; ++k)
    {
        uint32_t best = INVALID_ID;
        float best_d = FLT_MAX;
        uint32_t best_pos = INVALID_ID;
        for (uint32_t i = 0; i < path_shortcut_count; ++i)
        {
            if (path_shortcut_ids[i] == INVALID_ID)
                continue;
            bool dup = false;
            for (uint32_t j = 0; j < out; ++j)
            {
                if (graph_work[work_base + j] == path_shortcut_ids[i])
                {
                    dup = true;
                    break;
                }
            }
            if (!dup &&
                (best == INVALID_ID ||
                 filtered_path_shortcut_better<MAX_PATH_SHORTCUT>(
                     path_shortcut_dists[i], path_shortcut_hits[i], best_d,
                     best_pos == INVALID_ID ? 0 : path_shortcut_hits[best_pos], path_shortcut_select_mode)))
            {
                best = path_shortcut_ids[i];
                best_d = path_shortcut_dists[i];
                best_pos = i;
            }
        }
        if (best == INVALID_ID)
            break;
        graph_work[work_base + out] = best;
        graph_dists[work_base + out] = best_d;
        if (graph_sources)
            graph_sources[work_base + out] = 6;
        path_shortcut_ids[best_pos] = INVALID_ID;
        ++out;
        ++path_shortcut_added;
    }
    for (uint32_t k = 0; k < shortcut_count && shortcut_added < label_local_shortcut_keep && out < C; ++k)
    {
        uint32_t best = INVALID_ID;
        const bool mixed_far_pick = label_local_shortcut_mode == 2 && (shortcut_added & 1u) != 0;
        float best_d = (label_local_shortcut_mode == 1 || mixed_far_pick) ? -1.0f : FLT_MAX;
        uint16_t best_hits = 0;
        uint32_t best_pos = INVALID_ID;
        for (uint32_t i = 0; i < shortcut_count; ++i)
        {
            if (shortcut_ids[i] == INVALID_ID)
                continue;
            bool dup = false;
            for (uint32_t j = 0; j < out; ++j)
            {
                if (graph_work[work_base + j] == shortcut_ids[i])
                {
                    dup = true;
                    break;
                }
            }
            if (!dup &&
                ((label_local_shortcut_mode == 3 &&
                  (best == INVALID_ID || shortcut_hits[i] > best_hits ||
                   (shortcut_hits[i] == best_hits && shortcut_dists[i] < best_d))) ||
                 ((label_local_shortcut_mode == 1 || mixed_far_pick) && shortcut_dists[i] > best_d) ||
                 ((label_local_shortcut_mode != 1 && label_local_shortcut_mode != 3 && !mixed_far_pick) &&
                  shortcut_dists[i] < best_d)))
            {
                best = shortcut_ids[i];
                best_d = shortcut_dists[i];
                best_hits = shortcut_hits[i];
                best_pos = i;
            }
        }
        if (best == INVALID_ID)
            break;
        graph_work[work_base + out] = best;
        graph_dists[work_base + out] = best_d;
        if (graph_sources)
            graph_sources[work_base + out] = 3;
        shortcut_ids[best_pos] = INVALID_ID;
        ++out;
        ++shortcut_added;
    }
    for (uint32_t k = 0; k < f_count && filtered_added < max_filtered && out < C; ++k)
    {
        uint32_t best = INVALID_ID;
        float best_d = FLT_MAX;
        uint32_t best_pos = INVALID_ID;
        for (uint32_t i = 0; i < f_count; ++i)
        {
            if (f_ids[i] == INVALID_ID)
                continue;
            bool dup = false;
            for (uint32_t j = 0; j < out; ++j)
            {
                if (graph_work[work_base + j] == f_ids[i])
                {
                    dup = true;
                    break;
                }
            }
            if (!dup && f_dists[i] < best_d)
            {
                best = f_ids[i];
                best_d = f_dists[i];
                best_pos = i;
            }
        }
        if (best == INVALID_ID)
            break;
        graph_work[work_base + out] = best;
        graph_dists[work_base + out] = best_d;
        if (graph_sources)
            graph_sources[work_base + out] = 3;
        f_ids[best_pos] = INVALID_ID;
        ++out;
        ++filtered_added;
    }
    uint32_t path_added = 0;
    uint32_t max_path = path_reserve;
    if (row_is_label_start && label_start_path_reserve > max_path)
        max_path = label_start_path_reserve;
    if (max_path > MAX_PATH)
        max_path = MAX_PATH;
    for (uint32_t k = 0; k < path_count && path_added < max_path && out < C; ++k)
    {
        uint32_t best = INVALID_ID;
        float best_d = FLT_MAX;
        uint32_t best_pos = INVALID_ID;
        for (uint32_t i = 0; i < path_count; ++i)
        {
            if (path_ids[i] == INVALID_ID)
                continue;
            bool dup = false;
            for (uint32_t j = 0; j < out; ++j)
            {
                if (graph_work[work_base + j] == path_ids[i])
                {
                    dup = true;
                    break;
                }
            }
            if (!dup && path_dists[i] < best_d)
            {
                best = path_ids[i];
                best_d = path_dists[i];
                best_pos = i;
            }
        }
        if (best == INVALID_ID)
            break;
        graph_work[work_base + out] = best;
        graph_dists[work_base + out] = best_d;
        if (graph_sources)
            graph_sources[work_base + out] = 6;
        path_ids[best_pos] = INVALID_ID;
        ++out;
        ++path_added;
    }
    uint32_t effective_visited_reserve = visited_reserve;
    if (row_targeted && target_visited_reserve > effective_visited_reserve)
        effective_visited_reserve = target_visited_reserve;
    if (row_is_label_start && label_start_visited_reserve > effective_visited_reserve)
        effective_visited_reserve = label_start_visited_reserve;
    if (visited_low_common_threshold > 0 && visited_reserve > 0)
    {
        uint32_t row_common_degree = 0;
        for (uint32_t i = 0; i < cur_deg; ++i)
        {
            uint32_t nb = graph_cur[cur_base + i];
            if (nb != INVALID_ID && nb < N && nb != row &&
                has_common_label_device(row, nb, offsets, labels, universal_label, label_checks, universal_pass))
                ++row_common_degree;
        }
        if (!row_targeted && !row_is_label_start && row_common_degree >= visited_low_common_threshold)
            effective_visited_reserve = 0;
    }
    if (effective_visited_reserve > 32)
        effective_visited_reserve = 32;
    for (uint32_t k = 0; k < v_count && visited_added < effective_visited_reserve && out < C; ++k)
    {
        uint32_t best = INVALID_ID;
        bool pick_far = (visited_select_mode == 1 && (visited_added & 1u) != 0);
        float best_d = pick_far ? -1.0f : FLT_MAX;
        uint32_t best_pos = INVALID_ID;
        for (uint32_t i = 0; i < v_count; ++i)
        {
            if (v_ids[i] == INVALID_ID)
                continue;
            bool dup = false;
            for (uint32_t j = 0; j < out; ++j)
            {
                if (graph_work[work_base + j] == v_ids[i])
                {
                    dup = true;
                    break;
                }
            }
            if (!dup && ((!pick_far && v_dists[i] < best_d) || (pick_far && v_dists[i] > best_d)))
            {
                best = v_ids[i];
                best_d = v_dists[i];
                best_pos = i;
            }
        }
        if (best == INVALID_ID)
            break;
        graph_work[work_base + out] = best;
        graph_dists[work_base + out] = best_d;
        if (graph_sources)
            graph_sources[work_base + out] = 4;
        v_ids[best_pos] = INVALID_ID;
        ++out;
        ++visited_added;
    }
    atomicAdd(filtered_candidates, (unsigned long long)f_count);
    atomicAdd(filtered_visited_count, (unsigned long long)v_count);
    atomicAdd(filtered_reserved, (unsigned long long)(filtered_added + portal_added + shortcut_added));
    atomicAdd(filtered_top_reserved, (unsigned long long)(filtered_added + portal_added + shortcut_added));
    atomicAdd(filtered_path_reserved, (unsigned long long)(path_added + path_shortcut_added));
    atomicAdd(filtered_visited_reserved, (unsigned long long)visited_added);





    if (per_label_candidate_mode != 0 && label_point_offsets && le > lb && le - lb <= 16)
    {
        uint32_t assigned[16] = {};
        uint32_t row_label_count = le - lb;




        uint32_t quota = (R + row_label_count - 1) / row_label_count;
        if (quota < per_label_candidate_keep)
            quota = per_label_candidate_keep;
        uint32_t balanced_out = 0;
        for (uint32_t i = 0; i < out; ++i)
        {
            uint32_t candidate = graph_work[work_base + i];
            uint32_t assigned_label = INVALID_ID;
            uint32_t assigned_cardinality = UINT32_MAX;
            for (uint32_t q = 0; q < row_label_count; ++q)
            {
                uint32_t label = labels[lb + q];
                if (!point_has_exact_label_device(candidate, label, offsets, labels))
                    continue;
                uint32_t cardinality = label_point_offsets[label + 1] - label_point_offsets[label];
                if (assigned[q] < quota &&
                    (assigned_label == INVALID_ID || cardinality < assigned_cardinality ||
                     (cardinality == assigned_cardinality && label < labels[lb + assigned_label])))
                {
                    assigned_label = q;
                    assigned_cardinality = cardinality;
                }
            }
            if (assigned_label == INVALID_ID)
                continue;
            if (balanced_out != i)
            {
                graph_work[work_base + balanced_out] = candidate;
                graph_dists[work_base + balanced_out] = graph_dists[work_base + i];
                if (graph_sources)
                    graph_sources[work_base + balanced_out] = graph_sources[work_base + i];
            }
            ++assigned[assigned_label];
            ++balanced_out;
        }
        out = balanced_out;
    }
    if (row == build_trace_row)
    {
        printf("[gpu_filtered_build_trace] stage=merged row=%u out=%u labels=", row, out);
        for (uint32_t p = lb; p < le; ++p)
        {
            uint32_t count = 0;
            for (uint32_t i = 0; i < out; ++i)
                if (point_has_exact_label_device(graph_work[work_base + i], labels[p], offsets, labels))
                    ++count;
            printf("%u:%u ", labels[p], count);
        }
        printf("\n");
    }
    degree_work[work_row] = out;
}

__global__ void filtered_graph_diagnostics_kernel(const uint32_t *graph,
                                                  const uint32_t *degree,
                                                  uint32_t N,
                                                  uint32_t R,
                                                  const uint32_t *offsets,
                                                  const uint32_t *labels,
                                                  uint32_t universal_label,
                                                  unsigned long long *out_edges,
                                                  unsigned long long *common_edges,
                                                  unsigned long long *label_checks,
                                                  unsigned long long *label_rejects,
                                                  unsigned long long *universal_pass,
                                                  unsigned int *invalid_edges,
                                                  unsigned int *self_loops,
                                                  unsigned int *low_common_rows,
                                                  unsigned int *degree_min,
                                                  unsigned int *degree_max,
                                                  unsigned int *common_min,
                                                  unsigned int *common_max)
{
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N)
        return;

    uint32_t deg = degree[row];
    atomicMin(degree_min, deg);
    atomicMax(degree_max, deg);
    atomicAdd(out_edges, (unsigned long long)deg);

    uint32_t common = 0;
    for (uint32_t j = 0; j < deg && j < R; ++j)
    {
        uint32_t nb = graph[(size_t)row * R + j];
        if (nb == INVALID_ID || nb >= N)
        {
            atomicAdd(invalid_edges, 1u);
            continue;
        }
        if (nb == row)
        {
            atomicAdd(self_loops, 1u);
            continue;
        }
        if (has_common_label_device(row, nb, offsets, labels, universal_label, label_checks, universal_pass))
        {
            ++common;
        }
        else
        {
            atomicAdd(label_rejects, 1ull);
        }
    }

    atomicAdd(common_edges, (unsigned long long)common);
    atomicMin(common_min, common);
    atomicMax(common_max, common);
    if (common < 2)
        atomicAdd(low_common_rows, 1u);
}

__global__ void filtered_reverse_generate_kernel(uint32_t N,
                                                 uint32_t R,
                                                 uint32_t reverse_cap,
                                                 uint32_t reverse_dst_begin,
                                                 uint32_t reverse_dst_count,
                                                 uint32_t replacement_mode,
                                                 uint32_t touch_active_only,
                                                 uint32_t sample_permille,
                                                 uint32_t inactive_sample_permille,
                                                 uint32_t inactive_target_labels_only,
                                                 const uint8_t *active_rows,
                                                 const uint32_t *graph_cur,
                                                 const uint32_t *degree_cur,
                                                 const uint32_t *offsets,
                                                 const uint32_t *labels,
                                                 const uint8_t *target_labels,
                                                 uint32_t num_labels,
                                                 uint32_t universal_label,
                                                 uint32_t *reverse_ids,
                                                 uint32_t *reverse_counts,
                                                 uint8_t *touched_rows,
                                                 unsigned long long *reverse_edges,
                                                 unsigned long long *reverse_kept,
                                                 unsigned long long *label_checks,
                                                 unsigned long long *universal_pass)
{
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || active_rows[row] == 0)
        return;
    uint32_t deg = degree_cur[row];
    if (deg > R)
        deg = R;
    const size_t base = (size_t)row * R;
    for (uint32_t i = 0; i < deg; ++i)
    {
        uint32_t nb = graph_cur[base + i];
        if (nb == INVALID_ID || nb >= N || nb == row)
            continue;
        if (!has_common_label_device(row, nb, offsets, labels, universal_label, label_checks, universal_pass))
            continue;
        if (nb < reverse_dst_begin || nb >= reverse_dst_begin + reverse_dst_count)
            continue;
        uint32_t local_nb = nb - reverse_dst_begin;
        if (touch_active_only != 0 && active_rows[nb] == 0)
        {
            bool allow_inactive = false;
            if (inactive_target_labels_only != 0)
                allow_inactive = point_has_target_label_device(nb, offsets, labels, target_labels, num_labels);
            if (!allow_inactive && inactive_sample_permille == 0)
                continue;
            if (!allow_inactive)
            {
                uint32_t inactive_h = filtered_hash_u32(row * 0x7f4a7c15u + nb * 0x94d049bbu + 0x2545f491u);
                if ((inactive_h % 1000u) >= inactive_sample_permille)
                    continue;
            }
            else if (inactive_sample_permille > 0 && inactive_sample_permille < 1000)
            {
                uint32_t target_h = filtered_hash_u32(row * 0x517cc1b7u + nb * 0x6d2b79f5u + 0x9e3779b9u);
                if ((target_h % 1000u) >= inactive_sample_permille)
                    continue;
            }
        }
        if (sample_permille < 1000)
        {
            uint32_t h = filtered_hash_u32(row * 0x9e3779b1u + nb * 0x85ebca6bu + 0x165667b1u);
            if ((h % 1000u) >= sample_permille)
                continue;
        }
        atomicAdd(reverse_edges, 1ull);
        uint32_t stored_row = row;
        if (replacement_mode == 2 && N <= 0x00ffffffu)
        {







            stored_row = ((i & 0xffu) << 24) | (row & 0x00ffffffu);
            atomicAdd(&reverse_counts[local_nb], 1u);
            if (reverse_cap == 0)
                continue;





            const uint32_t target_lb = offsets[nb];
            const uint32_t target_le = offsets[nb + 1];
            uint32_t target_label_count = target_le - target_lb;
            if (target_label_count > reverse_cap)
                target_label_count = reverse_cap;
            for (uint32_t q = 0; q < target_label_count; ++q)
            {
                const uint32_t shared_label = labels[target_lb + q];
                if (shared_label == universal_label ||
                    !point_has_exact_label_device(row, shared_label, offsets, labels))
                    continue;
                const uint32_t base_width = reverse_cap / target_label_count;
                const uint32_t remainder = reverse_cap % target_label_count;
                const uint32_t width = base_width + (q < remainder ? 1u : 0u);
                const uint32_t slot_begin = q * base_width + (q < remainder ? q : remainder);
                if (width == 0)
                    continue;
                const uint32_t local_slot =
                    filtered_hash_u32(row * 0x9e3779b1u + nb * 0x85ebca6bu +
                                      shared_label * 0x7f4a7c15u) % width;
                uint32_t *slot_ptr =
                    &reverse_ids[(size_t)local_nb * reverse_cap + slot_begin + local_slot];
                uint32_t old = atomicMin(slot_ptr, stored_row);
                if (stored_row <= old)
                    atomicAdd(reverse_kept, 1ull);
            }
            touched_rows[local_nb] = 1;
            continue;
        }
        uint32_t pos = atomicAdd(&reverse_counts[local_nb], 1u);
        if (pos < reverse_cap)
        {
            reverse_ids[(size_t)local_nb * reverse_cap + pos] = stored_row;
            touched_rows[local_nb] = 1;
            atomicAdd(reverse_kept, 1ull);
        }
        else if (replacement_mode == 1 && reverse_cap > 0)
        {
            uint32_t slot = filtered_hash_u32(row * 0x9e3779b1u + nb * 0x85ebca6bu + 0x27d4eb2du) % reverse_cap;
            atomicExch(&reverse_ids[(size_t)local_nb * reverse_cap + slot], row);
            touched_rows[local_nb] = 1;
            atomicAdd(reverse_kept, 1ull);
        }
    }
}

__global__ void count_u8_flags_kernel(const uint8_t *flags, uint32_t N, unsigned long long *count)
{
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && flags[row] != 0)
        atomicAdd(count, 1ull);
}

__global__ void filtered_trace_graph_row_kernel(const uint32_t *graph, const uint32_t *degree,
                                                uint32_t N, uint32_t R, const uint32_t *offsets,
                                                const uint32_t *labels, uint32_t row,
                                                uint32_t iteration, uint32_t stage)
{
    if (blockIdx.x != 0 || threadIdx.x != 0 || row >= N)
        return;
    uint32_t deg = degree[row] > R ? R : degree[row];
    printf("[gpu_filtered_build_trace] stage=%s iter=%u row=%u degree=%u labels=",
           stage == 0 ? "forward_pruned" : "reverse_pruned", iteration, row, deg);
    for (uint32_t p = offsets[row]; p < offsets[row + 1]; ++p)
    {
        uint32_t count = 0;
        for (uint32_t i = 0; i < deg; ++i)
        {
            uint32_t candidate = graph[(size_t)row * R + i];
            if (candidate < N && point_has_exact_label_device(candidate, labels[p], offsets, labels))
                ++count;
        }
        printf("%u:%u ", labels[p], count);
    }
    printf("\n");
}

template <typename DataT>
__global__ void filtered_reverse_apply_to_work_kernel(const DataT *data,
                                                      uint32_t N,
                                                      uint32_t dim,
                                                      uint32_t R,
                                                      uint32_t C,
                                                      uint32_t reverse_cap,
                                                      uint32_t reverse_dst_begin,
                                                      uint32_t reverse_dst_count,
                                                      uint32_t reverse_apply_common_sources,
                                                      uint32_t reverse_dup_source_merge,
                                                      uint32_t compatible_only,
                                                      const uint8_t *touched_rows,
                                                      const uint32_t *touched_ids,
                                                      uint32_t touched_id_count,
                                                      const uint32_t *reverse_ids,
                                                      const uint32_t *reverse_counts,
                                                      uint32_t reverse_replacement_mode,
                                                      const uint32_t *graph_cur,
                                                      const uint32_t *degree_cur,
                                                      uint32_t *graph_work,
                                                      uint32_t *degree_work,
                                                      float *graph_dists,
                                                      uint8_t *graph_sources,
                                                      const uint32_t *offsets,
                                                      const uint32_t *labels,
                                                      uint32_t universal_label,
                                                      uint32_t build_trace_row,
                                                      unsigned long long *label_checks,
                                                      unsigned long long *universal_pass)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = tid;
    uint32_t work_row = tid;
    if (touched_ids)
    {
        if (tid >= touched_id_count)
            return;
        row = touched_ids[tid];
    }
    else if (row >= N || touched_rows[row] == 0)
    {
        return;
    }
    if (row >= N)
        return;
    const size_t cur_base = (size_t)row * R;
    const size_t work_base = (size_t)work_row * C;
    uint32_t out = 0;
    uint32_t deg = degree_cur[row];
    if (deg > R)
        deg = R;
    for (uint32_t i = 0; i < C; ++i)
    {
        graph_work[work_base + i] = INVALID_ID;
        graph_dists[work_base + i] = FLT_MAX;
        if (graph_sources)
            graph_sources[work_base + i] = 0;
    }





    uint32_t row_label_begin = offsets[row];
    uint32_t row_label_end = offsets[row + 1];
    for (uint32_t p = row_label_begin; p < row_label_end && out < C; ++p)
    {
        uint32_t label = labels[p];
        uint32_t best = INVALID_ID;
        float best_distance = FLT_MAX;
        for (uint32_t i = 0; i < deg; ++i)
        {
            uint32_t candidate = graph_cur[cur_base + i];
            if (candidate == INVALID_ID || candidate >= N || candidate == row ||
                !point_has_exact_label_device(candidate, label, offsets, labels))
                continue;
            float distance = point_distance_device<DataT>(data, dim, row, candidate);
            if (best == INVALID_ID || distance < best_distance ||
                (distance == best_distance && candidate < best))
            {
                best = candidate;
                best_distance = distance;
            }
        }
        if (best == INVALID_ID)
            continue;
        bool duplicate = false;
        for (uint32_t j = 0; j < out; ++j)
            duplicate = duplicate || graph_work[work_base + j] == best;
        if (duplicate)
            continue;
        graph_work[work_base + out] = best;
        graph_dists[work_base + out] = best_distance;
        if (graph_sources)
            graph_sources[work_base + out] = 3;
        ++out;
    }




    if (row < reverse_dst_begin)
        return;
    uint32_t reverse_local_row = row - reverse_dst_begin;
    if (reverse_local_row >= reverse_dst_count)
        return;
    uint32_t rc = reverse_counts[reverse_local_row];


    if (reverse_replacement_mode == 2 && rc > 0)
        rc = reverse_cap;
    else if (rc > reverse_cap)
        rc = reverse_cap;
    const size_t rb = (size_t)reverse_local_row * reverse_cap;
    for (uint32_t i = 0; i < rc && out < C; ++i)
    {
        uint32_t cand = reverse_ids[rb + i];
        if (reverse_replacement_mode == 2)
            cand &= 0x00ffffffu;
        if (cand == INVALID_ID || cand >= N || cand == row)
            continue;
        if (!has_common_label_device(row, cand, offsets, labels, universal_label, label_checks, universal_pass))
            continue;
        bool dup = false;
        for (uint32_t j = 0; j < out; ++j)
        {
            if (graph_work[work_base + j] == cand)
            {
                dup = true;
                break;
            }
        }
        if (dup)
            continue;
        graph_work[work_base + out] = cand;
        graph_dists[work_base + out] = point_distance_device<DataT>(data, dim, row, cand);
        if (graph_sources)
            graph_sources[work_base + out] = 5;
        ++out;
    }
    for (uint32_t i = 0; i < deg && out < C; ++i)
    {
        uint32_t nb = graph_cur[cur_base + i];
        if (nb == INVALID_ID || nb >= N || nb == row)
            continue;
        const bool common =
            has_common_label_device(row, nb, offsets, labels, universal_label, label_checks, universal_pass);
        if (compatible_only != 0 && !common)
            continue;
        bool dup = false;
        for (uint32_t j = 0; j < out; ++j)
            dup = dup || graph_work[work_base + j] == nb;
        if (dup)
            continue;
        graph_work[work_base + out] = nb;
        graph_dists[work_base + out] = point_distance_device<DataT>(data, dim, row, nb);
        if (graph_sources)
        {
            if (reverse_apply_common_sources != 0 && common)
                graph_sources[work_base + out] = 2;
            else
                graph_sources[work_base + out] = 1;
        }
        ++out;
    }
    degree_work[work_row] = out;
    if (row == build_trace_row)
    {
        printf("[gpu_filtered_build_trace] stage=reverse_work row=%u out=%u labels=", row, out);
        for (uint32_t p = row_label_begin; p < row_label_end; ++p)
        {
            uint32_t count = 0;
            for (uint32_t i = 0; i < out; ++i)
            {
                uint32_t candidate = graph_work[work_base + i];
                if (candidate < N && point_has_exact_label_device(candidate, labels[p], offsets, labels))
                    ++count;
            }
            printf("%u:%u ", labels[p], count);
        }
        printf("ids=");
        for (uint32_t i = 0; i < out; ++i)
            printf("%u%s", graph_work[work_base + i], i + 1 == out ? "" : ",");
        printf("\n");
    }
}

template <typename DataT>
__global__ void filtered_final_prune_to_compact_kernel(const DataT *data,
                                                       uint32_t dim,
                                                       const uint32_t *in_graph,
                                                       const uint32_t *in_degree,
                                                       uint32_t *out_graph,
                                                       uint32_t *out_degree,
                                                       uint32_t N,
                                                       uint32_t R,
                                                       float alpha,
                                                       const uint32_t *offsets,
                                                       const uint32_t *labels,
                                                       uint32_t universal_label,
                                                       unsigned long long *label_checks,
                                                       unsigned long long *label_rejects,
                                                       unsigned long long *prune_rejects,
                                                       unsigned long long *universal_pass)
{
    uint32_t row = blockIdx.x;
    if (row >= N || threadIdx.x != 0)
        return;

    const uint32_t MAX_LOCAL_R = 128;
    uint32_t cand_ids[MAX_LOCAL_R];
    float cand_dists[MAX_LOCAL_R];
    uint8_t removed[MAX_LOCAL_R];
    uint32_t cand_count = 0;
    uint32_t deg = in_degree[row];
    for (uint32_t j = 0; j < deg && j < R && cand_count < MAX_LOCAL_R; ++j)
    {
        uint32_t nb = in_graph[(size_t)row * R + j];
        if (nb >= N || nb == row)
            continue;
        bool dup = false;
        for (uint32_t k = 0; k < cand_count; ++k)
        {
            if (cand_ids[k] == nb)
            {
                dup = true;
                break;
            }
        }
        if (dup)
            continue;
        cand_ids[cand_count] = nb;
        cand_dists[cand_count] = point_distance_device<DataT>(data, dim, row, nb);
        removed[cand_count] = 0;
        ++cand_count;
    }

    for (uint32_t i = 0; i < cand_count; ++i)
    {
        for (uint32_t j = i + 1; j < cand_count; ++j)
        {
            if (cand_dists[j] < cand_dists[i] ||
                (cand_dists[j] == cand_dists[i] && cand_ids[j] < cand_ids[i]))
            {
                float td = cand_dists[i];
                cand_dists[i] = cand_dists[j];
                cand_dists[j] = td;
                uint32_t ti = cand_ids[i];
                cand_ids[i] = cand_ids[j];
                cand_ids[j] = ti;
            }
        }
    }

    uint32_t selected_ids[MAX_LOCAL_R];
    uint32_t selected_count = 0;
    for (uint32_t i = 0; i < cand_count && selected_count < R; ++i)
    {
        if (removed[i])
            continue;
        uint32_t cur = cand_ids[i];
        selected_ids[selected_count++] = cur;
        for (uint32_t j = i + 1; j < cand_count; ++j)
        {
            if (removed[j])
                continue;
            uint32_t other = cand_ids[j];
            bool cover = selected_covers_row_candidate_common_labels_device(row, cur, other, offsets, labels,
                                                                            universal_label, label_checks,
                                                                            universal_pass);
            if (!cover)
            {
                atomicAdd(label_rejects, 1ull);
                continue;
            }
            float djk = point_distance_device<DataT>(data, dim, other, cur);
            if (djk > 0.0f && alpha * djk <= cand_dists[j])
            {
                removed[j] = 1;
                atomicAdd(prune_rejects, 1ull);
            }
        }
    }

    for (uint32_t j = 0; j < R; ++j)
        out_graph[(size_t)row * R + j] = INVALID_ID;
    for (uint32_t j = 0; j < selected_count; ++j)
        out_graph[(size_t)row * R + j] = selected_ids[j];
    out_degree[row] = selected_count;
}

template <typename DataT>
__global__ void filtered_final_prune_to_compact_from_work_kernel(const DataT *data,
                                                                 uint32_t dim,
                                                                 const uint32_t *work_graph,
                                                                 const float *work_dists,
                                                                 const uint32_t *work_degree,
                                                                 uint32_t *out_graph,
                                                                 uint32_t *out_degree,
                                                                 uint32_t N,
                                                                 uint32_t R,
                                                                 uint32_t C,
                                                                 float alpha,
                                                                 uint32_t work_prune_cap,
                                                                 const uint32_t *offsets,
                                                                 const uint32_t *labels,
                                                                 uint32_t universal_label,
                                                                 unsigned long long *label_checks,
                                                                 unsigned long long *label_rejects,
                                                                 unsigned long long *prune_rejects,
                                                                 unsigned long long *label_occlusion_blocked,
                                                                 unsigned long long *geometry_occluded,
                                                                 unsigned long long *candidate_rejected_by_label,
                                                                 unsigned long long *refill_count,
                                                                 unsigned long long *degree_sum,
                                                                 unsigned long long *universal_pass)
{
    uint32_t row = blockIdx.x;
    if (row >= N || threadIdx.x != 0)
        return;

    const uint32_t MAX_LOCAL_C = 256;
    uint32_t cand_ids[MAX_LOCAL_C];
    float cand_dists[MAX_LOCAL_C];
    uint8_t removed[MAX_LOCAL_C];
    uint32_t cand_count = 0;
    uint32_t deg = work_degree[row];
    if (deg > C)
        deg = C;
    if (work_prune_cap == 0 || work_prune_cap > MAX_LOCAL_C)
        work_prune_cap = MAX_LOCAL_C;
    const size_t work_base = (size_t)row * C;
    for (uint32_t j = 0; j < deg && cand_count < work_prune_cap; ++j)
    {
        uint32_t nb = work_graph[work_base + j];
        if (nb >= N || nb == row || nb == INVALID_ID)
            continue;
        float d = work_dists ? work_dists[work_base + j] : FLT_MAX;
        if (d == FLT_MAX)
            d = point_distance_device<DataT>(data, dim, row, nb);
        bool dup = false;
        for (uint32_t k = 0; k < cand_count; ++k)
        {
            if (cand_ids[k] == nb)
            {
                dup = true;
                break;
            }
        }
        if (dup)
            continue;
        cand_ids[cand_count] = nb;
        cand_dists[cand_count] = d;
        removed[cand_count] = 0;
        ++cand_count;
    }

    for (uint32_t i = 0; i < cand_count; ++i)
    {
        for (uint32_t j = i + 1; j < cand_count; ++j)
        {
            if (cand_dists[j] < cand_dists[i] ||
                (cand_dists[j] == cand_dists[i] && cand_ids[j] < cand_ids[i]))
            {
                float td = cand_dists[i];
                cand_dists[i] = cand_dists[j];
                cand_dists[j] = td;
                uint32_t ti = cand_ids[i];
                cand_ids[i] = cand_ids[j];
                cand_ids[j] = ti;
            }
        }
    }

    const size_t out_base = (size_t)row * R;
    for (uint32_t j = 0; j < R; ++j)
        out_graph[out_base + j] = INVALID_ID;

    uint32_t selected_count = 0;
    for (uint32_t i = 0; i < cand_count && selected_count < R; ++i)
    {
        if (removed[i])
            continue;
        uint32_t cur = cand_ids[i];
        out_graph[out_base + selected_count++] = cur;
        for (uint32_t j = i + 1; j < cand_count; ++j)
        {
            if (removed[j])
                continue;
            uint32_t other = cand_ids[j];
            bool cover = selected_covers_row_candidate_common_labels_device(row, cur, other, offsets, labels,
                                                                            universal_label, label_checks,
                                                                            universal_pass);
            if (!cover)
            {
                atomicAdd(label_occlusion_blocked, 1ull);
                continue;
            }
            float djk = point_distance_device<DataT>(data, dim, other, cur);
            if (djk > 0.0f && alpha * djk <= cand_dists[j])
            {
                removed[j] = 1;
                atomicAdd(prune_rejects, 1ull);
                atomicAdd(geometry_occluded, 1ull);
            }
        }
    }



    atomicAdd(degree_sum, (unsigned long long)selected_count);
    (void)label_rejects;
    (void)candidate_rejected_by_label;
    out_degree[row] = selected_count;
}

template <typename DataT, uint32_t MAX_LOCAL_C>
__global__ void filtered_refine_prune_to_compact_kernel(const DataT *data,
                                                        uint32_t dim,
                                                        const uint8_t *active_rows,
                                                        const uint32_t *work_graph,
                                                        const float *work_dists,
                                                        const uint8_t *work_sources,
                                                        const uint32_t *work_degree,
                                                        uint32_t *graph_cur,
                                                        uint32_t *degree_cur,
                                                        uint32_t N,
                                                        uint32_t R,
                                                        uint32_t C,
                                                        float alpha,
                                                        uint32_t source_aware_alpha,
                                                        uint32_t work_prune_cap,
                                                        uint32_t bridge_protect,
                                                        uint32_t noncommon_bridge_protect,
                                                        uint32_t refill_target_degree,
                                                        uint32_t source_priority_mode,
                                                        uint32_t source_merge_mode,
                                                        uint32_t filtered_top_protect,
                                                        uint32_t common_degree_cap,
                                                        uint32_t target_common_degree_cap,
                                                        uint32_t position_top_protect,
                                                        uint32_t position_filtered_reserve,
                                                        uint32_t position_reverse_protect,
                                                        uint32_t position_reverse_reserve,
                                                        uint32_t path_protect,
                                                        uint32_t consensus_protect,
                                                        uint32_t consensus_require_filtered_top,
                                                        const uint32_t *active_ids,
                                                        uint32_t active_id_count,
                                                        const uint8_t *target_labels,
                                                        const uint32_t *offsets,
                                                        const uint32_t *labels,
                                                        uint32_t universal_label,
                                                        unsigned long long *label_checks,
                                                        unsigned long long *label_rejects,
                                                        unsigned long long *prune_rejects,
                                                        unsigned long long *label_occlusion_blocked,
                                                        unsigned long long *geometry_occluded,
                                                        unsigned long long *candidate_rejected_by_label,
                                                        unsigned long long *refill_count,
                                                        unsigned long long *degree_sum,
                                                        unsigned long long *source_selected_counts,
                                                        unsigned long long *source_refill_counts,
                                                        unsigned long long *universal_pass)
{
    uint32_t work_row = blockIdx.x;
    uint32_t row = work_row;
    if (active_ids)
    {
        if (work_row >= active_id_count)
            return;
        row = active_ids[work_row];
    }
    if (row >= N || threadIdx.x != 0 || (!active_ids && active_rows[row] == 0))
        return;

    uint32_t cand_ids[MAX_LOCAL_C];
    float cand_dists[MAX_LOCAL_C];
    float cand_keys[MAX_LOCAL_C];
    uint8_t cand_sources[MAX_LOCAL_C];
    uint8_t removed[MAX_LOCAL_C];
    uint32_t cand_count = 0;
    uint32_t deg = work_degree[work_row];
    if (deg > C)
        deg = C;
    if (work_prune_cap == 0 || work_prune_cap > MAX_LOCAL_C)
        work_prune_cap = MAX_LOCAL_C;
    const size_t work_base = (size_t)work_row * C;
    const size_t out_base = (size_t)row * R;
    uint32_t position_filtered_begin = degree_cur[row];
    if (position_filtered_begin > C)
        position_filtered_begin = C;
    uint32_t position_filtered_end = position_filtered_begin + position_filtered_reserve;
    if (position_filtered_end > C)
        position_filtered_end = C;
    uint32_t position_reverse_begin = degree_cur[row];
    if (position_reverse_begin > C)
        position_reverse_begin = C;
    uint32_t position_reverse_end = position_reverse_begin + position_reverse_reserve;
    if (position_reverse_end > C)
        position_reverse_end = C;
    for (uint32_t j = 0; j < deg && cand_count < work_prune_cap; ++j)
    {
        uint32_t nb = work_graph[work_base + j];
        if (nb >= N || nb == row || nb == INVALID_ID)
            continue;
        float d = work_dists ? work_dists[work_base + j] : FLT_MAX;
        if (d == FLT_MAX)
            d = point_distance_device<DataT>(data, dim, row, nb);
        bool dup = false;
        for (uint32_t k = 0; k < cand_count; ++k)
        {
            if (cand_ids[k] == nb)
            {
                dup = true;
                if (source_merge_mode)
                {
                    uint8_t src = work_sources[work_base + j];
                    cand_sources[k] = filtered_source_merge(cand_sources[k], src, 1);
                    if (d < cand_dists[k])
                        cand_dists[k] = d;
                }
                break;
            }
        }
        if (dup)
            continue;
        cand_ids[cand_count] = nb;
        cand_dists[cand_count] = d;
        uint8_t src = work_sources ? work_sources[work_base + j] : 0;
        if (!work_sources && position_top_protect != 0 && j >= position_filtered_begin && j < position_filtered_end)
            src = 3;
        if (!work_sources && position_reverse_protect != 0 && j >= position_reverse_begin && j < position_reverse_end)
            src = 5;
        cand_sources[cand_count] = source_merge_mode ? filtered_source_to_bit(src) : src;
        cand_keys[cand_count] = d;
        removed[cand_count] = 0;
        ++cand_count;
    }
    for (uint32_t i = 0; i < cand_count; ++i)
    {
        cand_keys[i] = cand_dists[i] *
                       filtered_source_key_factor(cand_sources[i], source_priority_mode,
                                                  source_merge_mode ? 1 : 0);
    }
    for (uint32_t i = 0; i < cand_count; ++i)
    {
        for (uint32_t j = i + 1; j < cand_count; ++j)
        {
            if (cand_keys[j] < cand_keys[i] ||
                (cand_keys[j] == cand_keys[i] && cand_ids[j] < cand_ids[i]))
            {
                float td = cand_dists[i];
                cand_dists[i] = cand_dists[j];
                cand_dists[j] = td;
                float tk = cand_keys[i];
                cand_keys[i] = cand_keys[j];
                cand_keys[j] = tk;
                uint8_t ts = cand_sources[i];
                cand_sources[i] = cand_sources[j];
                cand_sources[j] = ts;
                uint32_t ti = cand_ids[i];
                cand_ids[i] = cand_ids[j];
                cand_ids[j] = ti;
            }
        }
    }







    if (g_filtered_geometric_only_prune == 0 && cand_count > R)
    {
        constexpr uint32_t MAX_ROW_LABELS = 16;
        uint32_t label_begin = offsets[row];
        uint32_t label_count = offsets[row + 1] - label_begin;
        if (label_count > MAX_ROW_LABELS)
            label_count = MAX_ROW_LABELS;
        uint32_t cursors[MAX_ROW_LABELS] = {};
        uint32_t selected_per_label[MAX_ROW_LABELS] = {};
        uint32_t keep_count = 0;
        while (keep_count < R && label_count > 0)
        {
            bool progress = false;
            for (uint32_t q = 0; q < label_count && keep_count < R; ++q)
            {
                if (g_filtered_balance_per_label > 0 &&
                    selected_per_label[q] >= g_filtered_balance_per_label)
                    continue;
                const uint32_t label = labels[label_begin + q];
                while (cursors[q] < cand_count)
                {
                    const uint32_t idx = cursors[q]++;
                    if (removed[idx] != 0)
                        continue;
                    if (!point_has_exact_label_device(cand_ids[idx], label, offsets, labels))
                        continue;
                    removed[idx] = 1;
                    ++keep_count;
                    ++selected_per_label[q];
                    progress = true;
                    break;
                }
            }
            if (!progress)
                break;
        }
        for (uint32_t i = 0; i < cand_count && keep_count < R; ++i)
        {
            if (removed[i] == 0)
            {
                removed[i] = 1;
                ++keep_count;
            }
        }
        for (uint32_t i = 0; i < cand_count; ++i)
            removed[i] = removed[i] != 0 ? 0 : 1;
    }
    for (uint32_t j = 0; j < R; ++j)
        graph_cur[out_base + j] = INVALID_ID;
    uint32_t selected_count = 0;
    if (bridge_protect > R)
        bridge_protect = R;
    if (noncommon_bridge_protect > R)
        noncommon_bridge_protect = R;
    if (refill_target_degree == 0 || refill_target_degree > R)
        refill_target_degree = R;
    if (common_degree_cap > R)
        common_degree_cap = R;
    if (target_common_degree_cap > R)
        target_common_degree_cap = R;
    if (target_labels && target_common_degree_cap > 0)
    {
        uint32_t lb = offsets[row];
        uint32_t le = offsets[row + 1];
        for (uint32_t p = lb; p < le; ++p)
        {
            uint32_t lbl = labels[p];
            if (target_labels[lbl] != 0)
            {
                common_degree_cap = target_common_degree_cap;
                break;
            }
        }
    }
    uint32_t selected_common_count = 0;
    for (uint32_t j = 0; j < deg && j < bridge_protect && selected_count < R; ++j)
    {
        uint32_t id = work_graph[work_base + j];
        if (id == INVALID_ID || id >= N || id == row)
            continue;
        bool dup = false;
        for (uint32_t k = 0; k < selected_count; ++k)
        {
            if (graph_cur[out_base + k] == id)
            {
                dup = true;
                break;
            }
        }
        if (!dup)
        {
            graph_cur[out_base + selected_count++] = id;
            if (common_degree_cap > 0 &&
                has_common_label_device(row, id, offsets, labels, universal_label, label_checks, universal_pass))
                ++selected_common_count;
            if (source_selected_counts && work_sources)
                filtered_source_count_add(source_selected_counts, work_sources[work_base + j], 0);
        }
    }
    uint32_t protected_noncommon_bridge = 0;
    if (noncommon_bridge_protect > 0)
    {
        uint32_t original_limit = degree_cur[row];
        if (original_limit > deg)
            original_limit = deg;
        for (uint32_t j = 0; j < original_limit && selected_count < R &&
                             protected_noncommon_bridge < noncommon_bridge_protect;
             ++j)
        {
            uint32_t id = work_graph[work_base + j];
            if (id == INVALID_ID || id >= N || id == row)
                continue;
            if (has_common_label_device(row, id, offsets, labels, universal_label, label_checks, universal_pass))
                continue;
            bool dup = false;
            for (uint32_t k = 0; k < selected_count; ++k)
            {
                if (graph_cur[out_base + k] == id)
                {
                    dup = true;
                    break;
                }
            }
            if (dup)
                continue;
            graph_cur[out_base + selected_count++] = id;
            ++protected_noncommon_bridge;
            if (source_selected_counts && work_sources)
                filtered_source_count_add(source_selected_counts, work_sources[work_base + j], 0);
        }
    }
    if (selected_count > 0)
    {
        for (uint32_t s = 0; s < selected_count; ++s)
        {
            uint32_t cur = graph_cur[out_base + s];
            for (uint32_t j = 0; j < cand_count; ++j)
            {
                if (removed[j] || cand_ids[j] == cur)
                {
                    if (cand_ids[j] == cur)
                        removed[j] = 1;
                    continue;
                }
                uint32_t other = cand_ids[j];
                bool cover = selected_covers_row_candidate_common_labels_device(row, cur, other, offsets, labels,
                                                                                universal_label, label_checks,
                                                                                universal_pass);
                if (!cover)
                {
                    atomicAdd(label_occlusion_blocked, 1ull);
                    continue;
                }
                float djk = point_distance_device<DataT>(data, dim, other, cur);
                float candidate_alpha = alpha;
                if (source_aware_alpha != 0 && work_sources)
                {
                    const bool cand_original_common =
                        filtered_source_has(cand_sources[j], source_merge_mode ? 1 : 0, 2);
                    const bool cand_filtered_top =
                        filtered_source_has(cand_sources[j], source_merge_mode ? 1 : 0, 3);
                    const bool cand_visited =
                        filtered_source_has(cand_sources[j], source_merge_mode ? 1 : 0, 4);
                    const bool cand_reverse =
                        filtered_source_has(cand_sources[j], source_merge_mode ? 1 : 0, 5);
                    const bool cand_path =
                        filtered_source_has(cand_sources[j], source_merge_mode ? 1 : 0, 6);
                    const bool strong_source = cand_original_common || cand_filtered_top || cand_reverse || cand_path;
                    if (cand_visited && !strong_source)
                        candidate_alpha = 1.0f;
                    else if (cand_visited && strong_source && candidate_alpha > 1.1f)
                        candidate_alpha = 1.1f;
                }
                if (djk > 0.0f && candidate_alpha * djk <= cand_dists[j])
                {
                    removed[j] = 1;
                    atomicAdd(prune_rejects, 1ull);
                    atomicAdd(geometry_occluded, 1ull);
                }
            }
        }
    }
    uint32_t protected_consensus = 0;
    if (consensus_protect > R)
        consensus_protect = R;
    if (work_sources && consensus_protect > 0)
    {
        for (uint32_t i = 0; i < cand_count && selected_count < R && protected_consensus < consensus_protect; ++i)
        {
            if (removed[i] || filtered_source_strong_count(cand_sources[i], source_merge_mode ? 1 : 0) < 2)
                continue;
            if (consensus_require_filtered_top != 0 &&
                !filtered_source_has(cand_sources[i], source_merge_mode ? 1 : 0, 3))
                continue;
            uint32_t cur = cand_ids[i];
            bool dup = false;
            for (uint32_t s = 0; s < selected_count; ++s)
            {
                if (graph_cur[out_base + s] == cur)
                {
                    dup = true;
                    break;
                }
            }
            if (dup)
            {
                removed[i] = 1;
                continue;
            }
            if (common_degree_cap > 0 &&
                has_common_label_device(row, cur, offsets, labels, universal_label, label_checks, universal_pass) &&
                selected_common_count >= common_degree_cap)
                continue;
            graph_cur[out_base + selected_count++] = cur;
            removed[i] = 1;
            ++protected_consensus;
            if (common_degree_cap > 0 &&
                has_common_label_device(row, cur, offsets, labels, universal_label, label_checks, universal_pass))
                ++selected_common_count;
            filtered_source_count_add(source_selected_counts, cand_sources[i], source_merge_mode ? 1 : 0);
            for (uint32_t j = i + 1; j < cand_count; ++j)
            {
                if (removed[j])
                    continue;
                uint32_t other = cand_ids[j];
                bool cover = selected_covers_row_candidate_common_labels_device(row, cur, other, offsets, labels,
                                                                                universal_label, label_checks,
                                                                                universal_pass);
                if (!cover)
                {
                    atomicAdd(label_occlusion_blocked, 1ull);
                    continue;
                }
                float djk = point_distance_device<DataT>(data, dim, other, cur);
                if (djk > 0.0f && alpha * djk <= cand_dists[j])
                {
                    removed[j] = 1;
                    atomicAdd(prune_rejects, 1ull);
                    atomicAdd(geometry_occluded, 1ull);
                }
            }
        }
    }
    uint32_t protected_filtered_top = 0;
    if (filtered_top_protect > R)
        filtered_top_protect = R;
    if ((work_sources || position_top_protect != 0) && filtered_top_protect > 0)
    {
        for (uint32_t i = 0; i < cand_count && selected_count < R && protected_filtered_top < filtered_top_protect; ++i)
        {
            if (removed[i] || !filtered_source_has(cand_sources[i], source_merge_mode ? 1 : 0, 3))
                continue;
            uint32_t cur = cand_ids[i];
            bool dup = false;
            for (uint32_t s = 0; s < selected_count; ++s)
            {
                if (graph_cur[out_base + s] == cur)
                {
                    dup = true;
                    break;
                }
            }
            if (dup)
            {
                removed[i] = 1;
                continue;
            }
            graph_cur[out_base + selected_count++] = cur;
            removed[i] = 1;
            ++protected_filtered_top;
            if (common_degree_cap > 0 &&
                has_common_label_device(row, cur, offsets, labels, universal_label, label_checks, universal_pass))
                ++selected_common_count;
            filtered_source_count_add(source_selected_counts, cand_sources[i], source_merge_mode ? 1 : 0);
            for (uint32_t j = i + 1; j < cand_count; ++j)
            {
                if (removed[j])
                    continue;
                uint32_t other = cand_ids[j];
                bool cover = selected_covers_row_candidate_common_labels_device(row, cur, other, offsets, labels,
                                                                                universal_label, label_checks,
                                                                                universal_pass);
                if (!cover)
                {
                    atomicAdd(label_occlusion_blocked, 1ull);
                    continue;
                }
                float djk = point_distance_device<DataT>(data, dim, other, cur);
                if (djk > 0.0f && alpha * djk <= cand_dists[j])
                {
                    removed[j] = 1;
                    atomicAdd(prune_rejects, 1ull);
                    atomicAdd(geometry_occluded, 1ull);
                }
            }
        }
    }
    uint32_t protected_reverse = 0;
    if (position_reverse_protect > R)
        position_reverse_protect = R;
    if ((work_sources || position_reverse_protect != 0) && position_reverse_protect > 0)
    {
        for (uint32_t i = 0; i < cand_count && selected_count < R && protected_reverse < position_reverse_protect; ++i)
        {
            if (removed[i] || !filtered_source_has(cand_sources[i], source_merge_mode ? 1 : 0, 5))
                continue;
            uint32_t cur = cand_ids[i];
            bool dup = false;
            for (uint32_t s = 0; s < selected_count; ++s)
            {
                if (graph_cur[out_base + s] == cur)
                {
                    dup = true;
                    break;
                }
            }
            if (dup)
            {
                removed[i] = 1;
                continue;
            }
            graph_cur[out_base + selected_count++] = cur;
            removed[i] = 1;
            ++protected_reverse;
            if (common_degree_cap > 0 &&
                has_common_label_device(row, cur, offsets, labels, universal_label, label_checks, universal_pass))
                ++selected_common_count;
            filtered_source_count_add(source_selected_counts, cand_sources[i], source_merge_mode ? 1 : 0);
            for (uint32_t j = i + 1; j < cand_count; ++j)
            {
                if (removed[j])
                    continue;
                uint32_t other = cand_ids[j];
                bool cover = selected_covers_row_candidate_common_labels_device(row, cur, other, offsets, labels,
                                                                                universal_label, label_checks,
                                                                                universal_pass);
                if (!cover)
                {
                    atomicAdd(label_occlusion_blocked, 1ull);
                    continue;
                }
                float djk = point_distance_device<DataT>(data, dim, other, cur);
                if (djk > 0.0f && alpha * djk <= cand_dists[j])
                {
                    removed[j] = 1;
                    atomicAdd(prune_rejects, 1ull);
                    atomicAdd(geometry_occluded, 1ull);
                }
            }
        }
    }
    uint32_t protected_path = 0;
    if (path_protect > R)
        path_protect = R;
    if (work_sources && path_protect > 0)
    {
        for (uint32_t i = 0; i < cand_count && selected_count < R && protected_path < path_protect; ++i)
        {
            if (removed[i] || !filtered_source_has(cand_sources[i], source_merge_mode ? 1 : 0, 6))
                continue;
            uint32_t cur = cand_ids[i];
            bool dup = false;
            for (uint32_t s = 0; s < selected_count; ++s)
            {
                if (graph_cur[out_base + s] == cur)
                {
                    dup = true;
                    break;
                }
            }
            if (dup)
            {
                removed[i] = 1;
                continue;
            }
            if (common_degree_cap > 0 &&
                has_common_label_device(row, cur, offsets, labels, universal_label, label_checks, universal_pass) &&
                selected_common_count >= common_degree_cap)
                continue;
            graph_cur[out_base + selected_count++] = cur;
            removed[i] = 1;
            ++protected_path;
            if (common_degree_cap > 0 &&
                has_common_label_device(row, cur, offsets, labels, universal_label, label_checks, universal_pass))
                ++selected_common_count;
            filtered_source_count_add(source_selected_counts, cand_sources[i], source_merge_mode ? 1 : 0);
            for (uint32_t j = i + 1; j < cand_count; ++j)
            {
                if (removed[j])
                    continue;
                uint32_t other = cand_ids[j];
                bool cover = selected_covers_row_candidate_common_labels_device(row, cur, other, offsets, labels,
                                                                                universal_label, label_checks,
                                                                                universal_pass);
                if (!cover)
                {
                    atomicAdd(label_occlusion_blocked, 1ull);
                    continue;
                }
                float djk = point_distance_device<DataT>(data, dim, other, cur);
                if (djk > 0.0f && alpha * djk <= cand_dists[j])
                {
                    removed[j] = 1;
                    atomicAdd(prune_rejects, 1ull);
                    atomicAdd(geometry_occluded, 1ull);
                }
            }
        }
    }
    for (uint32_t i = 0; i < cand_count && selected_count < refill_target_degree; ++i)
    {
        if (removed[i])
            continue;
        uint32_t cur = cand_ids[i];
        bool cur_common = false;
        if (common_degree_cap > 0)
        {
            cur_common = has_common_label_device(row, cur, offsets, labels, universal_label, label_checks, universal_pass);
            if (cur_common && selected_common_count >= common_degree_cap)
                continue;
        }
        graph_cur[out_base + selected_count++] = cur;
        if (cur_common)
            ++selected_common_count;
        filtered_source_count_add(source_selected_counts, cand_sources[i], source_merge_mode ? 1 : 0);
        for (uint32_t j = i + 1; j < cand_count; ++j)
        {
            if (removed[j])
                continue;
            uint32_t other = cand_ids[j];
            bool cover = selected_covers_row_candidate_common_labels_device(row, cur, other, offsets, labels,
                                                                            universal_label, label_checks,
                                                                            universal_pass);
            if (!cover)
            {
                atomicAdd(label_occlusion_blocked, 1ull);
                continue;
            }
            float djk = point_distance_device<DataT>(data, dim, other, cur);
            if (djk > 0.0f && alpha * djk <= cand_dists[j])
            {
                removed[j] = 1;
                atomicAdd(prune_rejects, 1ull);
                atomicAdd(geometry_occluded, 1ull);
            }
        }
    }



    degree_cur[row] = selected_count;
    atomicAdd(degree_sum, (unsigned long long)selected_count);
    (void)label_rejects;
    (void)candidate_rejected_by_label;
}








__device__ __forceinline__ float warp_l2_distance(const uint8_t *__restrict__ data, uint32_t dim,
                                                   uint32_t a, uint32_t b, uint32_t lane)
{
    if (dim == 128)
    {
        float tile_sum = 0.0f;
        if (lane < 8)
        {
            const uint4 xv = reinterpret_cast<const uint4 *>(data + (size_t)a * 128)[lane];
            const uint4 yv = reinterpret_cast<const uint4 *>(data + (size_t)b * 128)[lane];
            const uint8_t *xb = reinterpret_cast<const uint8_t *>(&xv);
            const uint8_t *yb = reinterpret_cast<const uint8_t *>(&yv);
#pragma unroll
            for (uint32_t i = 0; i < 16; ++i)
            {
                const int delta = (int)xb[i] - (int)yb[i];
                tile_sum += (float)(delta * delta);
            }
        }
        for (int offset = 16; offset; offset >>= 1)
            tile_sum += __shfl_down_sync(0xffffffffu, tile_sum, offset);
        return __shfl_sync(0xffffffffu, tile_sum, 0);
    }
    unsigned int xx_yy = 0, xy = 0;
    const uint32_t *x4 = reinterpret_cast<const uint32_t *>(data + (size_t)a * dim);
    const uint32_t *y4 = reinterpret_cast<const uint32_t *>(data + (size_t)b * dim);
    const uint32_t dim4 = dim >> 2;
    for (uint32_t d = lane; d < dim4; d += 32)
    {
        const uint32_t xv = x4[d], yv = y4[d];
        xx_yy = __dp4a(xv, xv, xx_yy);
        xx_yy = __dp4a(yv, yv, xx_yy);
        xy = __dp4a(xv, yv, xy);
    }
    int sum = (int)xx_yy - 2 * (int)xy;
    if ((dim & 3u) != 0)
    {
        const uint8_t *x = data + (size_t)a * dim;
        const uint8_t *y = data + (size_t)b * dim;
        for (uint32_t d = (dim & ~3u) + lane; d < dim; d += 32)
        {
            const int delta = (int)x[d] - (int)y[d];
            sum += delta * delta;
        }
    }
    for (uint32_t offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    return (float)__shfl_sync(0xffffffffu, sum, 0);
}

__device__ __forceinline__ float warp_l2_distance(const __half *__restrict__ data, uint32_t dim,
                                                   uint32_t a, uint32_t b, uint32_t lane)
{
    float sum = 0.0f;
    const half2 *x2 = reinterpret_cast<const half2 *>(data + (size_t)a * dim);
    const half2 *y2 = reinterpret_cast<const half2 *>(data + (size_t)b * dim);
    for (uint32_t d = lane; d < (dim >> 1); d += 32)
    {
        const half2 delta = __hsub2(x2[d], y2[d]);
        const float2 square = __half22float2(__hmul2(delta, delta));
        sum += square.x + square.y;
    }
    if ((dim & 1u) && lane == 0)
    {
        const float delta = __half2float(data[(size_t)a * dim + dim - 1]) -
                            __half2float(data[(size_t)b * dim + dim - 1]);
        sum += delta * delta;
    }
    for (uint32_t offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    return __shfl_sync(0xffffffffu, sum, 0);
}

__device__ __forceinline__ float warp_l2_distance(const float *__restrict__ data, uint32_t dim,
                                                   uint32_t a, uint32_t b, uint32_t lane)
{
    float sum = 0.0f;
    const float *x = data + (size_t)a * dim;
    const float *y = data + (size_t)b * dim;
    for (uint32_t d = lane; d < dim; d += 32)
    {
        const float delta = x[d] - y[d];
        sum += delta * delta;
    }
    for (uint32_t offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    return __shfl_sync(0xffffffffu, sum, 0);
}

template <int CacheDim, typename DataT>
__device__ __forceinline__ void init_node_vector_cache(const DataT *, uint32_t, uint32_t, uint32_t,
                                                        half2 *, uint32_t *kind)
{
    *kind = 0;
}

template <int CacheDim>
__device__ __forceinline__ void init_node_vector_cache(const __half *data, uint32_t dim, uint32_t node,
                                                        uint32_t lane, half2 *cache, uint32_t *kind)
{
    if constexpr (CacheDim > 0)
    {
        if (dim == (uint32_t)CacheDim && !(dim & 1u))
        {
            constexpr uint32_t slots = ((CacheDim / 2) + 31) / 32;
            const half2 *row = reinterpret_cast<const half2 *>(data + (size_t)node * dim);
#pragma unroll
            for (uint32_t i = 0; i < slots; ++i)
            {
                const uint32_t index = lane + i * 32;
                cache[i] = index < (uint32_t)CacheDim / 2 ? row[index] : __float2half2_rn(0.0f);
            }
            *kind = 1;
            return;
        }
    }
    *kind = 0;
}

template <int CacheDim, typename DataT>
__device__ __forceinline__ float warp_l2_distance_from_node_cache(const DataT *data, uint32_t dim,
                                                                   uint32_t node, uint32_t other,
                                                                   uint32_t lane, const half2 *, uint32_t)
{
    return warp_l2_distance(data, dim, node, other, lane);
}

template <int CacheDim>
__device__ __forceinline__ float warp_l2_distance_from_node_cache(const __half *data, uint32_t dim,
                                                                   uint32_t node, uint32_t other,
                                                                   uint32_t lane, const half2 *cache,
                                                                   uint32_t kind)
{
    if constexpr (CacheDim > 0)
    {
        if (kind == 1 && dim == (uint32_t)CacheDim)
        {
            float sum = 0.0f;
            constexpr uint32_t slots = ((CacheDim / 2) + 31) / 32;
            const half2 *row = reinterpret_cast<const half2 *>(data + (size_t)other * dim);
#pragma unroll
            for (uint32_t i = 0; i < slots; ++i)
            {
                const uint32_t index = lane + i * 32;
                if (index < (uint32_t)CacheDim / 2)
                {
                    const half2 delta = __hsub2(cache[i], row[index]);
                    const float2 square = __half22float2(__hmul2(delta, delta));
                    sum += square.x + square.y;
                }
            }
            for (int offset = 16; offset; offset >>= 1)
                sum += __shfl_down_sync(0xffffffffu, sum, offset);
            return __shfl_sync(0xffffffffu, sum, 0);
        }
    }
    return warp_l2_distance(data, dim, node, other, lane);
}

__global__ void canonical_make_task_rows(uint32_t N, const uint32_t *offsets, uint32_t *task_rows)
{
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N)
        return;
    for (uint32_t p = offsets[row]; p < offsets[row + 1]; ++p)
        task_rows[p] = row;
}

__global__ void canonical_convert_float_to_half(const float *__restrict__ input,
                                                 __half *__restrict__ output, size_t count)
{
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
        output[index] = __float2half_rn(input[index]);
}

__device__ __forceinline__ void canonical_recompute_worst_warp(uint32_t count, const uint32_t *ids,
                                                               const float *distances, int *worst_pos,
                                                               float *worst_distance, uint32_t lane)
{
    float local_distance = -FLT_MAX;
    uint32_t local_id = 0;
    int local_pos = -1;
    for (uint32_t begin = 0; begin < count; begin += 32)
    {
        const uint32_t pos = begin + lane;
        if (pos < count && (distances[pos] > local_distance ||
            (distances[pos] == local_distance && ids[pos] > local_id)))
        {
            local_distance = distances[pos]; local_id = ids[pos]; local_pos = (int)pos;
        }
    }
    for (int offset = 16; offset; offset >>= 1)
    {
        const float other_distance = __shfl_down_sync(0xffffffffu, local_distance, offset);
        const uint32_t other_id = __shfl_down_sync(0xffffffffu, local_id, offset);
        const int other_pos = __shfl_down_sync(0xffffffffu, local_pos, offset);
        if (other_pos >= 0 && (other_distance > local_distance ||
            (other_distance == local_distance && other_id > local_id)))
        {
            local_distance = other_distance; local_id = other_id; local_pos = other_pos;
        }
    }
    *worst_pos = __shfl_sync(0xffffffffu, local_pos, 0);
    *worst_distance = __shfl_sync(0xffffffffu, local_distance, 0);
}

__device__ __forceinline__ bool insert_top_l_state_warp(uint32_t id, float distance, uint32_t capacity,
                                                         uint32_t *ids, float *distances, uint8_t *expanded,
                                                         uint32_t *count, int *worst_pos,
                                                         float *worst_distance, uint32_t lane)
{
    if (id == INVALID_ID)
        return false;
    int duplicate = -1;
    for (uint32_t begin = 0; begin < *count && duplicate < 0; begin += 32)
    {
        const uint32_t pos = begin + lane;
        const uint32_t mask = __ballot_sync(0xffffffffu, pos < *count && ids[pos] == id);
        if (mask)
            duplicate = (int)(begin + __ffs(mask) - 1);
    }
    duplicate = __shfl_sync(0xffffffffu, duplicate, 0);
    int changed = 0;
    if (duplicate >= 0)
    {
        if (lane == 0 && distance < distances[duplicate])
        {
            distances[duplicate] = distance; expanded[duplicate] = 0; changed = 1;
        }
    }
    else if (*count < capacity)
    {
        if (lane == 0)
        {
            ids[*count] = id; distances[*count] = distance; expanded[*count] = 0; ++*count; changed = 1;
        }
    }
    else
    {
        canonical_recompute_worst_warp(*count, ids, distances, worst_pos, worst_distance, lane);
        const uint32_t worst_id = *worst_pos >= 0 ? ids[*worst_pos] : 0;
        if (*worst_pos >= 0 && (distance < *worst_distance ||
            (distance == *worst_distance && id < worst_id)) && lane == 0)
        {
            ids[*worst_pos] = id; distances[*worst_pos] = distance; expanded[*worst_pos] = 0; changed = 1;
        }
    }
    *count = __shfl_sync(0xffffffffu, *count, 0);
    changed = __shfl_sync(0xffffffffu, changed, 0);
    if (changed)
        canonical_recompute_worst_warp(*count, ids, distances, worst_pos, worst_distance, lane);
    return changed != 0;
}

__device__ __forceinline__ int select_next_unexpanded_size_warp(uint32_t count, const uint32_t *ids,
                                                                 const float *distances,
                                                                 const uint8_t *expanded, uint32_t lane)
{
    float best_distance = FLT_MAX;
    uint32_t best_id = UINT32_MAX;
    int best_pos = -1;
    for (uint32_t begin = 0; begin < count; begin += 32)
    {
        const uint32_t pos = begin + lane;
        if (pos < count && !expanded[pos] && (distances[pos] < best_distance ||
            (distances[pos] == best_distance && ids[pos] < best_id)))
        {
            best_distance = distances[pos]; best_id = ids[pos]; best_pos = (int)pos;
        }
    }
    for (int offset = 16; offset; offset >>= 1)
    {
        const float other_distance = __shfl_down_sync(0xffffffffu, best_distance, offset);
        const uint32_t other_id = __shfl_down_sync(0xffffffffu, best_id, offset);
        const int other_pos = __shfl_down_sync(0xffffffffu, best_pos, offset);
        if (other_pos >= 0 && (other_distance < best_distance ||
            (other_distance == best_distance && other_id < best_id)))
        {
            best_distance = other_distance; best_id = other_id; best_pos = other_pos;
        }
    }
    return __shfl_sync(0xffffffffu, best_pos, 0);
}

template <int CacheDim, typename DataT>
__global__ void canonical_filtered_task_search(const DataT *__restrict__ data, uint32_t N, uint32_t dim,
                                                uint32_t R, uint32_t task_count, uint32_t pool_capacity,
                                                uint32_t keep, uint32_t expansion_steps,
                                                const uint32_t *task_positions, const uint32_t *task_rows,
                                                const uint32_t *task_labels,
                                                const uint32_t *label_starts, uint32_t num_labels,
                                                const uint32_t *label_point_offsets,
                                                const uint32_t *label_points, uint32_t label_seed_count,
                                                uint32_t universal_label, const uint32_t *offsets,
                                                const uint32_t *labels, const uint32_t *graph,
                                                const uint32_t *degrees, uint32_t *task_candidates,
                                                float *task_candidate_distances, uint32_t fused_label_threshold,
                                                uint32_t retain_posting_anchor)
{
    extern __shared__ unsigned char shared_raw[];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp_in_block = threadIdx.x >> 5;
    const uint32_t warps_per_block = blockDim.x >> 5;
    const uint32_t task = blockIdx.x * warps_per_block + warp_in_block;
    uint32_t *shared_ids = reinterpret_cast<uint32_t *>(shared_raw) + (size_t)warp_in_block * pool_capacity;
    float *shared_distances = reinterpret_cast<float *>(shared_raw +
        (size_t)warps_per_block * pool_capacity * sizeof(uint32_t)) + (size_t)warp_in_block * pool_capacity;
    uint8_t *shared_expanded = reinterpret_cast<uint8_t *>(shared_raw +
        (size_t)warps_per_block * pool_capacity * (sizeof(uint32_t) + sizeof(float))) +
        (size_t)warp_in_block * pool_capacity;
    for (uint32_t i = lane; i < pool_capacity; i += 32)
    {
        shared_ids[i] = INVALID_ID;
        shared_distances[i] = FLT_MAX;
        shared_expanded[i] = 0;
    }
    if (task >= task_count)
        return;
    __syncwarp();
    const uint32_t position = task_positions ? task_positions[task] : task;
    const uint32_t row = task_rows[position];
    const uint32_t label = task_labels[position];
    if (fused_label_threshold && offsets[row + 1] - offsets[row] <= fused_label_threshold)
        return;
    uint32_t count = 0;
    int worst_pos = -1;
    float worst_distance = -FLT_MAX;
    half2 row_cache[CacheDim > 0 ? ((CacheDim / 2 + 31) / 32) : 1];
    uint32_t cache_kind = 0;
    init_node_vector_cache<CacheDim>(data, dim, row, lane, row_cache, &cache_kind);
    if (row >= N || label >= num_labels || label == universal_label)
    {
        for (uint32_t i = lane; i < keep; i += 32)
        {
            task_candidates[(size_t)position * keep + i] = INVALID_ID;
            task_candidate_distances[(size_t)position * keep + i] = FLT_MAX;
        }
        return;
    }
    const uint32_t start = label_starts[label];
    uint32_t posting_anchor = INVALID_ID;
    float posting_anchor_distance = FLT_MAX;
    if (start < N && start != row)
    {
        const float distance = warp_l2_distance_from_node_cache<CacheDim>(
            data, dim, row, start, lane, row_cache, cache_kind);
        if (g_canonical_profile && lane == 0) atomicAdd(&g_canonical_distance_counts[0], 1ull);
        insert_top_l_state_warp(start, distance, pool_capacity, shared_ids, shared_distances, shared_expanded,
                                &count, &worst_pos, &worst_distance, lane);
    }
    if (label_point_offsets && label_points && label_seed_count)
    {
        const uint32_t point_begin = label_point_offsets[label];
        const uint32_t point_count = label_point_offsets[label + 1] - point_begin;
        const uint32_t seed_count = min(min(label_seed_count, 16u), point_count);
        if (retain_posting_anchor && point_count > 0)
        {
            uint32_t anchor_position = filtered_hash_u32(row ^ (label * 0x9e3779b1u)) % point_count;
            posting_anchor = label_points[point_begin + anchor_position];
            if (posting_anchor == row && point_count > 1)
                posting_anchor = label_points[point_begin + ((anchor_position + 1u) % point_count)];
            if (posting_anchor >= N || posting_anchor == row)
                posting_anchor = INVALID_ID;
            if (posting_anchor < N)
                posting_anchor_distance = warp_l2_distance_from_node_cache<CacheDim>(
                    data, dim, row, posting_anchor, lane, row_cache, cache_kind);
        }
        for (uint32_t s = 0; s < seed_count; ++s)
        {
            const uint32_t seed = label_points[point_begin +
                (filtered_hash_u32(row + s * 7919u + label * 17u) % point_count)];
            if (seed >= N || seed == row)
                continue;
            const float distance = warp_l2_distance_from_node_cache<CacheDim>(
                data, dim, row, seed, lane, row_cache, cache_kind);
            if (g_canonical_profile && lane == 0) atomicAdd(&g_canonical_distance_counts[0], 1ull);
            insert_top_l_state_warp(seed, distance, pool_capacity, shared_ids, shared_distances,
                                    shared_expanded, &count, &worst_pos, &worst_distance, lane);
        }
    }
    uint32_t degree = degrees[row] > R ? R : degrees[row];
    for (uint32_t j = 0; j < degree; ++j)
    {
        const uint32_t candidate = graph[(size_t)row * R + j];
        if (candidate >= N || candidate == row ||
            !point_has_exact_label_device(candidate, label, offsets, labels))
            continue;
        const float distance = warp_l2_distance_from_node_cache<CacheDim>(
            data, dim, row, candidate, lane, row_cache, cache_kind);
        if (g_canonical_profile && lane == 0) atomicAdd(&g_canonical_distance_counts[0], 1ull);
        insert_top_l_state_warp(candidate, distance, pool_capacity, shared_ids, shared_distances, shared_expanded,
                                &count, &worst_pos, &worst_distance, lane);
    }
    for (uint32_t step = 0; step < expansion_steps; ++step)
    {
        const int expand_position = select_next_unexpanded_size_warp(
            count, shared_ids, shared_distances, shared_expanded, lane);
        if (expand_position < 0)
            break;
        if (lane == 0)
            shared_expanded[expand_position] = 1;
        __syncwarp();
        const uint32_t expand = shared_ids[expand_position];
        const uint32_t expand_degree = degrees[expand] > R ? R : degrees[expand];
        for (uint32_t j = 0; j < expand_degree; ++j)
        {
            const uint32_t candidate = graph[(size_t)expand * R + j];
            if (candidate >= N || candidate == row ||
                !point_has_exact_label_device(candidate, label, offsets, labels))
                continue;
            const float distance = warp_l2_distance_from_node_cache<CacheDim>(
                data, dim, row, candidate, lane, row_cache, cache_kind);
            if (g_canonical_profile && lane == 0) atomicAdd(&g_canonical_distance_counts[0], 1ull);
            insert_top_l_state_warp(candidate, distance, pool_capacity, shared_ids, shared_distances,
                                    shared_expanded, &count, &worst_pos, &worst_distance, lane);
        }
    }
    for (uint32_t i = lane; i < keep; i += 32)
    {
        uint32_t output_id = INVALID_ID;
        float output_distance = FLT_MAX;
        if (retain_posting_anchor && posting_anchor < N && i + 1u == keep)
        {
            output_id = posting_anchor;
            output_distance = posting_anchor_distance;
        }
        else
        {
            const uint32_t wanted = i;
            uint32_t seen = 0;
            for (uint32_t p = 0; p < count; ++p)
            {
                if (retain_posting_anchor && shared_ids[p] == posting_anchor)
                    continue;
                if (seen++ == wanted)
                {
                    output_id = shared_ids[p];
                    output_distance = shared_distances[p];
                    break;
                }
            }
        }
        task_candidates[(size_t)position * keep + i] = output_id;
        task_candidate_distances[(size_t)position * keep + i] = output_distance;
    }
}

template <int CacheDim, typename DataT>
__global__ void canonical_fused_filtered_row_search(
    const DataT *__restrict__ data, uint32_t N, uint32_t dim, uint32_t R, uint32_t pool_capacity,
    uint32_t expansion_steps, uint32_t fused_label_threshold, const uint32_t *label_starts,
    uint32_t num_labels, uint32_t universal_label, const uint32_t *offsets, const uint32_t *labels,
    const uint32_t *graph, const uint32_t *degrees, uint32_t *task_candidates,
    float *task_candidate_distances)
{
    extern __shared__ unsigned char shared_raw[];
    const uint32_t lane = threadIdx.x & 31u, warp_in_block = threadIdx.x >> 5;
    const uint32_t warps_per_block = blockDim.x >> 5;
    const uint32_t row = blockIdx.x * warps_per_block + warp_in_block;
    uint32_t *ids = reinterpret_cast<uint32_t *>(shared_raw) + (size_t)warp_in_block * pool_capacity;
    float *distances = reinterpret_cast<float *>(shared_raw +
        (size_t)warps_per_block * pool_capacity * sizeof(uint32_t)) + (size_t)warp_in_block * pool_capacity;
    uint8_t *expanded = reinterpret_cast<uint8_t *>(shared_raw +
        (size_t)warps_per_block * pool_capacity * (sizeof(uint32_t) + sizeof(float))) +
        (size_t)warp_in_block * pool_capacity;
    for (uint32_t i = lane; i < pool_capacity; i += 32)
    {
        ids[i] = INVALID_ID; distances[i] = FLT_MAX; expanded[i] = 0;
    }
    if (row >= N)
        return;
    const uint32_t label_begin = offsets[row], label_end = offsets[row + 1];
    const uint32_t label_count = label_end - label_begin;
    if (!label_count || label_count > fused_label_threshold || label_count > 64)
        return;
    __syncwarp();
    half2 row_cache[CacheDim > 0 ? ((CacheDim / 2 + 31) / 32) : 1];
    uint32_t cache_kind = 0, count = 0;
    int worst_pos = -1;
    float worst_distance = -FLT_MAX;
    init_node_vector_cache<CacheDim>(data, dim, row, lane, row_cache, &cache_kind);
    for (uint32_t p = label_begin; p < label_end; ++p)
    {
        const uint32_t label = labels[p];
        const uint32_t start = label < num_labels ? label_starts[label] : INVALID_ID;
        if (label != universal_label && start < N && start != row)
        {
            const float distance = warp_l2_distance_from_node_cache<CacheDim>(
                data, dim, row, start, lane, row_cache, cache_kind);
            if (g_canonical_profile && lane == 0) atomicAdd(&g_canonical_distance_counts[0], 1ull);
            insert_top_l_state_warp(start, distance, pool_capacity, ids, distances, expanded,
                                    &count, &worst_pos, &worst_distance, lane);
        }
    }
    const auto covers_row_label = [&](uint32_t candidate) {
        bool covers = false;
        for (uint32_t p = label_begin; !covers && p < label_end; ++p)
            covers = labels[p] != universal_label && point_has_exact_label_device(candidate, labels[p], offsets, labels);
        return covers;
    };
    const uint32_t degree = min(degrees[row], R);
    for (uint32_t j = 0; j < degree; ++j)
    {
        const uint32_t candidate = graph[(size_t)row * R + j];
        if (candidate >= N || candidate == row || !covers_row_label(candidate))
            continue;
        const float distance = warp_l2_distance_from_node_cache<CacheDim>(
            data, dim, row, candidate, lane, row_cache, cache_kind);
        if (g_canonical_profile && lane == 0) atomicAdd(&g_canonical_distance_counts[0], 1ull);
        insert_top_l_state_warp(candidate, distance, pool_capacity, ids, distances, expanded,
                                &count, &worst_pos, &worst_distance, lane);
    }
    for (uint32_t step = 0; step < expansion_steps; ++step)
    {
        const int pos = select_next_unexpanded_size_warp(count, ids, distances, expanded, lane);
        if (pos < 0) break;
        if (lane == 0) expanded[pos] = 1;
        __syncwarp();
        const uint32_t expand = ids[pos], expand_degree = min(degrees[expand], R);
        for (uint32_t j = 0; j < expand_degree; ++j)
        {
            const uint32_t candidate = graph[(size_t)expand * R + j];
            if (candidate >= N || candidate == row || !covers_row_label(candidate))
                continue;
            const float distance = warp_l2_distance_from_node_cache<CacheDim>(
                data, dim, row, candidate, lane, row_cache, cache_kind);
            if (g_canonical_profile && lane == 0) atomicAdd(&g_canonical_distance_counts[0], 1ull);
            insert_top_l_state_warp(candidate, distance, pool_capacity, ids, distances, expanded,
                                    &count, &worst_pos, &worst_distance, lane);
        }
    }
    for (uint32_t i = lane; i < pool_capacity; i += 32)
    {
        const uint32_t candidate = i < count ? ids[i] : INVALID_ID;
        uint64_t coverage_mask = 0;
        for (uint32_t q = 0; q < label_count && candidate < N; ++q)
            if (point_has_exact_label_device(candidate, labels[label_begin + q], offsets, labels))
                coverage_mask |= 1ull << q;
        for (uint32_t q = 0; q < label_count; ++q)
        {
            const uint32_t p = label_begin + q;
            const bool covered = (coverage_mask & (1ull << q)) != 0;
            task_candidates[(size_t)p * pool_capacity + i] = covered ? candidate : INVALID_ID;
            task_candidate_distances[(size_t)p * pool_capacity + i] = covered ? distances[i] : FLT_MAX;
        }
    }
}

template <int CacheDim, typename DataT>
__global__ void canonical_ordinary_search(const DataT *__restrict__ data, uint32_t N, uint32_t dim, uint32_t R,
                                          uint32_t pool_capacity, uint32_t keep, uint32_t output_stride,
                                          uint32_t expansion_steps, uint32_t global_start,
                                          const uint32_t *row_ids, uint32_t row_count,
                                          const uint32_t *graph, const uint32_t *degrees,
                                          const uint32_t *offsets, const uint32_t *labels,
                                          const uint32_t *label_starts, uint32_t num_labels,
                                          const uint32_t *label_point_offsets,
                                          const uint32_t *label_points, uint32_t label_seed_count,
                                          uint32_t *ordinary_candidates, float *ordinary_distances,
                                          uint32_t unrestricted_traversal,
                                          uint32_t compatible_output_only)
{
    extern __shared__ unsigned char shared_raw[];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp_in_block = threadIdx.x >> 5;
    const uint32_t warps_per_block = blockDim.x >> 5;
    const uint32_t warp = blockIdx.x * warps_per_block + warp_in_block;
    uint32_t *ids = reinterpret_cast<uint32_t *>(shared_raw) + (size_t)warp_in_block * pool_capacity;
    float *distances = reinterpret_cast<float *>(shared_raw +
        (size_t)warps_per_block * pool_capacity * sizeof(uint32_t)) + (size_t)warp_in_block * pool_capacity;
    uint8_t *expanded = reinterpret_cast<uint8_t *>(shared_raw +
        (size_t)warps_per_block * pool_capacity * (sizeof(uint32_t) + sizeof(float))) +
        (size_t)warp_in_block * pool_capacity;
    for (uint32_t i = lane; i < pool_capacity; i += 32)
    {
        ids[i] = INVALID_ID;
        distances[i] = FLT_MAX;
        expanded[i] = 0;
    }
    if (warp >= row_count)
        return;
    const uint32_t row = row_ids ? row_ids[warp] : warp;
    __syncwarp();
    uint32_t count = 0;
    int worst_pos = -1;
    float worst_distance = -FLT_MAX;
    half2 row_cache[CacheDim > 0 ? ((CacheDim / 2 + 31) / 32) : 1];
    uint32_t cache_kind = 0;
    init_node_vector_cache<CacheDim>(data, dim, row, lane, row_cache, &cache_kind);
    const auto compatible = [&](uint32_t candidate) {
        bool common = false;
        for (uint32_t p = offsets[row]; !common && p < offsets[row + 1]; ++p)
            common = point_has_exact_label_device(candidate, labels[p], offsets, labels);
        return common;
    };
    if (!unrestricted_traversal)
    for (uint32_t p = offsets[row]; p < offsets[row + 1]; ++p)
    {
        const uint32_t label = labels[p];
        if (label >= num_labels)
            continue;
        const uint32_t start = label_starts[label];
        if (start < N && start != row)
        {
            const float distance = warp_l2_distance_from_node_cache<CacheDim>(
                data, dim, row, start, lane, row_cache, cache_kind);
            insert_top_l_state_warp(start, distance, pool_capacity, ids, distances, expanded,
                                    &count, &worst_pos, &worst_distance, lane);
        }
        const uint32_t point_begin = label_point_offsets[label];
        const uint32_t point_count = label_point_offsets[label + 1] - point_begin;
        const uint32_t seed_count = min(label_seed_count, point_count);
        for (uint32_t s = 0; s < seed_count; ++s)
        {
            const uint32_t seed = label_points[point_begin +
                (filtered_hash_u32(row + s * 7919u + label * 17u) % point_count)];
            if (seed >= N || seed == row)
                continue;
            const float distance = warp_l2_distance_from_node_cache<CacheDim>(
                data, dim, row, seed, lane, row_cache, cache_kind);
            insert_top_l_state_warp(seed, distance, pool_capacity, ids, distances, expanded,
                                    &count, &worst_pos, &worst_distance, lane);
        }
    }
    if (global_start < N && global_start != row &&
        (unrestricted_traversal || compatible(global_start)))
    {
        const float distance = warp_l2_distance_from_node_cache<CacheDim>(
            data, dim, row, global_start, lane, row_cache, cache_kind);
        if (g_canonical_profile && lane == 0) atomicAdd(&g_canonical_distance_counts[1], 1ull);
        insert_top_l_state_warp(global_start, distance, pool_capacity, ids, distances, expanded,
                                &count, &worst_pos, &worst_distance, lane);
    }
    uint32_t degree = degrees[row] > R ? R : degrees[row];
    for (uint32_t j = 0; j < degree; ++j)
    {
        const uint32_t candidate = graph[(size_t)row * R + j];
        if (candidate >= N || candidate == row ||
            (!unrestricted_traversal && !compatible(candidate)))
            continue;
        const float distance = warp_l2_distance_from_node_cache<CacheDim>(
            data, dim, row, candidate, lane, row_cache, cache_kind);
        if (g_canonical_profile && lane == 0) atomicAdd(&g_canonical_distance_counts[1], 1ull);
        insert_top_l_state_warp(candidate, distance, pool_capacity, ids, distances, expanded,
                                &count, &worst_pos, &worst_distance, lane);
    }
    for (uint32_t step = 0; step < expansion_steps; ++step)
    {
        const int expand_position = select_next_unexpanded_size_warp(count, ids, distances, expanded, lane);
        if (expand_position < 0)
            break;
        if (lane == 0)
            expanded[expand_position] = 1;
        __syncwarp();
        const uint32_t expand = ids[expand_position];
        const uint32_t expand_degree = degrees[expand] > R ? R : degrees[expand];
        for (uint32_t j = 0; j < expand_degree; ++j)
        {
            const uint32_t candidate = graph[(size_t)expand * R + j];
            if (candidate >= N || candidate == row ||
                (!unrestricted_traversal && !compatible(candidate)))
                continue;
            const float distance = warp_l2_distance_from_node_cache<CacheDim>(
                data, dim, row, candidate, lane, row_cache, cache_kind);
            if (g_canonical_profile && lane == 0) atomicAdd(&g_canonical_distance_counts[1], 1ull);
            insert_top_l_state_warp(candidate, distance, pool_capacity, ids, distances, expanded,
                                    &count, &worst_pos, &worst_distance, lane);
        }
    }
    for (uint32_t i = lane; i < keep; i += 32)
    {
        uint32_t output_id = INVALID_ID;
        float output_distance = FLT_MAX;
        if (!compatible_output_only)
        {
            if (i < count) { output_id = ids[i]; output_distance = distances[i]; }
        }
        else
        {
            uint32_t seen = 0;
            for (uint32_t p = 0; p < count; ++p)
                if (compatible(ids[p]) && seen++ == i)
                {
                    output_id = ids[p]; output_distance = distances[p]; break;
                }
        }
        ordinary_candidates[(size_t)row * output_stride + i] = output_id;
        ordinary_distances[(size_t)row * output_stride + i] = output_distance;
    }
}

__device__ __forceinline__ void canonical_merge_one_warp(uint32_t candidate, float distance, uint32_t row,
                                                          uint32_t N, uint32_t C, uint32_t *work,
                                                          float *work_distances, uint32_t &count,
                                                          uint32_t lane)
{
    if (candidate >= N || candidate == row || count >= C)
        return;
    int duplicate = -1;
    for (uint32_t begin = 0; begin < count && duplicate < 0; begin += 32)
    {
        const uint32_t pos = begin + lane;
        const uint32_t mask = __ballot_sync(0xffffffffu, pos < count && work[pos] == candidate);
        if (mask)
            duplicate = (int)(begin + __ffs(mask) - 1);
    }
    duplicate = __shfl_sync(0xffffffffu, duplicate, 0);
    if (duplicate >= 0)
    {
        if (lane == 0 && distance < work_distances[duplicate])
            work_distances[duplicate] = distance;
        return;
    }
    if (lane == 0)
    {
        work[count] = candidate;
        work_distances[count] = distance;
        ++count;
    }
    count = __shfl_sync(0xffffffffu, count, 0);
}

__global__ void canonical_merge_candidates(uint32_t N, uint32_t R, uint32_t C, uint32_t per_label_keep,
                                            uint32_t ordinary_keep, const uint32_t *offsets,
                                            const uint32_t *labels,
                                            const uint32_t *label_point_offsets,
                                            const uint32_t *label_points,
                                            const uint32_t *row_ids, uint32_t row_count,
                                            const uint32_t *graph, const uint32_t *degrees,
                                            const uint32_t *task_candidates, const float *task_distances,
                                            const uint32_t *ordinary_candidates, const float *ordinary_distances,
                                            uint32_t *work, float *work_distances, uint32_t *work_degrees,
                                            uint32_t preserve_unrestricted, uint32_t bridge_reserve,
                                            uint32_t additional_posting_anchor)
{
    __shared__ uint32_t ordinary_ids_cache[64];
    __shared__ float ordinary_distance_cache[64];
    const uint32_t lane = threadIdx.x & 31u;
    if (blockIdx.x >= row_count)
        return;
    const uint32_t row = row_ids ? row_ids[blockIdx.x] : blockIdx.x;
    uint32_t *row_work = work + (size_t)row * C;
    float *row_distances = work_distances + (size_t)row * C;


    for (uint32_t i = lane; i < ordinary_keep; i += 32)
    {
        ordinary_ids_cache[i] = ordinary_candidates[(size_t)row * C + i];
        ordinary_distance_cache[i] = ordinary_distances[(size_t)row * C + i];
    }
    __syncwarp();
    for (uint32_t i = lane; i < C; i += 32)
    {
        row_work[i] = INVALID_ID;
        row_distances[i] = FLT_MAX;
    }
    __syncwarp();
    uint32_t count = 0;



    for (uint32_t p = offsets[row]; p < offsets[row + 1]; ++p)
        for (uint32_t i = 0; i < per_label_keep; ++i)
            canonical_merge_one_warp(task_candidates[(size_t)p * per_label_keep + i],
                task_distances[(size_t)p * per_label_keep + i], row, N, C,
                row_work, row_distances, count, lane);




    if (additional_posting_anchor && label_points && label_point_offsets)
        for (uint32_t p = offsets[row]; p < offsets[row + 1]; ++p)
        {
            const uint32_t label = labels[p];
            const uint32_t begin = label_point_offsets[label];
            const uint32_t point_count = label_point_offsets[label + 1] - begin;
            if (!point_count) continue;
            uint32_t position = filtered_hash_u32(row ^ (label * 0x9e3779b1u)) % point_count;
            uint32_t anchor = label_points[begin + position];
            if (anchor == row && point_count > 1)
                anchor = label_points[begin + ((position + 1u) % point_count)];
            canonical_merge_one_warp(anchor, FLT_MAX, row, N, C,
                                     row_work, row_distances, count, lane);
        }
    const uint32_t degree = degrees[row] > R ? R : degrees[row];
    for (uint32_t i = 0; i < degree; ++i)
    {
        const uint32_t candidate = graph[(size_t)row * R + i];
        if (candidate < N && candidate != row &&
            (preserve_unrestricted ||
             has_common_label_device(row, candidate, offsets, labels, UINT32_MAX, nullptr, nullptr)))
            canonical_merge_one_warp(candidate, FLT_MAX, row, N, C,
                                     row_work, row_distances, count, lane);
    }
    for (uint32_t i = 0; i < ordinary_keep; ++i)
        canonical_merge_one_warp(ordinary_ids_cache[i], ordinary_distance_cache[i], row, N, C,
            row_work, row_distances, count, lane);




    count = __shfl_sync(0xffffffffu, count, 0);
    if (lane == 0)
        work_degrees[row] = count;
}

template <int CacheDim, typename DataT>
__global__ void canonical_compute_work_distances(const DataT *__restrict__ data, uint32_t N, uint32_t dim,
                                                  uint32_t C, const uint32_t *work,
                                                  const uint32_t *row_ids, uint32_t row_count,
                                                  const uint32_t *work_degrees, float *work_distances)
{
    if (blockIdx.x >= row_count)
        return;
    const uint32_t row = row_ids ? row_ids[blockIdx.x] : blockIdx.x;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= N)
        return;
    const uint32_t degree = work_degrees[row] > C ? C : work_degrees[row];
    half2 row_cache[CacheDim > 0 ? ((CacheDim / 2 + 31) / 32) : 1];
    uint32_t cache_kind = 0;
    init_node_vector_cache<CacheDim>(data, dim, row, lane, row_cache, &cache_kind);
    for (uint32_t i = 0; i < degree; ++i)
    {
        const uint32_t candidate = work[(size_t)row * C + i];
        if (work_distances[(size_t)row * C + i] != FLT_MAX)
            continue;
        const float distance = warp_l2_distance_from_node_cache<CacheDim>(
            data, dim, row, candidate, lane, row_cache, cache_kind);
        if (g_canonical_profile && lane == 0) atomicAdd(&g_canonical_distance_counts[2], 1ull);
        if (lane == 0)
            work_distances[(size_t)row * C + i] = distance;
    }
}




template <typename DataT>
__global__ void canonical_exact_prune(const DataT *__restrict__ data, uint32_t N, uint32_t dim, uint32_t R,
                                      uint32_t C, float alpha, const uint32_t *offsets, const uint32_t *labels,
                                      const uint32_t *row_ids, uint32_t row_count,
                                      const uint32_t *old_graph, const uint32_t *old_degrees,
                                      const uint32_t *work, const uint32_t *work_degrees,
                                      const float *work_distances, uint8_t *removed, uint32_t compact_work_rows,
                                      uint32_t scalar_pair_distance,
                                      uint32_t *new_graph, float *new_graph_distances,
                                      uint32_t *new_degrees, uint8_t *row_changed)
{
    if (blockIdx.x >= row_count)
        return;
    const uint32_t row = row_ids ? row_ids[blockIdx.x] : blockIdx.x;
    const uint32_t work_row = compact_work_rows ? blockIdx.x : row;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= N)
        return;
    const size_t work_base = (size_t)work_row * C;
    const size_t graph_base = (size_t)row * R;
    const uint32_t candidate_count = work_degrees[work_row] > C ? C : work_degrees[work_row];
    for (uint32_t i = lane; i < candidate_count; i += 32)
        removed[work_base + i] = 0;
    for (uint32_t i = lane; i < R; i += 32)
    {
        new_graph[graph_base + i] = INVALID_ID;
        if (new_graph_distances)
            new_graph_distances[graph_base + i] = FLT_MAX;
    }
    __syncwarp();
    uint32_t selected_count = 0;
    while (selected_count < R)
    {
        float best_distance = FLT_MAX;
        uint32_t best_id = UINT32_MAX;
        uint32_t best_index = UINT32_MAX;
        for (uint32_t i = lane; i < candidate_count; i += 32)
        {
            if (removed[work_base + i])
                continue;
            const float distance = work_distances[work_base + i];
            const uint32_t id = work[work_base + i];
            if (distance < best_distance ||
                (distance == best_distance && (id < best_id || (id == best_id && i < best_index))))
            {
                best_distance = distance;
                best_id = id;
                best_index = i;
            }
        }
        for (uint32_t offset = 16; offset > 0; offset >>= 1)
        {
            const float other_distance = __shfl_down_sync(0xffffffffu, best_distance, offset);
            const uint32_t other_id = __shfl_down_sync(0xffffffffu, best_id, offset);
            const uint32_t other_index = __shfl_down_sync(0xffffffffu, best_index, offset);
            if (other_distance < best_distance ||
                (other_distance == best_distance &&
                 (other_id < best_id || (other_id == best_id && other_index < best_index))))
            {
                best_distance = other_distance;
                best_id = other_id;
                best_index = other_index;
            }
        }
        best_id = __shfl_sync(0xffffffffu, best_id, 0);
        best_index = __shfl_sync(0xffffffffu, best_index, 0);
        if (best_index == UINT32_MAX)
            break;
        if (lane == 0)
        {
            new_graph[graph_base + selected_count] = best_id;
            if (new_graph_distances)
                new_graph_distances[graph_base + selected_count] = best_distance;
            removed[work_base + best_index] = 1;
        }
        ++selected_count;
        __syncwarp();
        for (uint32_t i = 0; i < candidate_count; ++i)
        {
            if (removed[work_base + i])
                continue;
            const uint32_t candidate = work[work_base + i];
            if (!filtered_selected_covers_common_labels(row, best_id, candidate, offsets, labels))
                continue;
            float pair_distance = 0.0f;
            if (scalar_pair_distance)
            {
                if (lane == 0)
                    pair_distance = point_distance_device<DataT>(data, dim, best_id, candidate);
                pair_distance = __shfl_sync(0xffffffffu, pair_distance, 0);
            }
            else
            {
                pair_distance = warp_l2_distance(data, dim, best_id, candidate, lane);
            }
            if (g_canonical_profile && lane == 0) atomicAdd(&g_canonical_distance_counts[3], 1ull);
            if (pair_distance > 0.0f &&
                alpha * pair_distance <= work_distances[work_base + i] && lane == 0)
                removed[work_base + i] = 1;
            __syncwarp();
        }
        __syncwarp();
    }
    if (lane == 0 && row_changed)
    {
        new_degrees[row] = selected_count;
        bool changed = !old_degrees || old_degrees[row] != selected_count;
        for (uint32_t i = 0; !changed && i < selected_count; ++i)
        {
            bool found = false;
            for (uint32_t j = 0; old_graph && !found && j < old_degrees[row]; ++j)
                found = old_graph[graph_base + j] == new_graph[graph_base + i];
            changed = !found;
        }
        row_changed[row] = changed ? 1 : 0;
    }
    else if (lane == 0)
        new_degrees[row] = selected_count;
}

__global__ void canonical_reverse_count(uint32_t N, uint32_t R, const uint32_t *graph,
                                        const uint32_t *degrees, const uint32_t *source_ids,
                                        uint32_t source_count, const uint32_t *label_offsets,
                                        const uint32_t *labels, uint32_t universal_label,
                                        uint32_t *reverse_offsets)
{
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= source_count)
        return;
    const uint32_t row = source_ids ? source_ids[index] : index;
    const uint32_t degree = degrees[row] > R ? R : degrees[row];
    for (uint32_t i = 0; i < degree; ++i)
    {
        const uint32_t destination = graph[(size_t)row * R + i];




        if (destination < N && destination != row)
            atomicAdd(reverse_offsets + destination + 1, 1u);
    }
}

__global__ void canonical_reverse_fill(uint32_t N, uint32_t R, const uint32_t *graph,
                                       const uint32_t *degrees, uint32_t *reverse_cursor,
                                       uint32_t *reverse_ids, const uint32_t *source_ids,
                                       uint32_t source_count, const uint32_t *label_offsets,
                                       const uint32_t *labels, uint32_t universal_label)
{
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= source_count)
        return;
    const uint32_t row = source_ids ? source_ids[index] : index;
    const uint32_t degree = degrees[row] > R ? R : degrees[row];
    for (uint32_t i = 0; i < degree; ++i)
    {
        const uint32_t destination = graph[(size_t)row * R + i];
        if (destination < N && destination != row)
            reverse_ids[atomicAdd(reverse_cursor + destination, 1u)] = row;
    }
}

__global__ void canonical_reverse_candidate_count(uint32_t N, uint32_t R, const uint32_t *graph,
                                                   const uint32_t *degrees,
                                                   const uint32_t *reverse_offsets,
                                                   const uint32_t *reverse_ids,
                                                   uint32_t *candidate_offsets, uint8_t *touched)
{
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N)
        return;
    const uint32_t degree = degrees[row] > R ? R : degrees[row];
    const uint32_t begin = reverse_offsets[row];
    const uint32_t end = reverse_offsets[row + 1];
    touched[row] = begin < end ? 1 : 0;
    if (begin == end)
    {
        candidate_offsets[row + 1] = 0;
        return;
    }
    uint32_t count = degree;
    for (uint32_t p = begin; p < end; ++p)
    {
        const uint32_t candidate = reverse_ids[p];
        bool duplicate = candidate >= N || candidate == row;
        for (uint32_t i = 0; !duplicate && i < degree; ++i)
            duplicate = graph[(size_t)row * R + i] == candidate;



        count += uint32_t(!duplicate);
    }
    candidate_offsets[row + 1] = count;
}

__global__ void canonical_reverse_candidate_fill(uint32_t N, uint32_t R, const uint32_t *graph,
                                                  const uint32_t *degrees,
                                                  const uint32_t *reverse_offsets,
                                                  const uint32_t *reverse_ids,
                                                  const uint32_t *candidate_offsets,
                                                  uint32_t *candidate_ids,
                                                  const float *graph_distances,
                                                  float *candidate_distances)
{
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N)
        return;
    uint32_t out = candidate_offsets[row];
    const uint32_t degree = degrees[row] > R ? R : degrees[row];
    for (uint32_t i = 0; i < degree; ++i)
    {
        candidate_ids[out++] = graph[(size_t)row * R + i];
        candidate_distances[out - 1] = graph_distances ?
            graph_distances[(size_t)row * R + i] : FLT_MAX;
    }
    const uint32_t begin = reverse_offsets[row];
    const uint32_t end = reverse_offsets[row + 1];
    for (uint32_t p = begin; p < end; ++p)
    {
        const uint32_t candidate = reverse_ids[p];
        bool duplicate = candidate >= N || candidate == row;
        for (uint32_t i = 0; !duplicate && i < degree; ++i)
            duplicate = graph[(size_t)row * R + i] == candidate;
        if (!duplicate)
        {
            candidate_ids[out++] = candidate;
            candidate_distances[out - 1] = FLT_MAX;
        }
    }
}

template <typename DataT>
__global__ void canonical_csr_candidate_distances(const DataT *__restrict__ data, uint32_t N, uint32_t dim,
                                                   const uint32_t *candidate_offsets,
                                                   const uint32_t *row_ids, uint32_t row_count,
                                                   const uint32_t *candidate_ids, float *candidate_distances)
{
    if (blockIdx.x >= row_count)
        return;
    const uint32_t row = row_ids ? row_ids[blockIdx.x] : blockIdx.x;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= N)
        return;
    for (uint32_t p = candidate_offsets[row]; p < candidate_offsets[row + 1]; ++p)
    {
        if (candidate_distances[p] != FLT_MAX)
            continue;
        const float distance = warp_l2_distance(data, dim, row, candidate_ids[p], lane);
        if (g_canonical_profile && lane == 0) atomicAdd(&g_canonical_distance_counts[4], 1ull);
        if (lane == 0)
            candidate_distances[p] = distance;
    }
}

template <typename DataT>
__global__ void canonical_exact_prune_csr(const DataT *__restrict__ data, uint32_t N, uint32_t dim, uint32_t R,
                                          float alpha, const uint32_t *label_offsets, const uint32_t *labels,
                                          const uint32_t *row_ids, uint32_t row_count,
                                          const uint32_t *candidate_offsets, const uint32_t *candidate_ids,
                                          const float *candidate_distances, uint8_t *removed,
                                          const uint32_t *old_graph, const uint32_t *old_degrees,
                                          uint32_t *new_graph, uint32_t *new_degrees, uint8_t *row_changed)
{
    if (blockIdx.x >= row_count)
        return;
    const uint32_t row = row_ids ? row_ids[blockIdx.x] : blockIdx.x;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= N)
        return;
    const uint32_t begin = candidate_offsets[row];
    const uint32_t end = candidate_offsets[row + 1];
    const size_t graph_base = (size_t)row * R;
    for (uint32_t p = begin + lane; p < end; p += 32)
        removed[p] = 0;
    for (uint32_t i = lane; i < R; i += 32)
        new_graph[graph_base + i] = INVALID_ID;
    __syncwarp();
    uint32_t selected_count = 0;




    while (selected_count < R)
    {
        float best_distance = FLT_MAX;
        uint32_t best_id = UINT32_MAX;
        uint32_t best_position = UINT32_MAX;
        for (uint32_t p = begin + lane; p < end; p += 32)
        {
            if (removed[p])
                continue;
            const float distance = candidate_distances[p];
            const uint32_t id = candidate_ids[p];
            if (distance < best_distance ||
                (distance == best_distance && (id < best_id || (id == best_id && p < best_position))))
            {
                best_distance = distance;
                best_id = id;
                best_position = p;
            }
        }
        for (uint32_t offset = 16; offset > 0; offset >>= 1)
        {
            const float other_distance = __shfl_down_sync(0xffffffffu, best_distance, offset);
            const uint32_t other_id = __shfl_down_sync(0xffffffffu, best_id, offset);
            const uint32_t other_position = __shfl_down_sync(0xffffffffu, best_position, offset);
            if (other_distance < best_distance ||
                (other_distance == best_distance &&
                 (other_id < best_id || (other_id == best_id && other_position < best_position))))
            {
                best_distance = other_distance;
                best_id = other_id;
                best_position = other_position;
            }
        }
        best_id = __shfl_sync(0xffffffffu, best_id, 0);
        best_position = __shfl_sync(0xffffffffu, best_position, 0);
        if (best_position == UINT32_MAX)
            break;
        if (lane == 0)
        {
            new_graph[graph_base + selected_count] = best_id;
            removed[best_position] = 1;
        }
        ++selected_count;
        __syncwarp();
        for (uint32_t p = begin; p < end; ++p)
        {
            if (removed[p])
                continue;
            const uint32_t candidate = candidate_ids[p];
            if (!filtered_selected_covers_common_labels(row, best_id, candidate, label_offsets, labels))
                continue;
            const float pair_distance = warp_l2_distance(data, dim, best_id, candidate, lane);
            if (g_canonical_profile && lane == 0) atomicAdd(&g_canonical_distance_counts[5], 1ull);
            if (lane == 0 && pair_distance > 0.0f &&
                alpha * pair_distance <= candidate_distances[p])
                removed[p] = 1;
            __syncwarp();
        }
    }
    if (lane == 0)
    {
        new_degrees[row] = selected_count;
        bool changed = old_degrees[row] != selected_count;
        for (uint32_t i = 0; !changed && i < selected_count; ++i)
        {
            bool found = false;
            for (uint32_t j = 0; !found && j < old_degrees[row]; ++j)
                found = old_graph[graph_base + j] == new_graph[graph_base + i];
            changed = !found;
        }
        row_changed[row] = changed ? 1 : 0;
    }
}

__global__ void canonical_init_ids(uint32_t count, uint32_t *ids)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        ids[i] = i;
}

__global__ void canonical_mark_task_flags(uint32_t task_count, const uint32_t *task_rows,
                                          const uint8_t *active_flags, uint8_t *task_flags)
{
    const uint32_t task = blockIdx.x * blockDim.x + threadIdx.x;
    if (task < task_count)
        task_flags[task] = active_flags[task_rows[task]];
}

__global__ void canonical_mark_deficient(uint32_t N, uint32_t R, uint32_t tau_F,
                                         const uint32_t *graph, const uint32_t *degrees,
                                         const uint32_t *offsets, const uint32_t *labels,
                                         const uint32_t *label_sizes, uint32_t num_labels,
                                         uint32_t universal_label, uint8_t *deficient)
{
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N)
        return;
    const uint32_t degree = degrees[row] > R ? R : degrees[row];
    bool row_deficient = false;
    for (uint32_t p = offsets[row]; !row_deficient && p < offsets[row + 1]; ++p)
    {
        const uint32_t label = labels[p];
        if (label == universal_label || label >= num_labels)
            continue;
        const uint32_t label_limit = label_sizes[label] > 0 ? label_sizes[label] - 1 : 0;
        const uint32_t tau = min(min(tau_F, label_limit), R);
        uint32_t label_degree = 0;
        for (uint32_t i = 0; i < degree && label_degree < tau; ++i)
        {
            const uint32_t neighbor = graph[(size_t)row * R + i];
            if (neighbor < N && point_has_exact_label_device(neighbor, label, offsets, labels))
                ++label_degree;
        }
        row_deficient = label_degree < tau;
    }
    deficient[row] = row_deficient ? 1 : 0;
}

__global__ void canonical_combine_frontier(uint32_t N, const uint8_t *changed,
                                           const uint8_t *touched, const uint8_t *deficient,
                                           uint8_t *active)
{
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N)
        active[row] = (changed[row] || touched[row] || deficient[row]) ? 1 : 0;
}






__device__ __forceinline__ void canonical_rare_label_shortlist_warp(
    uint32_t row, uint32_t R, uint32_t begin, uint32_t end,
    const uint32_t *candidate_ids, const float *candidate_distances,
    const uint32_t *offsets, const uint32_t *labels,
    const uint32_t *label_sizes, uint32_t num_labels, uint32_t universal_label,
    uint8_t *state, uint32_t *output_ids, float *output_distances, uint32_t *output_degree,
    uint32_t lane)
{
    for (uint32_t p = begin + lane; p < end; p += 32)
        state[p] = 0;
    for (uint32_t i = lane; i < R; i += 32)
    {
        output_ids[i] = INVALID_ID;
        output_distances[i] = FLT_MAX;
    }
    __syncwarp();
    uint32_t count = 0;
    const uint32_t row_label_count = offsets[row + 1] - offsets[row];
    const uint32_t quota = row_label_count ? max(1u, R / row_label_count) : 0;
    const uint64_t risk_limit = (uint64_t)R * R;
    for (uint32_t lp = offsets[row]; lp < offsets[row + 1] && count < R; ++lp)
    {
        const uint32_t label = labels[lp];
        if (label == universal_label || label >= num_labels || label_sizes[label] > risk_limit)
            continue;
        for (uint32_t p = begin + lane; p < end; p += 32)
            state[p] &= 1u;
        __syncwarp();
        for (uint32_t k = 0; k < quota && count < R; ++k)
        {
            float best_distance = FLT_MAX;
            uint32_t best_id = UINT32_MAX, best_pos = UINT32_MAX;
            for (uint32_t p = begin + lane; p < end; p += 32)
            {
                const uint32_t id = candidate_ids[p];
                const float distance = candidate_distances[p];
                if ((state[p] & 2u) || id == INVALID_ID ||
                    !point_has_exact_label_device(id, label, offsets, labels))
                    continue;
                if (distance < best_distance ||
                    (distance == best_distance && (id < best_id || (id == best_id && p < best_pos))))
                {
                    best_distance = distance; best_id = id; best_pos = p;
                }
            }
            for (uint32_t delta = 16; delta; delta >>= 1)
            {
                const float od = __shfl_down_sync(0xffffffffu, best_distance, delta);
                const uint32_t oi = __shfl_down_sync(0xffffffffu, best_id, delta);
                const uint32_t op = __shfl_down_sync(0xffffffffu, best_pos, delta);
                if (od < best_distance || (od == best_distance &&
                    (oi < best_id || (oi == best_id && op < best_pos))))
                { best_distance = od; best_id = oi; best_pos = op; }
            }
            best_pos = __shfl_sync(0xffffffffu, best_pos, 0);
            best_id = __shfl_sync(0xffffffffu, best_id, 0);
            best_distance = __shfl_sync(0xffffffffu, best_distance, 0);
            if (best_pos == UINT32_MAX)
                break;
            if (lane == 0)
            {
                state[best_pos] |= 2u;
                if (!(state[best_pos] & 1u))
                {
                    state[best_pos] |= 1u;
                    output_ids[count] = best_id;
                    output_distances[count] = best_distance;
                    ++count;
                }
            }
            count = __shfl_sync(0xffffffffu, count, 0);
            __syncwarp();
        }
    }
    while (count < R)
    {
        float best_distance = FLT_MAX;
        uint32_t best_id = UINT32_MAX, best_pos = UINT32_MAX;
        for (uint32_t p = begin + lane; p < end; p += 32)
        {
            const uint32_t id = candidate_ids[p];
            const float distance = candidate_distances[p];
            if ((state[p] & 1u) || id == INVALID_ID)
                continue;
            if (distance < best_distance ||
                (distance == best_distance && (id < best_id || (id == best_id && p < best_pos))))
            { best_distance = distance; best_id = id; best_pos = p; }
        }
        for (uint32_t delta = 16; delta; delta >>= 1)
        {
            const float od = __shfl_down_sync(0xffffffffu, best_distance, delta);
            const uint32_t oi = __shfl_down_sync(0xffffffffu, best_id, delta);
            const uint32_t op = __shfl_down_sync(0xffffffffu, best_pos, delta);
            if (od < best_distance || (od == best_distance &&
                (oi < best_id || (oi == best_id && op < best_pos))))
            { best_distance = od; best_id = oi; best_pos = op; }
        }
        best_pos = __shfl_sync(0xffffffffu, best_pos, 0);
        best_id = __shfl_sync(0xffffffffu, best_id, 0);
        best_distance = __shfl_sync(0xffffffffu, best_distance, 0);
        if (best_pos == UINT32_MAX)
            break;
        if (lane == 0)
        {
            state[best_pos] |= 1u;
            output_ids[count] = best_id;
            output_distances[count] = best_distance;
            ++count;
        }
        count = __shfl_sync(0xffffffffu, count, 0);
        __syncwarp();
    }
    if (lane == 0)
        *output_degree = count;
}

template <typename DataT>
__global__ void canonical_fixed_rare_shortlist(
    const DataT *data, uint32_t N, uint32_t dim, uint32_t input_R, uint32_t output_R,
    const uint32_t *input_graph, const uint32_t *input_degree,
    const uint32_t *offsets, const uint32_t *labels, const uint32_t *label_sizes,
    uint32_t num_labels, uint32_t universal_label, float *input_distances, uint8_t *state,
    uint32_t *work, float *work_distances, uint32_t *work_degree)
{
    const uint32_t row = blockIdx.x, lane = threadIdx.x & 31u;
    if (row >= N) return;
    const uint32_t degree = min(input_degree[row], input_R);
    for (uint32_t i = 0; i < degree; ++i)
    {
        const uint32_t id = input_graph[(size_t)row * input_R + i];
        const float distance = warp_l2_distance(data, dim, row, id, lane);
        if (lane == 0) input_distances[(size_t)row * input_R + i] = distance;
    }
    __syncwarp();
    canonical_rare_label_shortlist_warp(
        row, output_R, (uint32_t)((size_t)row * input_R),
        (uint32_t)((size_t)row * input_R + degree), input_graph, input_distances,
        offsets, labels, label_sizes, num_labels, universal_label, state,
        work + (size_t)row * output_R, work_distances + (size_t)row * output_R,
        work_degree + row, lane);
}

__global__ void canonical_csr_rare_shortlist(
    uint32_t N, uint32_t R, const uint32_t *row_ids, uint32_t row_count,
    const uint32_t *candidate_offsets, const uint32_t *candidate_ids, const float *candidate_distances,
    const uint32_t *offsets, const uint32_t *labels, const uint32_t *label_sizes,
    uint32_t num_labels, uint32_t universal_label, uint8_t *state,
    uint32_t *work, float *work_distances, uint32_t *work_degree)
{
    const uint32_t compact_row = blockIdx.x, lane = threadIdx.x & 31u;
    if (compact_row >= row_count) return;
    const uint32_t row = row_ids[compact_row];
    canonical_rare_label_shortlist_warp(
        row, R, candidate_offsets[row], candidate_offsets[row + 1], candidate_ids, candidate_distances,
        offsets, labels, label_sizes, num_labels, universal_label, state,
        work + (size_t)compact_row * R, work_distances + (size_t)compact_row * R,
        work_degree + compact_row, lane);
}







template <typename DataT>
__global__ void canonical_repair_label_coverage(
    const DataT *__restrict__ data, uint32_t N, uint32_t dim, uint32_t R,
    uint32_t per_label_keep, uint32_t coverage_quota, const uint32_t *offsets, const uint32_t *labels,
    const uint32_t *label_sizes, uint32_t num_labels, uint32_t universal_label,
    const uint32_t *task_candidates, const float *task_distances,
    uint32_t *graph, uint32_t *degrees, unsigned long long *counters)
{
    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= N)
        return;
    const size_t graph_base = (size_t)row * R;

    const uint32_t label_begin = offsets[row], label_end = offsets[row + 1];


    for (uint32_t rank = 0; rank < label_end - label_begin; ++rank)
    {
        uint32_t p = UINT32_MAX;
        uint32_t p_size = UINT32_MAX, p_label = UINT32_MAX;
        for (uint32_t q = label_begin; q < label_end; ++q)
        {
            const uint32_t qlabel = labels[q];
            if (qlabel == universal_label || qlabel >= num_labels)
                continue;
            uint32_t preceding = 0;
            for (uint32_t t = label_begin; t < label_end; ++t)
            {
                const uint32_t tlabel = labels[t];
                if (tlabel < num_labels &&
                    (label_sizes[tlabel] < label_sizes[qlabel] ||
                     (label_sizes[tlabel] == label_sizes[qlabel] && tlabel < qlabel)))
                    ++preceding;
            }
            if (preceding == rank)
            {
                p = q; p_size = label_sizes[qlabel]; p_label = qlabel;
                break;
            }
        }
        (void)p_size;
        if (p == UINT32_MAX)
            continue;
        const uint32_t label = p_label;
        if (label == universal_label || label >= num_labels || label_sizes[label] <= 1)
            continue;
        const uint32_t target = min(min(coverage_quota, label_sizes[label] - 1), R);
        while (true)
        {
            uint32_t degree = min(degrees[row], R);
            uint32_t local_coverage = 0;
            for (uint32_t i = lane; i < degree; i += 32)
                if (point_has_exact_label_device(graph[graph_base + i], label, offsets, labels))
                    ++local_coverage;
            for (uint32_t delta = 16; delta; delta >>= 1)
                local_coverage += __shfl_down_sync(0xffffffffu, local_coverage, delta);
            const uint32_t coverage = __shfl_sync(0xffffffffu, local_coverage, 0);
            if (coverage >= target)
                break;
            if (lane == 0 && counters) atomicAdd(counters + 0, 1ull);



        float best_distance = FLT_MAX;
        uint32_t best_id = UINT32_MAX;
        for (uint32_t i = lane; i < per_label_keep; i += 32)
        {
            const uint32_t candidate = task_candidates[(size_t)p * per_label_keep + i];
            const float distance = task_distances[(size_t)p * per_label_keep + i];
            bool duplicate = candidate >= N || candidate == row;
            for (uint32_t j = 0; !duplicate && j < degree; ++j)
                duplicate = graph[graph_base + j] == candidate;
            if (!duplicate && point_has_exact_label_device(candidate, label, offsets, labels) &&
                (distance < best_distance || (distance == best_distance && candidate < best_id)))
            {
                best_distance = distance;
                best_id = candidate;
            }
        }
        for (uint32_t delta = 16; delta; delta >>= 1)
        {
            const float other_distance = __shfl_down_sync(0xffffffffu, best_distance, delta);
            const uint32_t other_id = __shfl_down_sync(0xffffffffu, best_id, delta);
            if (other_distance < best_distance ||
                (other_distance == best_distance && other_id < best_id))
            {
                best_distance = other_distance;
                best_id = other_id;
            }
        }
        best_id = __shfl_sync(0xffffffffu, best_id, 0);
        best_distance = __shfl_sync(0xffffffffu, best_distance, 0);
            if (best_id == UINT32_MAX)
            {
                if (lane == 0 && counters) atomicAdd(counters + 2, 1ull);
                break;
            }
            if (degree < R)
            {
                if (lane == 0)
                {
                    graph[graph_base + degree] = best_id;
                    degrees[row] = degree + 1;
                    if (counters) atomicAdd(counters + 1, 1ull);
                }
                __syncwarp();
                continue;
            }



        float worst_distance = -1.0f;
        uint32_t worst_id = 0;
        uint32_t worst_pos = UINT32_MAX;






        for (uint32_t i = 0; i < degree; ++i)
        {
            const uint32_t victim = graph[graph_base + i];
            bool safe = false;
            if (lane == 0)
            {
                safe = victim < N;
                for (uint32_t q = offsets[row]; safe && q < offsets[row + 1]; ++q)
                {
                    const uint32_t qlabel = labels[q];
                    if (qlabel == universal_label || qlabel >= num_labels ||
                        !point_has_exact_label_device(victim, qlabel, offsets, labels))
                        continue;
                    uint32_t representatives = 0;
                    for (uint32_t j = 0; j < degree; ++j)
                        if (point_has_exact_label_device(graph[graph_base + j], qlabel, offsets, labels))
                            ++representatives;
                    const uint32_t qtarget = min(min(coverage_quota, label_sizes[qlabel] - 1), R);
                    if (representatives <= qtarget &&
                        !point_has_exact_label_device(best_id, qlabel, offsets, labels))
                        safe = false;
                }
            }
            safe = __shfl_sync(0xffffffffu, safe, 0);
            if (safe)
            {
                const float distance = warp_l2_distance(data, dim, row, victim, lane);
                if (lane == 0 && (distance > worst_distance ||
                    (distance == worst_distance && victim > worst_id)))
                {
                    worst_distance = distance;
                    worst_id = victim;
                    worst_pos = i;
                }
            }
        }
        worst_pos = __shfl_sync(0xffffffffu, worst_pos, 0);
            if (lane == 0)
            {
                if (worst_pos != UINT32_MAX)
                {
                    graph[graph_base + worst_pos] = best_id;
                    if (counters) atomicAdd(counters + 1, 1ull);
                }
                else
                    if (counters) atomicAdd(counters + 2, 1ull);
            }
            __syncwarp();
            if (worst_pos == UINT32_MAX)
                break;
        }
    }
}




__global__ void canonical_compare_graph_sets(uint32_t N, uint32_t R,
                                             const uint32_t *lhs_graph,
                                             const uint32_t *lhs_degree,
                                             const uint32_t *rhs_graph,
                                             const uint32_t *rhs_degree,
                                             unsigned long long *mismatch_rows,
                                             uint32_t *first_mismatch)
{
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N)
        return;
    const uint32_t ld = min(lhs_degree[row], R);
    const uint32_t rd = min(rhs_degree[row], R);
    bool mismatch = ld != rd;
    for (uint32_t i = 0; !mismatch && i < ld; ++i)
    {
        const uint32_t id = lhs_graph[(size_t)row * R + i];
        bool found = false;
        for (uint32_t j = 0; !found && j < rd; ++j)
            found = rhs_graph[(size_t)row * R + j] == id;
        mismatch = !found;
    }
    if (mismatch)
    {
        atomicAdd(mismatch_rows, 1ull);
        atomicMin(first_mismatch, row);
    }
}

template <typename SearchT, typename HostT>
static int canonical_round0_from_host_backbone_typed(const HostT *h_data, uint32_t N, uint32_t dim, uint32_t R,
                                                uint32_t C, uint32_t search_L, uint32_t filtered_L,
                                                uint32_t build_steps, float alpha, const uint32_t *h_offsets,
                                                const uint32_t *h_labels, uint32_t total_labels,
                                                const uint32_t *h_label_starts, uint32_t num_labels,
                                                uint32_t universal_label, uint32_t global_start,
                                                uint32_t *h_graph, uint32_t *h_degrees,
                                                HostT *d_handoff_data = nullptr,
                                                uint32_t *d_handoff_graph = nullptr,
                                                uint32_t *d_handoff_degrees = nullptr)
{
    const uint32_t warps_per_block = 8;
    const uint32_t threads = warps_per_block * 32;




    const uint32_t per_label_pool = filtered_L;
    const uint32_t per_label_keep = std::min(per_label_pool, 8u);
    const uint32_t ordinary_keep = std::min(R, search_L);



    const uint32_t ordinary_pool = std::min<uint32_t>(64u, filtered_L);
    const uint32_t expansion_steps = build_steps;
    const uint32_t true_ordinary_round0 = env_u32_clamped(
        "DISKANN_GPU_FILTERED_TRUE_ORDINARY_ROUND0", 0, 0, 1);
    const uint32_t pure_ordinary_round0 = env_u32_clamped(
        "DISKANN_GPU_FILTERED_PURE_ORDINARY_ROUND0", 0, 0, 1);
    const uint32_t true_ordinary_steps = env_u32_clamped(
        "DISKANN_GPU_FILTERED_TRUE_ORDINARY_STEPS", 32, 1, 64);
    const uint32_t retain_posting_anchor = env_u32_clamped(
        "DISKANN_GPU_FILTERED_RETAIN_POSTING_ANCHOR", 0, 0, 1);
    const uint32_t additional_round0_anchor = env_u32_clamped(
        "DISKANN_GPU_FILTERED_ADDITIONAL_ROUND0_ANCHOR", 0, 0, 1);
    const uint32_t bridge_reserve = env_u32_clamped(
        "DISKANN_GPU_FILTERED_BRIDGE_WORK_RESERVE", 8, 0, 16);
    const uint32_t coverage_quota = 1;



    const uint32_t configured_rounds = 3;
    const char *fused_threshold_env = getenv("DISKANN_GPU_FILTERED_FUSED_LABEL_THRESHOLD");
    const uint32_t fused_label_threshold = fused_threshold_env ?
        std::min(64u, (uint32_t)std::max(0, atoi(fused_threshold_env))) : 0u;
    const uint32_t canonical_profile = 0;


    const bool reverse_parity = false;
    const bool enable_per_label_search = true;
    const bool enable_union_label_search = true;
    const bool enable_exact_reverse_repair = true;
    const bool enable_coverage_repair = false;
    const uint32_t reverse_reference_output = 0u;
    CUDA_CHECK_FILTERED(cudaMemcpyToSymbol(g_canonical_profile, &canonical_profile, sizeof(uint32_t)));
    printf("[gpu_vamana_filtered_navigation_policy] per_label_pool=%u per_label_keep=%u "
           "true_ordinary_round0=%u pure_ordinary_round0=%u ordinary_pool=%u ordinary_steps=%u "
           "posting_anchor=%u additional_round0_anchor=%u "
           "bridge_work_reserve=%u\n",
           per_label_pool, per_label_keep, true_ordinary_round0, pure_ordinary_round0,
           ordinary_pool, true_ordinary_steps,
           retain_posting_anchor, additional_round0_anchor, bridge_reserve);
    SearchT *d_data = nullptr;
    HostT *d_source_data = d_handoff_data;
    uint32_t *d_offsets = nullptr, *d_labels = nullptr, *d_starts = nullptr, *d_task_rows = nullptr, *d_label_sizes = nullptr;
    uint32_t *d_label_point_offsets = nullptr, *d_label_points = nullptr;
    uint32_t *d_graph_a = nullptr, *d_graph_b = nullptr;
    uint32_t *d_degree_a = nullptr, *d_degree_b = nullptr;
    uint32_t *d_task_candidates = nullptr, *d_ordinary = nullptr, *d_work = nullptr, *d_work_degrees = nullptr;
    float *d_task_candidate_distances = nullptr, *d_ordinary_distances = nullptr, *d_work_distances = nullptr;
    uint8_t *d_removed = nullptr, *d_changed = nullptr, *d_forward_changed = nullptr;
    uint8_t *d_active = nullptr, *d_deficient = nullptr, *d_task_flags = nullptr;
    uint32_t *d_all_ids = nullptr, *d_active_ids = nullptr, *d_changed_ids = nullptr;
    uint32_t *d_touched_ids = nullptr, *d_deficient_ids = nullptr;
    uint32_t *d_all_task_ids = nullptr, *d_active_task_ids = nullptr, *d_selected_count = nullptr;
    uint32_t *d_reverse_offsets = nullptr, *d_reverse_cursor = nullptr, *d_reverse_ids = nullptr;
    uint32_t *d_reverse_candidate_offsets = nullptr, *d_reverse_candidate_ids = nullptr;
    float *d_reverse_candidate_distances = nullptr;
    uint8_t *d_reverse_removed = nullptr, *d_touched = nullptr;
    void *d_scan_temp = nullptr, *d_select_temp = nullptr;
    size_t scan_temp_bytes = 0, select_temp_bytes = 0, select_rows_bytes = 0, select_tasks_bytes = 0;
    const size_t max_forward_edges = (size_t)N * R;
    const size_t max_reverse_candidates = max_forward_edges * 2;
    const double start = now_sec_filtered();
    std::vector<uint32_t> h_label_sizes(std::max(1u, num_labels), 0);
    for (uint32_t p = 0; p < total_labels; ++p)
        if (h_labels[p] < num_labels)
            ++h_label_sizes[h_labels[p]];
    std::vector<uint32_t> h_label_point_offsets((size_t)std::max(1u, num_labels) + 1, 0);
    for (uint32_t label = 0; label < num_labels; ++label)
        h_label_point_offsets[label + 1] = h_label_point_offsets[label] + h_label_sizes[label];
    std::vector<uint32_t> h_label_points(h_label_point_offsets[num_labels], INVALID_ID);
    std::vector<uint32_t> h_label_point_cursor = h_label_point_offsets;
    for (uint32_t row = 0; row < N; ++row)
        for (uint32_t p = h_offsets[row]; p < h_offsets[row + 1]; ++p)
            if (h_labels[p] < num_labels)
                h_label_points[h_label_point_cursor[h_labels[p]]++] = row;
    d_graph_a = d_handoff_graph;
    d_degree_a = d_handoff_degrees;
    const size_t element_count = (size_t)N * dim;
    if constexpr (std::is_same<SearchT, HostT>::value)
    {
        d_data = reinterpret_cast<SearchT *>(d_handoff_data);
        if (!d_data)
        {
            CUDA_CHECK_FILTERED(cudaMalloc(&d_data, element_count * sizeof(SearchT)));
            CUDA_CHECK_FILTERED(cudaMemcpy(d_data, h_data, element_count * sizeof(HostT), cudaMemcpyHostToDevice));
        }
    }
    else
    {
        if (!d_source_data)
        {
            CUDA_CHECK_FILTERED(cudaMalloc(&d_source_data, element_count * sizeof(HostT)));
            CUDA_CHECK_FILTERED(cudaMemcpy(d_source_data, h_data, element_count * sizeof(HostT), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK_FILTERED(cudaMalloc(&d_data, element_count * sizeof(SearchT)));
        canonical_convert_float_to_half<<<(element_count + 255) / 256, 256>>>(
            reinterpret_cast<const float *>(d_source_data), reinterpret_cast<__half *>(d_data), element_count);
        CUDA_CHECK_FILTERED(cudaGetLastError());
        CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
        cudaFree(d_source_data);
        d_source_data = nullptr;
    }
    CUDA_CHECK_FILTERED(cudaMalloc(&d_offsets, (size_t)(N + 1) * sizeof(uint32_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_labels, (size_t)total_labels * sizeof(uint32_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_starts, (size_t)std::max(1u, num_labels) * sizeof(uint32_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_label_sizes, (size_t)std::max(1u, num_labels) * sizeof(uint32_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_label_point_offsets,
                                   ((size_t)std::max(1u, num_labels) + 1) * sizeof(uint32_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_label_points, (size_t)h_label_points.size() * sizeof(uint32_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_task_rows, (size_t)total_labels * sizeof(uint32_t)));
    if (!d_graph_a)
        CUDA_CHECK_FILTERED(cudaMalloc(&d_graph_a, (size_t)N * R * sizeof(uint32_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_graph_b, (size_t)N * R * sizeof(uint32_t)));
    if (!d_degree_a)
        CUDA_CHECK_FILTERED(cudaMalloc(&d_degree_a, (size_t)N * sizeof(uint32_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_degree_b, (size_t)N * sizeof(uint32_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_task_candidates, (size_t)total_labels * per_label_keep * sizeof(uint32_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_task_candidate_distances, (size_t)total_labels * per_label_keep * sizeof(float)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_work, (size_t)N * C * sizeof(uint32_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_work_degrees, (size_t)N * sizeof(uint32_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_work_distances, (size_t)N * C * sizeof(float)));
    d_ordinary = d_work;
    d_ordinary_distances = d_work_distances;
    CUDA_CHECK_FILTERED(cudaMalloc(&d_removed, (size_t)N * C * sizeof(uint8_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_changed, (size_t)N * sizeof(uint8_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_forward_changed, (size_t)N * sizeof(uint8_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_active, (size_t)N * sizeof(uint8_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_deficient, (size_t)N * sizeof(uint8_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_task_flags, (size_t)total_labels * sizeof(uint8_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_all_ids, (size_t)N * sizeof(uint32_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_active_ids, (size_t)N * sizeof(uint32_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_changed_ids, (size_t)N * sizeof(uint32_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_touched_ids, (size_t)N * sizeof(uint32_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_deficient_ids, (size_t)N * sizeof(uint32_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_all_task_ids, (size_t)total_labels * sizeof(uint32_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_active_task_ids, (size_t)total_labels * sizeof(uint32_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_selected_count, sizeof(uint32_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_reverse_offsets, (size_t)(N + 1) * sizeof(uint32_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_reverse_cursor, (size_t)N * sizeof(uint32_t)));




    CUDA_CHECK_FILTERED(cudaMalloc(&d_reverse_candidate_offsets, (size_t)(N + 1) * sizeof(uint32_t)));




    if ((size_t)N * C < max_reverse_candidates)
    {
        fprintf(stderr, "[gpu_vamana_filtered] GPU work capacity cannot hold exact reverse CSR\n");
        return 1;
    }
    d_reverse_candidate_ids = d_work;
    d_reverse_candidate_distances = d_work_distances;
    d_reverse_removed = d_removed;
    CUDA_CHECK_FILTERED(cudaMalloc(&d_touched, (size_t)N * sizeof(uint8_t)));
    CUDA_CHECK_FILTERED(cub::DeviceScan::InclusiveSum(nullptr, scan_temp_bytes, d_reverse_offsets,
                                                       d_reverse_offsets, N + 1));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_scan_temp, scan_temp_bytes));
    CUDA_CHECK_FILTERED(cub::DeviceSelect::Flagged(nullptr, select_rows_bytes, d_all_ids, d_active,
                                                    d_active_ids, d_selected_count, N));
    CUDA_CHECK_FILTERED(cub::DeviceSelect::Flagged(nullptr, select_tasks_bytes, d_all_task_ids, d_task_flags,
                                                    d_active_task_ids, d_selected_count, total_labels));
    select_temp_bytes = std::max(select_rows_bytes, select_tasks_bytes);
    CUDA_CHECK_FILTERED(cudaMalloc(&d_select_temp, select_temp_bytes));
    update_filtered_peak_gpu_memory();
    CUDA_CHECK_FILTERED(cudaMemcpy(d_offsets, h_offsets, (size_t)(N + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_FILTERED(cudaMemcpy(d_labels, h_labels, (size_t)total_labels * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_FILTERED(cudaMemcpy(d_starts, h_label_starts, (size_t)std::max(1u, num_labels) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_FILTERED(cudaMemcpy(d_label_sizes, h_label_sizes.data(),
                                   (size_t)std::max(1u, num_labels) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_FILTERED(cudaMemcpy(d_label_point_offsets, h_label_point_offsets.data(),
                                   ((size_t)std::max(1u, num_labels) + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_FILTERED(cudaMemcpy(d_label_points, h_label_points.data(),
                                   (size_t)h_label_points.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    if (!d_handoff_graph)
        CUDA_CHECK_FILTERED(cudaMemcpy(d_graph_a, h_graph, (size_t)N * R * sizeof(uint32_t), cudaMemcpyHostToDevice));
    if (!d_handoff_degrees)
        CUDA_CHECK_FILTERED(cudaMemcpy(d_degree_a, h_degrees, (size_t)N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    g_last_filtered_stats.task_generation_seconds = now_sec_filtered();
    canonical_make_task_rows<<<(N + 255) / 256, 256>>>(N, d_offsets, d_task_rows);
    canonical_init_ids<<<(N + 255) / 256, 256>>>(N, d_all_ids);
    canonical_init_ids<<<(total_labels + 255) / 256, 256>>>(total_labels, d_all_task_ids);
    if (!enable_per_label_search)
    {
        CUDA_CHECK_FILTERED(cudaMemset(d_task_candidates, 0xff,
                                       (size_t)total_labels * per_label_keep * sizeof(uint32_t)));
    }
    CUDA_CHECK_FILTERED(cudaGetLastError());
    CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
    g_last_filtered_stats.task_generation_seconds = now_sec_filtered() - g_last_filtered_stats.task_generation_seconds;
    const size_t filtered_shared = (size_t)warps_per_block * per_label_pool *
                             (sizeof(uint32_t) + sizeof(float) + sizeof(uint8_t));
    const size_t ordinary_shared = (size_t)warps_per_block * ordinary_pool *
                             (sizeof(uint32_t) + sizeof(float) + sizeof(uint8_t));
    uint32_t active_count = N;
    double work_distance_seconds = 0.0;
    const unsigned long long zero_distance_counts[6] = {};
    CUDA_CHECK_FILTERED(cudaMemcpyToSymbol(g_canonical_distance_counts, zero_distance_counts,
                                            sizeof(zero_distance_counts)));
    for (uint32_t round = 0; round < configured_rounds; ++round)
    {
        const uint32_t *round_ids = nullptr;
        uint32_t active_task_count = total_labels;
        const uint32_t *active_task_positions = nullptr;
        const float round_alpha = alpha;
        double stage = now_sec_filtered();
        if (enable_per_label_search && fused_label_threshold)
        {
#define CANONICAL_LAUNCH_FUSED(CACHE_DIM) canonical_fused_filtered_row_search<CACHE_DIM, SearchT>\
            <<<(N + warps_per_block - 1) / warps_per_block, threads, filtered_shared>>>(\
            d_data, N, dim, R, per_label_pool, expansion_steps, fused_label_threshold, d_starts,\
            num_labels, universal_label, d_offsets, d_labels, d_graph_a, d_degree_a,\
            d_task_candidates, d_task_candidate_distances)
            if (dim == 96) CANONICAL_LAUNCH_FUSED(96); else if (dim == 128) CANONICAL_LAUNCH_FUSED(128);
            else if (dim == 282) CANONICAL_LAUNCH_FUSED(282); else CANONICAL_LAUNCH_FUSED(0);
#undef CANONICAL_LAUNCH_FUSED
        }
        if (enable_per_label_search)
        {
#define CANONICAL_LAUNCH_FILTERED(CACHE_DIM) canonical_filtered_task_search<CACHE_DIM, SearchT>\
            <<<(active_task_count + warps_per_block - 1) / warps_per_block, threads, filtered_shared>>>(\
            d_data, N, dim, R, active_task_count, per_label_pool, per_label_keep, expansion_steps,\
            active_task_positions, d_task_rows, d_labels, d_starts, num_labels,\
            d_label_point_offsets, d_label_points, 16u, universal_label,\
            d_offsets, d_labels, d_graph_a, d_degree_a, d_task_candidates, d_task_candidate_distances,\
            fused_label_threshold, retain_posting_anchor)
        if (dim == 96) CANONICAL_LAUNCH_FILTERED(96); else if (dim == 128) CANONICAL_LAUNCH_FILTERED(128);
        else if (dim == 282) CANONICAL_LAUNCH_FILTERED(282); else CANONICAL_LAUNCH_FILTERED(0);
#undef CANONICAL_LAUNCH_FILTERED
        }
        CUDA_CHECK_FILTERED(cudaGetLastError());
        CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
        g_last_filtered_stats.filtered_search_seconds += now_sec_filtered() - stage;
        stage = now_sec_filtered();
        if (enable_union_label_search)
        {
#define CANONICAL_LAUNCH_ORDINARY(CACHE_DIM) canonical_ordinary_search<CACHE_DIM, SearchT>\
            <<<(active_count + warps_per_block - 1) / warps_per_block, threads, ordinary_shared>>>(\
            d_data, N, dim, R, ordinary_pool, ordinary_keep, C,\
            ((true_ordinary_round0 || pure_ordinary_round0) && round == 0) ? true_ordinary_steps : expansion_steps,\
            global_start, round_ids, active_count,\
            d_graph_a, d_degree_a, d_offsets, d_labels, d_starts, num_labels,\
            d_label_point_offsets, d_label_points, 64u, d_ordinary, d_ordinary_distances,\
            ((true_ordinary_round0 || pure_ordinary_round0) && round == 0) ? 1u : 0u,\
            (pure_ordinary_round0 && round == 0) ? 1u : 0u)
        if (dim == 96) CANONICAL_LAUNCH_ORDINARY(96); else if (dim == 128) CANONICAL_LAUNCH_ORDINARY(128);
        else if (dim == 282) CANONICAL_LAUNCH_ORDINARY(282); else CANONICAL_LAUNCH_ORDINARY(0);
#undef CANONICAL_LAUNCH_ORDINARY
        }
        CUDA_CHECK_FILTERED(cudaGetLastError());
        CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
        g_last_filtered_stats.ordinary_search_seconds += now_sec_filtered() - stage;
        stage = now_sec_filtered();
        canonical_merge_candidates<<<active_count, 32>>>(
            N, R, C, enable_per_label_search ? per_label_keep : 0u,
            enable_union_label_search ? ordinary_keep : 0u, d_offsets, d_labels,
            d_label_point_offsets, d_label_points, round_ids, active_count,
            d_graph_a, d_degree_a, d_task_candidates, d_task_candidate_distances,
            d_ordinary, d_ordinary_distances, d_work, d_work_distances, d_work_degrees,
            true_ordinary_round0, bridge_reserve,
            (additional_round0_anchor && round == 0) ? 1u : 0u);
        CUDA_CHECK_FILTERED(cudaGetLastError());
        CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
        g_last_filtered_stats.merge_seconds += now_sec_filtered() - stage;
        stage = now_sec_filtered();
#define CANONICAL_LAUNCH_WORK_DISTANCE(CACHE_DIM) canonical_compute_work_distances<CACHE_DIM, SearchT>\
            <<<active_count, 32>>>(d_data, N, dim, C, d_work, round_ids, active_count,\
                                   d_work_degrees, d_work_distances)
        if (dim == 96) CANONICAL_LAUNCH_WORK_DISTANCE(96); else if (dim == 128) CANONICAL_LAUNCH_WORK_DISTANCE(128);
        else if (dim == 282) CANONICAL_LAUNCH_WORK_DISTANCE(282); else CANONICAL_LAUNCH_WORK_DISTANCE(0);
#undef CANONICAL_LAUNCH_WORK_DISTANCE
        CUDA_CHECK_FILTERED(cudaGetLastError());
        CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
        work_distance_seconds += now_sec_filtered() - stage;
        stage = now_sec_filtered();


        if (active_count < N)
        {
            CUDA_CHECK_FILTERED(cudaMemcpy(d_graph_b, d_graph_a, max_forward_edges * sizeof(uint32_t),
                                           cudaMemcpyDeviceToDevice));
            CUDA_CHECK_FILTERED(cudaMemcpy(d_degree_b, d_degree_a, (size_t)N * sizeof(uint32_t),
                                           cudaMemcpyDeviceToDevice));
        }
        CUDA_CHECK_FILTERED(cudaMemset(d_forward_changed, 0, (size_t)N));
        canonical_exact_prune<SearchT><<<active_count, 32>>>(
            d_data, N, dim, R, C, round_alpha, d_offsets, d_labels, round_ids, active_count,
            d_graph_a, d_degree_a, d_work, d_work_degrees, d_work_distances, d_removed,
            0u, 0u, d_graph_b, nullptr, d_degree_b, d_forward_changed);
        CUDA_CHECK_FILTERED(cudaGetLastError());
        CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
        g_last_filtered_stats.forward_prune_seconds += now_sec_filtered() - stage;






        uint32_t *d_parity_graph = nullptr, *d_parity_degree = nullptr;
        uint32_t *d_parity_warp_graph = nullptr, *d_parity_warp_degree = nullptr;
        uint32_t *d_parity_reverse_ids = nullptr, *d_parity_reverse_counts = nullptr;
        uint32_t *d_parity_work = nullptr, *d_parity_work_degree = nullptr;
        float *d_parity_work_distances = nullptr;
        uint8_t *d_parity_sources = nullptr, *d_parity_touched = nullptr;
        unsigned long long *d_parity_counters = nullptr;
        uint32_t parity_touched_count = 0;
        if (reverse_parity || reverse_reference_output == 2)
        {
            const uint32_t reverse_cap = min(R, 64u);
            if (reverse_parity)
            {
                CUDA_CHECK_FILTERED(cudaMalloc(&d_parity_graph, max_forward_edges * sizeof(uint32_t)));
                CUDA_CHECK_FILTERED(cudaMalloc(&d_parity_degree, (size_t)N * sizeof(uint32_t)));
                CUDA_CHECK_FILTERED(cudaMemcpy(d_parity_graph, d_graph_b, max_forward_edges * sizeof(uint32_t),
                                               cudaMemcpyDeviceToDevice));
                CUDA_CHECK_FILTERED(cudaMemcpy(d_parity_degree, d_degree_b, (size_t)N * sizeof(uint32_t),
                                               cudaMemcpyDeviceToDevice));
            }
            CUDA_CHECK_FILTERED(cudaMalloc(&d_parity_reverse_ids, (size_t)N * reverse_cap * sizeof(uint32_t)));
            CUDA_CHECK_FILTERED(cudaMalloc(&d_parity_reverse_counts, (size_t)N * sizeof(uint32_t)));
            CUDA_CHECK_FILTERED(cudaMalloc(&d_parity_touched, (size_t)N));
            CUDA_CHECK_FILTERED(cudaMalloc(&d_parity_counters, 16 * sizeof(unsigned long long)));
            CUDA_CHECK_FILTERED(cudaMemset(d_parity_reverse_ids, 0xff,
                                           (size_t)N * reverse_cap * sizeof(uint32_t)));
            CUDA_CHECK_FILTERED(cudaMemset(d_parity_reverse_counts, 0, (size_t)N * sizeof(uint32_t)));
            CUDA_CHECK_FILTERED(cudaMemset(d_parity_touched, 0, (size_t)N));
            CUDA_CHECK_FILTERED(cudaMemset(d_parity_counters, 0, 16 * sizeof(unsigned long long)));


            CUDA_CHECK_FILTERED(cudaMemset(d_active, 1, (size_t)N));
            filtered_reverse_generate_kernel<<<(N + 255) / 256, 256>>>(
                N, R, reverse_cap, 0, N, 2u, 0u, 1000u, 0u, 0u,
                d_active, d_graph_b, d_degree_b, d_offsets, d_labels, nullptr,
                num_labels, universal_label, d_parity_reverse_ids, d_parity_reverse_counts,
                d_parity_touched, d_parity_counters + 0, d_parity_counters + 1,
                d_parity_counters + 2, d_parity_counters + 3);
            CUDA_CHECK_FILTERED(cub::DeviceSelect::Flagged(d_select_temp, select_temp_bytes,
                                                            d_all_ids, d_parity_touched,
                                                            d_touched_ids, d_selected_count, N));
            CUDA_CHECK_FILTERED(cudaMemcpy(&parity_touched_count, d_selected_count, sizeof(uint32_t),
                                           cudaMemcpyDeviceToHost));
            if (parity_touched_count > 0)
            {
                CUDA_CHECK_FILTERED(cudaMalloc(&d_parity_work, (size_t)parity_touched_count * C * sizeof(uint32_t)));
                CUDA_CHECK_FILTERED(cudaMalloc(&d_parity_work_degree,
                                               (size_t)parity_touched_count * sizeof(uint32_t)));
                CUDA_CHECK_FILTERED(cudaMalloc(&d_parity_work_distances,
                                               (size_t)parity_touched_count * C * sizeof(float)));
                CUDA_CHECK_FILTERED(cudaMalloc(&d_parity_sources,
                                               (size_t)parity_touched_count * C * sizeof(uint8_t)));
                filtered_reverse_apply_to_work_kernel<SearchT><<<(parity_touched_count + 255) / 256, 256>>>(
                    d_data, N, dim, R, C, reverse_cap, 0, N, 0u, 0u, 1u,
                    d_parity_touched, d_touched_ids, parity_touched_count,
                    d_parity_reverse_ids, d_parity_reverse_counts, 2u,
                    d_graph_b, d_degree_b, d_parity_work, d_parity_work_degree,
                    d_parity_work_distances, d_parity_sources, d_offsets, d_labels,
                    universal_label, UINT32_MAX, d_parity_counters + 2, d_parity_counters + 3);
                if (reverse_parity)
                filtered_refine_prune_to_compact_kernel<SearchT, 128><<<parity_touched_count, 1>>>(
                    d_data, dim, d_parity_touched, d_parity_work, d_parity_work_distances,
                    d_parity_sources, d_parity_work_degree, d_parity_graph, d_parity_degree,
                    N, R, C, round_alpha, 0u, C, 0u, 0u, R, 0u, 0u, 0u,
                    0u, 0u, 0u, 0u, 0u, reverse_cap, 0u, 0u, 0u,
                    d_touched_ids, parity_touched_count, nullptr, d_offsets, d_labels,
                    universal_label, d_parity_counters + 2, d_parity_counters + 4,
                    d_parity_counters + 5, d_parity_counters + 6, d_parity_counters + 7,
                    d_parity_counters + 8, d_parity_counters + 9, d_parity_counters + 10,
                    d_parity_counters + 11, d_parity_counters + 12, d_parity_counters + 3);
                CUDA_CHECK_FILTERED(cudaGetLastError());
                CUDA_CHECK_FILTERED(cudaDeviceSynchronize());




                uint8_t *d_parity_warp_removed = nullptr;
                CUDA_CHECK_FILTERED(cudaMalloc(&d_parity_warp_graph,
                                               max_forward_edges * sizeof(uint32_t)));
                CUDA_CHECK_FILTERED(cudaMalloc(&d_parity_warp_degree, (size_t)N * sizeof(uint32_t)));
                CUDA_CHECK_FILTERED(cudaMemcpy(d_parity_warp_graph, d_graph_b,
                                               max_forward_edges * sizeof(uint32_t),
                                               cudaMemcpyDeviceToDevice));
                CUDA_CHECK_FILTERED(cudaMemcpy(d_parity_warp_degree, d_degree_b,
                                               (size_t)N * sizeof(uint32_t),
                                               cudaMemcpyDeviceToDevice));
                CUDA_CHECK_FILTERED(cudaMalloc(&d_parity_warp_removed,
                                               (size_t)parity_touched_count * C));
                CUDA_CHECK_FILTERED(cudaMemcpy(d_changed, d_forward_changed, (size_t)N,
                                               cudaMemcpyDeviceToDevice));
                canonical_exact_prune<SearchT><<<parity_touched_count, 32>>>(
                    d_data, N, dim, R, C, round_alpha, d_offsets, d_labels,
                    d_touched_ids, parity_touched_count, d_graph_b, d_degree_b,
                    d_parity_work, d_parity_work_degree, d_parity_work_distances,
                    d_parity_warp_removed, 1u, 1u, d_parity_warp_graph, nullptr,
                    d_parity_warp_degree, d_changed);
                CUDA_CHECK_FILTERED(cudaGetLastError());
                CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
                cudaFree(d_parity_warp_removed);
            }
        }

        uint32_t touched_count = 0;
        uint32_t round_reverse_proposals = 0;
        if (reverse_reference_output == 2)
        {
            touched_count = parity_touched_count;
            CUDA_CHECK_FILTERED(cudaMemcpy(d_graph_b, d_parity_warp_graph,
                                           max_forward_edges * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
            CUDA_CHECK_FILTERED(cudaMemcpy(d_degree_b, d_parity_warp_degree,
                                           (size_t)N * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
            CUDA_CHECK_FILTERED(cudaMemcpy(d_touched, d_parity_touched,
                                           (size_t)N, cudaMemcpyDeviceToDevice));
        }
        else
        {
        stage = now_sec_filtered();
        CUDA_CHECK_FILTERED(cudaMemcpy(d_changed, d_forward_changed, (size_t)N, cudaMemcpyDeviceToDevice));
        CUDA_CHECK_FILTERED(cudaMemset(d_touched, 0, (size_t)N));
        if (enable_exact_reverse_repair)
        {
        CUDA_CHECK_FILTERED(cudaMemset(d_reverse_offsets, 0, (size_t)(N + 1) * sizeof(uint32_t)));
        canonical_reverse_count<<<(active_count + 255) / 256, 256>>>(
            N, R, d_graph_b, d_degree_b, round_ids, active_count,
            d_offsets, d_labels, universal_label, d_reverse_offsets);
        CUDA_CHECK_FILTERED(cub::DeviceScan::InclusiveSum(d_scan_temp, scan_temp_bytes, d_reverse_offsets,
                                                           d_reverse_offsets, N + 1));
        CUDA_CHECK_FILTERED(cudaMemcpy(&round_reverse_proposals, d_reverse_offsets + N,
                                       sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_reverse_ids,
                                       (size_t)std::max(1u, round_reverse_proposals) * sizeof(uint32_t)));
        CUDA_CHECK_FILTERED(cudaMemcpy(d_reverse_cursor, d_reverse_offsets, (size_t)N * sizeof(uint32_t),
                                       cudaMemcpyDeviceToDevice));
        canonical_reverse_fill<<<(active_count + 255) / 256, 256>>>(
            N, R, d_graph_b, d_degree_b, d_reverse_cursor, d_reverse_ids,
            round_ids, active_count, d_offsets, d_labels, universal_label);
        CUDA_CHECK_FILTERED(cudaMemset(d_reverse_candidate_offsets, 0, (size_t)(N + 1) * sizeof(uint32_t)));
        canonical_reverse_candidate_count<<<(N + 255) / 256, 256>>>(
            N, R, d_graph_b, d_degree_b, d_reverse_offsets, d_reverse_ids,
            d_reverse_candidate_offsets, d_touched);
        CUDA_CHECK_FILTERED(cub::DeviceScan::InclusiveSum(d_scan_temp, scan_temp_bytes,
                                                           d_reverse_candidate_offsets,
                                                           d_reverse_candidate_offsets, N + 1));
        canonical_reverse_candidate_fill<<<(N + 255) / 256, 256>>>(
            N, R, d_graph_b, d_degree_b, d_reverse_offsets, d_reverse_ids,
            d_reverse_candidate_offsets, d_reverse_candidate_ids,
            nullptr, d_reverse_candidate_distances);
        CUDA_CHECK_FILTERED(cub::DeviceSelect::Flagged(d_select_temp, select_temp_bytes, d_all_ids, d_touched,
                                                        d_touched_ids, d_selected_count, N));
        CUDA_CHECK_FILTERED(cudaMemcpy(&touched_count, d_selected_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
        g_last_filtered_stats.csr_reverse_seconds += now_sec_filtered() - stage;

        stage = now_sec_filtered();
        if (touched_count > 0)
        {
            canonical_csr_candidate_distances<SearchT><<<touched_count, 32>>>(
                d_data, N, dim, d_reverse_candidate_offsets, d_touched_ids, touched_count,
                d_reverse_candidate_ids, d_reverse_candidate_distances);
            canonical_exact_prune_csr<SearchT><<<touched_count, 32>>>(
                d_data, N, dim, R, round_alpha, d_offsets, d_labels, d_touched_ids, touched_count,
                d_reverse_candidate_offsets, d_reverse_candidate_ids, d_reverse_candidate_distances,
                d_reverse_removed, d_graph_a, d_degree_a,
                d_graph_b, d_degree_b, d_changed);
        }
        CUDA_CHECK_FILTERED(cudaGetLastError());
        CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
        g_last_filtered_stats.reverse_prune_seconds += now_sec_filtered() - stage;
        cudaFree(d_reverse_ids);
        d_reverse_ids = nullptr;
        }
        else
        {
            touched_count = 0;
            round_reverse_proposals = 0;
        }
        }

        if (reverse_parity)
        {
            unsigned long long *d_mismatch_rows = nullptr;
            uint32_t *d_first_mismatch = nullptr;
            CUDA_CHECK_FILTERED(cudaMalloc(&d_mismatch_rows, sizeof(unsigned long long)));
            CUDA_CHECK_FILTERED(cudaMalloc(&d_first_mismatch, sizeof(uint32_t)));
            CUDA_CHECK_FILTERED(cudaMemset(d_mismatch_rows, 0, sizeof(unsigned long long)));
            const uint32_t no_row = UINT32_MAX;
            CUDA_CHECK_FILTERED(cudaMemcpy(d_first_mismatch, &no_row, sizeof(uint32_t), cudaMemcpyHostToDevice));
            canonical_compare_graph_sets<<<(N + 255) / 256, 256>>>(
                N, R, d_parity_graph, d_parity_degree,
                d_parity_warp_graph, d_parity_warp_degree,
                d_mismatch_rows, d_first_mismatch);
            unsigned long long prune_mismatch_rows = 0;
            uint32_t prune_first_mismatch = UINT32_MAX;
            CUDA_CHECK_FILTERED(cudaMemcpy(&prune_mismatch_rows, d_mismatch_rows,
                                           sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            CUDA_CHECK_FILTERED(cudaMemcpy(&prune_first_mismatch, d_first_mismatch,
                                           sizeof(uint32_t), cudaMemcpyDeviceToHost));
            if (prune_first_mismatch != UINT32_MAX)
            {
                std::vector<uint32_t> scalar_row(R), warp_row(R);
                uint32_t scalar_degree = 0, warp_degree = 0;
                CUDA_CHECK_FILTERED(cudaMemcpy(scalar_row.data(),
                    d_parity_graph + (size_t)prune_first_mismatch * R,
                    (size_t)R * sizeof(uint32_t), cudaMemcpyDeviceToHost));
                CUDA_CHECK_FILTERED(cudaMemcpy(warp_row.data(),
                    d_parity_warp_graph + (size_t)prune_first_mismatch * R,
                    (size_t)R * sizeof(uint32_t), cudaMemcpyDeviceToHost));
                CUDA_CHECK_FILTERED(cudaMemcpy(&scalar_degree,
                    d_parity_degree + prune_first_mismatch, sizeof(uint32_t), cudaMemcpyDeviceToHost));
                CUDA_CHECK_FILTERED(cudaMemcpy(&warp_degree,
                    d_parity_warp_degree + prune_first_mismatch, sizeof(uint32_t), cudaMemcpyDeviceToHost));
                printf("[gpu_vamana_filtered_prune_row] row=%u scalar_degree=%u warp_degree=%u scalar=",
                       prune_first_mismatch, scalar_degree, warp_degree);
                for (uint32_t i = 0; i < scalar_degree; ++i) printf("%u,", scalar_row[i]);
                printf(" warp=");
                for (uint32_t i = 0; i < warp_degree; ++i) printf("%u,", warp_row[i]);
                printf("\n");
            }
            CUDA_CHECK_FILTERED(cudaMemset(d_mismatch_rows, 0, sizeof(unsigned long long)));
            CUDA_CHECK_FILTERED(cudaMemcpy(d_first_mismatch, &no_row, sizeof(uint32_t), cudaMemcpyHostToDevice));
            canonical_compare_graph_sets<<<(N + 255) / 256, 256>>>(
                N, R, d_parity_graph, d_parity_degree, d_graph_b, d_degree_b,
                d_mismatch_rows, d_first_mismatch);
            unsigned long long mismatch_rows = 0;
            uint32_t first_mismatch = UINT32_MAX;
            CUDA_CHECK_FILTERED(cudaMemcpy(&mismatch_rows, d_mismatch_rows,
                                           sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            CUDA_CHECK_FILTERED(cudaMemcpy(&first_mismatch, d_first_mismatch,
                                           sizeof(uint32_t), cudaMemcpyDeviceToHost));
            printf("[gpu_vamana_filtered_reverse_parity] round=%u frozen_forward=1 "
                   "reference_touched=%u canonical_touched=%u warp_prune_mismatch_rows=%llu "
                   "warp_prune_first_row=%u final_prune_mismatch_rows=%llu first_row=%u\n",
                   round, parity_touched_count, touched_count,
                   prune_mismatch_rows, prune_first_mismatch, mismatch_rows, first_mismatch);
            if (reverse_reference_output)
            {
                const uint32_t *reference_graph = reverse_reference_output == 2 ?
                    d_parity_warp_graph : d_parity_graph;
                const uint32_t *reference_degree = reverse_reference_output == 2 ?
                    d_parity_warp_degree : d_parity_degree;
                CUDA_CHECK_FILTERED(cudaMemcpy(d_graph_b, reference_graph,
                                               max_forward_edges * sizeof(uint32_t),
                                               cudaMemcpyDeviceToDevice));
                CUDA_CHECK_FILTERED(cudaMemcpy(d_degree_b, reference_degree,
                                               (size_t)N * sizeof(uint32_t),
                                               cudaMemcpyDeviceToDevice));
                printf("[gpu_vamana_filtered_reverse_parity] round=%u output=%s\n", round,
                       reverse_reference_output == 2 ? "warp_prune" : "reference");
            }
            cudaFree(d_mismatch_rows); cudaFree(d_first_mismatch);
            cudaFree(d_parity_graph); cudaFree(d_parity_degree);
            cudaFree(d_parity_warp_graph); cudaFree(d_parity_warp_degree);
            cudaFree(d_parity_reverse_ids); cudaFree(d_parity_reverse_counts);
            cudaFree(d_parity_work); cudaFree(d_parity_work_degree);
            cudaFree(d_parity_work_distances); cudaFree(d_parity_sources);
            cudaFree(d_parity_touched); cudaFree(d_parity_counters);
        }
        else if (reverse_reference_output == 2)
        {
            cudaFree(d_parity_warp_graph); cudaFree(d_parity_warp_degree);
            cudaFree(d_parity_reverse_ids); cudaFree(d_parity_reverse_counts);
            cudaFree(d_parity_work); cudaFree(d_parity_work_degree);
            cudaFree(d_parity_work_distances); cudaFree(d_parity_sources);
            cudaFree(d_parity_touched); cudaFree(d_parity_counters);
        }

        uint32_t changed_count = 0;
        CUDA_CHECK_FILTERED(cub::DeviceSelect::Flagged(d_select_temp, select_temp_bytes, d_all_ids, d_changed,
                                                        d_changed_ids, d_selected_count, N));
        CUDA_CHECK_FILTERED(cudaMemcpy(&changed_count, d_selected_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        printf("[gpu_vamana_filtered_canonical_round] round=%u kind=full alpha=%.3f active=%u tasks=%u changed=%u touched=%u reverse_proposals=%u\n",
               round, round_alpha, active_count, active_task_count, changed_count, touched_count,
               round_reverse_proposals);
        std::swap(d_graph_a, d_graph_b);
        std::swap(d_degree_a, d_degree_b);
        active_count = N;
    }




    if (enable_coverage_repair)
    {
        unsigned long long *d_coverage_repair_counters = nullptr;
        unsigned long long coverage_repair_counters[3] = {};
        CUDA_CHECK_FILTERED(cudaMalloc(&d_coverage_repair_counters, sizeof(coverage_repair_counters)));
        CUDA_CHECK_FILTERED(cudaMemset(d_coverage_repair_counters, 0, sizeof(coverage_repair_counters)));
        canonical_repair_label_coverage<SearchT><<<N, 32>>>(
            d_data, N, dim, R, per_label_keep, coverage_quota, d_offsets, d_labels, d_label_sizes,
            num_labels, universal_label, d_task_candidates, d_task_candidate_distances,
            d_graph_a, d_degree_a, d_coverage_repair_counters);
        CUDA_CHECK_FILTERED(cudaGetLastError());
        CUDA_CHECK_FILTERED(cudaMemcpy(coverage_repair_counters, d_coverage_repair_counters,
                                       sizeof(coverage_repair_counters), cudaMemcpyDeviceToHost));
        printf("[gpu_vamana_filtered_label_coverage_repair] missing_before=%llu repaired=%llu unresolved=%llu\n",
               coverage_repair_counters[0], coverage_repair_counters[1], coverage_repair_counters[2]);
        cudaFree(d_coverage_repair_counters);
    }
    else
    {
        printf("[gpu_vamana_filtered_label_coverage_repair] disabled=1\n");
    }

    uint32_t publish_R = R;
    const char *publish_env = getenv("DISKANN_GPU_FILTERED_PUBLISH_DEGREE");
    if (publish_env && atoi(publish_env) > 0)
        publish_R = min(R, (uint32_t)atoi(publish_env));
    uint32_t *d_output_graph = d_graph_a, *d_output_degree = d_degree_a;
    uint32_t *d_publish_graph = nullptr, *d_publish_final_graph = nullptr;
    uint32_t *d_publish_degree = nullptr, *d_publish_final_degree = nullptr;
    float *d_publish_input_distances = nullptr, *d_publish_work_distances = nullptr;
    uint32_t *d_publish_work = nullptr, *d_publish_work_degree = nullptr;
    uint8_t *d_publish_state = nullptr, *d_publish_removed = nullptr, *d_publish_changed = nullptr;
    if (publish_R < R)
    {
        const double publish_start = now_sec_filtered();
        const size_t output_edges = (size_t)N * publish_R;


        cudaFree(d_graph_b); d_graph_b = nullptr;
        cudaFree(d_degree_b); d_degree_b = nullptr;
        cudaFree(d_task_candidates); d_task_candidates = nullptr;
        cudaFree(d_task_candidate_distances); d_task_candidate_distances = nullptr;
        CUDA_CHECK_FILTERED(cudaMalloc(&d_reverse_ids, max_forward_edges * sizeof(uint32_t)));
        d_publish_input_distances = reinterpret_cast<float *>(d_reverse_ids);
        d_publish_state = d_removed;
        d_publish_work = d_work;
        d_publish_work_distances = d_work_distances;
        d_publish_work_degree = d_work_degrees;
        d_publish_removed = d_removed;
        CUDA_CHECK_FILTERED(cudaMalloc(&d_publish_graph, output_edges * sizeof(uint32_t)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_publish_degree, (size_t)N * sizeof(uint32_t)));
        update_filtered_peak_gpu_memory();
        d_publish_changed = d_changed;
        canonical_fixed_rare_shortlist<SearchT><<<N, 32>>>(
            d_data, N, dim, R, publish_R, d_graph_a, d_degree_a,
            d_offsets, d_labels, d_label_sizes, num_labels, universal_label,
            d_publish_input_distances, d_publish_state, d_publish_work,
            d_publish_work_distances, d_publish_work_degree);
        canonical_exact_prune<SearchT><<<N, 32>>>(
            d_data, N, dim, publish_R, publish_R, alpha, d_offsets, d_labels,
            nullptr, N, d_publish_work, d_publish_work_degree,
            d_publish_work, d_publish_work_degree, d_publish_work_distances,
            d_publish_removed, 0u, 0u, d_publish_graph, nullptr,
            d_publish_degree, d_publish_changed);
        CUDA_CHECK_FILTERED(cudaGetLastError());
        CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
        const double forward_done = now_sec_filtered();


        CUDA_CHECK_FILTERED(cudaMemset(d_reverse_offsets, 0, (size_t)(N + 1) * sizeof(uint32_t)));
        canonical_reverse_count<<<(N + 255) / 256, 256>>>(
            N, publish_R, d_publish_graph, d_publish_degree, nullptr, N,
            d_offsets, d_labels, universal_label, d_reverse_offsets);
        CUDA_CHECK_FILTERED(cub::DeviceScan::InclusiveSum(
            d_scan_temp, scan_temp_bytes, d_reverse_offsets, d_reverse_offsets, N + 1));
        CUDA_CHECK_FILTERED(cudaMemcpy(d_reverse_cursor, d_reverse_offsets,
                                       (size_t)N * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
        canonical_reverse_fill<<<(N + 255) / 256, 256>>>(
            N, publish_R, d_publish_graph, d_publish_degree, d_reverse_cursor,
            d_reverse_ids, nullptr, N, d_offsets, d_labels, universal_label);
        CUDA_CHECK_FILTERED(cudaMemset(d_reverse_candidate_offsets, 0,
                                       (size_t)(N + 1) * sizeof(uint32_t)));
        canonical_reverse_candidate_count<<<(N + 255) / 256, 256>>>(
            N, publish_R, d_publish_graph, d_publish_degree, d_reverse_offsets,
            d_reverse_ids, d_reverse_candidate_offsets, d_touched);
        CUDA_CHECK_FILTERED(cub::DeviceScan::InclusiveSum(
            d_scan_temp, scan_temp_bytes, d_reverse_candidate_offsets,
            d_reverse_candidate_offsets, N + 1));
        canonical_reverse_candidate_fill<<<(N + 255) / 256, 256>>>(
            N, publish_R, d_publish_graph, d_publish_degree, d_reverse_offsets,
            d_reverse_ids, d_reverse_candidate_offsets, d_reverse_candidate_ids,
            nullptr, d_reverse_candidate_distances);
        CUDA_CHECK_FILTERED(cub::DeviceSelect::Flagged(
            d_select_temp, select_temp_bytes, d_all_ids, d_touched,
            d_touched_ids, d_selected_count, N));
        uint32_t publish_touched = 0;
        CUDA_CHECK_FILTERED(cudaMemcpy(&publish_touched, d_selected_count,
                                       sizeof(uint32_t), cudaMemcpyDeviceToHost));
        canonical_csr_candidate_distances<SearchT><<<publish_touched, 32>>>(
            d_data, N, dim, d_reverse_candidate_offsets, d_touched_ids,
            publish_touched, d_reverse_candidate_ids, d_reverse_candidate_distances);




        cudaFree(d_graph_a); d_graph_a = nullptr;
        cudaFree(d_degree_a); d_degree_a = nullptr;
        cudaFree(d_reverse_ids); d_reverse_ids = nullptr;
        CUDA_CHECK_FILTERED(cudaMalloc(&d_publish_work, output_edges * sizeof(uint32_t)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_publish_work_distances, output_edges * sizeof(float)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_publish_work_degree, (size_t)N * sizeof(uint32_t)));
        update_filtered_peak_gpu_memory();
        canonical_csr_rare_shortlist<<<publish_touched, 32>>>(
            N, publish_R, d_touched_ids, publish_touched,
            d_reverse_candidate_offsets, d_reverse_candidate_ids,
            d_reverse_candidate_distances, d_offsets, d_labels, d_label_sizes,
            num_labels, universal_label, d_publish_state, d_publish_work,
            d_publish_work_distances, d_publish_work_degree);
        canonical_exact_prune<SearchT><<<publish_touched, 32>>>(
            d_data, N, dim, publish_R, publish_R, alpha, d_offsets, d_labels,
            d_touched_ids, publish_touched, d_publish_graph, d_publish_degree,
            d_publish_work, d_publish_work_degree, d_publish_work_distances,
            d_publish_removed, 1u, 0u, d_publish_graph, nullptr,
            d_publish_degree, nullptr);
        CUDA_CHECK_FILTERED(cudaGetLastError());
        CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
        printf("[gpu_vamana_filtered_gpu_publish] internal_R=%u publish_R=%u touched=%u "
               "forward_prune=%.6f reverse_total=%.6f total=%.6f\n",
               R, publish_R, publish_touched, forward_done - publish_start,
               now_sec_filtered() - forward_done, now_sec_filtered() - publish_start);
        d_output_graph = d_publish_graph;
        d_output_degree = d_publish_degree;
    }

    const double d2h_start = now_sec_filtered();
    CUDA_CHECK_FILTERED(cudaMemcpy(h_graph, d_output_graph,
                                   (size_t)N * publish_R * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_FILTERED(cudaMemcpy(h_degrees, d_output_degree,
                                   (size_t)N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    g_last_filtered_stats.d2h_seconds += now_sec_filtered() - d2h_start;
    const double canonical_diagnostic_start = now_sec_filtered();
    canonical_print_round_diagnostics(N, publish_R, h_graph, h_degrees, h_offsets, h_labels, universal_label);
    g_last_filtered_stats.diagnostics_seconds += now_sec_filtered() - canonical_diagnostic_start;
    cudaFree(d_data); cudaFree(d_offsets); cudaFree(d_labels); cudaFree(d_starts); cudaFree(d_label_sizes); cudaFree(d_task_rows);
    cudaFree(d_label_point_offsets); cudaFree(d_label_points);
    cudaFree(d_graph_a); cudaFree(d_graph_b); cudaFree(d_degree_a); cudaFree(d_degree_b);
    cudaFree(d_task_candidates); cudaFree(d_task_candidate_distances);
    cudaFree(d_work); cudaFree(d_work_degrees);
    cudaFree(d_work_distances); cudaFree(d_removed); cudaFree(d_changed); cudaFree(d_forward_changed);
    cudaFree(d_active); cudaFree(d_deficient); cudaFree(d_task_flags); cudaFree(d_all_ids); cudaFree(d_active_ids);
    cudaFree(d_changed_ids); cudaFree(d_touched_ids); cudaFree(d_deficient_ids); cudaFree(d_all_task_ids);
    cudaFree(d_active_task_ids); cudaFree(d_selected_count);
    cudaFree(d_reverse_offsets); cudaFree(d_reverse_cursor); cudaFree(d_reverse_ids);
    cudaFree(d_reverse_candidate_offsets); cudaFree(d_touched);
    cudaFree(d_publish_graph); cudaFree(d_publish_final_graph);
    cudaFree(d_publish_degree); cudaFree(d_publish_final_degree);


    cudaFree(d_publish_work_distances);
    cudaFree(d_publish_work); cudaFree(d_publish_work_degree);
    cudaFree(d_scan_temp); cudaFree(d_select_temp);
    unsigned long long distance_counts[6] = {};
    CUDA_CHECK_FILTERED(cudaMemcpyFromSymbol(distance_counts, g_canonical_distance_counts,
                                              sizeof(distance_counts)));
    const double bytes_per_vector = (double)dim * sizeof(SearchT);
    printf("[gpu_vamana_filtered_canonical_distances] filtered=%llu ordinary=%llu work_only=%llu forward_prune=%llu reverse_only=%llu reverse_prune=%llu dram_bytes_estimate=%.0f\n",
           distance_counts[0], distance_counts[1], distance_counts[2], distance_counts[3],
           distance_counts[4], distance_counts[5],
           (distance_counts[0] + distance_counts[1] + distance_counts[2] + distance_counts[3] +
            distance_counts[4] + distance_counts[5]) * 2.0 * bytes_per_vector);
    printf("[gpu_vamana_filtered_canonical] rounds=%u graph_buffers=%u fused_label_threshold=%u total=%.6f task=%.6f filtered=%.6f ordinary=%.6f merge=%.6f work_distance=%.6f forward_prune=%.6f csr_reverse=%.6f reverse_prune=%.6f\n",
           configured_rounds, 2u, fused_label_threshold,
           now_sec_filtered() - start, g_last_filtered_stats.task_generation_seconds,
           g_last_filtered_stats.filtered_search_seconds, g_last_filtered_stats.ordinary_search_seconds,
           g_last_filtered_stats.merge_seconds, work_distance_seconds, g_last_filtered_stats.forward_prune_seconds,
           g_last_filtered_stats.csr_reverse_seconds, g_last_filtered_stats.reverse_prune_seconds);
    return 0;
}

template <typename DataT>
static int canonical_round0_from_host_backbone(const DataT *h_data, uint32_t N, uint32_t dim, uint32_t R,
                                                uint32_t C, uint32_t search_L, uint32_t filtered_L,
                                                uint32_t build_steps, float alpha, const uint32_t *h_offsets,
                                                const uint32_t *h_labels, uint32_t total_labels,
                                                const uint32_t *h_label_starts, uint32_t num_labels,
                                                uint32_t universal_label, uint32_t global_start,
                                                uint32_t *h_graph, uint32_t *h_degrees,
                                                DataT *d_handoff_data = nullptr,
                                                uint32_t *d_handoff_graph = nullptr,
                                                uint32_t *d_handoff_degrees = nullptr)
{
    if constexpr (std::is_same<DataT, float>::value)
    {
        return canonical_round0_from_host_backbone_typed<__half, float>(
                h_data, N, dim, R, C, search_L, filtered_L, build_steps, alpha, h_offsets, h_labels,
                total_labels, h_label_starts, num_labels, universal_label, global_start, h_graph, h_degrees,
                d_handoff_data, d_handoff_graph, d_handoff_degrees);
    }
    else
        return canonical_round0_from_host_backbone_typed<DataT, DataT>(
            h_data, N, dim, R, C, search_L, filtered_L, build_steps, alpha, h_offsets, h_labels,
            total_labels, h_label_starts, num_labels, universal_label, global_start, h_graph, h_degrees,
            d_handoff_data, d_handoff_graph, d_handoff_degrees);
}

template <typename DataT>
static int gpu_vamana_filtered_build_impl(const DataT *h_data,
                                          uint32_t num_points,
                                          uint32_t dim,
                                          uint32_t R,
                                          uint32_t L,
                                          uint32_t filtered_L,
                                          uint32_t C,
                                          uint32_t STEPS,
                                          float build_alpha,
                                          const uint32_t *point_label_offsets,
                                          const uint32_t *point_labels,
                                          uint32_t total_label_count,
                                          const uint32_t *label_to_start_id,
                                          uint32_t num_labels,
                                          uint32_t universal_label_id,
                                          uint32_t global_start_id,
                                          const uint32_t *query_anchor_offsets,
                                          const uint32_t *query_anchor_ids,
                                          uint32_t query_anchor_count,
                                          uint32_t *h_graph,
                                          uint32_t *h_degree,
                                          const char *type_name)
{
    memset(&g_last_filtered_stats, 0, sizeof(g_last_filtered_stats));
    g_last_filtered_stats.semantic_complete = 0;
    double t0 = now_sec_filtered();
    uint32_t stage = GPU_FILTERED_STAGE_LABEL_SYNC;
    bool run_label_prune = false;
    bool run_two_pool = false;
    bool run_work_label_prune = false;
    bool run_refine_vnew2 = true;
    bool run_label_sync = true;
    const bool run_canonical_v2 = true;
    const bool filtered_mode_full = true;
    const bool filtered_mode_active = false;
    uint32_t geometric_only_prune = 0;
    CUDA_CHECK_FILTERED(cudaMemcpyToSymbol(g_filtered_geometric_only_prune, &geometric_only_prune,
                                           sizeof(geometric_only_prune)));
    uint32_t balance_per_label = 0;
    CUDA_CHECK_FILTERED(cudaMemcpyToSymbol(g_filtered_balance_per_label, &balance_per_label,
                                           sizeof(balance_per_label)));
    const char *stage_name = "label_sync";
    g_last_filtered_stats.semantic_stage = stage;
    g_last_filtered_stats.semantic_complete =
        (run_work_label_prune || filtered_mode_full || filtered_mode_active) ? 1 : 0;

    if (!h_data || !h_graph || !h_degree || !point_label_offsets || !point_labels || !label_to_start_id)
    {
        fprintf(stderr, "[gpu_vamana_filtered] null input pointer.\n");
        return -1;
    }
    if (num_points == 0 || dim == 0 || R == 0 || L == 0 || C < R)
    {
        fprintf(stderr, "[gpu_vamana_filtered] invalid parameters N=%u dim=%u R=%u L=%u C=%u\n",
                num_points, dim, R, L, C);
        return -1;
    }
    const uint32_t cpu_input_C = C;
    const uint32_t default_backbone_C = std::min(cpu_input_C, 96u);
    const uint32_t default_work_C = std::min(cpu_input_C, 128u);
    const uint32_t gpu_backbone_C =
        env_u32_clamped("DISKANN_GPU_FILTERED_BACKBONE_C", default_backbone_C, R, cpu_input_C);
    const uint32_t gpu_work_C =
        env_u32_clamped("DISKANN_GPU_FILTERED_WORK_C", default_work_C, R, cpu_input_C);

    printf("[gpu_vamana_filtered] enabled type=%s N=%u dim=%u R=%u L=%u FilteredL=%u C=%u labels=%u total_point_labels=%u universal=%u global_start=%u\n",
           type_name, num_points, dim, R, L, filtered_L, C, num_labels, total_label_count, universal_label_id,
           global_start_id);
    printf("[gpu_vamana_filtered] cpu_input_C=%u gpu_backbone_C=%u gpu_work_C=%u\n",
           cpu_input_C, gpu_backbone_C, gpu_work_C);
    printf("[gpu_vamana_filtered] semantic_stage=%s semantic_complete=%u%s\n", stage_name,
           g_last_filtered_stats.semantic_complete,
           run_label_sync ? " using GPU bulk-synchronous label refinement." :
           (run_refine_vnew2 ? " using vnew2 graph then filtered refine." :
                              (run_two_pool ? " using opt-in two-pool filtered path." :
                                              " using vnew2 GPU graph substrate.")));
    printf("[gpu_vamana_filtered] implementation=canonical_v2\n");

    uint32_t *d_offsets = nullptr;
    uint32_t *d_labels = nullptr;
    uint32_t *d_label_starts = nullptr;
    double label_t0 = now_sec_filtered();
    CUDA_CHECK_FILTERED(cudaMalloc(&d_offsets, (size_t)(num_points + 1) * sizeof(uint32_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_labels, (size_t)total_label_count * sizeof(uint32_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_label_starts, (size_t)std::max(1u, num_labels) * sizeof(uint32_t)));
    CUDA_CHECK_FILTERED(cudaMemcpy(d_offsets, point_label_offsets, (size_t)(num_points + 1) * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice));
    CUDA_CHECK_FILTERED(cudaMemcpy(d_labels, point_labels, (size_t)total_label_count * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice));
    CUDA_CHECK_FILTERED(cudaMemcpy(d_label_starts, label_to_start_id, (size_t)std::max(1u, num_labels) * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice));
    CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
    g_last_filtered_stats.label_h2d_seconds = now_sec_filtered() - label_t0;

    double graph_t0 = now_sec_filtered();
    int ret = 0;
    DataT *d_handoff_data = nullptr;
    uint32_t *d_handoff_graph = nullptr, *d_handoff_degree = nullptr;
    bool used_device_handoff = false;
    unsigned long long pre_label_checks = 0;
    unsigned long long pre_label_rejects = 0;
    unsigned long long pre_universal_pass = 0;




    const char *backbone_input = getenv("DISKANN_GPU_FILTERED_BACKBONE_INPUT");
    const char *handoff_env = getenv("DISKANN_GPU_FILTERED_HANDOFF");
    const bool canonical_device_handoff = !handoff_env || atoi(handoff_env) != 0;
    if (backbone_input && backbone_input[0] != '\0')
    {
        FILE *fp = fopen(backbone_input, "rb");
        uint32_t header[3] = {0, 0, 0};
        const uint32_t expected_magic = 0x46424231u;
        if (!fp || fread(header, sizeof(uint32_t), 3, fp) != 3 || header[0] != expected_magic ||
            header[1] != num_points || header[2] != R ||
            fread(h_graph, sizeof(uint32_t), (size_t)num_points * R, fp) != (size_t)num_points * R ||
            fread(h_degree, sizeof(uint32_t), num_points, fp) != num_points)
        {
            fprintf(stderr, "[gpu_vamana_filtered] invalid frozen backbone input: %s\n", backbone_input);
            if (fp)
                fclose(fp);
            cudaFree(d_offsets);
            cudaFree(d_labels);
            cudaFree(d_label_starts);
            return -1;
        }
        fclose(fp);
        printf("[gpu_vamana_filtered] loaded frozen diagnostic backbone from %s\n", backbone_input);
    }
    else if (run_two_pool)
    {
        DataT *d_data_build = nullptr;
        uint32_t *d_graph_cur_build = nullptr;
        uint32_t *d_graph_work_build = nullptr;
        uint32_t *d_degree_cur_build = nullptr;
        uint32_t *d_degree_work_build = nullptr;
        float *d_graph_dists_build = nullptr;
        uint32_t *d_label_multi_starts = nullptr;
        uint32_t *d_label_point_offsets = nullptr;
        uint32_t *d_label_points = nullptr;
        unsigned long long *d_filtered_dist = nullptr;
        unsigned long long *d_unfiltered_dist = nullptr;
        unsigned long long *d_build_label_checks = nullptr;
        unsigned long long *d_build_label_rejects = nullptr;
        unsigned long long *d_build_universal = nullptr;
        unsigned long long *d_filtered_cands = nullptr;
        unsigned long long *d_unfiltered_cands = nullptr;
        unsigned long long *d_merged_cands = nullptr;
        unsigned long long *d_filtered_seed_count = nullptr;
        unsigned long long *d_reserved_cands = nullptr;
        unsigned long long *d_unfiltered_reserved = nullptr;
        unsigned long long *d_prune_rejects = nullptr;
        unsigned long long *d_label_occlusion_blocked = nullptr;
        unsigned long long *d_geometry_occluded = nullptr;
        unsigned long long *d_candidate_rejected_by_label = nullptr;
        unsigned long long *d_refill_count = nullptr;
        unsigned long long *d_prune_degree_sum = nullptr;

        uint32_t block = 128;
        uint32_t grid = (num_points + block - 1) / block;
        uint32_t filtered_pool_cap = filtered_L < 64 ? filtered_L : 64;
        const char *pool_env = getenv("DISKANN_GPU_FILTERED_POOL_SIZE");
        if (pool_env && atoi(pool_env) > 0)
            filtered_pool_cap = (uint32_t)atoi(pool_env);
        if (filtered_pool_cap == 0)
            filtered_pool_cap = 1;
        if (filtered_pool_cap > 96)
            filtered_pool_cap = 96;
        uint32_t inverted_seeds_enabled = 0;
        const char *inv_env = getenv("DISKANN_GPU_FILTERED_INVERTED_SEEDS");
        if (inv_env && atoi(inv_env) > 0)
            inverted_seeds_enabled = 1;
        uint32_t starts_per_label = 1;
        const char *starts_env = getenv("DISKANN_GPU_FILTERED_STARTS_PER_LABEL");
        if (starts_env && atoi(starts_env) > 0)
            starts_per_label = (uint32_t)atoi(starts_env);
        if (starts_per_label != 1 && starts_per_label != 2 && starts_per_label != 4)
            starts_per_label = 1;
        uint32_t label_anchor_mode = 0;
        const char *anchor_mode_env = getenv("DISKANN_GPU_FILTERED_LABEL_ANCHOR_MODE");
        if (anchor_mode_env && strcmp(anchor_mode_env, "spread") == 0)
            label_anchor_mode = 1;
        uint32_t label_seed_count = 0;
        const char *seed_env = getenv("DISKANN_GPU_FILTERED_LABEL_SEEDS");
        if (seed_env && atoi(seed_env) >= 0)
            label_seed_count = (uint32_t)atoi(seed_env);
        if (label_seed_count > 4)
            label_seed_count = 4;
        uint32_t filtered_reserve = 0;
        if (stage == GPU_FILTERED_STAGE_B2_TWO_POOL_RESERVE_ORDINARY_PRUNE ||
            stage == GPU_FILTERED_STAGE_B3_TWO_POOL_WORK_LABEL_PRUNE)
            filtered_reserve = 8;
        const char *reserve_env = getenv("DISKANN_GPU_FILTERED_RESERVE");
        if (reserve_env && atoi(reserve_env) >= 0)
            filtered_reserve = (uint32_t)atoi(reserve_env);
        if (filtered_reserve > R)
            filtered_reserve = R;
        uint32_t unfiltered_reserve = 24;
        const char *ureserve_env = getenv("DISKANN_GPU_UNFILTERED_RESERVE");
        if (ureserve_env && atoi(ureserve_env) >= 0)
            unfiltered_reserve = (uint32_t)atoi(ureserve_env);
        if (unfiltered_reserve > gpu_work_C)
            unfiltered_reserve = gpu_work_C;
        if (filtered_reserve + unfiltered_reserve > gpu_work_C)
        {
            uint32_t old_filtered = filtered_reserve;
            filtered_reserve = gpu_work_C > unfiltered_reserve ? gpu_work_C - unfiltered_reserve : 0;
            printf("[gpu_vamana_filtered] clamp two_pool filtered_reserve %u -> %u because gpu_work_C=%u unfiltered_reserve=%u\n",
                   old_filtered, filtered_reserve, gpu_work_C, unfiltered_reserve);
        }
        uint32_t max_expand_steps = 8;
        const char *expand_env = getenv("DISKANN_GPU_FILTERED_MAX_EXPAND_STEPS");
        if (expand_env && atoi(expand_env) > 0)
            max_expand_steps = (uint32_t)atoi(expand_env);
        uint32_t target_max_expand_steps = 0;
        const char *target_expand_env = getenv("DISKANN_GPU_FILTERED_TARGET_MAX_EXPAND_STEPS");
        if (target_expand_env && atoi(target_expand_env) > 0)
            target_max_expand_steps = (uint32_t)atoi(target_expand_env);
        if (target_max_expand_steps < max_expand_steps)
            target_max_expand_steps = 0;
        uint32_t work_prune_cap = 256;
        const char *cap_env = getenv("DISKANN_GPU_FILTERED_WORK_PRUNE_CAP");
        if (cap_env && atoi(cap_env) > 0)
            work_prune_cap = (uint32_t)atoi(cap_env);
        if (work_prune_cap > 256)
            work_prune_cap = 256;
        uint32_t local_c_cap = 256;
        const char *local_c_env = getenv("DISKANN_GPU_FILTERED_LOCAL_C");
        if (local_c_env && atoi(local_c_env) > 0)
            local_c_cap = (uint32_t)atoi(local_c_env);
        if (local_c_cap <= 128)
            local_c_cap = 128;
        else
            local_c_cap = 256;
        if (work_prune_cap > local_c_cap)
            work_prune_cap = local_c_cap;

        printf("[gpu_vamana_filtered] two_pool path pool_size=%u filtered_reserve=%u unfiltered_reserve=%u starts_per_label=%u inverted_seeds=%u inverted_seed_limit=%u max_expand_steps=%u work_prune_cap=%u iterations=%u work_C=%u\n",
               filtered_pool_cap, filtered_reserve, unfiltered_reserve, starts_per_label, inverted_seeds_enabled,
               label_seed_count, max_expand_steps, work_prune_cap, STEPS, gpu_work_C);

        std::vector<uint32_t> h_label_point_offsets((size_t)std::max(1u, num_labels) + 1, 0);
        for (uint32_t p = 0; p < num_points; ++p)
        {
            for (uint32_t j = point_label_offsets[p]; j < point_label_offsets[p + 1]; ++j)
            {
                uint32_t lbl = point_labels[j];
                if (lbl < num_labels)
                    ++h_label_point_offsets[(size_t)lbl + 1];
            }
        }
        for (uint32_t l = 1; l <= num_labels; ++l)
            h_label_point_offsets[l] += h_label_point_offsets[l - 1];
        std::vector<uint32_t> h_label_points(h_label_point_offsets[num_labels], INVALID_ID);
        std::vector<uint32_t> h_label_cursor = h_label_point_offsets;
        for (uint32_t p = 0; p < num_points; ++p)
        {
            for (uint32_t j = point_label_offsets[p]; j < point_label_offsets[p + 1]; ++j)
            {
                uint32_t lbl = point_labels[j];
                if (lbl < num_labels)
                    h_label_points[h_label_cursor[lbl]++] = p;
            }
        }
        std::vector<uint32_t> h_label_multi_starts((size_t)std::max(1u, num_labels) * starts_per_label, INVALID_ID);
        for (uint32_t l = 0; l < num_labels; ++l)
        {
            uint32_t base_start = label_to_start_id[l];
            if (base_start != INVALID_ID)
                h_label_multi_starts[(size_t)l * starts_per_label] = base_start;
            uint32_t pb = h_label_point_offsets[l];
            uint32_t pe = h_label_point_offsets[l + 1];
            uint32_t pc = pe > pb ? pe - pb : 0;
            for (uint32_t s = 1; s < starts_per_label; ++s)
            {
                if (pc == 0)
                    break;
                uint32_t best = INVALID_ID;
                float best_d = FLT_MAX;
                uint32_t stride = pc / 64;
                if (stride == 0)
                    stride = 1;
                for (uint32_t idx = pb + s; idx < pe; idx += stride)
                {
                    uint32_t cand = h_label_points[idx];
                    if (cand == INVALID_ID || cand >= num_points || cand == base_start)
                        continue;
                    float d = base_start < num_points ? 0.0f : FLT_MAX;
                    if (base_start < num_points)
                    {
                        d = 0.0f;
                        for (uint32_t dd = 0; dd < dim; ++dd)
                        {
                            float diff = (float)h_data[(size_t)cand * dim + dd] -
                                         (float)h_data[(size_t)base_start * dim + dd];
                            d += diff * diff;
                        }
                    }
                    if (d < best_d)
                    {
                        best_d = d;
                        best = cand;
                    }
                }
                h_label_multi_starts[(size_t)l * starts_per_label + s] = best;
            }
        }

        CUDA_CHECK_FILTERED(cudaMalloc(&d_data_build, (size_t)num_points * dim * sizeof(DataT)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_graph_cur_build, (size_t)num_points * R * sizeof(uint32_t)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_graph_work_build, (size_t)num_points * gpu_work_C * sizeof(uint32_t)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_degree_cur_build, (size_t)num_points * sizeof(uint32_t)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_degree_work_build, (size_t)num_points * sizeof(uint32_t)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_graph_dists_build, (size_t)num_points * gpu_work_C * sizeof(float)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_label_multi_starts,
                                       std::max((size_t)1, h_label_multi_starts.size()) * sizeof(uint32_t)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_label_point_offsets, ((size_t)num_labels + 1) * sizeof(uint32_t)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_label_points, std::max((size_t)1, h_label_points.size()) * sizeof(uint32_t)));
        CUDA_CHECK_FILTERED(cudaMemcpy(d_data_build, h_data, (size_t)num_points * dim * sizeof(DataT),
                                       cudaMemcpyHostToDevice));
        CUDA_CHECK_FILTERED(cudaMemcpy(d_label_point_offsets, h_label_point_offsets.data(),
                                       ((size_t)num_labels + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK_FILTERED(cudaMemcpy(d_label_multi_starts, h_label_multi_starts.data(),
                                       h_label_multi_starts.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        if (!h_label_points.empty())
        {
            CUDA_CHECK_FILTERED(cudaMemcpy(d_label_points, h_label_points.data(),
                                           h_label_points.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK_FILTERED(cudaMalloc(&d_filtered_dist, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_unfiltered_dist, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_build_label_checks, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_build_label_rejects, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_build_universal, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_filtered_cands, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_unfiltered_cands, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_merged_cands, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_filtered_seed_count, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_reserved_cands, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_unfiltered_reserved, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_prune_rejects, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_label_occlusion_blocked, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_geometry_occluded, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_candidate_rejected_by_label, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_refill_count, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_prune_degree_sum, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMemset(d_filtered_dist, 0, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMemset(d_unfiltered_dist, 0, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMemset(d_build_label_checks, 0, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMemset(d_build_label_rejects, 0, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMemset(d_build_universal, 0, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMemset(d_filtered_cands, 0, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMemset(d_unfiltered_cands, 0, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMemset(d_merged_cands, 0, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMemset(d_filtered_seed_count, 0, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMemset(d_reserved_cands, 0, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMemset(d_unfiltered_reserved, 0, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMemset(d_prune_rejects, 0, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMemset(d_label_occlusion_blocked, 0, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMemset(d_geometry_occluded, 0, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMemset(d_candidate_rejected_by_label, 0, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMemset(d_refill_count, 0, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMemset(d_prune_degree_sum, 0, sizeof(unsigned long long)));

        filtered_init_graph_kernel<DataT><<<grid, block>>>(d_data_build, dim, d_graph_cur_build, d_degree_cur_build,
                                                           d_graph_work_build, d_degree_work_build,
                                                           d_graph_dists_build, num_points, R, gpu_work_C);
        CUDA_CHECK_FILTERED(cudaGetLastError());
        CUDA_CHECK_FILTERED(cudaDeviceSynchronize());

        uint32_t iterations = STEPS == 0 ? 1 : STEPS;
        for (uint32_t iter = 0; iter < iterations; ++iter)
        {
            filtered_two_pool_inject_kernel<DataT><<<grid, block>>>(
                d_data_build, num_points, dim, R, L, filtered_L, gpu_work_C, filtered_pool_cap, filtered_reserve,
                unfiltered_reserve, max_expand_steps, inverted_seeds_enabled, starts_per_label, global_start_id,
                d_graph_cur_build, d_degree_cur_build, d_graph_work_build, d_degree_work_build, d_graph_dists_build,
                d_offsets, d_labels, d_label_starts, d_label_multi_starts, d_label_point_offsets, d_label_points,
                label_seed_count, num_labels, universal_label_id, d_filtered_dist, d_unfiltered_dist,
                d_build_label_checks, d_build_label_rejects, d_filtered_cands, d_unfiltered_cands, d_merged_cands,
                d_filtered_seed_count, d_reserved_cands, d_unfiltered_reserved, d_build_universal);
            CUDA_CHECK_FILTERED(cudaGetLastError());
            CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
            if (run_work_label_prune)
            {
                filtered_final_prune_to_compact_from_work_kernel<DataT><<<num_points, 1>>>(
                    d_data_build, dim, d_graph_work_build, d_graph_dists_build, d_degree_work_build,
                    d_graph_cur_build, d_degree_cur_build, num_points, R, gpu_work_C, 1.2f, work_prune_cap, d_offsets,
                    d_labels, universal_label_id, d_build_label_checks, d_build_label_rejects, d_prune_rejects,
                    d_label_occlusion_blocked, d_geometry_occluded, d_candidate_rejected_by_label, d_refill_count,
                    d_prune_degree_sum, d_build_universal);
            }
            else
            {
                filtered_copy_first_from_work_kernel<<<grid, block>>>(num_points, R, gpu_work_C, d_graph_work_build,
                                                                      d_graph_dists_build, d_degree_work_build,
                                                                      d_graph_cur_build, d_degree_cur_build);
            }
            CUDA_CHECK_FILTERED(cudaGetLastError());
            CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
        }

        CUDA_CHECK_FILTERED(cudaMemcpy(h_graph, d_graph_cur_build, (size_t)num_points * R * sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(h_degree, d_degree_cur_build, (size_t)num_points * sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.filtered_search_distance_count, d_filtered_dist,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.unfiltered_search_distance_count, d_unfiltered_dist,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&pre_label_checks, d_build_label_checks,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&pre_label_rejects, d_build_label_rejects,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&pre_universal_pass, d_build_universal,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.filtered_candidate_count, d_filtered_cands,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.unfiltered_candidate_count, d_unfiltered_cands,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.merged_candidate_count, d_merged_cands,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.filtered_seed_count, d_filtered_seed_count,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.filtered_reserved_count, d_reserved_cands,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.unfiltered_reserved_count, d_unfiltered_reserved,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.filtered_prune_reject_count, d_prune_rejects,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.label_occlusion_blocked_count,
                                       d_label_occlusion_blocked, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.geometry_occluded_count, d_geometry_occluded,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.candidate_rejected_by_label_count,
                                       d_candidate_rejected_by_label, sizeof(unsigned long long),
                                       cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.refill_count, d_refill_count,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        unsigned long long prune_degree_sum = 0;
        CUDA_CHECK_FILTERED(cudaMemcpy(&prune_degree_sum, d_prune_degree_sum, sizeof(unsigned long long),
                                       cudaMemcpyDeviceToHost));
        g_last_filtered_stats.prune_output_degree_avg =
            (run_work_label_prune && num_points) ? (double)prune_degree_sum / (double)num_points : 0.0;
        g_last_filtered_stats.filtered_seed_source = inverted_seeds_enabled ? 2u : 1u;

        cudaFree(d_data_build);
        cudaFree(d_graph_cur_build);
        cudaFree(d_graph_work_build);
        cudaFree(d_degree_cur_build);
        cudaFree(d_degree_work_build);
        cudaFree(d_graph_dists_build);
        cudaFree(d_label_multi_starts);
        cudaFree(d_label_point_offsets);
        cudaFree(d_label_points);
        cudaFree(d_filtered_dist);
        cudaFree(d_unfiltered_dist);
        cudaFree(d_build_label_checks);
        cudaFree(d_build_label_rejects);
        cudaFree(d_build_universal);
        cudaFree(d_filtered_cands);
        cudaFree(d_unfiltered_cands);
        cudaFree(d_merged_cands);
        cudaFree(d_filtered_seed_count);
        cudaFree(d_reserved_cands);
        cudaFree(d_unfiltered_reserved);
        cudaFree(d_prune_rejects);
        cudaFree(d_label_occlusion_blocked);
        cudaFree(d_geometry_occluded);
        cudaFree(d_candidate_rejected_by_label);
        cudaFree(d_refill_count);
        cudaFree(d_prune_degree_sum);
    }
    else if (run_canonical_v2 && run_refine_vnew2 && canonical_device_handoff)
    {
        used_device_handoff = true;
        const uint32_t random_seed = 42;
        const uint32_t random_degree = std::min(16u, R);
        CUDA_CHECK_FILTERED(cudaMalloc(&d_handoff_graph, (size_t)num_points * R * sizeof(uint32_t)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_handoff_degree, (size_t)num_points * sizeof(uint32_t)));
        filtered_deterministic_random_graph_kernel<<<(num_points + 255) / 256, 256>>>(
            d_handoff_graph, d_handoff_degree, num_points, R, random_degree, random_seed);
        CUDA_CHECK_FILTERED(cudaGetLastError());
        CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
        printf("[gpu_vamana_filtered] backbone_handoff=deterministic_random stride_R=%u initial_degree=%u seed=%u\n",
               R, random_degree, random_seed);
    }
    else if constexpr (std::is_same<DataT, uint8_t>::value)
    {
        ret = gpu_vamana_vnew2_build(h_data, num_points, dim, R, L, gpu_backbone_C, STEPS, h_graph, h_degree);
    }
    else
    {
        ret = gpu_vamana_vnew2_build_float(h_data, num_points, dim, R, L, gpu_backbone_C, STEPS, h_graph, h_degree);
    }
    g_last_filtered_stats.graph_build_seconds = now_sec_filtered() - graph_t0;
    if (ret != 0)
    {
        cudaFree(d_offsets);
        cudaFree(d_labels);
        cudaFree(d_label_starts);
        fprintf(stderr, "[gpu_vamana_filtered] graph build failed ret=%d\n", ret);
        return ret;
    }
    const char *backbone_output = getenv("DISKANN_GPU_FILTERED_BACKBONE_OUTPUT");
    if (!used_device_handoff && (!backbone_input || backbone_input[0] == '\0') &&
        backbone_output && backbone_output[0] != '\0')
    {
        FILE *fp = fopen(backbone_output, "wb");
        const uint32_t header[3] = {0x46424231u, num_points, R};
        if (!fp || fwrite(header, sizeof(uint32_t), 3, fp) != 3 ||
            fwrite(h_graph, sizeof(uint32_t), (size_t)num_points * R, fp) != (size_t)num_points * R ||
            fwrite(h_degree, sizeof(uint32_t), num_points, fp) != num_points)
        {
            fprintf(stderr, "[gpu_vamana_filtered] failed to write frozen diagnostic backbone: %s\n",
                    backbone_output);
            if (fp)
                fclose(fp);
            cudaFree(d_offsets);
            cudaFree(d_labels);
            cudaFree(d_label_starts);
            return -1;
        }
        fclose(fp);
        printf("[gpu_vamana_filtered] wrote frozen diagnostic backbone to %s\n", backbone_output);
    }
    const char *trace_env = getenv("DISKANN_GPU_FILTERED_TRACE_MINV1");
    if (!used_device_handoff && trace_env && atoi(trace_env) > 0)
    {
        const char *trace_path_env = getenv("DISKANN_GPU_FILTERED_TRACE_OUT");
        const char *trace_path = trace_path_env ? trace_path_env : "/tmp/diskann_gpu_filtered_trace_minv1.csv";
        write_trace_minv1_diagnostic(trace_path, h_graph, h_degree, num_points, R, point_label_offsets, point_labels,
                                     universal_label_id);
    }

    if (run_canonical_v2 && run_refine_vnew2)
    {
        const int canonical_ret = canonical_round0_from_host_backbone(
            h_data, num_points, dim, R, gpu_work_C, L, filtered_L, STEPS, build_alpha,
            point_label_offsets, point_labels,
            total_label_count, label_to_start_id, num_labels, universal_label_id, global_start_id, h_graph, h_degree,
            d_handoff_data, d_handoff_graph, d_handoff_degree);
        if (canonical_ret != 0)
        {
            cudaFree(d_offsets);
            cudaFree(d_labels);
            cudaFree(d_label_starts);
            return canonical_ret;
        }
        run_refine_vnew2 = false;
        run_label_sync = false;
        const char *published_degree_env = getenv("DISKANN_GPU_FILTERED_PUBLISH_DEGREE");
        if (published_degree_env && atoi(published_degree_env) > 0)
            R = min(R, (uint32_t)atoi(published_degree_env));
    }

    if (run_refine_vnew2)
    {
        DataT *d_data_refine = nullptr;
        uint32_t *d_graph_cur_refine = nullptr;
        uint32_t *d_graph_work_refine = nullptr;
        uint32_t *d_degree_cur_refine = nullptr;
        uint32_t *d_degree_work_refine = nullptr;
        float *d_graph_dists_refine = nullptr;
        uint8_t *d_graph_sources_refine = nullptr;
        uint8_t *d_active_rows = nullptr;
        uint8_t *d_deficient_label_flags = nullptr;
        uint32_t *d_active_ids = nullptr;
        uint32_t *d_label_multi_starts = nullptr;
        uint8_t *d_target_labels = nullptr;
        uint32_t *d_label_point_offsets = nullptr;
        uint32_t *d_label_points = nullptr;
        uint32_t *d_query_anchor_offsets = nullptr;
        uint32_t *d_query_anchor_ids = nullptr;
        unsigned long long *d_active_count = nullptr;
        unsigned long long *d_common_before_sum = nullptr;
        unsigned int *d_low_common_before = nullptr;
        unsigned long long *d_refine_label_checks = nullptr;
        unsigned long long *d_refine_label_rejects = nullptr;
        unsigned long long *d_refine_universal = nullptr;
        unsigned long long *d_filtered_dist = nullptr;
        unsigned long long *d_bridge_dist = nullptr;
        unsigned long long *d_filtered_cands = nullptr;
        unsigned long long *d_filtered_visited_count = nullptr;
        unsigned long long *d_original_neighbors = nullptr;
        unsigned long long *d_filtered_reserved = nullptr;
        unsigned long long *d_filtered_path_reserved = nullptr;
        unsigned long long *d_filtered_visited_reserved = nullptr;
        unsigned long long *d_filtered_top_reserved = nullptr;
        unsigned long long *d_bridge_reserved = nullptr;
        unsigned long long *d_filtered_seed_count = nullptr;
        uint32_t *d_reverse_ids = nullptr;
        uint32_t *d_reverse_counts = nullptr;
        uint8_t *d_reverse_touched = nullptr;
        unsigned long long *d_reverse_edges = nullptr;
        unsigned long long *d_reverse_kept = nullptr;
        unsigned long long *d_reverse_pruned_rows = nullptr;
        unsigned long long *d_prune_rejects = nullptr;
        unsigned long long *d_label_occlusion_blocked = nullptr;
        unsigned long long *d_geometry_occluded = nullptr;
        unsigned long long *d_candidate_rejected_by_label = nullptr;
        unsigned long long *d_refill_count = nullptr;
        unsigned long long *d_prune_degree_sum = nullptr;
        unsigned long long *d_source_selected_counts = nullptr;
        unsigned long long *d_source_refill_counts = nullptr;

        uint32_t block = 128;
        uint32_t grid = (num_points + block - 1) / block;
        uint32_t label_sync_iters = filtered_mode_full ? 4u : 1u;
        if (!run_label_sync)
            label_sync_iters = 1;



        uint32_t build_trace_row = INVALID_ID;
        const char *build_trace_row_env = getenv("DISKANN_GPU_FILTERED_BUILD_TRACE_ROW");
        if (build_trace_row_env && build_trace_row_env[0] != '\0')
            build_trace_row = (uint32_t)strtoul(build_trace_row_env, nullptr, 10);
        uint32_t common_threshold = 20;
        const char *thr_env = run_label_sync ? getenv("DISKANN_GPU_FILTERED_COMMON_THRESHOLD")
                                             : getenv("DISKANN_GPU_FILTERED_REFINE_COMMON_THRESHOLD");
        if (!thr_env)
            thr_env = getenv("DISKANN_GPU_FILTERED_REFINE_COMMON_THRESHOLD");
        if (thr_env && atoi(thr_env) > 0)
            common_threshold = (uint32_t)atoi(thr_env);
        if (filtered_mode_full)
            common_threshold = R + 1;
        uint32_t per_label_candidate_mode = 1;
        uint32_t per_label_candidate_keep = 8;
        uint32_t backbone_admission_mode = 1;
        uint32_t backbone_per_label_quota = 0;
        printf("[gpu_vamana_filtered_policy] candidate_mode=%s per_label_keep=%u backbone_admission=%s "
               "backbone_per_label_quota=%u balance_per_label=%u\n",
               per_label_candidate_mode ? "per-label" : "union-label", per_label_candidate_keep,
               backbone_admission_mode == 0 ? "all" :
               (backbone_admission_mode == 1 ? "compatible-only" : "per-label-quota"),
               backbone_per_label_quota, balance_per_label);
        uint32_t active_permille = 1000;
        const char *ap_env = getenv("DISKANN_GPU_FILTERED_ACTIVE_PERMILLE");
        if (ap_env && atoi(ap_env) > 0)
            active_permille = (uint32_t)atoi(ap_env);
        if (active_permille > 1000)
            active_permille = 1000;
        uint32_t path_active_permille = 0;
        const char *path_ap_env = getenv("DISKANN_GPU_FILTERED_PATH_ACTIVE_PERMILLE");
        if (path_ap_env && atoi(path_ap_env) > 0)
            path_active_permille = (uint32_t)atoi(path_ap_env);
        if (path_active_permille > 1000)
            path_active_permille = 1000;
        uint32_t filtered_reserve = 8;
        const char *reserve_env = run_label_sync ? getenv("DISKANN_GPU_FILTERED_FILTERED_RESERVE")
                                                 : getenv("DISKANN_GPU_FILTERED_RESERVE");
        if (!reserve_env)
            reserve_env = getenv("DISKANN_GPU_FILTERED_RESERVE");
        if (reserve_env && atoi(reserve_env) >= 0)
            filtered_reserve = (uint32_t)atoi(reserve_env);
        if (filtered_reserve > R)
            filtered_reserve = R;
        uint32_t bridge_reserve = 24;
        const char *bridge_env = run_label_sync ? getenv("DISKANN_GPU_FILTERED_BRIDGE_RESERVE")
                                                : getenv("DISKANN_GPU_UNFILTERED_BRIDGE_RESERVE");
        if (!bridge_env)
            bridge_env = getenv("DISKANN_GPU_UNFILTERED_BRIDGE_RESERVE");
        if (bridge_env && atoi(bridge_env) > 0)
            bridge_reserve = (uint32_t)atoi(bridge_env);
        if (bridge_reserve > R)
            bridge_reserve = R;
        uint32_t visited_reserve = 0;
        const char *visited_env = getenv("DISKANN_GPU_FILTERED_VISITED_RESERVE");
        if (visited_env && atoi(visited_env) >= 0)
            visited_reserve = (uint32_t)atoi(visited_env);
        if (visited_reserve > 32)
            visited_reserve = 32;
        uint32_t visited_select_mode = 0;
        const char *visited_select_env = getenv("DISKANN_GPU_FILTERED_VISITED_SELECT");
        if (visited_select_env && strcmp(visited_select_env, "spread") == 0)
            visited_select_mode = 1;
        uint32_t visited_low_common_threshold = 0;
        const char *visited_low_env = getenv("DISKANN_GPU_FILTERED_VISITED_LOW_COMMON_THRESHOLD");
        if (visited_low_env && atoi(visited_low_env) > 0)
            visited_low_common_threshold = (uint32_t)atoi(visited_low_env);
        if (visited_low_common_threshold > R)
            visited_low_common_threshold = R;
        uint32_t requested_filtered_reserve = filtered_reserve;
        uint32_t requested_bridge_reserve = bridge_reserve;
        uint32_t requested_visited_reserve = visited_reserve;
        uint32_t target_visited_reserve = 0;
        const char *target_visited_env = getenv("DISKANN_GPU_FILTERED_TARGET_VISITED_RESERVE");
        if (target_visited_env && atoi(target_visited_env) > 0)
            target_visited_reserve = (uint32_t)atoi(target_visited_env);
        if (target_visited_reserve > 32)
            target_visited_reserve = 32;
        uint32_t path_reserve = 0;
        const char *path_reserve_env = getenv("DISKANN_GPU_FILTERED_PATH_RESERVE");
        if (path_reserve_env && atoi(path_reserve_env) > 0)
            path_reserve = (uint32_t)atoi(path_reserve_env);
        if (path_reserve > 16)
            path_reserve = 16;
        uint32_t target_active_labels = 0;
        const char *target_active_env = getenv("DISKANN_GPU_FILTERED_TARGET_ACTIVE_LABELS");
        if (target_active_env && atoi(target_active_env) > 0)
            target_active_labels = 1;
        if (filtered_mode_full)
        {
            filtered_reserve = R;
            bridge_reserve = 0;
            visited_reserve = 0;
            path_reserve = 0;
        }
        else if (filtered_mode_active)
        {
            filtered_reserve = std::min<uint32_t>(R, 32);
            bridge_reserve = 0;
        }
        if (bridge_reserve > gpu_work_C)
            bridge_reserve = gpu_work_C;
        if (filtered_reserve + bridge_reserve + visited_reserve > gpu_work_C)
        {
            uint32_t remaining = gpu_work_C;
            uint32_t old_bridge = bridge_reserve;
            bridge_reserve = std::min(bridge_reserve, remaining);
            remaining -= bridge_reserve;
            uint32_t old_filtered = filtered_reserve;
            filtered_reserve = std::min(filtered_reserve, remaining);
            remaining -= filtered_reserve;
            uint32_t old_visited = visited_reserve;
            visited_reserve = std::min(visited_reserve, remaining);
            printf("[gpu_vamana_filtered] clamp reserves for gpu_work_C=%u: filtered %u->%u bridge %u->%u visited %u->%u\n",
                   gpu_work_C, old_filtered, filtered_reserve, old_bridge, bridge_reserve, old_visited,
                   visited_reserve);
        }
        uint32_t reverse_repair = 1;
        uint32_t reverse_cap = 64;
        const char *reverse_cap_env = getenv("DISKANN_GPU_FILTERED_REVERSE_CAP");
        if (reverse_cap_env && atoi(reverse_cap_env) > 0)
            reverse_cap = (uint32_t)atoi(reverse_cap_env);
        if (reverse_cap > 128)
            reverse_cap = 128;
        if (filtered_mode_full)
            reverse_cap = std::min<uint32_t>(R, 128);


        uint32_t reverse_replacement_mode =
            (filtered_mode_full && num_points <= 0x00ffffffu) ? 2u : 0u;
        const char *reverse_repl_env = getenv("DISKANN_GPU_FILTERED_REVERSE_REPLACEMENT");
        if (reverse_repl_env && strcmp(reverse_repl_env, "hash") == 0)
            reverse_replacement_mode = 1;
        else if (reverse_repl_env && strcmp(reverse_repl_env, "rank") == 0)
            reverse_replacement_mode = 2;
        if (reverse_replacement_mode == 2 && num_points > 0x00ffffffu)
        {
            printf("[gpu_vamana_filtered] reverse replacement rank requires N<=16777215; fallback to hash for N=%u\n",
                   num_points);
            reverse_replacement_mode = 1;
        }
        uint32_t reverse_touch_active_only = 0;
        const char *reverse_touch_active_env = getenv("DISKANN_GPU_FILTERED_REVERSE_TOUCH_ACTIVE_ONLY");
        if (reverse_touch_active_env && atoi(reverse_touch_active_env) > 0)
            reverse_touch_active_only = 1;
        if (filtered_mode_full)
            reverse_touch_active_only = 0;
        uint32_t reverse_sample_permille = 1000;
        const char *reverse_sample_env = getenv("DISKANN_GPU_FILTERED_REVERSE_SAMPLE_PERMILLE");
        if (reverse_sample_env && atoi(reverse_sample_env) > 0)
            reverse_sample_permille = (uint32_t)atoi(reverse_sample_env);
        if (reverse_sample_permille > 1000)
            reverse_sample_permille = 1000;
        if (filtered_mode_full)
            reverse_sample_permille = 1000;
        uint32_t reverse_inactive_sample_permille = 0;
        const char *reverse_inactive_sample_env = getenv("DISKANN_GPU_FILTERED_REVERSE_INACTIVE_SAMPLE_PERMILLE");
        if (reverse_inactive_sample_env && atoi(reverse_inactive_sample_env) > 0)
            reverse_inactive_sample_permille = (uint32_t)atoi(reverse_inactive_sample_env);
        if (reverse_inactive_sample_permille > 1000)
            reverse_inactive_sample_permille = 1000;
        uint32_t reverse_inactive_target_labels = 0;
        const char *reverse_inactive_target_env = getenv("DISKANN_GPU_FILTERED_REVERSE_INACTIVE_TARGET_LABELS");
        if (reverse_inactive_target_env && atoi(reverse_inactive_target_env) > 0)
            reverse_inactive_target_labels = 1;
        uint32_t filtered_pool_cap = filtered_L < 64 ? filtered_L : 64;
        const char *pool_env = getenv("DISKANN_GPU_FILTERED_POOL_SIZE");
        if (pool_env && atoi(pool_env) > 0)
            filtered_pool_cap = (uint32_t)atoi(pool_env);
        if (filtered_pool_cap > 96)
            filtered_pool_cap = 96;
        uint32_t starts_per_label = 1;
        const char *starts_env = getenv("DISKANN_GPU_FILTERED_STARTS_PER_LABEL");
        if (starts_env && atoi(starts_env) > 0)
            starts_per_label = (uint32_t)atoi(starts_env);
        if (starts_per_label != 1 && starts_per_label != 2 && starts_per_label != 4)
            starts_per_label = 1;
        uint32_t label_anchor_mode = 0;
        const char *anchor_mode_env = getenv("DISKANN_GPU_FILTERED_LABEL_ANCHOR_MODE");
        if (anchor_mode_env && strcmp(anchor_mode_env, "spread") == 0)
            label_anchor_mode = 1;
        uint32_t inverted_seeds_enabled = 0;
        const char *inv_env = getenv("DISKANN_GPU_FILTERED_INVERTED_SEEDS");
        if (inv_env && atoi(inv_env) > 0)
            inverted_seeds_enabled = 1;
        uint32_t label_seed_count = 0;
        const char *seed_env = getenv("DISKANN_GPU_FILTERED_LABEL_SEEDS");
        if (seed_env && atoi(seed_env) >= 0)
            label_seed_count = (uint32_t)atoi(seed_env);
        if (label_seed_count > 64)
            label_seed_count = 64;
        if (filtered_mode_full)
        {
            inverted_seeds_enabled = 1;
            label_seed_count = 64;
        }
        else if (filtered_mode_active)
        {
            inverted_seeds_enabled = 1;
            label_seed_count = 16;
        }
        std::vector<uint32_t> target_label_ids = parse_u32_csv_env("DISKANN_GPU_FILTERED_TARGET_LABEL_IDS");
        uint32_t max_expand_steps = filtered_mode_full ? 32u : 8u;
        const char *expand_env = getenv("DISKANN_GPU_FILTERED_MAX_EXPAND_STEPS");
        if (expand_env && atoi(expand_env) > 0)
            max_expand_steps = (uint32_t)atoi(expand_env);
        uint32_t target_max_expand_steps = 0;
        const char *target_expand_env = getenv("DISKANN_GPU_FILTERED_TARGET_MAX_EXPAND_STEPS");
        if (target_expand_env && atoi(target_expand_env) > 0)
            target_max_expand_steps = (uint32_t)atoi(target_expand_env);
        if (target_max_expand_steps < max_expand_steps)
            target_max_expand_steps = 0;
        uint32_t force_label_start_active = 0;
        const char *force_label_start_env = getenv("DISKANN_GPU_FILTERED_FORCE_LABEL_START_ACTIVE");
        if (force_label_start_env && atoi(force_label_start_env) > 0)
            force_label_start_active = 1;
        uint32_t label_start_max_expand_steps = 0;
        const char *label_start_expand_env = getenv("DISKANN_GPU_FILTERED_LABEL_START_MAX_EXPAND_STEPS");
        if (label_start_expand_env && atoi(label_start_expand_env) > 0)
            label_start_max_expand_steps = (uint32_t)atoi(label_start_expand_env);
        if (label_start_max_expand_steps < max_expand_steps)
            label_start_max_expand_steps = 0;
        uint32_t label_start_filtered_reserve = 0;
        const char *label_start_filtered_env = getenv("DISKANN_GPU_FILTERED_LABEL_START_FILTERED_RESERVE");
        if (label_start_filtered_env && atoi(label_start_filtered_env) > 0)
            label_start_filtered_reserve = (uint32_t)atoi(label_start_filtered_env);
        if (label_start_filtered_reserve > R)
            label_start_filtered_reserve = R;
        uint32_t label_start_visited_reserve = 0;
        const char *label_start_visited_env = getenv("DISKANN_GPU_FILTERED_LABEL_START_VISITED_RESERVE");
        if (label_start_visited_env && atoi(label_start_visited_env) > 0)
            label_start_visited_reserve = (uint32_t)atoi(label_start_visited_env);
        if (label_start_visited_reserve > 32)
            label_start_visited_reserve = 32;
        uint32_t label_start_path_reserve = 0;
        const char *label_start_path_env = getenv("DISKANN_GPU_FILTERED_LABEL_START_PATH_RESERVE");
        if (label_start_path_env && atoi(label_start_path_env) > 0)
            label_start_path_reserve = (uint32_t)atoi(label_start_path_env);
        if (label_start_path_reserve > 16)
            label_start_path_reserve = 16;
        uint32_t label_start_portal_samples = 0;
        const char *label_start_portal_samples_env = getenv("DISKANN_GPU_FILTERED_LABEL_START_PORTAL_SAMPLES");
        if (label_start_portal_samples_env && atoi(label_start_portal_samples_env) > 0)
            label_start_portal_samples = (uint32_t)atoi(label_start_portal_samples_env);
        if (label_start_portal_samples > 64)
            label_start_portal_samples = 64;
        uint32_t label_start_portal_keep = 0;
        const char *label_start_portal_keep_env = getenv("DISKANN_GPU_FILTERED_LABEL_START_PORTAL_KEEP");
        if (label_start_portal_keep_env && atoi(label_start_portal_keep_env) > 0)
            label_start_portal_keep = (uint32_t)atoi(label_start_portal_keep_env);
        if (label_start_portal_keep > 16)
            label_start_portal_keep = 16;
        uint32_t label_start_portal_mode = 1;
        const char *label_start_portal_mode_env = getenv("DISKANN_GPU_FILTERED_LABEL_START_PORTAL_MODE");
        if (label_start_portal_mode_env && strcmp(label_start_portal_mode_env, "closest") == 0)
            label_start_portal_mode = 0;
        else if (label_start_portal_mode_env && strcmp(label_start_portal_mode_env, "far") == 0)
            label_start_portal_mode = 1;
        uint32_t query_anchor_keep = 0;
        const char *query_anchor_keep_env = getenv("DISKANN_GPU_FILTERED_QUERY_ANCHOR_KEEP");
        if (query_anchor_keep_env && atoi(query_anchor_keep_env) > 0)
            query_anchor_keep = (uint32_t)atoi(query_anchor_keep_env);
        if (query_anchor_keep > 16)
            query_anchor_keep = 16;
        uint32_t label_local_shortcut_samples = 0;
        const char *label_local_shortcut_samples_env = getenv("DISKANN_GPU_FILTERED_LABEL_LOCAL_SHORTCUT_SAMPLES");
        if (label_local_shortcut_samples_env && atoi(label_local_shortcut_samples_env) > 0)
            label_local_shortcut_samples = (uint32_t)atoi(label_local_shortcut_samples_env);
        if (label_local_shortcut_samples > 64)
            label_local_shortcut_samples = 64;
        uint32_t label_local_shortcut_keep = 0;
        const char *label_local_shortcut_keep_env = getenv("DISKANN_GPU_FILTERED_LABEL_LOCAL_SHORTCUT_KEEP");
        if (label_local_shortcut_keep_env && atoi(label_local_shortcut_keep_env) > 0)
            label_local_shortcut_keep = (uint32_t)atoi(label_local_shortcut_keep_env);
        if (label_local_shortcut_keep > 8)
            label_local_shortcut_keep = 8;
        uint32_t label_local_shortcut_mode = 0;
        const char *label_local_shortcut_mode_env = getenv("DISKANN_GPU_FILTERED_LABEL_LOCAL_SHORTCUT_MODE");
        if (label_local_shortcut_mode_env && strcmp(label_local_shortcut_mode_env, "far") == 0)
            label_local_shortcut_mode = 1;
        else if (label_local_shortcut_mode_env && strcmp(label_local_shortcut_mode_env, "mixed") == 0)
            label_local_shortcut_mode = 2;
        else if (label_local_shortcut_mode_env && strcmp(label_local_shortcut_mode_env, "agreement") == 0)
            label_local_shortcut_mode = 3;
        uint32_t label_local_shortcut_sampler = 0;
        const char *label_local_shortcut_sampler_env = getenv("DISKANN_GPU_FILTERED_LABEL_LOCAL_SHORTCUT_SAMPLER");
        if (label_local_shortcut_sampler_env && strcmp(label_local_shortcut_sampler_env, "spread") == 0)
            label_local_shortcut_sampler = 1;
        uint32_t label_path_shortcut_fanout = 0;
        const char *label_path_shortcut_fanout_env = getenv("DISKANN_GPU_FILTERED_LABEL_PATH_SHORTCUT_FANOUT");
        if (label_path_shortcut_fanout_env && atoi(label_path_shortcut_fanout_env) > 0)
            label_path_shortcut_fanout = (uint32_t)atoi(label_path_shortcut_fanout_env);
        if (label_path_shortcut_fanout > 32)
            label_path_shortcut_fanout = 32;
        uint32_t label_path_shortcut_keep = 0;
        const char *label_path_shortcut_keep_env = getenv("DISKANN_GPU_FILTERED_LABEL_PATH_SHORTCUT_KEEP");
        if (label_path_shortcut_keep_env && atoi(label_path_shortcut_keep_env) > 0)
            label_path_shortcut_keep = (uint32_t)atoi(label_path_shortcut_keep_env);
        if (label_path_shortcut_keep > 8)
            label_path_shortcut_keep = 8;
        uint32_t expanded_path_shortcut_fanout = 0;
        const char *expanded_path_shortcut_fanout_env = getenv("DISKANN_GPU_FILTERED_EXPANDED_PATH_SHORTCUT_FANOUT");
        if (expanded_path_shortcut_fanout_env && atoi(expanded_path_shortcut_fanout_env) > 0)
            expanded_path_shortcut_fanout = (uint32_t)atoi(expanded_path_shortcut_fanout_env);
        if (expanded_path_shortcut_fanout > 32)
            expanded_path_shortcut_fanout = 32;
        uint32_t expanded_path_shortcut_keep = 0;
        const char *expanded_path_shortcut_keep_env = getenv("DISKANN_GPU_FILTERED_EXPANDED_PATH_SHORTCUT_KEEP");
        if (expanded_path_shortcut_keep_env && atoi(expanded_path_shortcut_keep_env) > 0)
            expanded_path_shortcut_keep = (uint32_t)atoi(expanded_path_shortcut_keep_env);
        if (expanded_path_shortcut_keep > 8)
            expanded_path_shortcut_keep = 8;
        uint32_t twohop_path_shortcut_fanout = 0;
        const char *twohop_path_shortcut_fanout_env = getenv("DISKANN_GPU_FILTERED_TWOHOP_PATH_SHORTCUT_FANOUT");
        if (twohop_path_shortcut_fanout_env && atoi(twohop_path_shortcut_fanout_env) > 0)
            twohop_path_shortcut_fanout = (uint32_t)atoi(twohop_path_shortcut_fanout_env);
        if (twohop_path_shortcut_fanout > 8)
            twohop_path_shortcut_fanout = 8;
        uint32_t twohop_path_shortcut_keep = 0;
        const char *twohop_path_shortcut_keep_env = getenv("DISKANN_GPU_FILTERED_TWOHOP_PATH_SHORTCUT_KEEP");
        if (twohop_path_shortcut_keep_env && atoi(twohop_path_shortcut_keep_env) > 0)
            twohop_path_shortcut_keep = (uint32_t)atoi(twohop_path_shortcut_keep_env);
        if (twohop_path_shortcut_keep > 8)
            twohop_path_shortcut_keep = 8;
        uint32_t path_shortcut_select_mode = 0;
        const char *path_shortcut_select_env = getenv("DISKANN_GPU_FILTERED_PATH_SHORTCUT_SELECT");
        if (path_shortcut_select_env && strcmp(path_shortcut_select_env, "consensus") == 0)
            path_shortcut_select_mode = 1;
        uint32_t label_local_shortcut_target_only = 0;
        const char *label_local_shortcut_target_env = getenv("DISKANN_GPU_FILTERED_LABEL_LOCAL_SHORTCUT_TARGET_ONLY");
        if (label_local_shortcut_target_env && atoi(label_local_shortcut_target_env) > 0)
            label_local_shortcut_target_only = 1;
        uint32_t label_local_shortcut_as_frontier = 0;
        const char *label_local_shortcut_frontier_env = getenv("DISKANN_GPU_FILTERED_LABEL_LOCAL_SHORTCUT_AS_FRONTIER");
        if (label_local_shortcut_frontier_env && atoi(label_local_shortcut_frontier_env) > 0)
            label_local_shortcut_as_frontier = 1;
        uint32_t target_shortcut_samples = 0;
        const char *target_shortcut_samples_env = getenv("DISKANN_GPU_FILTERED_TARGET_SHORTCUT_SAMPLES");
        if (target_shortcut_samples_env && atoi(target_shortcut_samples_env) > 0)
            target_shortcut_samples = (uint32_t)atoi(target_shortcut_samples_env);
        if (target_shortcut_samples > 32)
            target_shortcut_samples = 32;
        uint32_t target_shortcut_keep = 0;
        const char *target_shortcut_keep_env = getenv("DISKANN_GPU_FILTERED_TARGET_SHORTCUT_KEEP");
        if (target_shortcut_keep_env && atoi(target_shortcut_keep_env) > 0)
            target_shortcut_keep = (uint32_t)atoi(target_shortcut_keep_env);
        if (target_shortcut_keep > 8)
            target_shortcut_keep = 8;
        uint32_t work_prune_cap = 256;
        const char *cap_env = getenv("DISKANN_GPU_FILTERED_WORK_PRUNE_CAP");
        if (cap_env && atoi(cap_env) > 0)
            work_prune_cap = (uint32_t)atoi(cap_env);
        if (work_prune_cap > 256)
            work_prune_cap = 256;
        uint32_t local_c_cap = 256;
        const char *local_c_env = getenv("DISKANN_GPU_FILTERED_LOCAL_C");
        if (local_c_env && atoi(local_c_env) > 0)
            local_c_cap = (uint32_t)atoi(local_c_env);
        if (local_c_cap <= 128)
            local_c_cap = 128;
        else
            local_c_cap = 256;
        if (work_prune_cap > local_c_cap)
            work_prune_cap = local_c_cap;
        uint32_t prune_bridge_protect = 0;
        const char *protect_env = getenv("DISKANN_GPU_FILTERED_PRUNE_BRIDGE_PROTECT");
        if (protect_env && atoi(protect_env) > 0)
            prune_bridge_protect = (uint32_t)atoi(protect_env);
        if (prune_bridge_protect > R)
            prune_bridge_protect = R;
        uint32_t prune_noncommon_bridge_protect = 0;
        const char *noncommon_protect_env = getenv("DISKANN_GPU_FILTERED_PRUNE_NONCOMMON_BRIDGE_PROTECT");
        if (noncommon_protect_env && atoi(noncommon_protect_env) > 0)
            prune_noncommon_bridge_protect = (uint32_t)atoi(noncommon_protect_env);
        if (prune_noncommon_bridge_protect > R)
            prune_noncommon_bridge_protect = R;
        uint32_t refill_target_degree = R;
        const char *refill_target_env = getenv("DISKANN_GPU_FILTERED_REFILL_TARGET_DEGREE");
        if (refill_target_env && atoi(refill_target_env) > 0)
            refill_target_degree = (uint32_t)atoi(refill_target_env);
        if (refill_target_degree > R)
            refill_target_degree = R;
        uint32_t source_priority_mode = 0;
        const char *source_priority_env = getenv("DISKANN_GPU_FILTERED_SOURCE_PRIORITY");
        if (source_priority_env && atoi(source_priority_env) > 0)
            source_priority_mode = (uint32_t)atoi(source_priority_env);
        if (source_priority_mode > 6)
            source_priority_mode = 6;
        uint32_t source_merge_mode = 0;
        const char *source_merge_env = getenv("DISKANN_GPU_FILTERED_SOURCE_MERGE");
        if (source_merge_env && atoi(source_merge_env) > 0)
            source_merge_mode = 1;
        uint32_t source_diagnostics = 0;
        const char *source_diag_env = getenv("DISKANN_GPU_FILTERED_SOURCE_DIAGNOSTICS");
        if (source_diag_env && atoi(source_diag_env) > 0)
            source_diagnostics = 1;
        uint32_t filtered_top_protect = 0;
        const char *top_protect_env = getenv("DISKANN_GPU_FILTERED_TOP_PROTECT");
        if (top_protect_env && atoi(top_protect_env) > 0)
            filtered_top_protect = (uint32_t)atoi(top_protect_env);
        if (filtered_top_protect > R)
            filtered_top_protect = R;
        uint32_t path_protect = 0;
        const char *path_protect_env = getenv("DISKANN_GPU_FILTERED_PATH_PROTECT");
        if (path_protect_env && atoi(path_protect_env) > 0)
            path_protect = (uint32_t)atoi(path_protect_env);
        if (path_protect > R)
            path_protect = R;
        uint32_t consensus_protect = 0;
        const char *consensus_protect_env = getenv("DISKANN_GPU_FILTERED_CONSENSUS_PROTECT");
        if (consensus_protect_env && atoi(consensus_protect_env) > 0)
            consensus_protect = (uint32_t)atoi(consensus_protect_env);
        if (consensus_protect > R)
            consensus_protect = R;
        uint32_t consensus_require_filtered_top = 0;
        const char *consensus_require_top_env = getenv("DISKANN_GPU_FILTERED_CONSENSUS_REQUIRE_FILTERED_TOP");
        if (consensus_require_top_env && atoi(consensus_require_top_env) > 0)
            consensus_require_filtered_top = 1;
        uint32_t position_top_protect = 0;
        const char *position_top_env = getenv("DISKANN_GPU_FILTERED_POSITION_TOP_PROTECT");
        if (position_top_env && atoi(position_top_env) > 0)
            position_top_protect = 1;
        uint32_t position_reverse_protect = 0;
        const char *position_reverse_env = getenv("DISKANN_GPU_FILTERED_POSITION_REVERSE_PROTECT");
        if (position_reverse_env && atoi(position_reverse_env) > 0)
            position_reverse_protect = (uint32_t)atoi(position_reverse_env);
        if (position_reverse_protect > R)
            position_reverse_protect = R;
        uint32_t common_degree_cap = 0;
        const char *common_cap_env = getenv("DISKANN_GPU_FILTERED_COMMON_DEGREE_CAP");
        if (common_cap_env && atoi(common_cap_env) > 0)
            common_degree_cap = (uint32_t)atoi(common_cap_env);
        if (common_degree_cap > R)
            common_degree_cap = R;
        uint32_t target_common_degree_cap = 0;
        const char *target_common_cap_env = getenv("DISKANN_GPU_FILTERED_TARGET_COMMON_DEGREE_CAP");
        if (target_common_cap_env && atoi(target_common_cap_env) > 0)
            target_common_degree_cap = (uint32_t)atoi(target_common_cap_env);
        if (target_common_degree_cap > R)
            target_common_degree_cap = R;
        float filtered_prune_alpha = build_alpha;
        const char *filtered_alpha_env = getenv("DISKANN_GPU_FILTERED_PRUNE_ALPHA");
        if (filtered_alpha_env && atof(filtered_alpha_env) > 0.0)
            filtered_prune_alpha = (float)atof(filtered_alpha_env);
        if (filtered_prune_alpha < 1.0f)
            filtered_prune_alpha = 1.0f;
        uint32_t source_aware_alpha = 0;
        const char *source_alpha_env = getenv("DISKANN_GPU_FILTERED_SOURCE_AWARE_ALPHA");
        if (source_alpha_env && atoi(source_alpha_env) > 0)
            source_aware_alpha = 1;
        uint32_t reverse_apply_common_sources = 0;
        const char *reverse_common_src_env = getenv("DISKANN_GPU_FILTERED_REVERSE_APPLY_COMMON_SOURCES");
        if (reverse_common_src_env && atoi(reverse_common_src_env) > 0)
            reverse_apply_common_sources = 1;
        uint32_t reverse_dup_source_merge = 0;
        const char *reverse_dup_merge_env = getenv("DISKANN_GPU_FILTERED_REVERSE_DUP_SOURCE_MERGE");
        if (reverse_dup_merge_env && atoi(reverse_dup_merge_env) > 0)
            reverse_dup_source_merge = 1;
        const uint32_t source_tracking_enabled =
            (source_priority_mode != 0 || source_diagnostics != 0 ||
             source_merge_mode != 0 ||
             (filtered_top_protect != 0 && position_top_protect == 0) || path_protect != 0 ||
             consensus_protect != 0 ||
             reverse_apply_common_sources != 0 || reverse_dup_source_merge != 0) ? 1u : 0u;
        uint32_t active_only_work_requested = 0;
        const char *active_only_env = getenv("DISKANN_GPU_FILTERED_ACTIVE_ONLY_WORK");
        if (active_only_env && atoi(active_only_env) > 0)
            active_only_work_requested = 1;
        if (filtered_mode_full)
            active_only_work_requested = 0;
        else if (filtered_mode_active)
            active_only_work_requested = 1;
        uint32_t active_only_work_effective = active_only_work_requested;
        uint32_t active_reverse_chunk_rows = 1000000;
        const char *active_reverse_chunk_env = getenv("DISKANN_GPU_FILTERED_ACTIVE_REVERSE_CHUNK_ROWS");
        if (active_reverse_chunk_env && atoi(active_reverse_chunk_env) > 0)
            active_reverse_chunk_rows = (uint32_t)atoi(active_reverse_chunk_env);
        if (active_reverse_chunk_rows < 1024)
            active_reverse_chunk_rows = 1024;
        uint32_t active_work_chunk_rows = 0;
        const char *active_work_chunk_env = getenv("DISKANN_GPU_FILTERED_ACTIVE_WORK_CHUNK_ROWS");
        if (active_work_chunk_env && atoi(active_work_chunk_env) > 0)
            active_work_chunk_rows = (uint32_t)atoi(active_work_chunk_env);
        if (active_work_chunk_rows > 0 && active_work_chunk_rows < 1024)
            active_work_chunk_rows = 1024;




        if (filtered_mode_full && active_work_chunk_rows > 0)
            active_only_work_effective = 1;
        uint32_t reverse_id_chunk_rows = 0;
        const char *reverse_id_chunk_env = getenv("DISKANN_GPU_FILTERED_REVERSE_ID_CHUNK_ROWS");
        if (reverse_id_chunk_env && atoi(reverse_id_chunk_env) > 0)
            reverse_id_chunk_rows = (uint32_t)atoi(reverse_id_chunk_env);
        if (reverse_id_chunk_rows > 0 && reverse_id_chunk_rows < 1024)
            reverse_id_chunk_rows = 1024;
        const size_t full_work_graph_bytes = (size_t)num_points * gpu_work_C * sizeof(uint32_t);
        const size_t full_work_dist_bytes = (size_t)num_points * gpu_work_C * sizeof(float);
        const size_t full_work_source_bytes =
            source_tracking_enabled != 0 ? (size_t)num_points * gpu_work_C * sizeof(uint8_t) : 0;
        const size_t full_reverse_bytes = reverse_repair ? (size_t)num_points * reverse_cap * sizeof(uint32_t) : 0;
        const size_t full_work_total_bytes =
            full_work_graph_bytes + full_work_dist_bytes + full_work_source_bytes + full_reverse_bytes;
        if (active_only_work_requested && !active_only_work_effective)
        {
            printf("[gpu_vamana_filtered] active_only_work fallback full_work_graph_gib=%.3f full_work_dists_gib=%.3f full_work_sources_gib=%.3f full_reverse_ids_gib=%.3f full_total_gib=%.3f.\n",
                   bytes_to_gib_filtered(full_work_graph_bytes), bytes_to_gib_filtered(full_work_dist_bytes),
                   bytes_to_gib_filtered(full_work_source_bytes), bytes_to_gib_filtered(full_reverse_bytes),
                   bytes_to_gib_filtered(full_work_total_bytes));
        }

        printf("[gpu_vamana_filtered] %s common_threshold=%u label_sync_iters=%u active_permille=%u path_active_permille=%u active_only_work_requested=%u active_only_work_effective=%u active_work_chunk_rows=%u reverse_id_chunk_rows=%u filtered_reserve=%u visited_reserve=%u target_visited_reserve=%u path_reserve=%u path_protect=%u consensus_protect=%u consensus_require_filtered_top=%u target_active_labels=%u target_label_count=%zu target_shortcut_samples=%u target_shortcut_keep=%u query_anchor_count=%u query_anchor_keep=%u force_label_start_active=%u label_start_expand=%u label_start_filtered_reserve=%u label_start_visited_reserve=%u label_start_path_reserve=%u label_start_portal_samples=%u label_start_portal_keep=%u label_start_portal_mode=%s label_local_shortcut_samples=%u label_local_shortcut_keep=%u label_local_shortcut_mode=%s label_local_shortcut_sampler=%s label_path_shortcut_fanout=%u label_path_shortcut_keep=%u expanded_path_shortcut_fanout=%u expanded_path_shortcut_keep=%u twohop_path_shortcut_fanout=%u twohop_path_shortcut_keep=%u path_shortcut_select=%s label_local_shortcut_target_only=%u label_local_shortcut_as_frontier=%u visited_select=%s visited_low_common_threshold=%u bridge_reserve=%u prune_bridge_protect=%u prune_noncommon_bridge_protect=%u refill_target_degree=%u source_priority=%u source_merge=%u source_diagnostics=%u filtered_top_protect=%u position_top_protect=%u position_reverse_protect=%u common_degree_cap=%u target_common_degree_cap=%u filtered_prune_alpha=%.3f source_aware_alpha=%u reverse_apply_common_sources=%u reverse_dup_source_merge=%u reverse_repair=%u reverse_cap=%u reverse_replacement=%u reverse_touch_active_only=%u reverse_sample_permille=%u reverse_inactive_sample_permille=%u reverse_inactive_target_labels=%u starts_per_label=%u label_anchor_mode=%u inverted_seeds=%u inverted_seed_limit=%u max_expand_steps=%u target_max_expand_steps=%u work_prune_cap=%u local_C=%u work_C=%u full_work_total_gib=%.3f requested_reserves=(filtered=%u bridge=%u visited=%u)\n",
               stage_name, common_threshold, label_sync_iters, active_permille, path_active_permille,
               active_only_work_requested, active_only_work_effective, active_work_chunk_rows, reverse_id_chunk_rows, filtered_reserve, visited_reserve,
               target_visited_reserve, path_reserve, path_protect, consensus_protect, consensus_require_filtered_top,
               target_active_labels, target_label_ids.size(),
               target_shortcut_samples, target_shortcut_keep, query_anchor_count, query_anchor_keep,
               force_label_start_active, label_start_max_expand_steps,
               label_start_filtered_reserve, label_start_visited_reserve, label_start_path_reserve,
               label_start_portal_samples, label_start_portal_keep,
               label_start_portal_mode == 1 ? "far" : "closest",
               label_local_shortcut_samples, label_local_shortcut_keep,
               label_local_shortcut_mode == 1 ? "far" : (label_local_shortcut_mode == 2 ? "mixed" : (label_local_shortcut_mode == 3 ? "agreement" : "closest")),
               label_local_shortcut_sampler == 1 ? "spread" : "hash",
               label_path_shortcut_fanout, label_path_shortcut_keep,
               expanded_path_shortcut_fanout, expanded_path_shortcut_keep,
               twohop_path_shortcut_fanout, twohop_path_shortcut_keep,
               path_shortcut_select_mode == 1 ? "consensus" : "closest",
               label_local_shortcut_target_only,
               label_local_shortcut_as_frontier,
               visited_select_mode == 1 ? "spread" : "closest",
               visited_low_common_threshold, bridge_reserve, prune_bridge_protect, prune_noncommon_bridge_protect, refill_target_degree,
               source_priority_mode, source_merge_mode, source_diagnostics, filtered_top_protect, position_top_protect, position_reverse_protect,
               common_degree_cap, target_common_degree_cap, filtered_prune_alpha, source_aware_alpha,
               reverse_apply_common_sources, reverse_dup_source_merge,
               reverse_repair, reverse_cap, reverse_replacement_mode,
               reverse_touch_active_only, reverse_sample_permille, reverse_inactive_sample_permille,
               reverse_inactive_target_labels,
               starts_per_label, label_anchor_mode, inverted_seeds_enabled,
               label_seed_count, max_expand_steps, target_max_expand_steps, work_prune_cap, local_c_cap, gpu_work_C,
               bytes_to_gib_filtered(full_work_total_bytes), requested_filtered_reserve, requested_bridge_reserve,
               requested_visited_reserve);

        std::vector<uint32_t> h_label_point_offsets((size_t)std::max(1u, num_labels) + 1, 0);
        for (uint32_t p = 0; p < num_points; ++p)
            for (uint32_t j = point_label_offsets[p]; j < point_label_offsets[p + 1]; ++j)
                if (point_labels[j] < num_labels)
                    ++h_label_point_offsets[(size_t)point_labels[j] + 1];
        for (uint32_t l = 1; l <= num_labels; ++l)
            h_label_point_offsets[l] += h_label_point_offsets[l - 1];
        std::vector<uint32_t> h_label_points(h_label_point_offsets[num_labels], INVALID_ID);
        std::vector<uint32_t> h_label_cursor = h_label_point_offsets;
        for (uint32_t p = 0; p < num_points; ++p)
            for (uint32_t j = point_label_offsets[p]; j < point_label_offsets[p + 1]; ++j)
                if (point_labels[j] < num_labels)
                    h_label_points[h_label_cursor[point_labels[j]]++] = p;
        std::vector<uint32_t> h_label_multi_starts((size_t)std::max(1u, num_labels) * starts_per_label, INVALID_ID);
        std::vector<uint8_t> h_target_labels((size_t)std::max(1u, num_labels), 0);
        for (uint32_t lbl : target_label_ids)
            if (lbl < num_labels)
                h_target_labels[lbl] = 1;
        for (uint32_t l = 0; l < num_labels; ++l)
        {
            uint32_t base_start = label_to_start_id[l];
            h_label_multi_starts[(size_t)l * starts_per_label] = base_start;
            uint32_t pb = h_label_point_offsets[l];
            uint32_t pe = h_label_point_offsets[l + 1];
            uint32_t pc = pe > pb ? pe - pb : 0;
            for (uint32_t s = 1; s < starts_per_label; ++s)
            {
                if (pc == 0 || base_start >= num_points)
                    break;
                uint32_t best = INVALID_ID;
                float best_score = label_anchor_mode == 1 ? -1.0f : FLT_MAX;
                uint32_t stride = pc / 64;
                if (stride == 0)
                    stride = 1;
                for (uint32_t idx = pb + s; idx < pe; idx += stride)
                {
                    uint32_t cand = h_label_points[idx];
                    if (cand == INVALID_ID || cand >= num_points || cand == base_start)
                        continue;
                    bool duplicate_anchor = false;
                    for (uint32_t prev = 0; prev < s; ++prev)
                    {
                        if (h_label_multi_starts[(size_t)l * starts_per_label + prev] == cand)
                        {
                            duplicate_anchor = true;
                            break;
                        }
                    }
                    if (duplicate_anchor)
                        continue;
                    float score = label_anchor_mode == 1 ? FLT_MAX : 0.0f;
                    uint32_t anchor_count = label_anchor_mode == 1 ? s : 1;
                    for (uint32_t prev = 0; prev < anchor_count; ++prev)
                    {
                        uint32_t anchor = h_label_multi_starts[(size_t)l * starts_per_label + prev];
                        if (anchor == INVALID_ID || anchor >= num_points)
                            continue;
                        float d = 0.0f;
                        for (uint32_t dd = 0; dd < dim; ++dd)
                        {
                            float diff = (float)h_data[(size_t)cand * dim + dd] -
                                         (float)h_data[(size_t)anchor * dim + dd];
                            d += diff * diff;
                        }
                        if (label_anchor_mode == 1)
                        {
                            if (d < score)
                                score = d;
                        }
                        else
                        {
                            score = d;
                        }
                    }
                    bool better = label_anchor_mode == 1 ? (score > best_score) : (score < best_score);
                    if (better)
                    {
                        best_score = score;
                        best = cand;
                    }
                }
                h_label_multi_starts[(size_t)l * starts_per_label + s] = best;
            }
        }

        CUDA_CHECK_FILTERED(cudaMalloc(&d_data_refine, (size_t)num_points * dim * sizeof(DataT)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_graph_cur_refine, (size_t)num_points * R * sizeof(uint32_t)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_degree_cur_refine, (size_t)num_points * sizeof(uint32_t)));
        if (!active_only_work_effective)
        {
            CUDA_CHECK_FILTERED(cudaMalloc(&d_graph_work_refine, (size_t)num_points * gpu_work_C * sizeof(uint32_t)));
            CUDA_CHECK_FILTERED(cudaMalloc(&d_degree_work_refine, (size_t)num_points * sizeof(uint32_t)));
            CUDA_CHECK_FILTERED(cudaMalloc(&d_graph_dists_refine, (size_t)num_points * gpu_work_C * sizeof(float)));
            if (source_tracking_enabled != 0)
                CUDA_CHECK_FILTERED(cudaMalloc(&d_graph_sources_refine, (size_t)num_points * gpu_work_C * sizeof(uint8_t)));
        }
        CUDA_CHECK_FILTERED(cudaMalloc(&d_active_rows, (size_t)num_points * sizeof(uint8_t)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_deficient_label_flags,
                                       std::max<size_t>(1, total_label_count) * sizeof(uint8_t)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_label_multi_starts,
                                       std::max((size_t)1, h_label_multi_starts.size()) * sizeof(uint32_t)));
        if (!target_label_ids.empty())
            CUDA_CHECK_FILTERED(cudaMalloc(&d_target_labels, std::max((size_t)1, h_target_labels.size()) * sizeof(uint8_t)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_label_point_offsets, ((size_t)num_labels + 1) * sizeof(uint32_t)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_label_points, std::max((size_t)1, h_label_points.size()) * sizeof(uint32_t)));
        if (query_anchor_offsets && query_anchor_ids && query_anchor_count > 0 && query_anchor_keep > 0)
        {
            CUDA_CHECK_FILTERED(cudaMalloc(&d_query_anchor_offsets, ((size_t)num_labels + 1) * sizeof(uint32_t)));
            CUDA_CHECK_FILTERED(cudaMalloc(&d_query_anchor_ids, (size_t)query_anchor_count * sizeof(uint32_t)));
        }
        CUDA_CHECK_FILTERED(cudaMemcpy(d_data_refine, h_data, (size_t)num_points * dim * sizeof(DataT),
                                       cudaMemcpyHostToDevice));
        CUDA_CHECK_FILTERED(cudaMemcpy(d_graph_cur_refine, h_graph, (size_t)num_points * R * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice));
        CUDA_CHECK_FILTERED(cudaMemcpy(d_degree_cur_refine, h_degree, (size_t)num_points * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice));
        CUDA_CHECK_FILTERED(cudaMemcpy(d_label_multi_starts, h_label_multi_starts.data(),
                                       h_label_multi_starts.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        if (d_target_labels)
            CUDA_CHECK_FILTERED(cudaMemcpy(d_target_labels, h_target_labels.data(),
                                           h_target_labels.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK_FILTERED(cudaMemcpy(d_label_point_offsets, h_label_point_offsets.data(),
                                       ((size_t)num_labels + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
        if (!h_label_points.empty())
            CUDA_CHECK_FILTERED(cudaMemcpy(d_label_points, h_label_points.data(),
                                           h_label_points.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        if (d_query_anchor_offsets && d_query_anchor_ids)
        {
            CUDA_CHECK_FILTERED(cudaMemcpy(d_query_anchor_offsets, query_anchor_offsets,
                                           ((size_t)num_labels + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK_FILTERED(cudaMemcpy(d_query_anchor_ids, query_anchor_ids,
                                           (size_t)query_anchor_count * sizeof(uint32_t), cudaMemcpyHostToDevice));
        }

#define ALLOC_ZERO_U64(ptr)                                                                       \
        CUDA_CHECK_FILTERED(cudaMalloc(&(ptr), sizeof(unsigned long long)));                      \
        CUDA_CHECK_FILTERED(cudaMemset((ptr), 0, sizeof(unsigned long long)))
        ALLOC_ZERO_U64(d_active_count);
        ALLOC_ZERO_U64(d_common_before_sum);
        CUDA_CHECK_FILTERED(cudaMalloc(&d_low_common_before, sizeof(unsigned int)));
        CUDA_CHECK_FILTERED(cudaMemset(d_low_common_before, 0, sizeof(unsigned int)));
        ALLOC_ZERO_U64(d_refine_label_checks);
        ALLOC_ZERO_U64(d_refine_label_rejects);
        ALLOC_ZERO_U64(d_refine_universal);
        ALLOC_ZERO_U64(d_filtered_dist);
        ALLOC_ZERO_U64(d_bridge_dist);
        ALLOC_ZERO_U64(d_filtered_cands);
        ALLOC_ZERO_U64(d_filtered_visited_count);
        ALLOC_ZERO_U64(d_original_neighbors);
        ALLOC_ZERO_U64(d_filtered_reserved);
        ALLOC_ZERO_U64(d_filtered_path_reserved);
        ALLOC_ZERO_U64(d_filtered_visited_reserved);
        ALLOC_ZERO_U64(d_filtered_top_reserved);
        ALLOC_ZERO_U64(d_bridge_reserved);
        ALLOC_ZERO_U64(d_filtered_seed_count);
        ALLOC_ZERO_U64(d_reverse_edges);
        ALLOC_ZERO_U64(d_reverse_kept);
        ALLOC_ZERO_U64(d_reverse_pruned_rows);
        ALLOC_ZERO_U64(d_prune_rejects);
        ALLOC_ZERO_U64(d_label_occlusion_blocked);
        ALLOC_ZERO_U64(d_geometry_occluded);
        ALLOC_ZERO_U64(d_candidate_rejected_by_label);
        ALLOC_ZERO_U64(d_refill_count);
        ALLOC_ZERO_U64(d_prune_degree_sum);
#undef ALLOC_ZERO_U64
        if (source_tracking_enabled != 0)
        {
            CUDA_CHECK_FILTERED(cudaMalloc(&d_source_selected_counts, 8 * sizeof(unsigned long long)));
            CUDA_CHECK_FILTERED(cudaMalloc(&d_source_refill_counts, 8 * sizeof(unsigned long long)));
            CUDA_CHECK_FILTERED(cudaMemset(d_source_selected_counts, 0, 8 * sizeof(unsigned long long)));
            CUDA_CHECK_FILTERED(cudaMemset(d_source_refill_counts, 0, 8 * sizeof(unsigned long long)));
        }

        if (!active_only_work_effective)
        {
            init_refine_graph_dists_kernel<DataT><<<grid, block>>>(d_data_refine, num_points, dim, R, gpu_work_C,
                                                                   d_graph_cur_refine, d_degree_cur_refine,
                                                                   d_graph_work_refine, d_degree_work_refine,
                                                                   d_graph_dists_refine);
            CUDA_CHECK_FILTERED(cudaGetLastError());
            CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
        }
        for (uint32_t sync_iter = 0; sync_iter < label_sync_iters; ++sync_iter)
        {
            if (active_only_work_effective)
            {
                cudaFree(d_active_ids);
                cudaFree(d_graph_work_refine);
                cudaFree(d_degree_work_refine);
                cudaFree(d_graph_dists_refine);
                cudaFree(d_graph_sources_refine);
                d_active_ids = nullptr;
                d_graph_work_refine = nullptr;
                d_degree_work_refine = nullptr;
                d_graph_dists_refine = nullptr;
                d_graph_sources_refine = nullptr;
            }
            CUDA_CHECK_FILTERED(cudaMemset(d_active_count, 0, sizeof(unsigned long long)));
            CUDA_CHECK_FILTERED(cudaMemset(d_common_before_sum, 0, sizeof(unsigned long long)));
            CUDA_CHECK_FILTERED(cudaMemset(d_low_common_before, 0, sizeof(unsigned int)));
            mark_filtered_refine_active_rows_kernel<<<grid, block>>>(
                d_graph_cur_refine, d_degree_cur_refine, num_points, R, d_offsets, d_labels,
                d_target_labels, d_label_starts, d_label_multi_starts, d_label_point_offsets, num_labels, target_active_labels,
                force_label_start_active, starts_per_label, universal_label_id,
                common_threshold, active_permille, path_active_permille,
                filtered_mode_full ? 2u : (filtered_mode_active ? 1u : 0u), d_active_rows, d_deficient_label_flags,
                d_active_count, d_common_before_sum,
                d_low_common_before, d_refine_label_checks, d_refine_universal);
            CUDA_CHECK_FILTERED(cudaGetLastError());
            CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
            uint32_t active_work_rows = num_points;
            uint32_t active_grid = grid;
            bool forward_chunked = false;
            if (active_only_work_effective)
            {
                unsigned long long h_iter_active_count_ull = 0;
                CUDA_CHECK_FILTERED(cudaMemcpy(&h_iter_active_count_ull, d_active_count,
                                               sizeof(h_iter_active_count_ull), cudaMemcpyDeviceToHost));
                if (h_iter_active_count_ull > (unsigned long long)num_points)
                    h_iter_active_count_ull = num_points;
                active_work_rows = (uint32_t)h_iter_active_count_ull;
                std::vector<uint8_t> h_active_rows((size_t)num_points);
                CUDA_CHECK_FILTERED(cudaMemcpy(h_active_rows.data(), d_active_rows, (size_t)num_points * sizeof(uint8_t),
                                               cudaMemcpyDeviceToHost));
                std::vector<uint32_t> h_active_ids;
                h_active_ids.reserve(active_work_rows);
                for (uint32_t p = 0; p < num_points; ++p)
                    if (h_active_rows[p] != 0)
                        h_active_ids.push_back(p);
                active_work_rows = (uint32_t)h_active_ids.size();
                if (active_work_rows > 0 && active_work_chunk_rows > 0 && active_work_chunk_rows < active_work_rows)
                {
                    forward_chunked = true;
                    uint32_t *d_graph_round_snapshot = nullptr;
                    uint32_t *d_degree_round_snapshot = nullptr;
                    CUDA_CHECK_FILTERED(cudaMalloc(&d_graph_round_snapshot,
                                                   (size_t)num_points * R * sizeof(uint32_t)));
                    CUDA_CHECK_FILTERED(cudaMalloc(&d_degree_round_snapshot,
                                                   (size_t)num_points * sizeof(uint32_t)));
                    CUDA_CHECK_FILTERED(cudaMemcpy(d_graph_round_snapshot, d_graph_cur_refine,
                                                   (size_t)num_points * R * sizeof(uint32_t),
                                                   cudaMemcpyDeviceToDevice));
                    CUDA_CHECK_FILTERED(cudaMemcpy(d_degree_round_snapshot, d_degree_cur_refine,
                                                   (size_t)num_points * sizeof(uint32_t),
                                                   cudaMemcpyDeviceToDevice));
                    printf("[gpu_vamana_filtered] active_only_forward_chunked iter=%u active_rows=%u chunk_rows=%u full_work_total_gib=%.3f\n",
                           sync_iter, active_work_rows, active_work_chunk_rows,
                           bytes_to_gib_filtered(full_work_total_bytes));
                    for (uint32_t chunk_begin = 0; chunk_begin < active_work_rows; chunk_begin += active_work_chunk_rows)
                    {
                        uint32_t chunk_rows = active_work_rows - chunk_begin;
                        if (chunk_rows > active_work_chunk_rows)
                            chunk_rows = active_work_chunk_rows;
                        uint32_t chunk_grid = (chunk_rows + block - 1) / block;
                        CUDA_CHECK_FILTERED(cudaMalloc(&d_active_ids, (size_t)chunk_rows * sizeof(uint32_t)));
                        CUDA_CHECK_FILTERED(cudaMemcpy(d_active_ids, h_active_ids.data() + chunk_begin,
                                                       (size_t)chunk_rows * sizeof(uint32_t), cudaMemcpyHostToDevice));
                        CUDA_CHECK_FILTERED(cudaMalloc(&d_graph_work_refine,
                                                       (size_t)chunk_rows * gpu_work_C * sizeof(uint32_t)));
                        CUDA_CHECK_FILTERED(cudaMalloc(&d_degree_work_refine,
                                                       (size_t)chunk_rows * sizeof(uint32_t)));
                        CUDA_CHECK_FILTERED(cudaMalloc(&d_graph_dists_refine,
                                                       (size_t)chunk_rows * gpu_work_C * sizeof(float)));
                        if (source_tracking_enabled != 0)
                            CUDA_CHECK_FILTERED(cudaMalloc(&d_graph_sources_refine,
                                                           (size_t)chunk_rows * gpu_work_C * sizeof(uint8_t)));
                        printf("[gpu_vamana_filtered] active_only_forward_chunk iter=%u begin=%u rows=%u work_gib=%.3f\n",
                               sync_iter, chunk_begin, chunk_rows,
                               bytes_to_gib_filtered((size_t)chunk_rows * gpu_work_C *
                                                     (sizeof(uint32_t) + sizeof(float) +
                                                      (source_tracking_enabled != 0 ? sizeof(uint8_t) : 0))));
                        filtered_refine_inject_kernel<DataT><<<chunk_grid, block>>>(
                            d_data_refine, num_points, dim, R, filtered_L, gpu_work_C, filtered_pool_cap, filtered_reserve,
                            bridge_reserve, visited_reserve, target_visited_reserve, path_reserve, visited_select_mode, visited_low_common_threshold, max_expand_steps,
                            target_max_expand_steps, label_start_max_expand_steps, label_start_filtered_reserve,
                            label_start_visited_reserve, label_start_path_reserve,
                            starts_per_label, inverted_seeds_enabled, target_shortcut_samples, target_shortcut_keep,
                            label_start_portal_samples, label_start_portal_keep, label_start_portal_mode,
                            label_local_shortcut_samples, label_local_shortcut_keep, label_local_shortcut_mode,
                            label_local_shortcut_sampler, label_path_shortcut_fanout, label_path_shortcut_keep,
                            expanded_path_shortcut_fanout, expanded_path_shortcut_keep,
                            twohop_path_shortcut_fanout, twohop_path_shortcut_keep,
                            path_shortcut_select_mode,
                            label_local_shortcut_target_only, label_local_shortcut_as_frontier,
                            per_label_candidate_mode, per_label_candidate_keep,
                            backbone_admission_mode, backbone_per_label_quota,
                            backbone_admission_mode != 0 ? 1u : 0u,
                            d_deficient_label_flags, filtered_mode_active ? 1u : 0u,
                            filtered_mode_full ? 8u : (filtered_mode_active ? 4u : 0u),
                            d_active_rows, d_active_ids, chunk_rows, d_graph_round_snapshot, d_degree_round_snapshot, d_graph_work_refine, d_degree_work_refine,
                            d_graph_dists_refine, d_graph_sources_refine, d_offsets, d_labels, d_label_starts, d_label_multi_starts, d_target_labels,
                            d_label_point_offsets, d_label_points, d_query_anchor_offsets, d_query_anchor_ids,
                            query_anchor_keep, label_seed_count, num_labels, universal_label_id, build_trace_row,
                            d_filtered_dist, d_bridge_dist, d_refine_label_checks, d_refine_label_rejects, d_filtered_cands,
                            d_filtered_visited_count, d_original_neighbors, d_filtered_reserved, d_filtered_path_reserved, d_filtered_visited_reserved,
                            d_filtered_top_reserved, d_bridge_reserved, d_filtered_seed_count, d_refine_universal);
                        CUDA_CHECK_FILTERED(cudaGetLastError());
                        CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
                        if (local_c_cap <= 128)
                        {
                            filtered_refine_prune_to_compact_kernel<DataT, 128><<<chunk_rows, 1>>>(
                                d_data_refine, dim, d_active_rows, d_graph_work_refine, d_graph_dists_refine, d_graph_sources_refine, d_degree_work_refine,
                                d_graph_cur_refine, d_degree_cur_refine, num_points, R, gpu_work_C, filtered_prune_alpha, source_aware_alpha, work_prune_cap,
                                prune_bridge_protect, prune_noncommon_bridge_protect, refill_target_degree, source_priority_mode, source_merge_mode, filtered_top_protect, common_degree_cap,
                                target_common_degree_cap, position_top_protect, filtered_reserve, 0, 0, path_protect, consensus_protect, consensus_require_filtered_top, d_active_ids, chunk_rows, d_target_labels, d_offsets, d_labels, universal_label_id, d_refine_label_checks,
                                d_refine_label_rejects, d_prune_rejects, d_label_occlusion_blocked, d_geometry_occluded,
                                d_candidate_rejected_by_label, d_refill_count, d_prune_degree_sum, d_source_selected_counts,
                                d_source_refill_counts, d_refine_universal);
                        }
                        else
                        {
                            filtered_refine_prune_to_compact_kernel<DataT, 256><<<chunk_rows, 1>>>(
                                d_data_refine, dim, d_active_rows, d_graph_work_refine, d_graph_dists_refine, d_graph_sources_refine, d_degree_work_refine,
                                d_graph_cur_refine, d_degree_cur_refine, num_points, R, gpu_work_C, filtered_prune_alpha, source_aware_alpha, work_prune_cap,
                                prune_bridge_protect, prune_noncommon_bridge_protect, refill_target_degree, source_priority_mode, source_merge_mode, filtered_top_protect, common_degree_cap,
                                target_common_degree_cap, position_top_protect, filtered_reserve, 0, 0, path_protect, consensus_protect, consensus_require_filtered_top, d_active_ids, chunk_rows, d_target_labels, d_offsets, d_labels, universal_label_id, d_refine_label_checks,
                                d_refine_label_rejects, d_prune_rejects, d_label_occlusion_blocked, d_geometry_occluded,
                                d_candidate_rejected_by_label, d_refill_count, d_prune_degree_sum, d_source_selected_counts,
                                d_source_refill_counts, d_refine_universal);
                        }
                        CUDA_CHECK_FILTERED(cudaGetLastError());
                        CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
                        cudaFree(d_active_ids);
                        cudaFree(d_graph_work_refine);
                        cudaFree(d_degree_work_refine);
                        cudaFree(d_graph_dists_refine);
                        cudaFree(d_graph_sources_refine);
                        d_active_ids = nullptr;
                        d_graph_work_refine = nullptr;
                        d_degree_work_refine = nullptr;
                        d_graph_dists_refine = nullptr;
                        d_graph_sources_refine = nullptr;
                    }
                    cudaFree(d_graph_round_snapshot);
                    cudaFree(d_degree_round_snapshot);
                }
                else if (active_work_rows > 0)
                {
                    CUDA_CHECK_FILTERED(cudaMalloc(&d_active_ids, (size_t)active_work_rows * sizeof(uint32_t)));
                    CUDA_CHECK_FILTERED(cudaMemcpy(d_active_ids, h_active_ids.data(),
                                                   (size_t)active_work_rows * sizeof(uint32_t), cudaMemcpyHostToDevice));
                    CUDA_CHECK_FILTERED(cudaMalloc(&d_graph_work_refine,
                                                   (size_t)active_work_rows * gpu_work_C * sizeof(uint32_t)));
                    CUDA_CHECK_FILTERED(cudaMalloc(&d_degree_work_refine,
                                                   (size_t)active_work_rows * sizeof(uint32_t)));
                    CUDA_CHECK_FILTERED(cudaMalloc(&d_graph_dists_refine,
                                                   (size_t)active_work_rows * gpu_work_C * sizeof(float)));
                    if (source_tracking_enabled != 0)
                        CUDA_CHECK_FILTERED(cudaMalloc(&d_graph_sources_refine,
                                                       (size_t)active_work_rows * gpu_work_C * sizeof(uint8_t)));
                    active_grid = (active_work_rows + block - 1) / block;
                    printf("[gpu_vamana_filtered] active_only_work iter=%u active_rows=%u active_work_total_gib=%.3f full_work_total_gib=%.3f\n",
                           sync_iter, active_work_rows,
                           bytes_to_gib_filtered((size_t)active_work_rows * gpu_work_C *
                                                 (sizeof(uint32_t) + sizeof(float) +
                                                  (source_tracking_enabled != 0 ? sizeof(uint8_t) : 0))),
                           bytes_to_gib_filtered(full_work_total_bytes));
                }
            }
            if (active_work_rows == 0)
                continue;
            if (!forward_chunked)
            {
            filtered_refine_inject_kernel<DataT><<<active_grid, block>>>(
                d_data_refine, num_points, dim, R, filtered_L, gpu_work_C, filtered_pool_cap, filtered_reserve,
                bridge_reserve, visited_reserve, target_visited_reserve, path_reserve, visited_select_mode, visited_low_common_threshold, max_expand_steps,
                target_max_expand_steps, label_start_max_expand_steps, label_start_filtered_reserve,
                label_start_visited_reserve, label_start_path_reserve,
                starts_per_label, inverted_seeds_enabled, target_shortcut_samples, target_shortcut_keep,
                label_start_portal_samples, label_start_portal_keep, label_start_portal_mode,
                label_local_shortcut_samples, label_local_shortcut_keep, label_local_shortcut_mode,
                label_local_shortcut_sampler, label_path_shortcut_fanout, label_path_shortcut_keep,
                expanded_path_shortcut_fanout, expanded_path_shortcut_keep,
                twohop_path_shortcut_fanout, twohop_path_shortcut_keep,
                path_shortcut_select_mode,
                label_local_shortcut_target_only, label_local_shortcut_as_frontier,
                per_label_candidate_mode, per_label_candidate_keep,
                backbone_admission_mode, backbone_per_label_quota,
                backbone_admission_mode != 0 ? 1u : 0u,
                d_deficient_label_flags, filtered_mode_active ? 1u : 0u,
                filtered_mode_full ? 8u : (filtered_mode_active ? 4u : 0u),
                d_active_rows, d_active_ids, active_work_rows, d_graph_cur_refine, d_degree_cur_refine, d_graph_work_refine, d_degree_work_refine,
                d_graph_dists_refine, d_graph_sources_refine, d_offsets, d_labels, d_label_starts, d_label_multi_starts, d_target_labels,
                d_label_point_offsets, d_label_points, d_query_anchor_offsets, d_query_anchor_ids,
                query_anchor_keep, label_seed_count, num_labels, universal_label_id, build_trace_row,
                d_filtered_dist, d_bridge_dist, d_refine_label_checks, d_refine_label_rejects, d_filtered_cands,
                d_filtered_visited_count, d_original_neighbors, d_filtered_reserved, d_filtered_path_reserved, d_filtered_visited_reserved,
                d_filtered_top_reserved, d_bridge_reserved, d_filtered_seed_count, d_refine_universal);
            CUDA_CHECK_FILTERED(cudaGetLastError());
            CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
            if (local_c_cap <= 128)
            {
                filtered_refine_prune_to_compact_kernel<DataT, 128><<<active_work_rows, 1>>>(
                    d_data_refine, dim, d_active_rows, d_graph_work_refine, d_graph_dists_refine, d_graph_sources_refine, d_degree_work_refine,
                    d_graph_cur_refine, d_degree_cur_refine, num_points, R, gpu_work_C, filtered_prune_alpha, source_aware_alpha, work_prune_cap,
                    prune_bridge_protect, prune_noncommon_bridge_protect, refill_target_degree, source_priority_mode, source_merge_mode, filtered_top_protect, common_degree_cap,
                    target_common_degree_cap, position_top_protect, filtered_reserve, 0, 0, path_protect, consensus_protect, consensus_require_filtered_top, d_active_ids, active_work_rows, d_target_labels, d_offsets, d_labels, universal_label_id, d_refine_label_checks,
                    d_refine_label_rejects, d_prune_rejects, d_label_occlusion_blocked, d_geometry_occluded,
                    d_candidate_rejected_by_label, d_refill_count, d_prune_degree_sum, d_source_selected_counts,
                    d_source_refill_counts, d_refine_universal);
            }
            else
            {
                filtered_refine_prune_to_compact_kernel<DataT, 256><<<active_work_rows, 1>>>(
                    d_data_refine, dim, d_active_rows, d_graph_work_refine, d_graph_dists_refine, d_graph_sources_refine, d_degree_work_refine,
                    d_graph_cur_refine, d_degree_cur_refine, num_points, R, gpu_work_C, filtered_prune_alpha, source_aware_alpha, work_prune_cap,
                    prune_bridge_protect, prune_noncommon_bridge_protect, refill_target_degree, source_priority_mode, source_merge_mode, filtered_top_protect, common_degree_cap,
                    target_common_degree_cap, position_top_protect, filtered_reserve, 0, 0, path_protect, consensus_protect, consensus_require_filtered_top, d_active_ids, active_work_rows, d_target_labels, d_offsets, d_labels, universal_label_id, d_refine_label_checks,
                    d_refine_label_rejects, d_prune_rejects, d_label_occlusion_blocked, d_geometry_occluded,
                    d_candidate_rejected_by_label, d_refill_count, d_prune_degree_sum, d_source_selected_counts,
                    d_source_refill_counts, d_refine_universal);
            }
            CUDA_CHECK_FILTERED(cudaGetLastError());
            CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
            }
            if (build_trace_row != INVALID_ID)
            {
                filtered_trace_graph_row_kernel<<<1, 1>>>(d_graph_cur_refine, d_degree_cur_refine,
                                                          num_points, R, d_offsets, d_labels,
                                                          build_trace_row, sync_iter, 0);
                CUDA_CHECK_FILTERED(cudaGetLastError());
                CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
            }
            if (reverse_repair)
            {
                if (reverse_id_chunk_rows > 0 && active_only_work_effective)
                {
                    cudaFree(d_active_ids);
                    cudaFree(d_graph_work_refine);
                    cudaFree(d_degree_work_refine);
                    cudaFree(d_graph_dists_refine);
                    cudaFree(d_graph_sources_refine);
                    d_active_ids = nullptr;
                    d_graph_work_refine = nullptr;
                    d_degree_work_refine = nullptr;
                    d_graph_dists_refine = nullptr;
                    d_graph_sources_refine = nullptr;
                    printf("[gpu_vamana_filtered] reverse_id_chunked iter=%u dst_chunk_rows=%u reverse_cap=%u full_reverse_ids_gib=%.3f\n",
                           sync_iter, reverse_id_chunk_rows, reverse_cap, bytes_to_gib_filtered(full_reverse_bytes));




                    uint32_t *d_reverse_graph_snapshot = nullptr;
                    uint32_t *d_reverse_degree_snapshot = nullptr;
                    CUDA_CHECK_FILTERED(cudaMalloc(&d_reverse_graph_snapshot,
                                                   (size_t)num_points * R * sizeof(uint32_t)));
                    CUDA_CHECK_FILTERED(cudaMalloc(&d_reverse_degree_snapshot,
                                                   (size_t)num_points * sizeof(uint32_t)));
                    CUDA_CHECK_FILTERED(cudaMemcpy(d_reverse_graph_snapshot, d_graph_cur_refine,
                                                   (size_t)num_points * R * sizeof(uint32_t),
                                                   cudaMemcpyDeviceToDevice));
                    CUDA_CHECK_FILTERED(cudaMemcpy(d_reverse_degree_snapshot, d_degree_cur_refine,
                                                   (size_t)num_points * sizeof(uint32_t),
                                                   cudaMemcpyDeviceToDevice));
                    for (uint32_t dst_begin = 0; dst_begin < num_points; dst_begin += reverse_id_chunk_rows)
                    {
                        uint32_t dst_count = num_points - dst_begin;
                        if (dst_count > reverse_id_chunk_rows)
                            dst_count = reverse_id_chunk_rows;
                        CUDA_CHECK_FILTERED(cudaMalloc(&d_reverse_ids, (size_t)dst_count * reverse_cap * sizeof(uint32_t)));
                        CUDA_CHECK_FILTERED(cudaMalloc(&d_reverse_counts, (size_t)dst_count * sizeof(uint32_t)));
                        CUDA_CHECK_FILTERED(cudaMalloc(&d_reverse_touched, (size_t)dst_count * sizeof(uint8_t)));
                        CUDA_CHECK_FILTERED(cudaMemset(d_reverse_ids, 0xFF,
                                                       (size_t)dst_count * reverse_cap * sizeof(uint32_t)));
                        CUDA_CHECK_FILTERED(cudaMemset(d_reverse_counts, 0, (size_t)dst_count * sizeof(uint32_t)));
                        CUDA_CHECK_FILTERED(cudaMemset(d_reverse_touched, 0, (size_t)dst_count * sizeof(uint8_t)));
                        filtered_reverse_generate_kernel<<<grid, block>>>(
                            num_points, R, reverse_cap, dst_begin, dst_count, reverse_replacement_mode, reverse_touch_active_only,
                            reverse_sample_permille, reverse_inactive_sample_permille, reverse_inactive_target_labels,
                            d_active_rows, d_reverse_graph_snapshot,
                            d_reverse_degree_snapshot, d_offsets, d_labels, d_target_labels, num_labels, universal_label_id, d_reverse_ids, d_reverse_counts,
                            d_reverse_touched, d_reverse_edges, d_reverse_kept, d_refine_label_checks, d_refine_universal);
                        CUDA_CHECK_FILTERED(cudaGetLastError());
                        CUDA_CHECK_FILTERED(cudaDeviceSynchronize());

                        std::vector<uint8_t> h_touched_rows((size_t)dst_count);
                        CUDA_CHECK_FILTERED(cudaMemcpy(h_touched_rows.data(), d_reverse_touched,
                                                       (size_t)dst_count * sizeof(uint8_t), cudaMemcpyDeviceToHost));
                        std::vector<uint32_t> h_touched_ids;
                        h_touched_ids.reserve(dst_count / 8);
                        for (uint32_t i = 0; i < dst_count; ++i)
                            if (h_touched_rows[i] != 0)
                                h_touched_ids.push_back(dst_begin + i);
                        uint32_t total_touched_rows = (uint32_t)h_touched_ids.size();
                        printf("[gpu_vamana_filtered] reverse_id_chunk iter=%u dst_begin=%u dst_rows=%u touched_rows=%u reverse_ids_gib=%.3f\n",
                               sync_iter, dst_begin, dst_count, total_touched_rows,
                               bytes_to_gib_filtered((size_t)dst_count * reverse_cap * sizeof(uint32_t)));

                        for (uint32_t chunk_begin = 0; chunk_begin < total_touched_rows; chunk_begin += active_reverse_chunk_rows)
                        {
                            uint32_t reverse_work_rows = total_touched_rows - chunk_begin;
                            if (reverse_work_rows > active_reverse_chunk_rows)
                                reverse_work_rows = active_reverse_chunk_rows;
                            uint32_t reverse_grid = (reverse_work_rows + block - 1) / block;
                            CUDA_CHECK_FILTERED(cudaMalloc(&d_active_ids, (size_t)reverse_work_rows * sizeof(uint32_t)));
                            CUDA_CHECK_FILTERED(cudaMemcpy(d_active_ids, h_touched_ids.data() + chunk_begin,
                                                           (size_t)reverse_work_rows * sizeof(uint32_t), cudaMemcpyHostToDevice));
                            CUDA_CHECK_FILTERED(cudaMalloc(&d_graph_work_refine,
                                                           (size_t)reverse_work_rows * gpu_work_C * sizeof(uint32_t)));
                            CUDA_CHECK_FILTERED(cudaMalloc(&d_degree_work_refine,
                                                           (size_t)reverse_work_rows * sizeof(uint32_t)));
                            CUDA_CHECK_FILTERED(cudaMalloc(&d_graph_dists_refine,
                                                           (size_t)reverse_work_rows * gpu_work_C * sizeof(float)));
                            if (source_tracking_enabled != 0)
                                CUDA_CHECK_FILTERED(cudaMalloc(&d_graph_sources_refine,
                                                               (size_t)reverse_work_rows * gpu_work_C * sizeof(uint8_t)));
                            filtered_reverse_apply_to_work_kernel<DataT><<<reverse_grid, block>>>(
                                d_data_refine, num_points, dim, R, gpu_work_C, reverse_cap, dst_begin, dst_count,
                                reverse_apply_common_sources, reverse_dup_source_merge,
                                (filtered_mode_full || filtered_mode_active) ? 1u : 0u, d_reverse_touched,
                                d_active_ids, reverse_work_rows, d_reverse_ids,
                                d_reverse_counts, reverse_replacement_mode, d_graph_cur_refine, d_degree_cur_refine, d_graph_work_refine,
                                d_degree_work_refine, d_graph_dists_refine, d_graph_sources_refine,
                                d_offsets, d_labels, universal_label_id, build_trace_row,
                                d_refine_label_checks, d_refine_universal);
                            CUDA_CHECK_FILTERED(cudaGetLastError());
                            CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
                            if (local_c_cap <= 128)
                            {
                                filtered_refine_prune_to_compact_kernel<DataT, 128><<<reverse_work_rows, 1>>>(
                                    d_data_refine, dim, d_reverse_touched, d_graph_work_refine, d_graph_dists_refine, d_graph_sources_refine,
                                    d_degree_work_refine, d_graph_cur_refine, d_degree_cur_refine, num_points, R, gpu_work_C, filtered_prune_alpha, source_aware_alpha,
                                work_prune_cap, prune_bridge_protect, prune_noncommon_bridge_protect, refill_target_degree, source_priority_mode, source_merge_mode, filtered_top_protect,
                                    common_degree_cap, target_common_degree_cap, 0, 0, position_reverse_protect, reverse_cap, 0, consensus_protect, consensus_require_filtered_top,
                                    d_active_ids, reverse_work_rows, d_target_labels, d_offsets, d_labels, universal_label_id,
                                    d_refine_label_checks, d_refine_label_rejects, d_prune_rejects, d_label_occlusion_blocked,
                                    d_geometry_occluded, d_candidate_rejected_by_label, d_refill_count, d_prune_degree_sum,
                                    d_source_selected_counts, d_source_refill_counts, d_refine_universal);
                            }
                            else
                            {
                                filtered_refine_prune_to_compact_kernel<DataT, 256><<<reverse_work_rows, 1>>>(
                                    d_data_refine, dim, d_reverse_touched, d_graph_work_refine, d_graph_dists_refine, d_graph_sources_refine,
                                    d_degree_work_refine, d_graph_cur_refine, d_degree_cur_refine, num_points, R, gpu_work_C, filtered_prune_alpha, source_aware_alpha,
                                work_prune_cap, prune_bridge_protect, prune_noncommon_bridge_protect, refill_target_degree, source_priority_mode, source_merge_mode, filtered_top_protect,
                                    common_degree_cap, target_common_degree_cap, 0, 0, position_reverse_protect, reverse_cap, 0, consensus_protect, consensus_require_filtered_top,
                                    d_active_ids, reverse_work_rows, d_target_labels, d_offsets, d_labels, universal_label_id,
                                    d_refine_label_checks, d_refine_label_rejects, d_prune_rejects, d_label_occlusion_blocked,
                                    d_geometry_occluded, d_candidate_rejected_by_label, d_refill_count, d_prune_degree_sum,
                                    d_source_selected_counts, d_source_refill_counts, d_refine_universal);
                            }
                            CUDA_CHECK_FILTERED(cudaGetLastError());
                            CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
                            cudaFree(d_active_ids);
                            cudaFree(d_graph_work_refine);
                            cudaFree(d_degree_work_refine);
                            cudaFree(d_graph_dists_refine);
                            cudaFree(d_graph_sources_refine);
                            d_active_ids = nullptr;
                            d_graph_work_refine = nullptr;
                            d_degree_work_refine = nullptr;
                            d_graph_dists_refine = nullptr;
                            d_graph_sources_refine = nullptr;
                        }
                        count_u8_flags_kernel<<<(dst_count + block - 1) / block, block>>>(d_reverse_touched, dst_count,
                                                                                           d_reverse_pruned_rows);
                        CUDA_CHECK_FILTERED(cudaGetLastError());
                        CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
                        cudaFree(d_reverse_ids);
                        cudaFree(d_reverse_counts);
                        cudaFree(d_reverse_touched);
                        d_reverse_ids = nullptr;
                        d_reverse_counts = nullptr;
                        d_reverse_touched = nullptr;
                    }
                    cudaFree(d_reverse_graph_snapshot);
                    cudaFree(d_reverse_degree_snapshot);
                }
                else
                {
                CUDA_CHECK_FILTERED(cudaMalloc(&d_reverse_ids, (size_t)num_points * reverse_cap * sizeof(uint32_t)));
                CUDA_CHECK_FILTERED(cudaMalloc(&d_reverse_counts, (size_t)num_points * sizeof(uint32_t)));
                CUDA_CHECK_FILTERED(cudaMalloc(&d_reverse_touched, (size_t)num_points * sizeof(uint8_t)));
                CUDA_CHECK_FILTERED(cudaMemset(d_reverse_ids, 0xFF,
                                               (size_t)num_points * reverse_cap * sizeof(uint32_t)));
                CUDA_CHECK_FILTERED(cudaMemset(d_reverse_counts, 0, (size_t)num_points * sizeof(uint32_t)));
                CUDA_CHECK_FILTERED(cudaMemset(d_reverse_touched, 0, (size_t)num_points * sizeof(uint8_t)));
                filtered_reverse_generate_kernel<<<grid, block>>>(
                    num_points, R, reverse_cap, 0, num_points, reverse_replacement_mode, reverse_touch_active_only,
                    reverse_sample_permille, reverse_inactive_sample_permille, reverse_inactive_target_labels,
                    d_active_rows, d_graph_cur_refine,
                    d_degree_cur_refine, d_offsets, d_labels, d_target_labels, num_labels, universal_label_id, d_reverse_ids, d_reverse_counts,
                    d_reverse_touched, d_reverse_edges, d_reverse_kept, d_refine_label_checks, d_refine_universal);
                CUDA_CHECK_FILTERED(cudaGetLastError());
                CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
                if (active_only_work_effective)
                {
                    cudaFree(d_active_ids);
                    cudaFree(d_graph_work_refine);
                    cudaFree(d_degree_work_refine);
                    cudaFree(d_graph_dists_refine);
                    cudaFree(d_graph_sources_refine);
                    d_active_ids = nullptr;
                    d_graph_work_refine = nullptr;
                    d_degree_work_refine = nullptr;
                    d_graph_dists_refine = nullptr;
                    d_graph_sources_refine = nullptr;
                    std::vector<uint8_t> h_touched_rows((size_t)num_points);
                    CUDA_CHECK_FILTERED(cudaMemcpy(h_touched_rows.data(), d_reverse_touched,
                                                   (size_t)num_points * sizeof(uint8_t), cudaMemcpyDeviceToHost));
                    std::vector<uint32_t> h_touched_ids;
                    h_touched_ids.reserve(num_points / 8);
                    for (uint32_t p = 0; p < num_points; ++p)
                        if (h_touched_rows[p] != 0)
                            h_touched_ids.push_back(p);
                    uint32_t total_touched_rows = (uint32_t)h_touched_ids.size();
                    printf("[gpu_vamana_filtered] active_only_reverse iter=%u touched_rows=%u chunk_rows=%u full_work_total_gib=%.3f reverse_ids_gib=%.3f\n",
                           sync_iter, total_touched_rows, active_reverse_chunk_rows,
                           bytes_to_gib_filtered(full_work_total_bytes), bytes_to_gib_filtered(full_reverse_bytes));
                    for (uint32_t chunk_begin = 0; chunk_begin < total_touched_rows; chunk_begin += active_reverse_chunk_rows)
                    {
                        uint32_t reverse_work_rows = total_touched_rows - chunk_begin;
                        if (reverse_work_rows > active_reverse_chunk_rows)
                            reverse_work_rows = active_reverse_chunk_rows;
                        uint32_t reverse_grid = (reverse_work_rows + block - 1) / block;
                        CUDA_CHECK_FILTERED(cudaMalloc(&d_active_ids, (size_t)reverse_work_rows * sizeof(uint32_t)));
                        CUDA_CHECK_FILTERED(cudaMemcpy(d_active_ids, h_touched_ids.data() + chunk_begin,
                                                       (size_t)reverse_work_rows * sizeof(uint32_t), cudaMemcpyHostToDevice));
                        CUDA_CHECK_FILTERED(cudaMalloc(&d_graph_work_refine,
                                                       (size_t)reverse_work_rows * gpu_work_C * sizeof(uint32_t)));
                        CUDA_CHECK_FILTERED(cudaMalloc(&d_degree_work_refine,
                                                       (size_t)reverse_work_rows * sizeof(uint32_t)));
                        CUDA_CHECK_FILTERED(cudaMalloc(&d_graph_dists_refine,
                                                       (size_t)reverse_work_rows * gpu_work_C * sizeof(float)));
                        if (source_tracking_enabled != 0)
                            CUDA_CHECK_FILTERED(cudaMalloc(&d_graph_sources_refine,
                                                           (size_t)reverse_work_rows * gpu_work_C * sizeof(uint8_t)));
                        printf("[gpu_vamana_filtered] active_only_reverse_chunk iter=%u begin=%u rows=%u reverse_work_gib=%.3f\n",
                               sync_iter, chunk_begin, reverse_work_rows,
                               bytes_to_gib_filtered((size_t)reverse_work_rows * gpu_work_C *
                                                     (sizeof(uint32_t) + sizeof(float) +
                                                      (source_tracking_enabled != 0 ? sizeof(uint8_t) : 0))));
                        filtered_reverse_apply_to_work_kernel<DataT><<<reverse_grid, block>>>(
                            d_data_refine, num_points, dim, R, gpu_work_C, reverse_cap, 0, num_points,
                            reverse_apply_common_sources, reverse_dup_source_merge,
                            (filtered_mode_full || filtered_mode_active) ? 1u : 0u, d_reverse_touched,
                            d_active_ids, reverse_work_rows, d_reverse_ids,
                            d_reverse_counts, reverse_replacement_mode, d_graph_cur_refine, d_degree_cur_refine, d_graph_work_refine,
                            d_degree_work_refine, d_graph_dists_refine, d_graph_sources_refine,
                            d_offsets, d_labels, universal_label_id, build_trace_row,
                            d_refine_label_checks, d_refine_universal);
                        CUDA_CHECK_FILTERED(cudaGetLastError());
                        CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
                        if (local_c_cap <= 128)
                        {
                            filtered_refine_prune_to_compact_kernel<DataT, 128><<<reverse_work_rows, 1>>>(
                                d_data_refine, dim, d_reverse_touched, d_graph_work_refine, d_graph_dists_refine, d_graph_sources_refine,
                                d_degree_work_refine, d_graph_cur_refine, d_degree_cur_refine, num_points, R, gpu_work_C, filtered_prune_alpha, source_aware_alpha,
                                work_prune_cap, prune_bridge_protect, prune_noncommon_bridge_protect, refill_target_degree, source_priority_mode, source_merge_mode, filtered_top_protect,
                                common_degree_cap, target_common_degree_cap, 0, 0, position_reverse_protect, reverse_cap, 0, consensus_protect, consensus_require_filtered_top,
                                d_active_ids, reverse_work_rows, d_target_labels, d_offsets, d_labels, universal_label_id,
                                d_refine_label_checks, d_refine_label_rejects, d_prune_rejects, d_label_occlusion_blocked,
                                d_geometry_occluded, d_candidate_rejected_by_label, d_refill_count, d_prune_degree_sum,
                                d_source_selected_counts, d_source_refill_counts, d_refine_universal);
                        }
                        else
                        {
                            filtered_refine_prune_to_compact_kernel<DataT, 256><<<reverse_work_rows, 1>>>(
                                d_data_refine, dim, d_reverse_touched, d_graph_work_refine, d_graph_dists_refine, d_graph_sources_refine,
                                d_degree_work_refine, d_graph_cur_refine, d_degree_cur_refine, num_points, R, gpu_work_C, filtered_prune_alpha, source_aware_alpha,
                                work_prune_cap, prune_bridge_protect, prune_noncommon_bridge_protect, refill_target_degree, source_priority_mode, source_merge_mode, filtered_top_protect,
                                common_degree_cap, target_common_degree_cap, 0, 0, position_reverse_protect, reverse_cap, 0, consensus_protect, consensus_require_filtered_top,
                                d_active_ids, reverse_work_rows, d_target_labels, d_offsets, d_labels, universal_label_id,
                                d_refine_label_checks, d_refine_label_rejects, d_prune_rejects, d_label_occlusion_blocked,
                                d_geometry_occluded, d_candidate_rejected_by_label, d_refill_count, d_prune_degree_sum,
                                d_source_selected_counts, d_source_refill_counts, d_refine_universal);
                        }
                        CUDA_CHECK_FILTERED(cudaGetLastError());
                        CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
                        cudaFree(d_active_ids);
                        cudaFree(d_graph_work_refine);
                        cudaFree(d_degree_work_refine);
                        cudaFree(d_graph_dists_refine);
                        cudaFree(d_graph_sources_refine);
                        d_active_ids = nullptr;
                        d_graph_work_refine = nullptr;
                        d_degree_work_refine = nullptr;
                        d_graph_dists_refine = nullptr;
                        d_graph_sources_refine = nullptr;
                    }
                }
                else
                {
                    uint32_t reverse_work_rows = num_points;
                    uint32_t reverse_grid = grid;
                    const uint32_t *d_reverse_work_ids = nullptr;
                    filtered_reverse_apply_to_work_kernel<DataT><<<reverse_grid, block>>>(
                        d_data_refine, num_points, dim, R, gpu_work_C, reverse_cap, 0, num_points,
                        reverse_apply_common_sources, reverse_dup_source_merge,
                        (filtered_mode_full || filtered_mode_active) ? 1u : 0u, d_reverse_touched,
                        d_reverse_work_ids, reverse_work_rows, d_reverse_ids,
                        d_reverse_counts, reverse_replacement_mode, d_graph_cur_refine, d_degree_cur_refine, d_graph_work_refine,
                        d_degree_work_refine, d_graph_dists_refine, d_graph_sources_refine,
                        d_offsets, d_labels, universal_label_id, build_trace_row,
                        d_refine_label_checks, d_refine_universal);
                    CUDA_CHECK_FILTERED(cudaGetLastError());
                    CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
                    if (local_c_cap <= 128)
                    {
                        filtered_refine_prune_to_compact_kernel<DataT, 128><<<reverse_work_rows, 1>>>(
                            d_data_refine, dim, d_reverse_touched, d_graph_work_refine, d_graph_dists_refine, d_graph_sources_refine,
                            d_degree_work_refine, d_graph_cur_refine, d_degree_cur_refine, num_points, R, gpu_work_C, filtered_prune_alpha, source_aware_alpha,
                                work_prune_cap, prune_bridge_protect, prune_noncommon_bridge_protect, refill_target_degree, source_priority_mode, source_merge_mode, filtered_top_protect,
                            common_degree_cap, target_common_degree_cap, 0, 0, position_reverse_protect, reverse_cap, 0, consensus_protect, consensus_require_filtered_top,
                            d_reverse_work_ids, reverse_work_rows, d_target_labels, d_offsets, d_labels, universal_label_id,
                            d_refine_label_checks, d_refine_label_rejects, d_prune_rejects, d_label_occlusion_blocked,
                            d_geometry_occluded, d_candidate_rejected_by_label, d_refill_count, d_prune_degree_sum,
                            d_source_selected_counts, d_source_refill_counts, d_refine_universal);
                    }
                    else
                    {
                        filtered_refine_prune_to_compact_kernel<DataT, 256><<<reverse_work_rows, 1>>>(
                            d_data_refine, dim, d_reverse_touched, d_graph_work_refine, d_graph_dists_refine, d_graph_sources_refine,
                            d_degree_work_refine, d_graph_cur_refine, d_degree_cur_refine, num_points, R, gpu_work_C, filtered_prune_alpha, source_aware_alpha,
                                work_prune_cap, prune_bridge_protect, prune_noncommon_bridge_protect, refill_target_degree, source_priority_mode, source_merge_mode, filtered_top_protect,
                            common_degree_cap, target_common_degree_cap, 0, 0, position_reverse_protect, reverse_cap, 0, consensus_protect, consensus_require_filtered_top,
                            d_reverse_work_ids, reverse_work_rows, d_target_labels, d_offsets, d_labels, universal_label_id,
                            d_refine_label_checks, d_refine_label_rejects, d_prune_rejects, d_label_occlusion_blocked,
                            d_geometry_occluded, d_candidate_rejected_by_label, d_refill_count, d_prune_degree_sum,
                            d_source_selected_counts, d_source_refill_counts, d_refine_universal);
                    }
                    CUDA_CHECK_FILTERED(cudaGetLastError());
                    CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
                }
                count_u8_flags_kernel<<<grid, block>>>(d_reverse_touched, num_points, d_reverse_pruned_rows);
                CUDA_CHECK_FILTERED(cudaGetLastError());
                CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
                cudaFree(d_reverse_ids);
                cudaFree(d_reverse_counts);
                cudaFree(d_reverse_touched);
                d_reverse_ids = nullptr;
                d_reverse_counts = nullptr;
                d_reverse_touched = nullptr;
                }
                if (build_trace_row != INVALID_ID)
                {
                    filtered_trace_graph_row_kernel<<<1, 1>>>(d_graph_cur_refine, d_degree_cur_refine,
                                                              num_points, R, d_offsets, d_labels,
                                                              build_trace_row, sync_iter, 1);
                    CUDA_CHECK_FILTERED(cudaGetLastError());
                    CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
                }
            }
        }
        CUDA_CHECK_FILTERED(cudaMemcpy(h_graph, d_graph_cur_refine, (size_t)num_points * R * sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(h_degree, d_degree_cur_refine, (size_t)num_points * sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost));

        unsigned long long active_count = 0, common_before_sum = 0, prune_degree_sum = 0;
        unsigned int low_common_before = 0;
        CUDA_CHECK_FILTERED(cudaMemcpy(&active_count, d_active_count, sizeof(active_count), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&common_before_sum, d_common_before_sum, sizeof(common_before_sum),
                                       cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&low_common_before, d_low_common_before, sizeof(low_common_before),
                                       cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&pre_label_checks, d_refine_label_checks, sizeof(unsigned long long),
                                       cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&pre_label_rejects, d_refine_label_rejects, sizeof(unsigned long long),
                                       cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&pre_universal_pass, d_refine_universal, sizeof(unsigned long long),
                                       cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.filtered_search_distance_count, d_filtered_dist,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.unfiltered_search_distance_count, d_bridge_dist,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.filtered_candidate_count, d_filtered_cands,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.filtered_visited_count, d_filtered_visited_count,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.original_neighbor_count, d_original_neighbors,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.filtered_reserved_count, d_filtered_reserved,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        unsigned long long h_filtered_path_reserved = 0;
        CUDA_CHECK_FILTERED(cudaMemcpy(&h_filtered_path_reserved, d_filtered_path_reserved,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.filtered_visited_reserved_count,
                                       d_filtered_visited_reserved, sizeof(unsigned long long),
                                       cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.filtered_top_reserved_count, d_filtered_top_reserved,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.bridge_reserved_count, d_bridge_reserved,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.filtered_seed_count, d_filtered_seed_count,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.filtered_prune_reject_count, d_prune_rejects,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.label_occlusion_blocked_count,
                                       d_label_occlusion_blocked, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.geometry_occluded_count, d_geometry_occluded,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.candidate_rejected_by_label_count,
                                       d_candidate_rejected_by_label, sizeof(unsigned long long),
                                       cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.refill_count, d_refill_count,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.filtered_reverse_edges, d_reverse_edges,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.filtered_reverse_candidates_kept, d_reverse_kept,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.filtered_reverse_touched_rows, d_reverse_pruned_rows,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        g_last_filtered_stats.filtered_reverse_pruned_rows = g_last_filtered_stats.filtered_reverse_touched_rows;
        CUDA_CHECK_FILTERED(cudaMemcpy(&prune_degree_sum, d_prune_degree_sum, sizeof(prune_degree_sum),
                                       cudaMemcpyDeviceToHost));
        g_last_filtered_stats.active_rows = active_count;
        g_last_filtered_stats.active_rows_frac = num_points ? (double)active_count / (double)num_points : 0.0;
        g_last_filtered_stats.common_label_degree_before_avg =
            num_points ? (double)common_before_sum / (double)num_points : 0.0;
        g_last_filtered_stats.low_common_before = low_common_before;
        g_last_filtered_stats.prune_output_degree_avg =
            active_count ? (double)prune_degree_sum / (double)active_count : 0.0;
        g_last_filtered_stats.filtered_seed_source = inverted_seeds_enabled ? 2u : 1u;
        g_last_filtered_stats.merged_candidate_count =
            g_last_filtered_stats.original_neighbor_count + g_last_filtered_stats.filtered_reserved_count;
        if (source_tracking_enabled != 0)
        {
            unsigned long long h_source_selected[8] = {0};
            unsigned long long h_source_refill[8] = {0};
            CUDA_CHECK_FILTERED(cudaMemcpy(h_source_selected, d_source_selected_counts,
                                           8 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            CUDA_CHECK_FILTERED(cudaMemcpy(h_source_refill, d_source_refill_counts,
                                           8 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            printf("[gpu_vamana_filtered_source_stats] source_priority=%u source_diagnostics=%u source_hit_counts=1 selected_unknown=%llu selected_original=%llu selected_original_common=%llu selected_filtered_top=%llu selected_visited=%llu selected_reverse=%llu selected_path=%llu refill_unknown=%llu refill_original=%llu refill_original_common=%llu refill_filtered_top=%llu refill_visited=%llu refill_reverse=%llu refill_path=%llu\n",
                   source_priority_mode, source_diagnostics,
                   (unsigned long long)h_source_selected[0],
                   (unsigned long long)h_source_selected[1],
                   (unsigned long long)h_source_selected[2],
                   (unsigned long long)h_source_selected[3],
                   (unsigned long long)h_source_selected[4],
                   (unsigned long long)h_source_selected[5],
                   (unsigned long long)h_source_selected[6],
                   (unsigned long long)h_source_refill[0],
                   (unsigned long long)h_source_refill[1],
                   (unsigned long long)h_source_refill[2],
                   (unsigned long long)h_source_refill[3],
                   (unsigned long long)h_source_refill[4],
                   (unsigned long long)h_source_refill[5],
                   (unsigned long long)h_source_refill[6]);
        }
        printf("[gpu_vamana_filtered_path_stats] path_reserve=%u path_protect=%u filtered_path_reserved_count=%llu\n",
               path_reserve, path_protect, (unsigned long long)h_filtered_path_reserved);

        cudaFree(d_data_refine);
        cudaFree(d_graph_cur_refine);
        cudaFree(d_graph_work_refine);
        cudaFree(d_degree_cur_refine);
        cudaFree(d_degree_work_refine);
        cudaFree(d_graph_dists_refine);
        cudaFree(d_graph_sources_refine);
        cudaFree(d_source_selected_counts);
        cudaFree(d_source_refill_counts);
        cudaFree(d_active_rows);
        cudaFree(d_deficient_label_flags);
        cudaFree(d_active_ids);
        cudaFree(d_label_multi_starts);
        cudaFree(d_target_labels);
        cudaFree(d_label_point_offsets);
        cudaFree(d_label_points);
        cudaFree(d_query_anchor_offsets);
        cudaFree(d_query_anchor_ids);
        cudaFree(d_active_count);
        cudaFree(d_common_before_sum);
        cudaFree(d_low_common_before);
        cudaFree(d_refine_label_checks);
        cudaFree(d_refine_label_rejects);
        cudaFree(d_refine_universal);
        cudaFree(d_filtered_dist);
        cudaFree(d_bridge_dist);
        cudaFree(d_filtered_cands);
        cudaFree(d_filtered_visited_count);
        cudaFree(d_original_neighbors);
        cudaFree(d_filtered_reserved);
        cudaFree(d_filtered_path_reserved);
        cudaFree(d_filtered_visited_reserved);
        cudaFree(d_filtered_top_reserved);
        cudaFree(d_bridge_reserved);
        cudaFree(d_filtered_seed_count);
        cudaFree(d_reverse_ids);
        cudaFree(d_reverse_counts);
        cudaFree(d_reverse_touched);
        cudaFree(d_reverse_edges);
        cudaFree(d_reverse_kept);
        cudaFree(d_reverse_pruned_rows);
        cudaFree(d_prune_rejects);
        cudaFree(d_label_occlusion_blocked);
        cudaFree(d_geometry_occluded);
        cudaFree(d_candidate_rejected_by_label);
        cudaFree(d_refill_count);
        cudaFree(d_prune_degree_sum);
    }

    uint32_t *d_graph = nullptr;
    uint32_t *d_degree = nullptr;
    uint32_t *d_pruned_graph = nullptr;
    uint32_t *d_pruned_degree = nullptr;
    unsigned long long *d_out_edges = nullptr;
    unsigned long long *d_common_edges = nullptr;
    unsigned long long *d_label_checks = nullptr;
    unsigned long long *d_label_rejects = nullptr;
    unsigned long long *d_universal_pass = nullptr;
    unsigned int *d_invalid = nullptr;
    unsigned int *d_self = nullptr;
    unsigned int *d_low_common = nullptr;
    unsigned int *d_degree_min = nullptr;
    unsigned int *d_degree_max = nullptr;
    unsigned int *d_common_min = nullptr;
    unsigned int *d_common_max = nullptr;

    double diag_t0 = now_sec_filtered();
    CUDA_CHECK_FILTERED(cudaMalloc(&d_graph, (size_t)num_points * R * sizeof(uint32_t)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_degree, (size_t)num_points * sizeof(uint32_t)));
    CUDA_CHECK_FILTERED(cudaMemcpy(d_graph, h_graph, (size_t)num_points * R * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_FILTERED(cudaMemcpy(d_degree, h_degree, (size_t)num_points * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_label_checks, sizeof(unsigned long long)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_label_rejects, sizeof(unsigned long long)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_universal_pass, sizeof(unsigned long long)));
    CUDA_CHECK_FILTERED(cudaMemset(d_label_checks, 0, sizeof(unsigned long long)));
    CUDA_CHECK_FILTERED(cudaMemset(d_label_rejects, 0, sizeof(unsigned long long)));
    CUDA_CHECK_FILTERED(cudaMemset(d_universal_pass, 0, sizeof(unsigned long long)));
    if (run_label_prune)
    {
        CUDA_CHECK_FILTERED(cudaMalloc(&d_pruned_graph, (size_t)num_points * R * sizeof(uint32_t)));
        CUDA_CHECK_FILTERED(cudaMalloc(&d_pruned_degree, (size_t)num_points * sizeof(uint32_t)));
        DataT *d_data_for_prune = nullptr;
        CUDA_CHECK_FILTERED(cudaMalloc(&d_data_for_prune, (size_t)num_points * dim * sizeof(DataT)));
        CUDA_CHECK_FILTERED(cudaMemcpy(d_data_for_prune, h_data, (size_t)num_points * dim * sizeof(DataT),
                                       cudaMemcpyHostToDevice));
        unsigned long long *d_prune_rejects = nullptr;
        CUDA_CHECK_FILTERED(cudaMalloc(&d_prune_rejects, sizeof(unsigned long long)));
        CUDA_CHECK_FILTERED(cudaMemset(d_prune_rejects, 0, sizeof(unsigned long long)));
        filtered_final_prune_to_compact_kernel<DataT><<<num_points, 1>>>(
            d_data_for_prune, dim, d_graph, d_degree, d_pruned_graph, d_pruned_degree, num_points, R, 1.2f,
            d_offsets, d_labels, universal_label_id, d_label_checks, d_label_rejects, d_prune_rejects,
            d_universal_pass);
        CUDA_CHECK_FILTERED(cudaGetLastError());
        CUDA_CHECK_FILTERED(cudaDeviceSynchronize());
        CUDA_CHECK_FILTERED(cudaMemcpy(h_graph, d_pruned_graph, (size_t)num_points * R * sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(h_degree, d_pruned_degree, (size_t)num_points * sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost));
        CUDA_CHECK_FILTERED(cudaMemcpy(d_graph, d_pruned_graph, (size_t)num_points * R * sizeof(uint32_t),
                                       cudaMemcpyDeviceToDevice));
        CUDA_CHECK_FILTERED(cudaMemcpy(d_degree, d_pruned_degree, (size_t)num_points * sizeof(uint32_t),
                                       cudaMemcpyDeviceToDevice));
        CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.filtered_prune_reject_count, d_prune_rejects,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        cudaFree(d_prune_rejects);
        cudaFree(d_data_for_prune);
    }
    CUDA_CHECK_FILTERED(cudaMalloc(&d_out_edges, sizeof(unsigned long long)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_common_edges, sizeof(unsigned long long)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_invalid, sizeof(unsigned int)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_self, sizeof(unsigned int)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_low_common, sizeof(unsigned int)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_degree_min, sizeof(unsigned int)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_degree_max, sizeof(unsigned int)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_common_min, sizeof(unsigned int)));
    CUDA_CHECK_FILTERED(cudaMalloc(&d_common_max, sizeof(unsigned int)));

    unsigned long long zero64 = 0;
    unsigned int zero32 = 0;
    unsigned int big32 = 0xFFFFFFFFu;
    CUDA_CHECK_FILTERED(cudaMemcpy(d_out_edges, &zero64, sizeof(zero64), cudaMemcpyHostToDevice));
    CUDA_CHECK_FILTERED(cudaMemcpy(d_common_edges, &zero64, sizeof(zero64), cudaMemcpyHostToDevice));
    CUDA_CHECK_FILTERED(cudaMemcpy(d_invalid, &zero32, sizeof(zero32), cudaMemcpyHostToDevice));
    CUDA_CHECK_FILTERED(cudaMemcpy(d_self, &zero32, sizeof(zero32), cudaMemcpyHostToDevice));
    CUDA_CHECK_FILTERED(cudaMemcpy(d_low_common, &zero32, sizeof(zero32), cudaMemcpyHostToDevice));
    CUDA_CHECK_FILTERED(cudaMemcpy(d_degree_min, &big32, sizeof(big32), cudaMemcpyHostToDevice));
    CUDA_CHECK_FILTERED(cudaMemcpy(d_degree_max, &zero32, sizeof(zero32), cudaMemcpyHostToDevice));
    CUDA_CHECK_FILTERED(cudaMemcpy(d_common_min, &big32, sizeof(big32), cudaMemcpyHostToDevice));
    CUDA_CHECK_FILTERED(cudaMemcpy(d_common_max, &zero32, sizeof(zero32), cudaMemcpyHostToDevice));

    uint32_t block = 256;
    uint32_t grid = (num_points + block - 1) / block;
    filtered_graph_diagnostics_kernel<<<grid, block>>>(d_graph, d_degree, num_points, R, d_offsets, d_labels,
                                                       universal_label_id, d_out_edges, d_common_edges,
                                                       d_label_checks, d_label_rejects, d_universal_pass, d_invalid,
                                                       d_self, d_low_common, d_degree_min, d_degree_max, d_common_min,
                                                       d_common_max);
    CUDA_CHECK_FILTERED(cudaGetLastError());
    CUDA_CHECK_FILTERED(cudaDeviceSynchronize());

    unsigned long long out_edges = 0, common_edges = 0;
    CUDA_CHECK_FILTERED(cudaMemcpy(&out_edges, d_out_edges, sizeof(out_edges), cudaMemcpyDeviceToHost));
    CUDA_CHECK_FILTERED(cudaMemcpy(&common_edges, d_common_edges, sizeof(common_edges), cudaMemcpyDeviceToHost));
    CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.label_check_count, d_label_checks,
                                   sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.label_reject_count, d_label_rejects,
                                   sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.universal_label_pass_count, d_universal_pass,
                                   sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    g_last_filtered_stats.label_check_count += pre_label_checks;
    g_last_filtered_stats.label_reject_count += pre_label_rejects;
    g_last_filtered_stats.universal_label_pass_count += pre_universal_pass;
    CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.invalid_neighbor_count, d_invalid, sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost));
    CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.self_loop_count, d_self, sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost));
    CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.low_common_label_degree_count, d_low_common,
                                   sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.out_degree_min, d_degree_min, sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost));
    CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.out_degree_max, d_degree_max, sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost));
    CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.common_label_degree_min, d_common_min, sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost));
    CUDA_CHECK_FILTERED(cudaMemcpy(&g_last_filtered_stats.common_label_degree_max, d_common_max, sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost));
    g_last_filtered_stats.out_degree_avg = num_points ? (double)out_edges / (double)num_points : 0.0;
    g_last_filtered_stats.avg_common_label_out_degree = num_points ? (double)common_edges / (double)num_points : 0.0;
    if (!run_two_pool && !run_refine_vnew2)
    {
        g_last_filtered_stats.filtered_search_distance_count = 0;
        g_last_filtered_stats.unfiltered_search_distance_count = 0;
        g_last_filtered_stats.filtered_candidate_count = common_edges;
        g_last_filtered_stats.unfiltered_candidate_count = out_edges;
        g_last_filtered_stats.merged_candidate_count = out_edges;
        g_last_filtered_stats.filtered_seed_count = 0;
        g_last_filtered_stats.filtered_visited_count = 0;
        g_last_filtered_stats.filtered_visited_reserved_count = 0;
        g_last_filtered_stats.filtered_top_reserved_count = 0;
        g_last_filtered_stats.filtered_reserved_count = 0;
        g_last_filtered_stats.unfiltered_reserved_count = 0;
        g_last_filtered_stats.bridge_reserved_count = 0;
        g_last_filtered_stats.filtered_reverse_edges = 0;
        g_last_filtered_stats.filtered_reverse_touched_rows = 0;
        g_last_filtered_stats.filtered_reverse_candidates_kept = 0;
        g_last_filtered_stats.filtered_reverse_pruned_rows = 0;
        g_last_filtered_stats.label_occlusion_blocked_count = 0;
        g_last_filtered_stats.geometry_occluded_count = 0;
        g_last_filtered_stats.candidate_rejected_by_label_count = 0;
        g_last_filtered_stats.refill_count = 0;
        g_last_filtered_stats.prune_output_degree_avg = 0.0;
        g_last_filtered_stats.filtered_seed_source = 0;
    }
    if (run_refine_vnew2)
    {
        g_last_filtered_stats.common_label_degree_after_avg = g_last_filtered_stats.avg_common_label_out_degree;
        g_last_filtered_stats.low_common_after = g_last_filtered_stats.low_common_label_degree_count;
        g_last_filtered_stats.unfiltered_reserved_count = g_last_filtered_stats.bridge_reserved_count;
    }
    g_last_filtered_stats.final_prune_rows = num_points;
    g_last_filtered_stats.reverse_edges = 0;
    g_last_filtered_stats.reverse_touched_rows = 0;
    g_last_filtered_stats.diagnostics_seconds = now_sec_filtered() - diag_t0;

    size_t free_mem = 0, total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess)
    {
        g_last_filtered_stats.peak_gpu_mem_gb = std::max(
            g_last_filtered_stats.peak_gpu_mem_gb,
            (double)(total_mem - free_mem) / (1024.0 * 1024.0 * 1024.0));
    }

    cudaFree(d_offsets);
    cudaFree(d_labels);
    cudaFree(d_label_starts);
    cudaFree(d_graph);
    cudaFree(d_degree);
    cudaFree(d_pruned_graph);
    cudaFree(d_pruned_degree);
    cudaFree(d_out_edges);
    cudaFree(d_common_edges);
    cudaFree(d_label_checks);
    cudaFree(d_label_rejects);
    cudaFree(d_universal_pass);
    cudaFree(d_invalid);
    cudaFree(d_self);
    cudaFree(d_low_common);
    cudaFree(d_degree_min);
    cudaFree(d_degree_max);
    cudaFree(d_common_min);
    cudaFree(d_common_max);

    g_last_filtered_stats.gpu_build_seconds = now_sec_filtered() - t0;
    printf("[gpu_vamana_filtered_stats] semantic_stage=%s semantic_complete=%u total=%.6f label_h2d=%.6f graph_build=%.6f diagnostics=%.6f active_rows=%llu active_rows_frac=%.6f common_label_degree_before_avg=%.3f common_label_degree_after_avg=%.3f low_common_before=%u low_common_after=%u out_degree_avg=%.3f common_label_out_degree_avg=%.3f filtered_search_distance_count=%llu unfiltered_bridge_distance_count=%llu filtered_candidate_count=%llu filtered_visited_count=%llu filtered_visited_reserved_count=%llu filtered_top_reserved_count=%llu original_neighbor_count=%llu bridge_reserved_count=%llu unfiltered_candidate_count=%llu merged_candidate_count=%llu filtered_seed_count=%llu filtered_seed_source=%u filtered_reserved_count=%llu unfiltered_reserved_count=%llu filtered_reverse_edges=%llu filtered_reverse_touched_rows=%llu filtered_reverse_candidates_kept=%llu filtered_reverse_pruned_rows=%llu label_checks=%llu label_rejects=%llu label_occlusion_blocked_count=%llu geometry_occluded_count=%llu candidate_rejected_by_label_count=%llu filtered_prune_rejects=%llu prune_output_degree_avg=%.3f refill_count=%llu invalid=%u self_loops=%u low_common_rows=%u peak_gpu_mem_gb=%.3f\n",
           stage_name, g_last_filtered_stats.semantic_complete,
           g_last_filtered_stats.gpu_build_seconds,
           g_last_filtered_stats.label_h2d_seconds, g_last_filtered_stats.graph_build_seconds,
           g_last_filtered_stats.diagnostics_seconds,
           (unsigned long long)g_last_filtered_stats.active_rows,
           g_last_filtered_stats.active_rows_frac,
           g_last_filtered_stats.common_label_degree_before_avg,
           g_last_filtered_stats.common_label_degree_after_avg,
           g_last_filtered_stats.low_common_before,
           g_last_filtered_stats.low_common_after,
           g_last_filtered_stats.out_degree_avg,
           g_last_filtered_stats.avg_common_label_out_degree,
           (unsigned long long)g_last_filtered_stats.filtered_search_distance_count,
           (unsigned long long)g_last_filtered_stats.unfiltered_search_distance_count,
           (unsigned long long)g_last_filtered_stats.filtered_candidate_count,
           (unsigned long long)g_last_filtered_stats.filtered_visited_count,
           (unsigned long long)g_last_filtered_stats.filtered_visited_reserved_count,
           (unsigned long long)g_last_filtered_stats.filtered_top_reserved_count,
           (unsigned long long)g_last_filtered_stats.original_neighbor_count,
           (unsigned long long)g_last_filtered_stats.bridge_reserved_count,
           (unsigned long long)g_last_filtered_stats.unfiltered_candidate_count,
           (unsigned long long)g_last_filtered_stats.merged_candidate_count,
           (unsigned long long)g_last_filtered_stats.filtered_seed_count,
           g_last_filtered_stats.filtered_seed_source,
           (unsigned long long)g_last_filtered_stats.filtered_reserved_count,
           (unsigned long long)g_last_filtered_stats.unfiltered_reserved_count,
           (unsigned long long)g_last_filtered_stats.filtered_reverse_edges,
           (unsigned long long)g_last_filtered_stats.filtered_reverse_touched_rows,
           (unsigned long long)g_last_filtered_stats.filtered_reverse_candidates_kept,
           (unsigned long long)g_last_filtered_stats.filtered_reverse_pruned_rows,
           (unsigned long long)g_last_filtered_stats.label_check_count,
           (unsigned long long)g_last_filtered_stats.label_reject_count,
           (unsigned long long)g_last_filtered_stats.label_occlusion_blocked_count,
           (unsigned long long)g_last_filtered_stats.geometry_occluded_count,
           (unsigned long long)g_last_filtered_stats.candidate_rejected_by_label_count,
           (unsigned long long)g_last_filtered_stats.filtered_prune_reject_count,
           g_last_filtered_stats.prune_output_degree_avg,
           (unsigned long long)g_last_filtered_stats.refill_count,
           g_last_filtered_stats.invalid_neighbor_count, g_last_filtered_stats.self_loop_count,
           g_last_filtered_stats.low_common_label_degree_count, g_last_filtered_stats.peak_gpu_mem_gb);
    return 0;
}

extern "C" int gpu_vamana_filtered_build_uint8(const uint8_t *h_data,
                                                uint32_t num_points,
                                                uint32_t dim,
                                                uint32_t R,
                                                uint32_t L,
                                                uint32_t filtered_L,
                                                uint32_t C,
                                                uint32_t STEPS,
                                                float alpha,
                                                const uint32_t *point_label_offsets,
                                                const uint32_t *point_labels,
                                                uint32_t total_label_count,
                                                const uint32_t *label_to_start_id,
                                                uint32_t num_labels,
                                                uint32_t universal_label_id,
                                                uint32_t global_start_id,
                                                const uint32_t *query_anchor_offsets,
                                                const uint32_t *query_anchor_ids,
                                                uint32_t query_anchor_count,
                                                uint32_t *h_graph,
                                                uint32_t *h_degree)
{
    return gpu_vamana_filtered_build_impl<uint8_t>(h_data, num_points, dim, R, L, filtered_L, C, STEPS, alpha,
                                                   point_label_offsets, point_labels, total_label_count,
                                                   label_to_start_id, num_labels, universal_label_id, global_start_id,
                                                   query_anchor_offsets, query_anchor_ids, query_anchor_count,
                                                   h_graph, h_degree, "uint8");
}

extern "C" int gpu_vamana_filtered_build_float(const float *h_data,
                                                uint32_t num_points,
                                                uint32_t dim,
                                                uint32_t R,
                                                uint32_t L,
                                                uint32_t filtered_L,
                                                uint32_t C,
                                                uint32_t STEPS,
                                                float alpha,
                                                const uint32_t *point_label_offsets,
                                                const uint32_t *point_labels,
                                                uint32_t total_label_count,
                                                const uint32_t *label_to_start_id,
                                                uint32_t num_labels,
                                                uint32_t universal_label_id,
                                                uint32_t global_start_id,
                                                const uint32_t *query_anchor_offsets,
                                                const uint32_t *query_anchor_ids,
                                                uint32_t query_anchor_count,
                                                uint32_t *h_graph,
                                                uint32_t *h_degree)
{
    return gpu_vamana_filtered_build_impl<float>(h_data, num_points, dim, R, L, filtered_L, C, STEPS, alpha,
                                                 point_label_offsets, point_labels, total_label_count,
                                                 label_to_start_id, num_labels, universal_label_id, global_start_id,
                                                 query_anchor_offsets, query_anchor_ids, query_anchor_count,
                                                 h_graph, h_degree, "float");
}

extern "C" void gpu_vamana_filtered_get_last_stats(GPUFilteredVamanaStats *stats)
{
    if (stats)
        *stats = g_last_filtered_stats;
}
