// Copyright (c) 2026 DANCE Authors. Licensed under the MIT License.



#include <cuda_runtime.h>
#include <stdint.h>
#include "../include/gpu_vamana_builder.h"

namespace
{
struct HandoffContext
{
    size_t data_bytes, graph_bytes, degree_bytes;
    void *data = nullptr, *graph = nullptr, *degree = nullptr;
    int stage = 0;
};
static thread_local HandoffContext *context = nullptr;

static cudaError_t handoff_malloc(void **pointer, size_t bytes)
{
    const cudaError_t error = ::cudaMalloc(pointer, bytes);
    if (error != cudaSuccess || context == nullptr)
        return error;
    if (context->stage == 0 && bytes == context->data_bytes)
    {
        context->data = *pointer;
        context->stage = 1;
    }
    else if (context->stage == 1 && bytes == context->graph_bytes)
    {
        context->graph = *pointer;
        context->stage = 2;
    }
    else if (context->stage == 2)
    {

        context->stage = 3;
    }
    else if (context->stage == 3 && bytes == context->degree_bytes)
    {
        context->degree = *pointer;
        context->stage = 4;
    }
    return error;
}

static cudaError_t handoff_free(void *pointer)
{
    if (context && (pointer == context->data || pointer == context->graph || pointer == context->degree))
        return cudaSuccess;
    return ::cudaFree(pointer);
}

static cudaError_t handoff_memcpy(void *destination, const void *source, size_t bytes, cudaMemcpyKind kind)
{
    if (context && kind == cudaMemcpyDeviceToHost &&
        (source == context->graph || source == context->degree))
        return cudaSuccess;
    return ::cudaMemcpy(destination, source, bytes, kind);
}
}

#define cudaMalloc handoff_malloc
#define cudaFree handoff_free
#define cudaMemcpy handoff_memcpy
#define gpu_vamana_vnew2_build handoff_internal_build_u8
#define gpu_vamana_vnew2_build_float handoff_internal_build_float
#define gpu_vamana_vnew2_get_last_medoid handoff_internal_get_last_medoid
#define gpu_vamana_vnew2_get_last_stats handoff_internal_get_last_stats
#define vnew2_convert_float_to_half_kernel handoff_vnew2_convert_float_to_half_kernel
#define vnew2_final_prune_kernel_shared handoff_vnew2_final_prune_kernel_shared
#define vnew2_init_random_graph handoff_vnew2_init_random_graph
#define vnew2_init_touched_from_active_kernel handoff_vnew2_init_touched_from_active_kernel
#define vnew2_count_touched_rows_kernel handoff_vnew2_count_touched_rows_kernel
#define vnew2_mark_important_nodes_budgeted_kernel handoff_vnew2_mark_important_nodes_budgeted_kernel
#define vnew2_mark_important_nodes_kernel handoff_vnew2_mark_important_nodes_kernel
#define vnew2_medoid_score_kernel_warp handoff_vnew2_medoid_score_kernel_warp
#define vnew2_reverse_count_csr_active_kernel handoff_vnew2_reverse_count_csr_active_kernel
#define vnew2_reverse_count_csr_kernel handoff_vnew2_reverse_count_csr_kernel
#define vnew2_reverse_fill_csr_active_kernel handoff_vnew2_reverse_fill_csr_active_kernel
#define vnew2_reverse_fill_csr_kernel handoff_vnew2_reverse_fill_csr_kernel
#define vnew2_reverse_generate_kernel handoff_vnew2_reverse_generate_kernel
#define vnew2_reverse_merge_csr_kernel handoff_vnew2_reverse_merge_csr_kernel
#define vnew2_reverse_merge_owner_kernel handoff_vnew2_reverse_merge_owner_kernel
#define vnew2_search_prune_kernel_shared handoff_vnew2_search_prune_kernel_shared
#include "gpu_vamana_vnew2.cu"
#undef vnew2_search_prune_kernel_shared
#undef vnew2_reverse_merge_owner_kernel
#undef vnew2_reverse_merge_csr_kernel
#undef vnew2_reverse_generate_kernel
#undef vnew2_reverse_fill_csr_kernel
#undef vnew2_reverse_fill_csr_active_kernel
#undef vnew2_reverse_count_csr_kernel
#undef vnew2_reverse_count_csr_active_kernel
#undef vnew2_medoid_score_kernel_warp
#undef vnew2_mark_important_nodes_kernel
#undef vnew2_mark_important_nodes_budgeted_kernel
#undef vnew2_init_touched_from_active_kernel
#undef vnew2_init_random_graph
#undef vnew2_final_prune_kernel_shared
#undef vnew2_convert_float_to_half_kernel
#undef gpu_vamana_vnew2_get_last_stats
#undef gpu_vamana_vnew2_get_last_medoid
#undef gpu_vamana_vnew2_build_float
#undef gpu_vamana_vnew2_build
#undef cudaMemcpy
#undef cudaFree
#undef cudaMalloc

template <typename DataT>
static int build_device(const DataT *host, uint32_t N, uint32_t dim, uint32_t R, uint32_t L,
                        uint32_t C, uint32_t steps, DataT **data, uint32_t **graph, uint32_t **degree)
{
    HandoffContext state{(size_t)N * dim * sizeof(DataT), (size_t)N * R * sizeof(uint32_t),
                         (size_t)N * sizeof(uint32_t)};
    context = &state;
    int result;
    uint32_t *const unused_host_output = reinterpret_cast<uint32_t *>(uintptr_t(1));
    if constexpr (sizeof(DataT) == sizeof(float))
        result = handoff_internal_build_float(reinterpret_cast<const float *>(host), N, dim, R, L, C,
                                              steps, unused_host_output, unused_host_output);
    else
        result = handoff_internal_build_u8(reinterpret_cast<const uint8_t *>(host), N, dim, R, L, C,
                                           steps, unused_host_output, unused_host_output);
    context = nullptr;
    if (result != 0 || !state.data || !state.graph || !state.degree)
    {
        if (state.data) ::cudaFree(state.data);
        if (state.graph) ::cudaFree(state.graph);
        if (state.degree) ::cudaFree(state.degree);
        return result != 0 ? result : -1;
    }
    *data = static_cast<DataT *>(state.data);
    *graph = static_cast<uint32_t *>(state.graph);
    *degree = static_cast<uint32_t *>(state.degree);
    return 0;
}

extern "C" int gpu_vamana_vnew2_build_device(const uint8_t *host, uint32_t N, uint32_t dim,
                                               uint32_t R, uint32_t L, uint32_t C, uint32_t steps,
                                               uint8_t **data, uint32_t **graph, uint32_t **degree)
{
    return build_device(host, N, dim, R, L, C, steps, data, graph, degree);
}

extern "C" int gpu_vamana_vnew2_build_device_float(const float *host, uint32_t N, uint32_t dim,
                                                     uint32_t R, uint32_t L, uint32_t C, uint32_t steps,
                                                     float **data, uint32_t **graph, uint32_t **degree)
{
    return build_device(host, N, dim, R, L, C, steps, data, graph, degree);
}
