// Copyright (c) 2026 DANCE Authors. Licensed under the MIT License.


















#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <type_traits>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "../include/gpu_vamana_builder.h"
#include "../include/gpu_vamana_config.h"

#define medoid_score_kernel_warp vnew2_medoid_score_kernel_warp
#define init_random_graph vnew2_init_random_graph
#define search_prune_kernel_shared vnew2_search_prune_kernel_shared
#define mark_important_nodes_kernel vnew2_mark_important_nodes_kernel
#define mark_important_nodes_budgeted_kernel vnew2_mark_important_nodes_budgeted_kernel
#define init_touched_from_active_kernel vnew2_init_touched_from_active_kernel
#define count_touched_rows_kernel vnew2_count_touched_rows_kernel
#define reverse_generate_kernel vnew2_reverse_generate_kernel
#define reverse_count_csr_kernel vnew2_reverse_count_csr_kernel
#define reverse_fill_csr_kernel vnew2_reverse_fill_csr_kernel
#define reverse_count_csr_active_kernel vnew2_reverse_count_csr_active_kernel
#define reverse_fill_csr_active_kernel vnew2_reverse_fill_csr_active_kernel
#define reverse_merge_csr_kernel vnew2_reverse_merge_csr_kernel
#define reverse_merge_owner_kernel vnew2_reverse_merge_owner_kernel
#define final_prune_kernel_shared vnew2_final_prune_kernel_shared
#define convert_float_to_half_kernel vnew2_convert_float_to_half_kernel

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef INVALID_ID
#define INVALID_ID 0xFFFFFFFFu
#endif

#ifndef VAMANA_ALPHA
#define VAMANA_ALPHA 1.2f
#endif

#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 8
#endif

#ifndef VAMANA_ITERS
#define VAMANA_ITERS 4
#endif

#ifndef INIT_RANDOM_SEEDS
#define INIT_RANDOM_SEEDS 8
#endif

#ifndef ENABLE_REVERSE_HEAVY_CAP
#define ENABLE_REVERSE_HEAVY_CAP 0
#endif

#ifndef ENABLE_RECOMPUTE_CURRENT_DISTS
#define ENABLE_RECOMPUTE_CURRENT_DISTS 0
#endif

#ifndef REVERSE_HEAVY_THRESHOLD
#define REVERSE_HEAVY_THRESHOLD 256
#endif

#ifndef REVERSE_HEAVY_SAMPLE
#define REVERSE_HEAVY_SAMPLE 256
#endif

#ifndef ENABLE_FAST_ONLINE_REPLACE
#define ENABLE_FAST_ONLINE_REPLACE 0
#endif

#ifndef ENABLE_TOPL_EARLY_REJECT
#define ENABLE_TOPL_EARLY_REJECT 0
#endif

#ifndef ENABLE_SEARCH_PRUNE_TIMING
#define ENABLE_SEARCH_PRUNE_TIMING 0
#endif

#ifndef ENABLE_NODE_VECTOR_REG_CACHE
#define ENABLE_NODE_VECTOR_REG_CACHE 1
#endif

#define NODE_VECTOR_CACHE_HALF2_SLOTS(cache_dim) (((cache_dim) > 0) ? ((((cache_dim) / 2) + WARP_SIZE - 1) / WARP_SIZE) : 1)

#ifndef USE_ALPHA_FROM_FIRST_ITER
#define USE_ALPHA_FROM_FIRST_ITER 0
#endif

#ifndef ENABLE_MULTI_ENTRY_MEDOIDS
#define ENABLE_MULTI_ENTRY_MEDOIDS 0
#endif

#ifndef MULTI_ENTRY_POINTS
#define MULTI_ENTRY_POINTS 1
#endif

#ifndef MAX_MULTI_ENTRY_POINTS
#define MAX_MULTI_ENTRY_POINTS 16
#endif

#ifndef ENABLE_SELECTIVE_HIGH_QUALITY_PASS
#define ENABLE_SELECTIVE_HIGH_QUALITY_PASS 0
#endif

#ifndef SELECTIVE_HQ_EXTRA_STEPS
#define SELECTIVE_HQ_EXTRA_STEPS 32
#endif

#ifndef ENABLE_SELECTIVE_HQ_BUDGET
#define ENABLE_SELECTIVE_HQ_BUDGET 0
#endif

#ifndef SELECTIVE_HQ_ACTIVE_PERMILLE
#define SELECTIVE_HQ_ACTIVE_PERMILLE 200
#endif

#ifndef ENABLE_SELECTIVE_HQ_ACTIVE_REVERSE_ONLY
#define ENABLE_SELECTIVE_HQ_ACTIVE_REVERSE_ONLY 0
#endif

#ifndef ENABLE_SELECTIVE_HQ_TOUCHED_FINAL_PRUNE
#define ENABLE_SELECTIVE_HQ_TOUCHED_FINAL_PRUNE 0
#endif

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                                          \
    do                                                                                            \
    {                                                                                             \
        cudaError_t err__ = (call);                                                               \
        if (err__ != cudaSuccess)                                                                 \
        {                                                                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                     \
                    cudaGetErrorString(err__));                                                   \
            return -1;                                                                            \
        }                                                                                         \
    } while (0)
#endif






__device__ __forceinline__ uint32_t hash_u32(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ uint32_t deterministic_neighbor(uint32_t node, uint32_t k, uint32_t N)
{
    if (N <= 1)
        return INVALID_ID;

    uint32_t h = hash_u32(node * 0x9e3779b1u + k * 0x85ebca6bu + 17u);
    uint32_t v = h % N;

    if (v == node)
        v = (v + 1u) % N;

    return v;
}

static size_t warp_candidate_smem_bytes(uint32_t warps_per_block,uint32_t L)
{
    size_t bytes = ((size_t)warps_per_block * L * sizeof(uint32_t)) +
                   ((size_t)warps_per_block * L * sizeof(float)) +
                   ((size_t)warps_per_block * L * sizeof(uint8_t));
    return ((bytes + 255) / 256) * 256;
}

static float gpu_vamana_alpha_for_iter(uint32_t iter,uint32_t iters)
{
#if USE_ALPHA_FROM_FIRST_ITER
    (void)iter;
    (void)iters;
    return (float)VAMANA_ALPHA;
#elif defined(VAMANA_ALPHA_FINAL_FROM_ITER)
    (void)iters;
    return iter >= (uint32_t)VAMANA_ALPHA_FINAL_FROM_ITER ? (float)VAMANA_ALPHA : 1.0f;
#else
    return (iter + 1 >= iters) ? (float)VAMANA_ALPHA : 1.0f;
#endif
}

__device__ __forceinline__ float warp_l2_distance_scalar_u8(const uint8_t *__restrict__ data,uint32_t dim,uint32_t a,uint32_t b,uint32_t lane)
{
    float sum = 0.0f;

    const uint8_t *__restrict__ x = data + (size_t)a * dim;
    const uint8_t *__restrict__ y = data + (size_t)b * dim;

    for (uint32_t d = lane; d < dim; d += WARP_SIZE)
    {
        int diff = (int)x[d] - (int)y[d];
        sum += (float)(diff * diff);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
    {
        sum += __shfl_down_sync(0xFFFFFFFF,sum,offset);
    }

    sum = __shfl_sync(0xFFFFFFFF,sum,0);
    return sum;
}

__device__ __forceinline__ float warp_l2_distance_uchar4(const uint8_t *__restrict__ data,uint32_t dim,uint32_t a,uint32_t b,uint32_t lane)
{
    float sum = 0.0f;

    const uchar4 *__restrict__ x4 = reinterpret_cast<const uchar4 *>(data + (size_t)a * dim);
    const uchar4 *__restrict__ y4 = reinterpret_cast<const uchar4 *>(data + (size_t)b * dim);

    uint32_t dim4 = dim >> 2;

    for (uint32_t d = lane; d < dim4; d += WARP_SIZE)
    {
        uchar4 xv = x4[d];
        uchar4 yv = y4[d];

        int dx = (int)xv.x - (int)yv.x;
        int dy = (int)xv.y - (int)yv.y;
        int dz = (int)xv.z - (int)yv.z;
        int dw = (int)xv.w - (int)yv.w;

        sum += (float)(dx * dx + dy * dy + dz * dz + dw * dw);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
    {
        sum += __shfl_down_sync(0xFFFFFFFF,sum,offset);
    }

    sum = __shfl_sync(0xFFFFFFFF,sum,0);
    return sum;
}

__device__ __forceinline__ float warp_l2_distance_dp4a(const uint8_t *__restrict__ data,uint32_t dim,uint32_t a,uint32_t b,uint32_t lane)
{
    unsigned int xx_yy_sum = 0;
    unsigned int xy_sum = 0;
    const uint32_t *__restrict__ x4 = reinterpret_cast<const uint32_t *>(data + (size_t)a * dim);
    const uint32_t *__restrict__ y4 = reinterpret_cast<const uint32_t *>(data + (size_t)b * dim);
    uint32_t dim4 = dim >> 2;

    for (uint32_t d = lane; d < dim4; d += WARP_SIZE)
    {
        uint32_t xv = x4[d];
        uint32_t yv = y4[d];
        xx_yy_sum = __dp4a(xv,xv,xx_yy_sum);
        xx_yy_sum = __dp4a(yv,yv,xx_yy_sum);
        xy_sum = __dp4a(xv,yv,xy_sum);
    }

    int sum = (int)xx_yy_sum - 2 * (int)xy_sum;
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF,sum,offset);

    return (float)__shfl_sync(0xFFFFFFFF,sum,0);
}

__device__ __forceinline__ float warp_l2_distance_uint4_tile8(const uint8_t *__restrict__ data,uint32_t a,uint32_t b,uint32_t lane)
{
    float sum = 0.0f;
    if (lane < 8)
    {
        const uint4 *__restrict__ x4 = reinterpret_cast<const uint4 *>(data + (size_t)a * 128);
        const uint4 *__restrict__ y4 = reinterpret_cast<const uint4 *>(data + (size_t)b * 128);
        uint4 xv = x4[lane];
        uint4 yv = y4[lane];

        const uint8_t *xb = reinterpret_cast<const uint8_t *>(&xv);
        const uint8_t *yb = reinterpret_cast<const uint8_t *>(&yv);
#pragma unroll
        for (uint32_t i = 0; i < 16; i++)
        {
            int diff = (int)xb[i] - (int)yb[i];
            sum += (float)(diff * diff);
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF,sum,offset);

    return __shfl_sync(0xFFFFFFFF,sum,0);
}

__device__ __forceinline__ float warp_l2_distance(const uint8_t *__restrict__ data,uint32_t dim,uint32_t a,uint32_t b,uint32_t lane)
{
    #if ENABLE_UINT4_DISTANCE_TILE
    if (dim == 128)
        return warp_l2_distance_uint4_tile8(data,a,b,lane);
    #endif

    #if ENABLE_DP4A_DISTANCE
    if ((dim & 3u) == 0)
        return warp_l2_distance_dp4a(data,dim,a,b,lane);
    #endif

    if ((dim & 3u) == 0)
        return warp_l2_distance_uchar4(data,dim,a,b,lane);

    return warp_l2_distance_scalar_u8(data,dim,a,b,lane);
}

__device__ __forceinline__ float warp_l2_distance_float4(const float *__restrict__ data,uint32_t dim,uint32_t a,uint32_t b,uint32_t lane)
{
    float sum = 0.0f;

    const float4 *__restrict__ x4 = reinterpret_cast<const float4 *>(data + (size_t)a * dim);
    const float4 *__restrict__ y4 = reinterpret_cast<const float4 *>(data + (size_t)b * dim);

    uint32_t dim4 = dim >> 2;

    for (uint32_t d = lane; d < dim4; d += WARP_SIZE)
    {
        float4 xv = x4[d];
        float4 yv = y4[d];

        float dx = xv.x - yv.x;
        float dy = xv.y - yv.y;
        float dz = xv.z - yv.z;
        float dw = xv.w - yv.w;

        sum += dx * dx + dy * dy + dz * dz + dw * dw;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF,sum,offset);

    return __shfl_sync(0xFFFFFFFF,sum,0);
}

__device__ __forceinline__ float warp_l2_distance_scalar_float(const float *__restrict__ data,uint32_t dim,uint32_t a,uint32_t b,uint32_t lane)
{
    float sum = 0.0f;

    const float *__restrict__ x = data + (size_t)a * dim;
    const float *__restrict__ y = data + (size_t)b * dim;

    for (uint32_t d = lane; d < dim; d += WARP_SIZE)
    {
        float diff = x[d] - y[d];
        sum += diff * diff;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF,sum,offset);

    return __shfl_sync(0xFFFFFFFF,sum,0);
}

__device__ __forceinline__ float warp_l2_distance(const float *__restrict__ data,uint32_t dim,uint32_t a,uint32_t b,uint32_t lane)
{
    if ((dim & 3u) == 0)
        return warp_l2_distance_float4(data,dim,a,b,lane);

    return warp_l2_distance_scalar_float(data,dim,a,b,lane);
}

__device__ __forceinline__ float warp_l2_distance_half2(const __half *__restrict__ data,uint32_t dim,uint32_t a,uint32_t b,uint32_t lane)
{
    float sum = 0.0f;
    const half2 *__restrict__ x2 = reinterpret_cast<const half2 *>(data + (size_t)a * dim);
    const half2 *__restrict__ y2 = reinterpret_cast<const half2 *>(data + (size_t)b * dim);
    uint32_t dim2 = dim >> 1;

    for (uint32_t d = lane; d < dim2; d += WARP_SIZE)
    {
        half2 diff = __hsub2(x2[d], y2[d]);
        half2 prod = __hmul2(diff, diff);
        float2 v = __half22float2(prod);
        sum += v.x + v.y;
    }

    if ((dim & 1u) != 0 && lane == 0)
    {
        float dx = __half2float(data[(size_t)a * dim + dim - 1]);
        float dy = __half2float(data[(size_t)b * dim + dim - 1]);
        float diff = dx - dy;
        sum += diff * diff;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF,sum,offset);

    return __shfl_sync(0xFFFFFFFF,sum,0);
}

__device__ __forceinline__ float warp_l2_distance(const __half *__restrict__ data,uint32_t dim,uint32_t a,uint32_t b,uint32_t lane)
{
    return warp_l2_distance_half2(data,dim,a,b,lane);
}

template <int CacheDim, typename DataT>
__device__ __forceinline__ void init_node_vector_cache(const DataT *__restrict__,uint32_t,uint32_t,uint32_t,half2 *,uint32_t *cache_kind)
{
    *cache_kind = 0;
}

template <int CacheDim>
__device__ __forceinline__ void init_node_vector_cache(const uint8_t *__restrict__ data,uint32_t dim,uint32_t node,uint32_t lane,half2 *,uint32_t *cache_kind)
{
    (void)data;
    (void)dim;
    (void)node;
    (void)lane;
    (void)CacheDim;
    *cache_kind = 0;
}

template <int CacheDim>
__device__ __forceinline__ void init_node_vector_cache(const __half *__restrict__ data,uint32_t dim,uint32_t node,uint32_t lane,half2 *half_cache,uint32_t *cache_kind)
{
    if constexpr (CacheDim <= 0)
    {
        *cache_kind = 0;
    }
    else
    {
#if ENABLE_NODE_VECTOR_REG_CACHE
        if (dim == (uint32_t)CacheDim && (dim & 1u) == 0)
        {
            constexpr uint32_t slots = (uint32_t)NODE_VECTOR_CACHE_HALF2_SLOTS(CacheDim);
            const uint32_t dim2 = (uint32_t)CacheDim >> 1;
            const half2 *__restrict__ x2 = reinterpret_cast<const half2 *>(data + (size_t)node * dim);
#pragma unroll
            for (uint32_t t = 0; t < slots; t++)
            {
                uint32_t idx = lane + t * WARP_SIZE;
                half_cache[t] = (idx < dim2) ? x2[idx] : __float2half2_rn(0.0f);
            }
            *cache_kind = 1;
            return;
        }
#endif
        *cache_kind = 0;
    }
}

template <int CacheDim, typename DataT>
__device__ __forceinline__ float warp_l2_distance_from_node_cache(const DataT *__restrict__ data,uint32_t dim,uint32_t node,uint32_t other,uint32_t lane,const half2 *,uint32_t)
{
    return warp_l2_distance(data,dim,node,other,lane);
}

template <int CacheDim>
__device__ __forceinline__ float warp_l2_distance_from_node_cache(const uint8_t *__restrict__ data,uint32_t dim,uint32_t node,uint32_t other,uint32_t lane,const half2 *,uint32_t)
{
    return warp_l2_distance(data,dim,node,other,lane);
}

template <int CacheDim>
__device__ __forceinline__ float warp_l2_distance_from_node_cache(const __half *__restrict__ data,uint32_t dim,uint32_t node,uint32_t other,uint32_t lane,const half2 *half_cache,uint32_t cache_kind)
{
    if constexpr (CacheDim <= 0)
    {
        return warp_l2_distance(data,dim,node,other,lane);
    }
    else
    {
        if (cache_kind != 1 || dim != (uint32_t)CacheDim || (dim & 1u) != 0)
            return warp_l2_distance(data,dim,node,other,lane);

        float sum = 0.0f;
        const half2 *__restrict__ y2 = reinterpret_cast<const half2 *>(data + (size_t)other * dim);
        constexpr uint32_t slots = (uint32_t)NODE_VECTOR_CACHE_HALF2_SLOTS(CacheDim);
        constexpr uint32_t dim2 = (uint32_t)CacheDim >> 1;

#pragma unroll
        for (uint32_t t = 0; t < slots; t++)
        {
            uint32_t idx = lane + t * WARP_SIZE;
            if (idx < dim2)
            {
                half2 diff = __hsub2(half_cache[t],y2[idx]);
                half2 prod = __hmul2(diff,diff);
                float2 v = __half22float2(prod);
                sum += v.x + v.y;
            }
        }

        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xFFFFFFFF,sum,offset);

        return __shfl_sync(0xFFFFFFFF,sum,0);
    }
}

__device__ __forceinline__ int find_id_pos_size_warp(uint32_t id,uint32_t size,const uint32_t *__restrict__ ids,uint32_t lane)
{
    for (uint32_t start = 0; start < size; start += WARP_SIZE)
    {
        uint32_t pos = start + lane;

        bool active = pos < size;
        bool match = active && (ids[pos] == id);

        unsigned int mask = __ballot_sync(0xFFFFFFFF,match);

        if (mask != 0)
        {
            int hit_lane = __ffs(mask) - 1;
            return (int)(start + hit_lane);
        }
    }

    return -1;
}

__device__ __forceinline__ void get_warp_candidate_smem(uint8_t *smem,uint32_t L,uint32_t warp_in_block,uint32_t warps_per_block,uint32_t **ids_out,float **dists_out,uint8_t **expanded_out)
{
    uint32_t *ids_base = reinterpret_cast<uint32_t *>(smem);
    float *dists_base = reinterpret_cast<float *>(ids_base + (size_t)warps_per_block * L);
    uint8_t *expanded_base = reinterpret_cast<uint8_t *>(dists_base + (size_t)warps_per_block * L);

    *ids_out = ids_base + (size_t)warp_in_block * L;
    *dists_out = dists_base + (size_t)warp_in_block * L;
    *expanded_out = expanded_base + (size_t)warp_in_block * L;
}

__device__ __forceinline__ void recompute_max_pos_size_warp(uint32_t size,const uint32_t *__restrict__ ids,const float *__restrict__ dists,int *max_pos_out,float *max_dist_out,uint32_t lane)
{
    float local_max = -FLT_MAX;
    int local_pos = -1;

    for (uint32_t start = 0; start < size; start += WARP_SIZE)
    {
        uint32_t pos = start + lane;

        if (pos < size && ids[pos] != INVALID_ID)
        {
            float v = dists[pos];

            if (v > local_max)
            {
                local_max = v;
                local_pos = (int)pos;
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
    {
        float other_val = __shfl_down_sync(0xFFFFFFFF,local_max,offset);
        int other_pos = __shfl_down_sync(0xFFFFFFFF,local_pos,offset);

        if (other_val > local_max || (other_val == local_max && other_pos >= 0 && (local_pos < 0 || other_pos < local_pos)))
        {
            local_max = other_val;
            local_pos = other_pos;
        }
    }

    local_max = __shfl_sync(0xFFFFFFFF,local_max,0);
    local_pos = __shfl_sync(0xFFFFFFFF,local_pos,0);

    if (lane == 0)
    {
        *max_pos_out = local_pos;
        *max_dist_out = local_max;
    }

    *max_pos_out = __shfl_sync(0xFFFFFFFF,*max_pos_out,0);
    *max_dist_out = __shfl_sync(0xFFFFFFFF,*max_dist_out,0);
}

__device__ __forceinline__ bool insert_top_l_state_warp(uint32_t id,float dist,uint32_t L,uint32_t *ids,float *dists,uint8_t *expanded,uint32_t *cand_size,int *current_max_pos,float *current_max_dist,uint32_t lane)
{
    if (id == INVALID_ID)
        return false;

    uint32_t size = __shfl_sync(0xFFFFFFFF,*cand_size,0);
    int max_pos = __shfl_sync(0xFFFFFFFF,*current_max_pos,0);
    float max_dist = __shfl_sync(0xFFFFFFFF,*current_max_dist,0);

#if ENABLE_TOPL_EARLY_REJECT
    if (size >= L && max_pos >= 0 && dist >= max_dist)
        return false;
#endif

    int same_pos = find_id_pos_size_warp(id,size,ids,lane);

    int changed = 0;
    int need_recompute_max = 0;

    if (same_pos >= 0)
    {
        if (lane == 0)
        {
            if (dist < dists[same_pos])
            {
                dists[same_pos] = dist;
                expanded[same_pos] = 0;
                changed = 1;

                if (same_pos == max_pos)
                    need_recompute_max = 1;
            }
        }

        changed = __shfl_sync(0xFFFFFFFF,changed,0);
        need_recompute_max = __shfl_sync(0xFFFFFFFF,need_recompute_max,0);

        if (need_recompute_max)
        {
            recompute_max_pos_size_warp(size,ids,dists,current_max_pos,current_max_dist,lane);
        }

        return changed != 0;
    }

    if (size < L)
    {
        if (lane == 0)
        {
            uint32_t pos = size;

            ids[pos] = id;
            dists[pos] = dist;
            expanded[pos] = 0;

            size++;

            *cand_size = size;

            if (max_pos < 0 || dist > max_dist)
            {
                *current_max_pos = (int)pos;
                *current_max_dist = dist;
            }

            changed = 1;
        }

        *cand_size = __shfl_sync(0xFFFFFFFF,*cand_size,0);
        *current_max_pos = __shfl_sync(0xFFFFFFFF,*current_max_pos,0);
        *current_max_dist = __shfl_sync(0xFFFFFFFF,*current_max_dist,0);

        changed = __shfl_sync(0xFFFFFFFF,changed,0);
        return changed != 0;
    }

    if (max_pos >= 0 && dist < max_dist)
    {
        if (lane == 0)
        {
            ids[max_pos] = id;
            dists[max_pos] = dist;
            expanded[max_pos] = 0;
            changed = 1;
        }

        changed = __shfl_sync(0xFFFFFFFF,changed,0);

        recompute_max_pos_size_warp(size,ids,dists,current_max_pos,current_max_dist,lane);

        return changed != 0;
    }

    return false;
}

__device__ __forceinline__ int select_next_unexpanded_size_warp(uint32_t size,const uint32_t *__restrict__ ids,const float *__restrict__ dists,const uint8_t *__restrict__ expanded,float *best_dist_out,uint32_t lane)
{
    float local_min = FLT_MAX;
    int local_pos = -1;

    for (uint32_t start = 0; start < size; start += WARP_SIZE)
    {
        uint32_t pos = start + lane;

        if (pos < size && ids[pos] != INVALID_ID && expanded[pos] == 0)
        {
            float v = dists[pos];

            if (v < local_min)
            {
                local_min = v;
                local_pos = (int)pos;
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
    {
        float other_val = __shfl_down_sync(0xFFFFFFFF,local_min,offset);
        int other_pos = __shfl_down_sync(0xFFFFFFFF,local_pos,offset);

        if (other_val < local_min || (other_val == local_min && other_pos >= 0 && (local_pos < 0 || other_pos < local_pos)))
        {
            local_min = other_val;
            local_pos = other_pos;
        }
    }

    local_min = __shfl_sync(0xFFFFFFFF,local_min,0);
    local_pos = __shfl_sync(0xFFFFFFFF,local_pos,0);

    if (best_dist_out != nullptr)
        *best_dist_out = local_min;

    return local_pos;
}

__device__ __forceinline__ int find_id_pos_R_warp(const uint32_t *__restrict__ row,uint32_t deg,uint32_t R,uint32_t x,uint32_t lane)
{
    if (deg > R)
        deg = R;

    for (uint32_t start = 0; start < deg; start += WARP_SIZE)
    {
        uint32_t pos = start + lane;

        bool active = pos < deg;
        bool match = active && (row[pos] == x);

        unsigned int mask = __ballot_sync(0xFFFFFFFF, match);

        if (mask != 0)
        {
            int hit_lane = __ffs(mask) - 1;
            return (int)(start + hit_lane);
        }
    }
    return -1;
}

template <typename DataT>
__device__ __forceinline__ void dynamic_robust_insert_C_warp(const DataT *__restrict__ data,uint32_t N,uint32_t dim,uint32_t C,uint32_t node,uint32_t x,float d_px,float alpha,uint32_t *__restrict__ graph_next,float *__restrict__ graph_dists,uint32_t *__restrict__ degree_next,uint32_t lane)
{
    if (x == INVALID_ID || x >= N || x == node)
        return;

    const size_t base = (size_t)node * C;

    uint32_t *row = graph_next + base;
    float *dist_row = graph_dists + base;

    uint32_t deg = degree_next[node];

    if (deg > C)
        deg = C;

    int same_pos = find_id_pos_R_warp(row,deg,C,x,lane);

    if (same_pos >= 0)
        return;

    int occluded = 0;

    for (uint32_t i = 0; i < deg; i++)
    {
        uint32_t y = row[i];

        if (y == INVALID_ID || y >= N || y == node)
            continue;

        float d_py = dist_row[i];

        if (d_py <= d_px)
        {
            float d_yx = warp_l2_distance(data,dim,y,x,lane);

            if (alpha * d_yx <= d_px)
                occluded = 1;
        }

        if (occluded)
            break;
    }

    occluded = __shfl_sync(0xFFFFFFFF,occluded,0);

    if (occluded)
        return;

    int worst_pos = -1;
    float worst_dist = -FLT_MAX;

#if !ENABLE_FAST_ONLINE_REPLACE
    int victim_pos = -1;
    float victim_dist = -FLT_MAX;
#endif

    for (uint32_t i = 0; i < deg; i++)
    {
        uint32_t y = row[i];

        if (y == INVALID_ID || y >= N || y == node)
            continue;

        float d_py = dist_row[i];

        if (d_py > worst_dist)
        {
            worst_dist = d_py;
            worst_pos = (int)i;
        }

#if !ENABLE_FAST_ONLINE_REPLACE
        if (d_py > d_px)
        {
            float d_xy = warp_l2_distance(data,dim,x,y,lane);

            if (alpha * d_xy <= d_py)
            {
                if (d_py > victim_dist)
                {
                    victim_dist = d_py;
                    victim_pos = (int)i;
                }
            }
        }
#endif
    }

#if !ENABLE_FAST_ONLINE_REPLACE
    victim_pos = __shfl_sync(0xFFFFFFFF,victim_pos,0);
#endif
    worst_pos = __shfl_sync(0xFFFFFFFF,worst_pos,0);
    worst_dist = __shfl_sync(0xFFFFFFFF,worst_dist,0);

    if (lane == 0)
    {
#if !ENABLE_FAST_ONLINE_REPLACE
        if (victim_pos >= 0)
        {
            row[victim_pos] = x;
            dist_row[victim_pos] = d_px;
        }
        else if (deg < C)
#else
        if (deg < C)
#endif
        {
            row[deg] = x;
            dist_row[deg] = d_px;
            degree_next[node] = deg + 1;
        }
        else if (worst_pos >= 0 && d_px < worst_dist)
        {
            row[worst_pos] = x;
            dist_row[worst_pos] = d_px;
        }
    }
}

template <typename DataT>
__device__ __forceinline__ void dynamic_fast_replace_insert_C_warp(const DataT *__restrict__ data,uint32_t N,uint32_t dim,uint32_t C,uint32_t node,uint32_t x,float d_px,float alpha,uint32_t *__restrict__ graph_next,float *__restrict__ graph_dists,uint32_t *__restrict__ degree_next,uint32_t lane)
{
    if (x == INVALID_ID || x >= N || x == node)
        return;

    const size_t base = (size_t)node * C;
    uint32_t *row = graph_next + base;
    float *dist_row = graph_dists + base;

    uint32_t deg = degree_next[node];
    if (deg > C)
        deg = C;

    if (find_id_pos_R_warp(row,deg,C,x,lane) >= 0)
        return;

    int occluded = 0;

    for (uint32_t i = 0; i < deg; i++)
    {
        uint32_t y = row[i];
        if (y == INVALID_ID || y >= N || y == node)
            continue;

        float d_py = dist_row[i];
        if (d_py <= d_px)
        {
            float d_yx = warp_l2_distance(data,dim,y,x,lane);
            if (alpha * d_yx <= d_px)
                occluded = 1;
        }

        if (occluded)
            break;
    }

    occluded = __shfl_sync(0xFFFFFFFF,occluded,0);
    if (occluded)
        return;

    int worst_pos = -1;
    float worst_dist = -FLT_MAX;
    for (uint32_t i = lane; i < deg; i += WARP_SIZE)
    {
        uint32_t y = row[i];
        float d_py = dist_row[i];
        if (y != INVALID_ID && y < N && y != node && d_py > worst_dist)
        {
            worst_dist = d_py;
            worst_pos = (int)i;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
    {
        float other_dist = __shfl_down_sync(0xFFFFFFFF,worst_dist,offset);
        int other_pos = __shfl_down_sync(0xFFFFFFFF,worst_pos,offset);
        if (other_dist > worst_dist || (other_dist == worst_dist && other_pos >= 0 && (worst_pos < 0 || other_pos < worst_pos)))
        {
            worst_dist = other_dist;
            worst_pos = other_pos;
        }
    }

    worst_pos = __shfl_sync(0xFFFFFFFF,worst_pos,0);
    worst_dist = __shfl_sync(0xFFFFFFFF,worst_dist,0);

    if (lane == 0)
    {
        if (deg < C)
        {
            row[deg] = x;
            dist_row[deg] = d_px;
            degree_next[node] = deg + 1;
        }
        else if (worst_pos >= 0 && d_px < worst_dist)
        {
            row[worst_pos] = x;
            dist_row[worst_pos] = d_px;
        }
    }
}

template <typename DataT>
__device__ __forceinline__ void dynamic_distance_only_insert_C_warp(const DataT *__restrict__ data,uint32_t N,uint32_t dim,uint32_t C,uint32_t node,uint32_t x,float d_px,float alpha,uint32_t *__restrict__ graph_next,float *__restrict__ graph_dists,uint32_t *__restrict__ degree_next,uint32_t lane)
{
    (void)data;
    (void)dim;
    (void)alpha;

    if (x == INVALID_ID || x >= N || x == node)
        return;

    const size_t base = (size_t)node * C;
    uint32_t *row = graph_next + base;
    float *dist_row = graph_dists + base;

    uint32_t deg = degree_next[node];
    if (deg > C)
        deg = C;

    if (find_id_pos_R_warp(row,deg,C,x,lane) >= 0)
        return;

    int worst_pos = -1;
    float worst_dist = -FLT_MAX;
    for (uint32_t i = lane; i < deg; i += WARP_SIZE)
    {
        uint32_t y = row[i];
        float d_py = dist_row[i];
        if (y != INVALID_ID && y < N && y != node && d_py > worst_dist)
        {
            worst_dist = d_py;
            worst_pos = (int)i;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
    {
        float other_dist = __shfl_down_sync(0xFFFFFFFF,worst_dist,offset);
        int other_pos = __shfl_down_sync(0xFFFFFFFF,worst_pos,offset);
        if (other_dist > worst_dist || (other_dist == worst_dist && other_pos >= 0 && (worst_pos < 0 || other_pos < worst_pos)))
        {
            worst_dist = other_dist;
            worst_pos = other_pos;
        }
    }

    worst_pos = __shfl_sync(0xFFFFFFFF,worst_pos,0);
    worst_dist = __shfl_sync(0xFFFFFFFF,worst_dist,0);

    if (lane == 0)
    {
        if (deg < C)
        {
            row[deg] = x;
            dist_row[deg] = d_px;
            degree_next[node] = deg + 1;
        }
        else if (worst_pos >= 0 && d_px < worst_dist)
        {
            row[worst_pos] = x;
            dist_row[worst_pos] = d_px;
        }
    }
}






template <typename DataT>
__global__ void medoid_score_kernel_warp(const DataT *__restrict__ data,uint32_t N,uint32_t dim,uint32_t num_candidates,uint32_t num_samples,float *__restrict__ scores,uint32_t *__restrict__ candidate_ids)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t warp_id = tid / WARP_SIZE;
    uint32_t lane = tid % WARP_SIZE;

    if (warp_id >= num_candidates || N == 0)
        return;

    uint32_t candidate = hash_u32(warp_id * 9973u + 17u) % N;

    if (lane == 0)
        candidate_ids[warp_id] = candidate;

    float sum = 0.0f;

    for (uint32_t s = 0; s < num_samples; s++)
    {
        uint32_t sample = hash_u32(s * 10007u + 131u) % N;

        if (sample == candidate)
            continue;

        float d = warp_l2_distance(data, dim, candidate, sample, lane);

        if (lane == 0)
            sum += d;
    }

    if (lane == 0)
        scores[warp_id] = sum / (float)num_samples;
}

template <typename DataT>
__global__ void init_random_graph(const DataT *__restrict__ data,uint32_t dim,uint32_t *__restrict__ graph_cur,uint32_t *__restrict__ degree_cur,uint32_t *__restrict__ graph_work,uint32_t *__restrict__ degree_work,float *__restrict__ graph_dists,uint32_t N,uint32_t R,uint32_t C)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t warp_id = tid / WARP_SIZE;
    uint32_t lane = tid % WARP_SIZE;
    uint32_t node = warp_id;

    if (node >= N)
        return;

    const size_t cur_base = (size_t)node * R;
    const size_t work_base = (size_t)node * C;

    for (uint32_t i = lane; i < C; i += WARP_SIZE)
    {
        graph_work[work_base + i] = INVALID_ID;
        graph_dists[work_base + i] = FLT_MAX;
        if (i < R)
        {
            graph_cur[cur_base + i] = INVALID_ID;
        }
    }


    for (uint32_t i = 0; i < R; i++)
    {
        uint32_t nb = deterministic_neighbor(node,i,N);

        if (N <= 1 || nb == INVALID_ID || nb >= N || nb == node)
            nb = INVALID_ID;

        float dist = FLT_MAX;

        if (nb != INVALID_ID)
            dist = warp_l2_distance(data,dim,node,nb,lane);

        if (lane == 0)
        {
            graph_cur[cur_base + i] = nb;
            graph_work[work_base + i] = nb;
            graph_dists[work_base + i] = dist;
        }
    }

    if (lane == 0)
    {
        uint32_t deg = (N > 1) ? R : 0;
        degree_cur[node] = deg;
        degree_work[node] = deg;
    }
}

template <typename DataT, int NodeCacheDim = 0>
__global__ void search_prune_kernel_shared(const DataT *__restrict__ data,uint32_t N,uint32_t dim,uint32_t R,uint32_t L,uint32_t C,uint32_t STEPS,uint32_t online_insert_steps,uint32_t online_fast_replace,uint32_t start_id,const uint32_t *__restrict__ entry_points,uint32_t num_entry_points,float alpha,const uint8_t *__restrict__ active_nodes,const uint8_t *__restrict__ important_nodes,uint32_t important_extra_steps,const uint32_t *__restrict__ graph_cur,const uint32_t *__restrict__ degree_cur,uint32_t *__restrict__ graph_work,uint32_t *__restrict__ degree_work,float *__restrict__ graph_dists
#if ENABLE_SEARCH_PRUNE_TIMING
                                      ,unsigned long long *__restrict__ timing_cycles
#endif
                                      )
{
    extern __shared__ uint8_t smem[];

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t warp_id = tid / WARP_SIZE;
    uint32_t lane = tid % WARP_SIZE;
    uint32_t warp_in_block = threadIdx.x / WARP_SIZE;

    uint32_t node = warp_id;

    if (node >= N)
        return;

    half2 node_half_cache[NODE_VECTOR_CACHE_HALF2_SLOTS(NodeCacheDim)];
    uint32_t node_cache_kind = 0;
    init_node_vector_cache<NodeCacheDim>(data,dim,node,lane,node_half_cache,&node_cache_kind);

#if ENABLE_SEARCH_PRUNE_TIMING
    unsigned long long timing_select = 0;
    unsigned long long timing_dynamic_insert = 0;
    unsigned long long timing_expand_candidates = 0;
    unsigned long long timing_initial_candidates = 0;
    unsigned long long timing_dynamic_calls = 0;
    unsigned long long timing_expand_candidates_count = 0;
    unsigned long long timing_expand_distance = 0;
    unsigned long long timing_expand_insert = 0;
#endif

    uint32_t *cand_ids = nullptr;
    float *cand_dists = nullptr;
    uint8_t *cand_expanded = nullptr;

    get_warp_candidate_smem(smem,L,warp_in_block,(uint32_t)WARPS_PER_BLOCK,&cand_ids,&cand_dists,&cand_expanded);

    const size_t cur_base = (size_t)node * R;
    const size_t work_base = (size_t)node * C;

    uint32_t cand_size = 0;
    int current_max_pos = -1;
    float current_max_dist = -FLT_MAX;

    for (uint32_t i = lane; i < L; i += WARP_SIZE)
    {
        cand_ids[i] = INVALID_ID;
        cand_dists[i] = FLT_MAX;
        cand_expanded[i] = 0;
    }

    __syncwarp();

    uint32_t deg_self = degree_cur[node];

    if (deg_self > R)
        deg_self = R;

    for (uint32_t i = lane; i < C; i += WARP_SIZE)
    {
        uint32_t nb = INVALID_ID;
        float dist = FLT_MAX;

        if (i < deg_self)
        {
            nb = graph_cur[cur_base + i];

            if (nb != INVALID_ID && nb < N && nb != node)
#if ENABLE_RECOMPUTE_CURRENT_DISTS
                dist = FLT_MAX;
#else
                dist = graph_dists[work_base + i];
#endif
            else
                nb = INVALID_ID;
        }

        graph_work[work_base + i] = nb;
        graph_dists[work_base + i] = dist;
    }

    if (lane == 0)
        degree_work[node] = deg_self;

    __syncwarp();

    if (active_nodes != NULL && active_nodes[node] == 0)
        return;

    uint32_t entry_count = num_entry_points;
    if (entry_count == 0 || entry_points == NULL)
        entry_count = 1;
    if (entry_count > (uint32_t)MAX_MULTI_ENTRY_POINTS)
        entry_count = (uint32_t)MAX_MULTI_ENTRY_POINTS;

    for (uint32_t ep = 0; ep < entry_count; ep++)
    {
        uint32_t entry = start_id;
        if (entry_points != NULL && ep < num_entry_points)
            entry = entry_points[ep];
        if (entry < N && entry != node)
        {
#if ENABLE_SEARCH_PRUNE_TIMING
            unsigned long long t0 = 0;
            if (lane == 0)
                t0 = clock64();
#endif
            float dist = warp_l2_distance_from_node_cache<NodeCacheDim>(data,dim,node,entry,lane,node_half_cache,node_cache_kind);
            insert_top_l_state_warp(entry,dist,L,cand_ids,cand_dists,cand_expanded,&cand_size,&current_max_pos,&current_max_dist,lane);
#if ENABLE_SEARCH_PRUNE_TIMING
            if (lane == 0)
                timing_initial_candidates += clock64() - t0;
#endif
        }
    }

    for (uint32_t start = 0; start < deg_self; start += WARP_SIZE)
    {
        uint32_t pos = start + lane;
        uint32_t local_nb = INVALID_ID;
        float local_dist = FLT_MAX;

        if (pos < deg_self)
        {
            local_nb = graph_cur[cur_base + pos];

            if (local_nb != INVALID_ID && local_nb < N && local_nb != node)
#if ENABLE_RECOMPUTE_CURRENT_DISTS
                local_dist = FLT_MAX;
#else
                local_dist = graph_dists[work_base + pos];
#endif
            else
                local_nb = INVALID_ID;
        }

        unsigned int valid_mask = __ballot_sync(0xFFFFFFFF,pos < deg_self && local_nb != INVALID_ID && local_nb < N && local_nb != node);

        while (valid_mask != 0)
        {
            int src_lane = __ffs(valid_mask) - 1;

            uint32_t nb = __shfl_sync(0xFFFFFFFF,local_nb,src_lane);
            uint32_t nb_pos = __shfl_sync(0xFFFFFFFF,pos,src_lane);
            float dist = __shfl_sync(0xFFFFFFFF,local_dist,src_lane);

#if ENABLE_RECOMPUTE_CURRENT_DISTS
            dist = warp_l2_distance_from_node_cache<NodeCacheDim>(data,dim,node,nb,lane,node_half_cache,node_cache_kind);
            if (lane == 0)
                graph_dists[work_base + nb_pos] = dist;
#endif

            insert_top_l_state_warp(nb,dist,L,cand_ids,cand_dists,cand_expanded,&cand_size,&current_max_pos,&current_max_dist,lane);

            valid_mask &= valid_mask - 1;
        }
    }

#pragma unroll
    for (uint32_t k = 0; k < INIT_RANDOM_SEEDS; k++)
    {
        uint32_t seed = deterministic_neighbor(node,k + 1024u,N);

        if (seed == INVALID_ID || seed >= N || seed == node)
            continue;

#if ENABLE_SEARCH_PRUNE_TIMING
        unsigned long long t0 = 0;
        if (lane == 0)
            t0 = clock64();
#endif
        float dist = warp_l2_distance_from_node_cache<NodeCacheDim>(data,dim,node,seed,lane,node_half_cache,node_cache_kind);

        insert_top_l_state_warp(seed,dist,L,cand_ids,cand_dists,cand_expanded,&cand_size,&current_max_pos,&current_max_dist,lane);
#if ENABLE_SEARCH_PRUNE_TIMING
        if (lane == 0)
            timing_initial_candidates += clock64() - t0;
#endif
    }

    uint32_t max_steps = STEPS;

    if (important_nodes != NULL && important_nodes[node] != 0)
        max_steps += important_extra_steps;

    if (max_steps > L)
        max_steps = L;

    uint32_t stagnation_seed_rounds = 0;
    (void)stagnation_seed_rounds;

    for (uint32_t step = 0; step < max_steps; step++)
    {
        float expand_dist = FLT_MAX;

        uint32_t size_now = __shfl_sync(0xFFFFFFFF,cand_size,0);

#if ENABLE_SEARCH_PRUNE_TIMING
        unsigned long long t_select = 0;
        if (lane == 0)
            t_select = clock64();
#endif
        int expand_pos = select_next_unexpanded_size_warp(size_now,cand_ids,cand_dists,cand_expanded,&expand_dist,lane);
#if ENABLE_SEARCH_PRUNE_TIMING
        if (lane == 0)
            timing_select += clock64() - t_select;
#endif

        uint32_t expand_id = INVALID_ID;
        int has_node = 0;

        if (lane == 0)
        {
            if (expand_pos >= 0)
            {
                expand_id = cand_ids[expand_pos];
                expand_dist = cand_dists[expand_pos];
                cand_expanded[expand_pos] = 1;
                has_node = 1;
            }
        }

        expand_id = __shfl_sync(0xFFFFFFFF,expand_id,0);
        expand_dist = __shfl_sync(0xFFFFFFFF,expand_dist,0);
        has_node = __shfl_sync(0xFFFFFFFF,has_node,0);

        if (!has_node || expand_id == INVALID_ID || expand_id >= N)
            break;

        {
#if ENABLE_SEARCH_PRUNE_TIMING
            unsigned long long t_dynamic = 0;
            if (lane == 0)
                t_dynamic = clock64();
#endif
            if (online_fast_replace != 0)
                dynamic_fast_replace_insert_C_warp(data,N,dim,C,node,expand_id,expand_dist,alpha,graph_work,graph_dists,degree_work,lane);
            else
                dynamic_robust_insert_C_warp(data,N,dim,C,node,expand_id,expand_dist,alpha,graph_work,graph_dists,degree_work,lane);
#if ENABLE_SEARCH_PRUNE_TIMING
            if (lane == 0)
            {
                timing_dynamic_insert += clock64() - t_dynamic;
                timing_dynamic_calls++;
            }
#endif
        }

        uint32_t deg = degree_cur[expand_id];

        if (deg > R)
            deg = R;

        const size_t expand_base = (size_t)expand_id * R;

        for (uint32_t start = 0; start < deg; start += WARP_SIZE)
        {
            uint32_t pos = start + lane;
            uint32_t local_cand = INVALID_ID;

            if (pos < deg)
                local_cand = graph_cur[expand_base + pos];

            unsigned int valid_mask = __ballot_sync(0xFFFFFFFF,pos < deg && local_cand != INVALID_ID && local_cand < N && local_cand != node);

            while (valid_mask != 0)
            {
                int src_lane = __ffs(valid_mask) - 1;

                uint32_t cand = __shfl_sync(0xFFFFFFFF,local_cand,src_lane);

#if ENABLE_SEARCH_PRUNE_TIMING
                unsigned long long t_expand = 0;
                if (lane == 0)
                    t_expand = clock64();
#endif
                float dist = warp_l2_distance_from_node_cache<NodeCacheDim>(data,dim,node,cand,lane,node_half_cache,node_cache_kind);
#if ENABLE_SEARCH_PRUNE_TIMING
                unsigned long long t_insert = 0;
                if (lane == 0)
                {
                    unsigned long long now = clock64();
                    timing_expand_distance += now - t_expand;
                    t_insert = now;
                }
#endif

                insert_top_l_state_warp(cand,dist,L,cand_ids,cand_dists,cand_expanded,&cand_size,&current_max_pos,&current_max_dist,lane);
#if ENABLE_SEARCH_PRUNE_TIMING
                if (lane == 0)
                {
                    unsigned long long now = clock64();
                    timing_expand_insert += now - t_insert;
                    timing_expand_candidates += now - t_expand;
                    timing_expand_candidates_count++;
                }
#endif

                valid_mask &= valid_mask - 1;
            }
        }
    }

#if ENABLE_SEARCH_PRUNE_TIMING
    if (lane == 0 && timing_cycles != NULL)
    {
        atomicAdd(&timing_cycles[0],timing_select);
        atomicAdd(&timing_cycles[1],timing_dynamic_insert);
        atomicAdd(&timing_cycles[2],timing_expand_candidates);
        atomicAdd(&timing_cycles[3],timing_initial_candidates);
        atomicAdd(&timing_cycles[4],timing_dynamic_calls);
        atomicAdd(&timing_cycles[5],timing_expand_candidates_count);
        atomicAdd(&timing_cycles[6],timing_expand_distance);
        atomicAdd(&timing_cycles[7],timing_expand_insert);
    }
#endif

}

__global__ void mark_important_nodes_kernel(uint32_t N,uint32_t R,uint32_t C,const uint32_t *__restrict__ degree_next,const uint32_t *__restrict__ reverse_pressure,uint8_t *__restrict__ important_nodes,uint32_t *__restrict__ important_count,uint32_t reverse_pressure_is_offsets,uint32_t low_degree_threshold,uint32_t reverse_indegree_threshold)
{
    uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= N)
        return;

    uint32_t degree = degree_next[node];
    if (degree > R)
        degree = R;

    uint32_t pressure = 0;
    if (reverse_pressure != NULL)
    {
        if (reverse_pressure_is_offsets != 0)
            pressure = reverse_pressure[node + 1] - reverse_pressure[node];
        else
            pressure = reverse_pressure[node];
    }

    bool important = degree < low_degree_threshold ||
                     pressure >= reverse_indegree_threshold;
    important_nodes[node] = important ? 1 : 0;
    if (important)
        atomicAdd(important_count, 1u);
}

__device__ __forceinline__ uint32_t selective_hq_hash_u32(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__global__ void mark_important_nodes_budgeted_kernel(uint32_t N,uint32_t R,uint32_t C,const uint32_t *__restrict__ degree_next,const uint32_t *__restrict__ reverse_pressure,uint8_t *__restrict__ important_nodes,uint32_t *__restrict__ important_count,uint32_t reverse_pressure_is_offsets,uint32_t low_degree_threshold,uint32_t reverse_indegree_threshold,uint32_t active_permille)
{
    uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= N)
        return;

    uint32_t degree = degree_next[node];
    if (degree > R)
        degree = R;

    uint32_t pressure = 0;
    if (reverse_pressure != NULL)
    {
        if (reverse_pressure_is_offsets != 0)
            pressure = reverse_pressure[node + 1] - reverse_pressure[node];
        else
            pressure = reverse_pressure[node];
    }

    bool low_degree = degree < low_degree_threshold;
    bool high_pressure = pressure >= reverse_indegree_threshold;
    if (active_permille > 1000)
        active_permille = 1000;

    uint32_t priority = 0;
    if (low_degree)
        priority += (low_degree_threshold - degree) * 32u;
    if (high_pressure)
    {
        uint32_t excess = pressure - reverse_indegree_threshold + 1u;
        if (excess > 512u)
            excess = 512u;
        priority += excess;
    }

    bool selected = false;
    if (priority > 0)
    {
        uint32_t h = selective_hq_hash_u32(node ^ (priority * 2654435761u));
        selected = (h % 1000u) < active_permille;
    }

    important_nodes[node] = selected ? 1 : 0;
    if (selected)
        atomicAdd(important_count, 1u);

    (void)C;
}

__global__ void init_touched_from_active_kernel(uint32_t N,const uint8_t *__restrict__ active_nodes,uint8_t *__restrict__ touched_nodes)
{
    uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= N)
        return;
    touched_nodes[node] = active_nodes[node];
}

__global__ void count_touched_rows_kernel(uint32_t N,const uint32_t *__restrict__ reverse_offsets,unsigned long long *__restrict__ count)
{
    uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node < N && reverse_offsets[node + 1] > reverse_offsets[node])
        atomicAdd(count, 1ULL);
}

__global__ void reverse_generate_kernel(uint32_t N,uint32_t R,uint32_t C,uint32_t reverse_capacity,const uint32_t *__restrict__ graph_next,const float *__restrict__ graph_dists,const uint32_t *__restrict__ degree_next,uint32_t *__restrict__ reverse_ids,float *__restrict__ reverse_dists,uint32_t *__restrict__ reverse_counts)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t warp_id = tid / WARP_SIZE;
    uint32_t lane = tid % WARP_SIZE;

    uint32_t src = warp_id;

    if (src >= N)
        return;

    uint32_t deg_src = degree_next[src];

    if (deg_src > R)
        deg_src = R;

    const size_t src_base = (size_t)src * C;

    for (uint32_t start = 0; start < deg_src; start += WARP_SIZE)
    {
        uint32_t src_pos = start + lane;
        uint32_t local_target = INVALID_ID;
        float local_dist = FLT_MAX;

        if (src_pos < deg_src)
        {
            local_target = graph_next[src_base + src_pos];
            local_dist = graph_dists[src_base + src_pos];
        }

        unsigned int valid_mask = __ballot_sync(0xFFFFFFFF,src_pos < deg_src && local_target != INVALID_ID && local_target < N && local_target != src);

        while (valid_mask != 0)
        {
            int src_lane = __ffs(valid_mask) - 1;

            uint32_t target = __shfl_sync(0xFFFFFFFF,local_target,src_lane);

            float dist_target_src = __shfl_sync(0xFFFFFFFF,local_dist,src_lane);

            uint32_t pos = 0;

            if (lane == 0)
            {
                pos = atomicAdd(&reverse_counts[target],1);
            }

            pos = __shfl_sync(0xFFFFFFFF,pos,0);

            if (pos < reverse_capacity)
            {
                if (lane == 0)
                {
                    size_t reverse_pos = (size_t)target * reverse_capacity + pos;
                    reverse_ids[reverse_pos] = src;
                    reverse_dists[reverse_pos] = dist_target_src;
                }
            }
            valid_mask &= valid_mask - 1;
        }
    }
}

__global__ void reverse_count_csr_kernel(uint32_t N,uint32_t R,uint32_t C,const uint32_t *__restrict__ graph_next,const uint32_t *__restrict__ degree_next,uint32_t *__restrict__ reverse_offsets)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t src = tid / WARP_SIZE;
    uint32_t lane = tid % WARP_SIZE;

    if (src >= N)
        return;

    uint32_t deg_src = degree_next[src];
    if (deg_src > R)
        deg_src = R;

    const size_t src_base = (size_t)src * C;

    for (uint32_t start = 0; start < deg_src; start += WARP_SIZE)
    {
        uint32_t src_pos = start + lane;
        uint32_t target = INVALID_ID;

        if (src_pos < deg_src)
            target = graph_next[src_base + src_pos];

        if (target != INVALID_ID && target < N && target != src)
            atomicAdd(&reverse_offsets[target + 1], 1u);
    }
}

__global__ void reverse_fill_csr_kernel(uint32_t N,uint32_t R,uint32_t C,const uint32_t *__restrict__ graph_next,const uint32_t *__restrict__ degree_next,uint32_t *__restrict__ reverse_cursor,uint32_t *__restrict__ reverse_ids)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t src = tid / WARP_SIZE;
    uint32_t lane = tid % WARP_SIZE;

    if (src >= N)
        return;

    uint32_t deg_src = degree_next[src];
    if (deg_src > R)
        deg_src = R;

    const size_t src_base = (size_t)src * C;

    for (uint32_t start = 0; start < deg_src; start += WARP_SIZE)
    {
        uint32_t src_pos = start + lane;
        uint32_t target = INVALID_ID;

        if (src_pos < deg_src)
            target = graph_next[src_base + src_pos];

        if (target != INVALID_ID && target < N && target != src)
        {
            uint32_t pos = atomicAdd(&reverse_cursor[target], 1u);
            reverse_ids[pos] = src;
        }
    }
}

__global__ void reverse_count_csr_active_kernel(uint32_t N,uint32_t R,uint32_t C,const uint32_t *__restrict__ graph_next,const uint32_t *__restrict__ degree_next,const uint8_t *__restrict__ active_sources,uint32_t *__restrict__ reverse_offsets)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t src = tid / WARP_SIZE;
    uint32_t lane = tid % WARP_SIZE;

    if (src >= N)
        return;
    if (active_sources != NULL && active_sources[src] == 0)
        return;

    uint32_t deg_src = degree_next[src];
    if (deg_src > R)
        deg_src = R;

    const size_t src_base = (size_t)src * C;
    for (uint32_t start = 0; start < deg_src; start += WARP_SIZE)
    {
        uint32_t src_pos = start + lane;
        uint32_t target = INVALID_ID;
        if (src_pos < deg_src)
            target = graph_next[src_base + src_pos];
        if (target != INVALID_ID && target < N && target != src)
            atomicAdd(&reverse_offsets[target + 1], 1u);
    }
}

__global__ void reverse_fill_csr_active_kernel(uint32_t N,uint32_t R,uint32_t C,const uint32_t *__restrict__ graph_next,const uint32_t *__restrict__ degree_next,const uint8_t *__restrict__ active_sources,uint32_t *__restrict__ reverse_cursor,uint32_t *__restrict__ reverse_ids,uint8_t *__restrict__ touched_nodes)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t src = tid / WARP_SIZE;
    uint32_t lane = tid % WARP_SIZE;

    if (src >= N)
        return;
    if (active_sources != NULL && active_sources[src] == 0)
        return;

    uint32_t deg_src = degree_next[src];
    if (deg_src > R)
        deg_src = R;

    const size_t src_base = (size_t)src * C;
    for (uint32_t start = 0; start < deg_src; start += WARP_SIZE)
    {
        uint32_t src_pos = start + lane;
        uint32_t target = INVALID_ID;
        if (src_pos < deg_src)
            target = graph_next[src_base + src_pos];
        if (target != INVALID_ID && target < N && target != src)
        {
            uint32_t pos = atomicAdd(&reverse_cursor[target], 1u);
            reverse_ids[pos] = src;
            if (touched_nodes != NULL)
                touched_nodes[target] = 1;
        }
    }
}

template <typename DataT>
__global__ void reverse_merge_csr_kernel(const DataT *__restrict__ data,uint32_t N,uint32_t dim,uint32_t C,float alpha,uint32_t *__restrict__ graph_next,float *__restrict__ graph_dists,uint32_t *__restrict__ degree_next,const uint32_t *__restrict__ reverse_offsets,const uint32_t *__restrict__ reverse_ids)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t target = tid / WARP_SIZE;
    uint32_t lane = tid % WARP_SIZE;

    if (target >= N)
        return;

    const size_t target_base = (size_t)target * C;
    uint32_t *row = graph_next + target_base;
    float *dist_row = graph_dists + target_base;
    uint32_t deg = degree_next[target];
    if (deg > C)
        deg = C;

    uint32_t begin = reverse_offsets[target];
    uint32_t end = reverse_offsets[target + 1];
    uint32_t indegree = end - begin;
    uint32_t process_count = indegree;
#if ENABLE_REVERSE_HEAVY_CAP
    if (indegree > (uint32_t)REVERSE_HEAVY_THRESHOLD)
    {
        process_count = (uint32_t)REVERSE_HEAVY_SAMPLE;
        if (process_count > indegree)
            process_count = indegree;
        if (process_count == 0)
            process_count = 1;
    }
#endif

    for (uint32_t sample_idx = 0; sample_idx < process_count; sample_idx++)
    {
        uint32_t i = begin + sample_idx;
#if ENABLE_REVERSE_HEAVY_CAP
        if (process_count < indegree)
            i = begin + (uint32_t)(((unsigned long long)sample_idx * (unsigned long long)indegree) / (unsigned long long)process_count);
#endif
        uint32_t src = reverse_ids[i];
        if (src == INVALID_ID || src >= N || src == target)
            continue;

        if (find_id_pos_R_warp(row,deg,C,src,lane) >= 0)
            continue;

        float dist = warp_l2_distance(data,dim,target,src,lane);
        if (deg < C)
        {
            if (lane == 0)
            {
                row[deg] = src;
                dist_row[deg] = dist;
            }
            deg++;
            __syncwarp();
            continue;
        }

        float local_worst_dist = -FLT_MAX;
        int local_worst_pos = -1;
        for (uint32_t p = lane; p < C; p += WARP_SIZE)
        {
            if (dist_row[p] > local_worst_dist)
            {
                local_worst_dist = dist_row[p];
                local_worst_pos = (int)p;
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            float other_dist = __shfl_down_sync(0xFFFFFFFF,local_worst_dist,offset);
            int other_pos = __shfl_down_sync(0xFFFFFFFF,local_worst_pos,offset);
            if (other_dist > local_worst_dist || (other_dist == local_worst_dist && other_pos >= 0 && (local_worst_pos < 0 || other_pos < local_worst_pos)))
            {
                local_worst_dist = other_dist;
                local_worst_pos = other_pos;
            }
        }
        local_worst_dist = __shfl_sync(0xFFFFFFFF,local_worst_dist,0);
        local_worst_pos = __shfl_sync(0xFFFFFFFF,local_worst_pos,0);
        if (lane == 0 && local_worst_pos >= 0 && dist < local_worst_dist)
        {
            row[local_worst_pos] = src;
            dist_row[local_worst_pos] = dist;
        }
        __syncwarp();
    }

    if (lane == 0)
        degree_next[target] = deg;
    (void)alpha;
}

__global__ void reverse_merge_owner_kernel(uint32_t N,uint32_t C,uint32_t reverse_capacity,uint32_t *__restrict__ graph_next,float *__restrict__ graph_dists,uint32_t *__restrict__ degree_next,const uint32_t *__restrict__ reverse_ids,const float *__restrict__ reverse_dists,const uint32_t *__restrict__ reverse_counts)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t target = tid / WARP_SIZE;
    uint32_t lane = tid % WARP_SIZE;

    if (target >= N)
        return;

    const size_t target_base = (size_t)target * C;
    const size_t reverse_base = (size_t)target * reverse_capacity;
    uint32_t *row = graph_next + target_base;
    float *dist_row = graph_dists + target_base;
    uint32_t deg = degree_next[target];
    if (deg > C)
        deg = C;

    uint32_t count = reverse_counts[target];
    if (count > reverse_capacity)
        count = reverse_capacity;

    for (uint32_t i = 0; i < count; i++)
    {
        uint32_t src = reverse_ids[reverse_base + i];
        float dist = reverse_dists[reverse_base + i];
        if (src == INVALID_ID || src >= N || src == target)
            continue;

        if (find_id_pos_R_warp(row,deg,C,src,lane) >= 0)
            continue;

        if (deg < C)
        {
            if (lane == 0)
            {
                row[deg] = src;
                dist_row[deg] = dist;
            }
            deg++;
            __syncwarp();
            continue;
        }

        float local_worst_dist = -FLT_MAX;
        int local_worst_pos = -1;
        for (uint32_t p = lane; p < C; p += WARP_SIZE)
        {
            if (dist_row[p] > local_worst_dist)
            {
                local_worst_dist = dist_row[p];
                local_worst_pos = (int)p;
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            float other_dist = __shfl_down_sync(0xFFFFFFFF,local_worst_dist,offset);
            int other_pos = __shfl_down_sync(0xFFFFFFFF,local_worst_pos,offset);
            if (other_dist > local_worst_dist || (other_dist == local_worst_dist && other_pos >= 0 && (local_worst_pos < 0 || other_pos < local_worst_pos)))
            {
                local_worst_dist = other_dist;
                local_worst_pos = other_pos;
            }
        }
        local_worst_dist = __shfl_sync(0xFFFFFFFF,local_worst_dist,0);
        local_worst_pos = __shfl_sync(0xFFFFFFFF,local_worst_pos,0);
        if (lane == 0 && local_worst_pos >= 0 && dist < local_worst_dist)
        {
            row[local_worst_pos] = src;
            dist_row[local_worst_pos] = dist;
        }
        __syncwarp();
    }

    if (lane == 0)
        degree_next[target] = deg;
}

template <typename DataT>
__global__ void final_prune_kernel_shared(const DataT *__restrict__ data,uint32_t N,uint32_t dim,uint32_t R,uint32_t C,uint32_t min_degree,float alpha,uint32_t *__restrict__ graph_work,float *__restrict__ graph_dists,uint32_t *__restrict__ degree_work,uint32_t *__restrict__ graph_cur,uint32_t *__restrict__ degree_cur,uint8_t *__restrict__ prune_loss,uint8_t *__restrict__ touched_nodes)
{
    extern __shared__ uint8_t smem[];

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t warp_id = tid / WARP_SIZE;
    uint32_t lane = tid % WARP_SIZE;
    uint32_t warp_in_block = threadIdx.x / WARP_SIZE;

    uint32_t node = warp_id;

    if (node >= N)
        return;

    if (touched_nodes != NULL && touched_nodes[node] == 0)
        return;

    uint32_t *scratch_ids = nullptr;
    float *scratch_dists = nullptr;
    uint8_t *scratch_removed = nullptr;

    get_warp_candidate_smem(smem,C,warp_in_block,(uint32_t)WARPS_PER_BLOCK,&scratch_ids,&scratch_dists,&scratch_removed);

    const size_t work_base = (size_t)node * C;
    const size_t cur_base = (size_t)node * R;

    uint32_t deg = degree_work[node];

    if (deg > C)
        deg = C;

    for (uint32_t i = lane; i < C; i += WARP_SIZE)
    {
        uint32_t id = INVALID_ID;
        float dist = FLT_MAX;
        uint8_t removed = 1;

        if (i < deg)
        {
            id = graph_work[work_base + i];
            dist = graph_dists[work_base + i];

            if (id != INVALID_ID && id < N && id != node && dist < FLT_MAX)
                removed = 0;
            else
                id = INVALID_ID;
        }

        scratch_ids[i] = id;
        scratch_dists[i] = dist;
        scratch_removed[i] = removed;
    }

    __syncwarp();

    for (uint32_t i = lane; i < C; i += WARP_SIZE)
    {
        graph_work[work_base + i] = INVALID_ID;
        graph_dists[work_base + i] = FLT_MAX;
        if (i < R && graph_cur != NULL)
        {
            graph_cur[cur_base + i] = INVALID_ID;
        }
    }

    if (lane == 0)
        degree_work[node] = 0;

    __syncwarp();

    uint32_t out_count = 0;
    for (uint32_t round = 0; round < R; round++)
    {
        float local_best_dist = FLT_MAX;
        int local_best_pos = -1;

        for (uint32_t start = 0; start < C; start += WARP_SIZE)
        {
            uint32_t pos = start + lane;

            if (pos < C && scratch_removed[pos] == 0)
            {
                float d = scratch_dists[pos];
                uint32_t id = scratch_ids[pos];

                if (id != INVALID_ID && id < N && id != node)
                {
                    if (d < local_best_dist || (d == local_best_dist && (local_best_pos < 0 || (int)pos < local_best_pos)))
                    {
                        local_best_dist = d;
                        local_best_pos = (int)pos;
                    }
                }
            }
        }

        for (int offset = 16; offset > 0; offset >>= 1)
        {
            float other_dist = __shfl_down_sync(0xFFFFFFFF,local_best_dist,offset);
            int other_pos = __shfl_down_sync(0xFFFFFFFF,local_best_pos,offset);

            if (other_dist < local_best_dist || (other_dist == local_best_dist && other_pos >= 0 && (local_best_pos < 0 || other_pos < local_best_pos)))
            {
                local_best_dist = other_dist;
                local_best_pos = other_pos;
            }
        }

        local_best_dist = __shfl_sync(0xFFFFFFFF,local_best_dist,0);
        local_best_pos = __shfl_sync(0xFFFFFFFF,local_best_pos,0);

        if (local_best_pos < 0)
            break;

        uint32_t selected_id = INVALID_ID;
        float selected_dist = FLT_MAX;

        if (lane == 0)
        {
            selected_id = scratch_ids[local_best_pos];
            selected_dist = scratch_dists[local_best_pos];
        }

        selected_id = __shfl_sync(0xFFFFFFFF,selected_id,0);
        selected_dist = __shfl_sync(0xFFFFFFFF,selected_dist,0);

        if (selected_id == INVALID_ID || selected_id >= N || selected_id == node)
            break;

        if (lane == 0)
        {
            graph_work[work_base + out_count] = selected_id;
            graph_dists[work_base + out_count] = selected_dist;
            if (graph_cur != NULL)
            {
                graph_cur[cur_base + out_count] = selected_id;
            }
        }

        out_count++;

        if (out_count >= R)
            break;

        for (uint32_t i = lane; i < C; i += WARP_SIZE)
        {
            if (scratch_removed[i] == 0 && scratch_ids[i] == selected_id)
                scratch_removed[i] = 1;
        }

        __syncwarp();

        for (uint32_t q = 0; q < C; q++)
        {
            uint32_t q_id = scratch_ids[q];
            float q_dist = scratch_dists[q];
            uint8_t q_removed = scratch_removed[q];

            q_id = __shfl_sync(0xFFFFFFFF,q_id,0);
            q_dist = __shfl_sync(0xFFFFFFFF,q_dist,0);
            q_removed = __shfl_sync(0xFFFFFFFF,q_removed,0);

            if (q_removed != 0 || q_id == INVALID_ID || q_id >= N || q_id == node)
                continue;

            if (selected_dist <= q_dist)
            {
                float d_selected_q = warp_l2_distance(data,dim,selected_id,q_id,lane);

                if (lane == 0)
                {
                    if (alpha * d_selected_q <= q_dist)
                    {
                        scratch_removed[q] = 1;
                    }
                }

                __syncwarp();
            }
        }
    }

    if (min_degree > R)
        min_degree = R;

    while (out_count < min_degree)
    {
        float local_best_dist = FLT_MAX;
        int local_best_pos = -1;
        for (uint32_t pos = lane; pos < C; pos += WARP_SIZE)
        {
            uint32_t id = scratch_ids[pos];
            float dist = scratch_dists[pos];
            bool duplicate = false;
            for (uint32_t j = 0; j < out_count; j++)
            {
                if (graph_work[work_base + j] == id)
                    duplicate = true;
            }
            if (id != INVALID_ID && id < N && id != node && !duplicate &&
                (dist < local_best_dist || (dist == local_best_dist && (local_best_pos < 0 || (int)pos < local_best_pos))))
            {
                local_best_dist = dist;
                local_best_pos = (int)pos;
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            float other_dist = __shfl_down_sync(0xFFFFFFFF,local_best_dist,offset);
            int other_pos = __shfl_down_sync(0xFFFFFFFF,local_best_pos,offset);
            if (other_dist < local_best_dist || (other_dist == local_best_dist && other_pos >= 0 && (local_best_pos < 0 || other_pos < local_best_pos)))
            {
                local_best_dist = other_dist;
                local_best_pos = other_pos;
            }
        }
        local_best_dist = __shfl_sync(0xFFFFFFFF,local_best_dist,0);
        local_best_pos = __shfl_sync(0xFFFFFFFF,local_best_pos,0);
        if (local_best_pos < 0)
            break;
        if (lane == 0)
        {
            graph_work[work_base + out_count] = scratch_ids[local_best_pos];
            graph_dists[work_base + out_count] = scratch_dists[local_best_pos];
            if (graph_cur != NULL)
            {
                graph_cur[cur_base + out_count] = scratch_ids[local_best_pos];
            }
            scratch_ids[local_best_pos] = INVALID_ID;
        }
        out_count++;
        __syncwarp();
    }

    if (lane == 0)
    {
        degree_work[node] = out_count;
        if (degree_cur != NULL)
            degree_cur[node] = out_count;
    }

    (void)prune_loss;

    __syncwarp();

    for (uint32_t i = lane; i < C; i += WARP_SIZE)
    {
        if (i >= out_count)
        {
            graph_work[work_base + i] = INVALID_ID;
            graph_dists[work_base + i] = FLT_MAX;
        }
    }
}

static uint32_t g_last_gpu_vamana_medoid = 0;
static GPUVamanaStats g_last_gpu_vamana_stats;

static double now_sec()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

extern "C" uint32_t gpu_vamana_vnew2_get_last_medoid()
{
    return g_last_gpu_vamana_medoid;
}

extern "C" void gpu_vamana_vnew2_get_last_stats(GPUVamanaStats *stats)
{
    if (stats != NULL)
        *stats = g_last_gpu_vamana_stats;
}

template <typename DataT>
static int gpu_vamana_build_impl(const DataT *h_data,uint32_t num_points,uint32_t dim,uint32_t R,uint32_t L,uint32_t C,uint32_t STEPS,uint32_t *h_graph,uint32_t *h_degree,const char *data_type_name,DataT *d_data_preloaded = NULL,double preloaded_h2d_seconds = 0.0)
{
    memset(&g_last_gpu_vamana_stats, 0, sizeof(g_last_gpu_vamana_stats));
    double gpu_build_t0 = now_sec();

    if ((h_data == NULL && d_data_preloaded == NULL) || h_graph == NULL || h_degree == NULL)
    {
        fprintf(stderr,"[gpu_vamana_build] null input pointer.\n");
        return -1;
    }

    if (num_points == 0 || dim == 0 || R == 0 || L == 0 || C == 0 || STEPS == 0)
    {
        fprintf(stderr,"[gpu_vamana_build] invalid parameters: N=%u dim=%u R=%u L=%u C=%u STEPS=%u\n",num_points,dim,R,L,C,STEPS);
        return -1;
    }

    if (C < R)
    {
        fprintf(stderr,"[gpu_vamana_build] invalid parameters: C=%u must be >= R=%u\n",C,R);
        return -1;
    }

    if (STEPS > L)
    {
        fprintf(stderr,"[gpu_vamana_build] STEPS=%u is larger than L=%u, clamp STEPS to L.\n",STEPS,L);
        STEPS = L;
    }

    if (WARPS_PER_BLOCK == 0 || WARPS_PER_BLOCK > 32)
    {
        fprintf(stderr,"[gpu_vamana_build] invalid WARPS_PER_BLOCK=%u. Use 1..32.\n",(uint32_t)WARPS_PER_BLOCK);
        return -1;
    }

    const uint32_t warps_per_block = (uint32_t)WARPS_PER_BLOCK;
    const uint32_t threads_per_block = warps_per_block * WARP_SIZE;
    const uint32_t num_blocks = (num_points + warps_per_block - 1) / warps_per_block;
    uint32_t start_id = 0;
    uint32_t h_entry_points[MAX_MULTI_ENTRY_POINTS];
    uint32_t h_num_entry_points = 1;
    for (uint32_t i = 0; i < (uint32_t)MAX_MULTI_ENTRY_POINTS; i++)
        h_entry_points[i] = 0;

    if (threads_per_block > 1024)
    {
        fprintf(stderr,"[gpu_vamana_build] threads_per_block=%u exceeds CUDA limit 1024.\n",threads_per_block);
        return -1;
    }

    size_t smem_search = warp_candidate_smem_bytes(warps_per_block,L);
    size_t smem_final = ((size_t)warps_per_block * C * sizeof(uint32_t)) + ((size_t)warps_per_block * C * sizeof(float)) + ((size_t)warps_per_block * C * sizeof(uint8_t));
    smem_search = ((smem_search + 255) / 256) * 256;
    smem_final = ((smem_final + 255) / 256) * 256;

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    int max_smem_optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem_optin,cudaDevAttrMaxSharedMemoryPerBlockOptin,device));

    if (smem_search > (size_t)max_smem_optin || smem_final > (size_t)max_smem_optin)
    {
        fprintf(stderr,"[gpu_vamana_build] shared memory too large: search=%zu final=%zu max_optin=%d\n",smem_search,smem_final,max_smem_optin);
        return -1;
    }

    if (smem_search > 49152)
    {
        CUDA_CHECK(cudaFuncSetAttribute(search_prune_kernel_shared<DataT,0>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_search));
        if (std::is_same<DataT,__half>::value)
        {
            CUDA_CHECK(cudaFuncSetAttribute(search_prune_kernel_shared<DataT,96>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_search));
            CUDA_CHECK(cudaFuncSetAttribute(search_prune_kernel_shared<DataT,128>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_search));
            CUDA_CHECK(cudaFuncSetAttribute(search_prune_kernel_shared<DataT,282>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_search));
        }
    }

    if (smem_final > 49152)
    {
        CUDA_CHECK(cudaFuncSetAttribute(final_prune_kernel_shared<DataT>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_final));
    }

    printf("[gpu_vamana_build] type=%s N=%u dim=%u R=%u L=%u C=%u STEPS=%u alpha=%.3f iters=%u smem_search=%zu smem_final=%zu\n",data_type_name,num_points,dim,R,L,C,STEPS,(float)VAMANA_ALPHA,(uint32_t)VAMANA_ITERS,smem_search,smem_final);
    printf("[gpu_vamana_build] profile=%s fp16=%u csr_reverse=%u heavy_cap=%u heavy_threshold=%u heavy_sample=%u warps_per_block=%u\n",
           DANCE_PROFILE_NAME,
           (uint32_t)ENABLE_FLOAT16_BUILD,
           (uint32_t)ENABLE_CSR_REVERSE,
           (uint32_t)ENABLE_REVERSE_HEAVY_CAP,
           (uint32_t)REVERSE_HEAVY_THRESHOLD,
           (uint32_t)REVERSE_HEAVY_SAMPLE,
           (uint32_t)WARPS_PER_BLOCK);
    printf("[gpu_vamana_opts] multi_entry=%u entry_points=%u selective_hq=%u selective_hq_extra_steps=%u selective_hq_budget=%u active_permille=%u active_reverse_only=%u touched_final_prune=%u top_l_early_reject=%u reverse_heavy_cap=%u reverse_heavy_threshold=%u reverse_heavy_sample=%u\n",
           (uint32_t)ENABLE_MULTI_ENTRY_MEDOIDS,
           (uint32_t)MULTI_ENTRY_POINTS,
           (uint32_t)ENABLE_SELECTIVE_HIGH_QUALITY_PASS,
           (uint32_t)SELECTIVE_HQ_EXTRA_STEPS,
           (uint32_t)ENABLE_SELECTIVE_HQ_BUDGET,
           (uint32_t)SELECTIVE_HQ_ACTIVE_PERMILLE,
           (uint32_t)ENABLE_SELECTIVE_HQ_ACTIVE_REVERSE_ONLY,
           (uint32_t)ENABLE_SELECTIVE_HQ_TOUCHED_FINAL_PRUNE,
           (uint32_t)ENABLE_TOPL_EARLY_REJECT,
           (uint32_t)ENABLE_REVERSE_HEAVY_CAP,
           (uint32_t)REVERSE_HEAVY_THRESHOLD,
           (uint32_t)REVERSE_HEAVY_SAMPLE);

    DataT *d_data = NULL;
    uint32_t *d_graph_cur = NULL;
    uint32_t *d_graph_work = NULL;
    uint32_t *d_degree_cur = NULL;
    uint32_t *d_degree_work = NULL;
    float *d_graph_dists = NULL;
    uint32_t *d_reverse_ids = NULL;
    float *d_reverse_dists = NULL;
    uint32_t *d_reverse_counts = NULL;
    uint32_t *d_reverse_offsets = NULL;
    uint32_t *d_reverse_cursor = NULL;
    uint8_t *d_selective_active = NULL;
    uint32_t *d_selective_active_count = NULL;
    uint32_t *d_entry_points = NULL;
    uint8_t *d_important_nodes = NULL;
    uint32_t *d_important_count = NULL;
    unsigned long long *d_reverse_touched_count = NULL;
#if ENABLE_SEARCH_PRUNE_TIMING
    unsigned long long *d_search_timing = NULL;
#endif
    size_t reverse_ids_capacity = 0;

    const size_t data_bytes = (size_t)num_points * dim * sizeof(DataT);
    const size_t graph_cur_bytes = (size_t)num_points * R * sizeof(uint32_t);
    const size_t graph_work_bytes = (size_t)num_points * C * sizeof(uint32_t);
    const size_t degree_bytes = (size_t)num_points * sizeof(uint32_t);
    const size_t graph_dists_bytes = (size_t)num_points * C * sizeof(float);
    const size_t total_graph_layout_bytes = graph_cur_bytes + graph_work_bytes + graph_dists_bytes;
    const uint32_t reverse_capacity = (uint32_t)REVERSE_CANDIDATE_CAPACITY;
    (void)reverse_capacity;
#if !ENABLE_REVERSE_CANDIDATE_INJECTION
    (void)reverse_ids_capacity;
#endif
#if !ENABLE_CSR_REVERSE
    const size_t reverse_edge_bytes = (size_t)num_points * reverse_capacity;
#endif

    if (d_data_preloaded != NULL)
    {
        d_data = d_data_preloaded;
    }
    else
    {
        CUDA_CHECK(cudaMalloc((void **)&d_data,data_bytes));
    }
    CUDA_CHECK(cudaMalloc((void **)&d_graph_cur,graph_cur_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_graph_work,graph_work_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_degree_cur,degree_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_degree_work,degree_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_graph_dists,graph_dists_bytes));
#if ENABLE_CSR_REVERSE
    CUDA_CHECK(cudaMalloc((void **)&d_reverse_offsets,((size_t)num_points + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void **)&d_reverse_cursor,(size_t)num_points * sizeof(uint32_t)));
#else
    CUDA_CHECK(cudaMalloc((void **)&d_reverse_ids,reverse_edge_bytes * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void **)&d_reverse_dists,reverse_edge_bytes * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_reverse_counts,degree_bytes));
#endif
    CUDA_CHECK(cudaMalloc((void **)&d_selective_active,(size_t)num_points * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc((void **)&d_selective_active_count,sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void **)&d_entry_points,(size_t)MAX_MULTI_ENTRY_POINTS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void **)&d_important_nodes,(size_t)num_points * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc((void **)&d_important_count,sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void **)&d_reverse_touched_count,sizeof(unsigned long long)));
#if ENABLE_SEARCH_PRUNE_TIMING
    CUDA_CHECK(cudaMalloc((void **)&d_search_timing,8 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_search_timing,0,8 * sizeof(unsigned long long)));
#endif
    CUDA_CHECK(cudaMemset(d_important_nodes,0,(size_t)num_points * sizeof(uint8_t)));
    if (d_data_preloaded != NULL)
    {
        g_last_gpu_vamana_stats.h2d_seconds = preloaded_h2d_seconds;
    }
    else
    {
        double h2d_t0 = now_sec();
        CUDA_CHECK(cudaMemcpy(d_data,h_data,data_bytes,cudaMemcpyHostToDevice));
        g_last_gpu_vamana_stats.h2d_seconds = now_sec() - h2d_t0;
    }



double medoid_t0 = now_sec();
{
    uint32_t medoid_candidates = (uint32_t)MEDOID_NUM_CANDIDATES;
    uint32_t medoid_samples = (uint32_t)MEDOID_NUM_SAMPLES;

    if (medoid_candidates > num_points)
        medoid_candidates = num_points;

    if (medoid_samples > num_points)
        medoid_samples = num_points;

    if (medoid_candidates > 0 && medoid_samples > 0)
    {
        float *d_medoid_scores = NULL;
        uint32_t *d_medoid_candidate_ids = NULL;

        CUDA_CHECK(cudaMalloc((void **)&d_medoid_scores,
                              (size_t)medoid_candidates * sizeof(float)));

        CUDA_CHECK(cudaMalloc((void **)&d_medoid_candidate_ids,
                              (size_t)medoid_candidates * sizeof(uint32_t)));

        uint32_t medoid_warps_per_block = (uint32_t)WARPS_PER_BLOCK;
        uint32_t medoid_threads_per_block = medoid_warps_per_block * WARP_SIZE;
        uint32_t medoid_blocks = (medoid_candidates + medoid_warps_per_block - 1) / medoid_warps_per_block;

        medoid_score_kernel_warp<DataT><<<medoid_blocks, medoid_threads_per_block>>>(
            d_data,
            num_points,
            dim,
            medoid_candidates,
            medoid_samples,
            d_medoid_scores,
            d_medoid_candidate_ids);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        float *h_medoid_scores = (float *)malloc((size_t)medoid_candidates * sizeof(float));
        uint32_t *h_medoid_candidate_ids = (uint32_t *)malloc((size_t)medoid_candidates * sizeof(uint32_t));

        if (h_medoid_scores == NULL || h_medoid_candidate_ids == NULL)
        {
            fprintf(stderr,"[gpu_vamana_build] medoid host malloc failed.\n");

            cudaFree(d_medoid_candidate_ids);
            cudaFree(d_medoid_scores);

            cudaFree(d_graph_dists);
            cudaFree(d_degree_work);
            cudaFree(d_degree_cur);
            cudaFree(d_graph_work);
            cudaFree(d_graph_cur);
            cudaFree(d_important_count);
            cudaFree(d_important_nodes);
            cudaFree(d_entry_points);
            if (d_data_preloaded == NULL)
                cudaFree(d_data);

            free(h_medoid_scores);
            free(h_medoid_candidate_ids);

            return -1;
        }

        CUDA_CHECK(cudaMemcpy(h_medoid_scores,
                              d_medoid_scores,
                              (size_t)medoid_candidates * sizeof(float),
                              cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemcpy(h_medoid_candidate_ids,
                              d_medoid_candidate_ids,
                              (size_t)medoid_candidates * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));

        uint32_t desired_entries = 1;
#if ENABLE_MULTI_ENTRY_MEDOIDS
        desired_entries = (uint32_t)MULTI_ENTRY_POINTS;
#endif
        if (desired_entries == 0)
            desired_entries = 1;
        if (desired_entries > (uint32_t)MAX_MULTI_ENTRY_POINTS)
            desired_entries = (uint32_t)MAX_MULTI_ENTRY_POINTS;
        if (desired_entries > medoid_candidates)
            desired_entries = medoid_candidates;

        h_num_entry_points = 0;
        float best_score = FLT_MAX;
        for (uint32_t k = 0; k < desired_entries; k++)
        {
            uint32_t best_pos = INVALID_ID;
            float kth_score = FLT_MAX;
            for (uint32_t i = 0; i < medoid_candidates; i++)
            {
                bool already_used = false;
                for (uint32_t j = 0; j < h_num_entry_points; j++)
                {
                    if (h_entry_points[j] == h_medoid_candidate_ids[i])
                        already_used = true;
                }
                if (!already_used && h_medoid_scores[i] < kth_score)
                {
                    kth_score = h_medoid_scores[i];
                    best_pos = i;
                }
            }
            if (best_pos == INVALID_ID)
                break;
            h_entry_points[h_num_entry_points++] = h_medoid_candidate_ids[best_pos];
            if (k == 0)
                best_score = kth_score;
        }
        if (h_num_entry_points == 0)
        {
            h_entry_points[0] = 0;
            h_num_entry_points = 1;
            best_score = 0.0f;
        }

        start_id = h_entry_points[0];
        g_last_gpu_vamana_medoid = start_id;

        printf("[gpu_vamana_build] approximate medoid start_id=%u entries=%u candidates=%u samples=%u score=%.3f\n",
               start_id,
               h_num_entry_points,
               medoid_candidates,
               medoid_samples,
               best_score);

        free(h_medoid_candidate_ids);
        free(h_medoid_scores);

        cudaFree(d_medoid_candidate_ids);
        cudaFree(d_medoid_scores);
    }
}
    g_last_gpu_vamana_stats.medoid_seconds = now_sec() - medoid_t0;
    CUDA_CHECK(cudaMemcpy(d_entry_points,h_entry_points,(size_t)MAX_MULTI_ENTRY_POINTS * sizeof(uint32_t),cudaMemcpyHostToDevice));
    printf("[gpu_vamana_build] entry_points");
    for (uint32_t i = 0; i < h_num_entry_points; i++)
        printf(" %u",h_entry_points[i]);
    printf("\n");


    dim3 block(threads_per_block);
    dim3 grid(num_blocks);

    printf("[gpu_vamana_layout] graph_cur_ids_bytes=%zu graph_work_ids_bytes=%zu graph_dists_bytes=%zu total_graph_layout_bytes=%zu\n",
           graph_cur_bytes,
           graph_work_bytes,
           graph_dists_bytes,
           total_graph_layout_bytes);

    init_random_graph<DataT><<<grid,block>>>(d_data,dim,d_graph_cur,d_degree_cur,d_graph_work,d_degree_work,d_graph_dists,num_points,R,C);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (uint32_t iter = 0; iter < (uint32_t)VAMANA_ITERS; iter++)
    {
        float alpha = gpu_vamana_alpha_for_iter(iter,(uint32_t)VAMANA_ITERS);

        printf("[gpu_vamana_build] iteration %u / %u, alpha=%.3f\n",iter + 1,(uint32_t)VAMANA_ITERS,alpha);

        double stage_t0 = now_sec();
        uint32_t iteration_graph_degree = R;
        uint32_t iteration_steps = STEPS;
#if ENABLE_COARSE_TO_FINE_STEPS
        if (iter < (uint32_t)COARSE_WARMUP_ITERS && iteration_steps > (uint32_t)WARMUP_EXPAND_STEPS)
            iteration_steps = (uint32_t)WARMUP_EXPAND_STEPS;
#endif
        uint32_t online_insert_steps = iteration_steps;
        uint32_t online_fast_replace = 0;
#if ENABLE_FAST_ONLINE_REPLACE_NONFINAL
        if (iter + 1 < (uint32_t)VAMANA_ITERS)
            online_fast_replace = 1;
#endif
        const uint8_t *active_nodes = NULL;
        const uint8_t *important_nodes = NULL;
        if (std::is_same<DataT,__half>::value && dim == 96)
            search_prune_kernel_shared<DataT,96><<<grid,block,smem_search>>>(d_data,num_points,dim,iteration_graph_degree,L,C,iteration_steps,online_insert_steps,online_fast_replace,start_id,d_entry_points,h_num_entry_points,alpha,active_nodes,important_nodes,0,d_graph_cur,d_degree_cur,d_graph_work,d_degree_work,d_graph_dists
#if ENABLE_SEARCH_PRUNE_TIMING
                                                                                   ,d_search_timing
#endif
                                                                                   );
        else if (std::is_same<DataT,__half>::value && dim == 128)
            search_prune_kernel_shared<DataT,128><<<grid,block,smem_search>>>(d_data,num_points,dim,iteration_graph_degree,L,C,iteration_steps,online_insert_steps,online_fast_replace,start_id,d_entry_points,h_num_entry_points,alpha,active_nodes,important_nodes,0,d_graph_cur,d_degree_cur,d_graph_work,d_degree_work,d_graph_dists
#if ENABLE_SEARCH_PRUNE_TIMING
                                                                                    ,d_search_timing
#endif
                                                                                    );
        else if (std::is_same<DataT,__half>::value && dim == 282)
            search_prune_kernel_shared<DataT,282><<<grid,block,smem_search>>>(d_data,num_points,dim,iteration_graph_degree,L,C,iteration_steps,online_insert_steps,online_fast_replace,start_id,d_entry_points,h_num_entry_points,alpha,active_nodes,important_nodes,0,d_graph_cur,d_degree_cur,d_graph_work,d_degree_work,d_graph_dists
#if ENABLE_SEARCH_PRUNE_TIMING
                                                                                    ,d_search_timing
#endif
                                                                                    );
        else
            search_prune_kernel_shared<DataT,0><<<grid,block,smem_search>>>(d_data,num_points,dim,iteration_graph_degree,L,C,iteration_steps,online_insert_steps,online_fast_replace,start_id,d_entry_points,h_num_entry_points,alpha,active_nodes,important_nodes,0,d_graph_cur,d_degree_cur,d_graph_work,d_degree_work,d_graph_dists
#if ENABLE_SEARCH_PRUNE_TIMING
                                                                                  ,d_search_timing
#endif
                                                                                  );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        g_last_gpu_vamana_stats.search_online_insert_seconds += now_sec() - stage_t0;

        stage_t0 = now_sec();
#if ENABLE_REVERSE_CANDIDATE_INJECTION
#if ENABLE_CSR_REVERSE
        const uint32_t reverse_source_limit = iteration_graph_degree;
        CUDA_CHECK(cudaMemset(d_reverse_offsets, 0, ((size_t)num_points + 1) * sizeof(uint32_t)));
        reverse_count_csr_kernel<<<grid,block>>>(num_points,reverse_source_limit,C,d_graph_work,d_degree_work,d_reverse_offsets);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        thrust::device_ptr<uint32_t> offsets_ptr(d_reverse_offsets);
        thrust::inclusive_scan(offsets_ptr, offsets_ptr + (size_t)num_points + 1, offsets_ptr);

        uint32_t total_reverse_edges = 0;
        CUDA_CHECK(cudaMemcpy(&total_reverse_edges,d_reverse_offsets + num_points,sizeof(uint32_t),cudaMemcpyDeviceToHost));
        g_last_gpu_vamana_stats.reverse_proposals += total_reverse_edges;
        CUDA_CHECK(cudaMemset(d_reverse_touched_count,0,sizeof(unsigned long long)));
        count_touched_rows_kernel<<<(num_points + 255) / 256,256>>>(num_points,d_reverse_offsets,d_reverse_touched_count);
        unsigned long long iteration_touched = 0;
        CUDA_CHECK(cudaMemcpy(&iteration_touched,d_reverse_touched_count,sizeof(iteration_touched),cudaMemcpyDeviceToHost));
        g_last_gpu_vamana_stats.reverse_touched_rows += iteration_touched;

        if ((size_t)total_reverse_edges > reverse_ids_capacity)
        {
            if (d_reverse_ids != NULL)
            {
                CUDA_CHECK(cudaFree(d_reverse_ids));
                d_reverse_ids = NULL;
            }
            if (total_reverse_edges > 0)
            {
                CUDA_CHECK(cudaMalloc((void **)&d_reverse_ids,(size_t)total_reverse_edges * sizeof(uint32_t)));
                reverse_ids_capacity = (size_t)total_reverse_edges;
            }
        }

        CUDA_CHECK(cudaMemcpy(d_reverse_cursor,d_reverse_offsets,(size_t)num_points * sizeof(uint32_t),cudaMemcpyDeviceToDevice));

        if (total_reverse_edges > 0)
        {
            reverse_fill_csr_kernel<<<grid,block>>>(num_points,reverse_source_limit,C,d_graph_work,d_degree_work,d_reverse_cursor,d_reverse_ids);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            reverse_merge_csr_kernel<DataT><<<grid,block>>>(d_data,num_points,dim,C,alpha,d_graph_work,d_graph_dists,d_degree_work,d_reverse_offsets,d_reverse_ids);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
#else
        CUDA_CHECK(cudaMemset(d_reverse_counts, 0, degree_bytes));
        reverse_generate_kernel<<<grid,block>>>(num_points,iteration_graph_degree,C,reverse_capacity,d_graph_work,d_graph_dists,d_degree_work,d_reverse_ids,d_reverse_dists,d_reverse_counts);
        CUDA_CHECK(cudaGetLastError());
        reverse_merge_owner_kernel<<<grid,block>>>(num_points,C,reverse_capacity,d_graph_work,d_graph_dists,d_degree_work,d_reverse_ids,d_reverse_dists,d_reverse_counts);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
#endif
        g_last_gpu_vamana_stats.reverse_seconds += now_sec() - stage_t0;
        g_last_gpu_vamana_stats.reverse_csr_seconds += now_sec() - stage_t0;
#endif

        stage_t0 = now_sec();
        uint32_t final_min_degree = 0;
#if ENABLE_FINAL_DEGREE_FILL
        if (iter + 1 == (uint32_t)VAMANA_ITERS)
            final_min_degree = (uint32_t)FINAL_MIN_DEGREE;
#endif
        final_prune_kernel_shared<DataT><<<grid,block,smem_final>>>(d_data,num_points,dim,iteration_graph_degree,C,final_min_degree,alpha,d_graph_work,d_graph_dists,d_degree_work,d_graph_cur,d_degree_cur,NULL,NULL);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        g_last_gpu_vamana_stats.final_prune_seconds += now_sec() - stage_t0;
    }

#if ENABLE_SELECTIVE_HIGH_QUALITY_PASS
    {
        CUDA_CHECK(cudaMemset(d_important_count,0,sizeof(uint32_t)));
        uint32_t mark_threads = 256;
        uint32_t mark_blocks = (num_points + mark_threads - 1) / mark_threads;
#if ENABLE_CSR_REVERSE
#if ENABLE_SELECTIVE_HQ_BUDGET
        mark_important_nodes_budgeted_kernel<<<mark_blocks,mark_threads>>>(num_points,R,C,d_degree_cur,d_reverse_offsets,d_important_nodes,d_important_count,1u,(uint32_t)SELECTIVE_HQ_LOW_DEGREE_THRESHOLD,(uint32_t)SELECTIVE_HQ_REVERSE_INDEGREE_THRESHOLD,(uint32_t)SELECTIVE_HQ_ACTIVE_PERMILLE);
#else
        mark_important_nodes_kernel<<<mark_blocks,mark_threads>>>(num_points,R,C,d_degree_cur,d_reverse_offsets,d_important_nodes,d_important_count,1u,(uint32_t)SELECTIVE_HQ_LOW_DEGREE_THRESHOLD,(uint32_t)SELECTIVE_HQ_REVERSE_INDEGREE_THRESHOLD);
#endif
#else
#if ENABLE_SELECTIVE_HQ_BUDGET
        mark_important_nodes_budgeted_kernel<<<mark_blocks,mark_threads>>>(num_points,R,C,d_degree_cur,d_reverse_counts,d_important_nodes,d_important_count,0u,(uint32_t)SELECTIVE_HQ_LOW_DEGREE_THRESHOLD,(uint32_t)SELECTIVE_HQ_REVERSE_INDEGREE_THRESHOLD,(uint32_t)SELECTIVE_HQ_ACTIVE_PERMILLE);
#else
        mark_important_nodes_kernel<<<mark_blocks,mark_threads>>>(num_points,R,C,d_degree_cur,d_reverse_counts,d_important_nodes,d_important_count,0u,(uint32_t)SELECTIVE_HQ_LOW_DEGREE_THRESHOLD,(uint32_t)SELECTIVE_HQ_REVERSE_INDEGREE_THRESHOLD);
#endif
#endif
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&g_last_gpu_vamana_stats.selective_active_nodes,d_important_count,sizeof(uint32_t),cudaMemcpyDeviceToHost));

        float alpha = (float)VAMANA_ALPHA;
        uint32_t selective_steps = STEPS + (uint32_t)SELECTIVE_HQ_EXTRA_STEPS;
        if (selective_steps > L)
            selective_steps = L;
        printf("[gpu_vamana_build] selective high-quality pass active_nodes=%u steps=%u alpha=%.3f low_degree=%u reverse_indegree=%u\n",
               g_last_gpu_vamana_stats.selective_active_nodes,
               selective_steps,
               alpha,
               (uint32_t)SELECTIVE_HQ_LOW_DEGREE_THRESHOLD,
               (uint32_t)SELECTIVE_HQ_REVERSE_INDEGREE_THRESHOLD);

        double stage_t0 = now_sec();
        const uint8_t *hq_extra_nodes = d_important_nodes;
        uint32_t hq_extra_steps = (uint32_t)SELECTIVE_HQ_EXTRA_STEPS;
        if (std::is_same<DataT,__half>::value && dim == 96)
            search_prune_kernel_shared<DataT,96><<<grid,block,smem_search>>>(d_data,num_points,dim,R,L,C,selective_steps,selective_steps,0,start_id,d_entry_points,h_num_entry_points,alpha,d_important_nodes,hq_extra_nodes,hq_extra_steps,d_graph_cur,d_degree_cur,d_graph_work,d_degree_work,d_graph_dists
#if ENABLE_SEARCH_PRUNE_TIMING
                                                                                   ,d_search_timing
#endif
                                                                                   );
        else if (std::is_same<DataT,__half>::value && dim == 128)
            search_prune_kernel_shared<DataT,128><<<grid,block,smem_search>>>(d_data,num_points,dim,R,L,C,selective_steps,selective_steps,0,start_id,d_entry_points,h_num_entry_points,alpha,d_important_nodes,hq_extra_nodes,hq_extra_steps,d_graph_cur,d_degree_cur,d_graph_work,d_degree_work,d_graph_dists
#if ENABLE_SEARCH_PRUNE_TIMING
                                                                                    ,d_search_timing
#endif
                                                                                    );
        else if (std::is_same<DataT,__half>::value && dim == 282)
            search_prune_kernel_shared<DataT,282><<<grid,block,smem_search>>>(d_data,num_points,dim,R,L,C,selective_steps,selective_steps,0,start_id,d_entry_points,h_num_entry_points,alpha,d_important_nodes,hq_extra_nodes,hq_extra_steps,d_graph_cur,d_degree_cur,d_graph_work,d_degree_work,d_graph_dists
#if ENABLE_SEARCH_PRUNE_TIMING
                                                                                    ,d_search_timing
#endif
                                                                                    );
        else
            search_prune_kernel_shared<DataT,0><<<grid,block,smem_search>>>(d_data,num_points,dim,R,L,C,selective_steps,selective_steps,0,start_id,d_entry_points,h_num_entry_points,alpha,d_important_nodes,hq_extra_nodes,hq_extra_steps,d_graph_cur,d_degree_cur,d_graph_work,d_degree_work,d_graph_dists
#if ENABLE_SEARCH_PRUNE_TIMING
                                                                                  ,d_search_timing
#endif
                                                                                  );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        g_last_gpu_vamana_stats.search_online_insert_seconds += now_sec() - stage_t0;

        stage_t0 = now_sec();
#if ENABLE_REVERSE_CANDIDATE_INJECTION
#if ENABLE_CSR_REVERSE
        CUDA_CHECK(cudaMemset(d_reverse_offsets, 0, ((size_t)num_points + 1) * sizeof(uint32_t)));
#if ENABLE_SELECTIVE_HQ_ACTIVE_REVERSE_ONLY
        reverse_count_csr_active_kernel<<<grid,block>>>(num_points,R,C,d_graph_work,d_degree_work,d_important_nodes,d_reverse_offsets);
#else
        reverse_count_csr_kernel<<<grid,block>>>(num_points,R,C,d_graph_work,d_degree_work,d_reverse_offsets);
#endif
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        thrust::device_ptr<uint32_t> offsets_ptr(d_reverse_offsets);
        thrust::inclusive_scan(offsets_ptr, offsets_ptr + (size_t)num_points + 1, offsets_ptr);

        uint32_t total_reverse_edges = 0;
        CUDA_CHECK(cudaMemcpy(&total_reverse_edges,d_reverse_offsets + num_points,sizeof(uint32_t),cudaMemcpyDeviceToHost));

        if ((size_t)total_reverse_edges > reverse_ids_capacity)
        {
            if (d_reverse_ids != NULL)
            {
                CUDA_CHECK(cudaFree(d_reverse_ids));
                d_reverse_ids = NULL;
            }
            if (total_reverse_edges > 0)
            {
                CUDA_CHECK(cudaMalloc((void **)&d_reverse_ids,(size_t)total_reverse_edges * sizeof(uint32_t)));
                reverse_ids_capacity = (size_t)total_reverse_edges;
            }
        }

        CUDA_CHECK(cudaMemcpy(d_reverse_cursor,d_reverse_offsets,(size_t)num_points * sizeof(uint32_t),cudaMemcpyDeviceToDevice));

        if (total_reverse_edges > 0)
        {
#if ENABLE_SELECTIVE_HQ_TOUCHED_FINAL_PRUNE
            init_touched_from_active_kernel<<<mark_blocks,mark_threads>>>(num_points,d_important_nodes,d_selective_active);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
#endif
#if ENABLE_SELECTIVE_HQ_ACTIVE_REVERSE_ONLY
            reverse_fill_csr_active_kernel<<<grid,block>>>(num_points,R,C,d_graph_work,d_degree_work,d_important_nodes,d_reverse_cursor,d_reverse_ids,
#if ENABLE_SELECTIVE_HQ_TOUCHED_FINAL_PRUNE
                                                          d_selective_active
#else
                                                          NULL
#endif
                                                          );
#else
            reverse_fill_csr_kernel<<<grid,block>>>(num_points,R,C,d_graph_work,d_degree_work,d_reverse_cursor,d_reverse_ids);
#endif
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            reverse_merge_csr_kernel<DataT><<<grid,block>>>(d_data,num_points,dim,C,alpha,d_graph_work,d_graph_dists,d_degree_work,d_reverse_offsets,d_reverse_ids);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
#else
        CUDA_CHECK(cudaMemset(d_reverse_counts, 0, degree_bytes));
        reverse_generate_kernel<<<grid,block>>>(num_points,R,C,reverse_capacity,d_graph_work,d_graph_dists,d_degree_work,d_reverse_ids,d_reverse_dists,d_reverse_counts);
        CUDA_CHECK(cudaGetLastError());
        reverse_merge_owner_kernel<<<grid,block>>>(num_points,C,reverse_capacity,d_graph_work,d_graph_dists,d_degree_work,d_reverse_ids,d_reverse_dists,d_reverse_counts);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
#endif
        g_last_gpu_vamana_stats.reverse_seconds += now_sec() - stage_t0;
#endif

        stage_t0 = now_sec();
        uint32_t final_min_degree = 0;
#if ENABLE_FINAL_DEGREE_FILL
        final_min_degree = (uint32_t)FINAL_MIN_DEGREE;
#endif
        final_prune_kernel_shared<DataT><<<grid,block,smem_final>>>(d_data,num_points,dim,R,C,final_min_degree,alpha,d_graph_work,d_graph_dists,d_degree_work,d_graph_cur,d_degree_cur,NULL,
#if ENABLE_SELECTIVE_HQ_TOUCHED_FINAL_PRUNE
                                                                    d_selective_active
#else
                                                                    NULL
#endif
                                                                    );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        g_last_gpu_vamana_stats.final_prune_seconds += now_sec() - stage_t0;
    }
#endif

    double d2h_t0 = now_sec();
    CUDA_CHECK(cudaMemcpy(h_graph,d_graph_cur,graph_cur_bytes,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_degree,d_degree_cur,degree_bytes,cudaMemcpyDeviceToHost));
    g_last_gpu_vamana_stats.d2h_seconds = now_sec() - d2h_t0;
    g_last_gpu_vamana_stats.distance_computations = 0;
    g_last_gpu_vamana_stats.online_insert_attempts = 0;
    g_last_gpu_vamana_stats.online_insert_mutations = 0;
    g_last_gpu_vamana_stats.reverse_candidate_overflows = 0;
#if ENABLE_SEARCH_PRUNE_TIMING
    {
        unsigned long long h_search_timing[8] = {0,0,0,0,0,0,0,0};
        CUDA_CHECK(cudaMemcpy(h_search_timing,d_search_timing,8 * sizeof(unsigned long long),cudaMemcpyDeviceToHost));
        unsigned long long total_cycles = h_search_timing[0] + h_search_timing[1] + h_search_timing[2] + h_search_timing[3];
        printf("[gpu_search_timing_cycles] select=%llu dynamic_insert=%llu expand_candidates=%llu initial_candidates=%llu dynamic_calls=%llu expand_candidates_count=%llu expand_distance=%llu expand_insert=%llu accounted=%llu\n",
               h_search_timing[0],
               h_search_timing[1],
               h_search_timing[2],
               h_search_timing[3],
               h_search_timing[4],
               h_search_timing[5],
               h_search_timing[6],
               h_search_timing[7],
               total_cycles);
    }
#endif
    cudaFree(d_important_count);
    cudaFree(d_reverse_touched_count);
    cudaFree(d_important_nodes);
    cudaFree(d_entry_points);
#if ENABLE_SEARCH_PRUNE_TIMING
    cudaFree(d_search_timing);
#endif
    cudaFree(d_selective_active_count);
    cudaFree(d_selective_active);
    cudaFree(d_reverse_counts);
    cudaFree(d_reverse_dists);
    cudaFree(d_reverse_ids);
    cudaFree(d_reverse_cursor);
    cudaFree(d_reverse_offsets);
    cudaFree(d_graph_dists);
    cudaFree(d_degree_work);
    cudaFree(d_degree_cur);
    cudaFree(d_graph_work);
    cudaFree(d_graph_cur);
    if (d_data_preloaded == NULL)
        cudaFree(d_data);

    CUDA_CHECK(cudaDeviceSynchronize());

    g_last_gpu_vamana_stats.gpu_build_seconds = now_sec() - gpu_build_t0 + (d_data_preloaded != NULL ? preloaded_h2d_seconds : 0.0);
    printf("[gpu_vamana_stats] selective_active_nodes=%u medoid=%.6f h2d=%.6f search_online_insert=%.6f reverse=%.6f reverse_csr=%.6f final_prune=%.6f d2h=%.6f total=%.6f reverse_proposals=%llu reverse_touched_rows=%llu\n",
           g_last_gpu_vamana_stats.selective_active_nodes,
           g_last_gpu_vamana_stats.medoid_seconds,
           g_last_gpu_vamana_stats.h2d_seconds,
           g_last_gpu_vamana_stats.search_online_insert_seconds,
           g_last_gpu_vamana_stats.reverse_seconds,
           g_last_gpu_vamana_stats.reverse_csr_seconds,
           g_last_gpu_vamana_stats.final_prune_seconds,
           g_last_gpu_vamana_stats.d2h_seconds,
           g_last_gpu_vamana_stats.gpu_build_seconds,
           (unsigned long long)g_last_gpu_vamana_stats.reverse_proposals,
           (unsigned long long)g_last_gpu_vamana_stats.reverse_touched_rows);
    printf("[gpu_vamana_hq_stats] important_nodes=%u\n",
           g_last_gpu_vamana_stats.selective_active_nodes);
    printf("[gpu_vamana_build] finished.\n");
    return 0;
}

extern "C" int gpu_vamana_vnew2_build(const uint8_t *h_data,uint32_t num_points,uint32_t dim,uint32_t R,uint32_t L,uint32_t C,uint32_t STEPS,uint32_t *h_graph,uint32_t *h_degree)
{
    return gpu_vamana_build_impl<uint8_t>(h_data,num_points,dim,R,L,C,STEPS,h_graph,h_degree,"uint8");
}

__global__ void convert_float_to_half_kernel(const float *__restrict__ src,__half *__restrict__ dst,size_t total)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = idx; i < total; i += stride)
        dst[i] = __float2half(src[i]);
}

extern "C" int gpu_vamana_vnew2_build_float(const float *h_data,uint32_t num_points,uint32_t dim,uint32_t R,uint32_t L,uint32_t C,uint32_t STEPS,uint32_t *h_graph,uint32_t *h_degree)
{
#if ENABLE_FLOAT16_BUILD
    if (h_data == NULL)
        return gpu_vamana_build_impl<float>(h_data,num_points,dim,R,L,C,STEPS,h_graph,h_degree,"float");
    size_t total = (size_t)num_points * (size_t)dim;
    float *d_float = NULL;
    __half *d_half = NULL;
    double convert_t0 = now_sec();
    cudaError_t err = cudaMalloc((void **)&d_float,total * sizeof(float));
    if (err == cudaSuccess)
        err = cudaMalloc((void **)&d_half,total * sizeof(__half));
    if (err == cudaSuccess)
        err = cudaMemcpy(d_float,h_data,total * sizeof(float),cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr,"[gpu_vamana_build_float] failed to allocate/copy GPU conversion buffers for N=%u dim=%u: %s\n",num_points,dim,cudaGetErrorString(err));
        cudaFree(d_float);
        cudaFree(d_half);
        return -1;
    }

    int block = 256;
    int grid = (int)((total + (size_t)block - 1) / (size_t)block);
    if (grid > 65535)
        grid = 65535;
    convert_float_to_half_kernel<<<grid,block>>>(d_float,d_half,total);
    err = cudaGetLastError();
    if (err == cudaSuccess)
        err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr,"[gpu_vamana_build_float] GPU float-to-fp16 conversion failed: %s\n",cudaGetErrorString(err));
        cudaFree(d_float);
        cudaFree(d_half);
        return -1;
    }
    cudaFree(d_float);

    double convert_seconds = now_sec() - convert_t0;
    printf("[gpu_vamana_build_float] converted float input to fp16 on GPU in %.6f seconds\n",convert_seconds);
    int status = gpu_vamana_build_impl<__half>(NULL,num_points,dim,R,L,C,STEPS,h_graph,h_degree,"float16",d_half,convert_seconds);
    cudaFree(d_half);
    return status;
#else
    return gpu_vamana_build_impl<float>(h_data,num_points,dim,R,L,C,STEPS,h_graph,h_degree,"float");
#endif
}
