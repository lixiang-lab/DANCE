// Copyright (c) 2026 DANCE Authors. Licensed under the MIT License.
#ifndef GPU_V1_DEF_H
#define GPU_V1_DEF_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GPUVamanaStats
{
    double medoid_seconds;
    double h2d_seconds;
    double search_online_insert_seconds;
    double reverse_seconds;
    double final_prune_seconds;
    double d2h_seconds;
    double gpu_build_seconds;
    uint64_t distance_computations;
    uint64_t online_insert_attempts;
    uint64_t online_insert_mutations;
    uint64_t reverse_candidate_overflows;
    uint64_t reverse_proposals;
    uint64_t reverse_touched_rows;
    double reverse_csr_seconds;
    uint32_t selective_active_nodes;
} GPUVamanaStats;

int gpu_vamana_build(const uint8_t *h_data,
                     uint32_t num_points,
                     uint32_t dim,
                     uint32_t R,
                     uint32_t L,
                     uint32_t C,
                     uint32_t STEPS,
                     uint32_t *h_graph,
                     uint32_t *h_degree);

int gpu_vamana_build_float(const float *h_data,
                           uint32_t num_points,
                           uint32_t dim,
                           uint32_t R,
                           uint32_t L,
                           uint32_t C,
                           uint32_t STEPS,
                           uint32_t *h_graph,
                           uint32_t *h_degree);

int gpu_vamana_vnew2_build(const uint8_t *h_data,
                           uint32_t num_points,
                           uint32_t dim,
                           uint32_t R,
                           uint32_t L,
                           uint32_t C,
                           uint32_t STEPS,
                           uint32_t *h_graph,
                           uint32_t *h_degree);

int gpu_vamana_vnew2_build_float(const float *h_data,
                                 uint32_t num_points,
                                 uint32_t dim,
                                 uint32_t R,
                                 uint32_t L,
                                 uint32_t C,
                                 uint32_t STEPS,
                                 uint32_t *h_graph,
                                 uint32_t *h_degree);

int gpu_vamana_vnew2_build_device(const uint8_t *h_data, uint32_t num_points, uint32_t dim,
                                  uint32_t R, uint32_t L, uint32_t C, uint32_t STEPS,
                                  uint8_t **d_data, uint32_t **d_graph, uint32_t **d_degree);
int gpu_vamana_vnew2_build_device_float(const float *h_data, uint32_t num_points, uint32_t dim,
                                        uint32_t R, uint32_t L, uint32_t C, uint32_t STEPS,
                                        float **d_data, uint32_t **d_graph, uint32_t **d_degree);


uint32_t gpu_vamana_get_last_medoid(void);
void gpu_vamana_get_last_stats(GPUVamanaStats *stats);
uint32_t gpu_vamana_vnew2_get_last_medoid(void);
void gpu_vamana_vnew2_get_last_stats(GPUVamanaStats *stats);

#ifdef __cplusplus
}
#endif

#endif
