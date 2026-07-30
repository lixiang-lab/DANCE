# Artifact scope

| Paper capability | Released implementation | Entry point |
|---|---|---|
| Standard in-memory GPU Vamana | DANCE standard builder | `gpu_vamana_memory_index` |
| Standard sharded GPU Vamana | DiskANN partition and shard pipeline with DANCE shard construction | `build_disk_index` |
| GPU-FilteredVamana | Fixed complete production implementation | `build_memory_index --gpu_filtered` |
| Sharded GPU-FilteredVamana | Fixed complete production implementation in the DiskANN SSD pipeline | `build_disk_index --gpu_filtered` |
| GPU-StitchedVamana | Per-label GPU construction, global-ID remapping, stitching, sidecar generation, and conditional final prune | `gpu_stitched_vamana_index` |
| CPU memory search | Upstream DiskANN search | `search_memory_index` |
| SSD search | Upstream DiskANN search | `search_disk_index` |

The release contains neither competitor implementations nor paper-only experimental variants. It does not expose ablation controls. Label balancing is disabled because the final evaluation found no benefit. GPU-FilteredVamana provides shared-graph overlapping-label construction. GPU-StitchedVamana supports both partitioned and overlapping-label construction through label-local graphs and stitching. Standard and filtered release gates are reported in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

The DANCE source files and the patch in `integrations/` are the authoritative artifact. Apply the patch to the supported upstream DiskANN commit as described in [INTEGRATION.md](INTEGRATION.md).
