# Baseline versions

The paper's reported comparison results use the following upstream revisions. Full commit identifiers are provided so that moving branches and later releases cannot silently change the baseline.

| System | Upstream version used | Repository |
|---|---|---|
| CPU DiskANN, CPU Vamana, CPU FilteredVamana, CPU StitchedVamana | commit [`78256bbab4685e1774e78d331e081a153be26823`](https://github.com/microsoft/DiskANN/commit/78256bbab4685e1774e78d331e081a153be26823), `cpp_main`, 36 commits after tag `0.7.0` | [microsoft/DiskANN](https://github.com/microsoft/DiskANN) |
| Tagore | commit [`110d480fb483f8a60d166d3e1e34144e57d2b3a3`](https://github.com/ZJU-DAILY/Tagore/commit/110d480fb483f8a60d166d3e1e34144e57d2b3a3), `master` | [ZJU-DAILY/Tagore](https://github.com/ZJU-DAILY/Tagore) |
| Jasper | commit [`2c71ca0d16370c9fa9967be4529747c53825c1c6`](https://github.com/saltsystemslab/Jasper/commit/2c71ca0d16370c9fa9967be4529747c53825c1c6), `main`, tested with the latest FP16 implementation available on 2026-07-29 | [saltsystemslab/Jasper](https://github.com/saltsystemslab/Jasper) |
| RAPIDS cuVS Vamana | release `v25.10.00`, commit [`f245c1529d460a374b92e8c8448de0b0b6b84ee6`](https://github.com/NVIDIA/cuvs/commit/f245c1529d460a374b92e8c8448de0b0b6b84ee6) | [NVIDIA/cuVS](https://github.com/NVIDIA/cuvs) |

The DANCE artifact itself is fixed by the repository release tag. The shared DiskANN partitioning, PQ, merge, serialization, and CPU-search components use the DiskANN revision above.

The Tagore, Jasper, and cuVS experiments use their listed upstream construction implementations through dataset-format and DiskANN-serialization adapters. Parameter mappings and timing boundaries follow the paper's experimental methodology; the full competitor repositories are not copied into this artifact.

Jasper results correspond to the listed latest FP16 repository, not the retired `JasperGPUANNS` repository. cuVS results correspond to the listed `v25.10.00` source revision, not a moving nightly build.
