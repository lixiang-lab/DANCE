# Third-party ownership

This repository does not redistribute complete third-party implementations.

## Microsoft DiskANN

DANCE integrates with [Microsoft DiskANN](https://github.com/microsoft/DiskANN), tested at commit `78256bbab4685e1774e78d331e081a153be26823`. DiskANN is Copyright Microsoft Corporation and is distributed under its own MIT License. Users must retain the DiskANN license and notices after applying the DANCE integration patch.

The integration patch necessarily contains limited DiskANN context and modifications to DiskANN integration points. Those upstream portions remain Copyright Microsoft Corporation. DANCE claims copyright only in DANCE-authored additions and modifications.

## Compared systems

The paper uses the following systems as experimental baselines:

- [Microsoft DiskANN](https://github.com/microsoft/DiskANN)
- [Tagore](https://github.com/ZJU-DAILY/Tagore)
- [Jasper](https://github.com/saltsystemslab/Jasper)
- [RAPIDS cuVS](https://github.com/rapidsai/cuvs)

No source code from Tagore, Jasper, or cuVS is included. Each project remains governed by its upstream license and ownership terms.
