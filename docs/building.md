# Building and running

These instructions describe the intended Linux CPU/MPI and NVIDIA CUDA routes. They have been checked against the pinned source interfaces; independent builds and runs for this standalone snapshot are still [pending](validation.md). Use fresh out-of-source build trees.

## Requirements

| Component | Requirement |
| --- | --- |
| Compiler | A C++20-capable compiler compatible with the pinned AMReX; a compatible C compiler and MPI implementation. |
| CMake | At least 3.20 for LBMReX CPU configuration; at least 3.26 for the CUDA C++20 route. |
| CUDA route | NVIDIA CUDA toolkit at least 12.2 as required by the pinned AMReX, a compatible host compiler/driver, and an architecture matching the intended devices. |
| AMReX | Official `4ceb814db7d79c5858b13dab153fb04ecc3b129d` with the [projected-tags patch](../patches/amrex/README.md), `AMReX_SPACEDIM=3`, `AMReX_EB=ON`, `AMReX_PRECISION=DOUBLE`. |
| Python | Python 3.9 or newer for formal tests; NumPy for numerical artifact consumers and several synthetic tests. Matplotlib is needed by plotting tools. ParaView's `pvpython` is needed only by the corresponding readers/renderers. |
| Shell tools | Git for clone/dependency preparation and source-identity Git fixtures; Bash and usual Linux utilities for runners. Git is not required to build an ordinary source archive with an existing package. |

The D2Q9 compositions also link `AMReX::amrex_3d`. A 2D AMReX installation is not a substitute. Keep dependency/application compilers, MPI, precision, GPU backend and ABI compatible. The configuration performs an EB header/link probe; do not disable it to conceal an incompatible package.

## Clone and prepare AMReX

```bash
git clone --recurse-submodules https://github.com/no-more-Bug/Open-LBMReX.git
cd Open-LBMReX
git submodule update --init -- external/amrex
git -C external/amrex rev-parse HEAD
mkdir -p build/dependencies
bash scripts/prepare-amrex.sh "$PWD/build/dependencies/amrex-source"
```

The displayed dependency HEAD must equal the pinned commit. The preparation command refuses an existing destination or a destination inside the submodule. It verifies the patch digest and writes a separate patched source tree with `.lbmrex-dependency` metadata; it does not build, install or modify the submodule. Use a new preparation path when intentionally rebuilding this material.

## CPU installation and application build

The following example enables MPI and OpenMP in the AMReX package, but uses one thread per rank for runs and disables optional OpenMP test identities. It does not select a cluster scheduler or a machine-specific installation directory.

```bash
cmake -S build/dependencies/amrex-source -B build/dependencies/amrex-cpu \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_INSTALL_PREFIX="$PWD/build/dependencies/install-cpu" \
  -DAMReX_INSTALL=ON -DAMReX_SPACEDIM=3 -DAMReX_PRECISION=DOUBLE \
  -DAMReX_MPI=ON -DAMReX_MPI_THREAD_MULTIPLE=OFF -DAMReX_OMP=ON \
  -DAMReX_EB=ON -DAMReX_FORTRAN=OFF -DAMReX_BUILD_SHARED_LIBS=OFF \
  -DAMReX_GPU_BACKEND=NONE -DAMReX_FASTMATH=OFF -DAMReX_ENABLE_TESTS=OFF
cmake --build build/dependencies/amrex-cpu --parallel 4
cmake --install build/dependencies/amrex-cpu

cmake -S . -B build/cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx \
  -DAMReX_ROOT="$PWD/build/dependencies/install-cpu" \
  -DLBM_ENABLE_CUDA=OFF -DBUILD_TESTING=ON \
  -DLBM_BUILD_SHOWCASES=ON -DLBM_TEST_OPENMP=OFF \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build/cpu --parallel 4
```

To build only examples, use `BUILD_TESTING=OFF`. To build the LBMReX library and tests without showcase targets, use `LBM_BUILD_SHOWCASES=OFF`. Performance executables are excluded from the default build and require [explicit targets](../tests/performance/README.md).

## CUDA installation and application build

Use a different dependency build/install prefix and application build tree. The example architecture `80` is for a matching compute-capability 8.0 device; replace it with the actual target architecture. Select a host compiler supported by the installed CUDA toolkit. All relevant executables must see the same compatible MPI installation.

```bash
CUDA_ARCH=80
cmake -S build/dependencies/amrex-source -B build/dependencies/amrex-cuda \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DMPI_C_COMPILER=mpicc -DMPI_CXX_COMPILER=mpicxx \
  -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" \
  -DCMAKE_CUDA_HOST_COMPILER="$(command -v g++)" \
  -DCMAKE_INSTALL_PREFIX="$PWD/build/dependencies/install-cuda" \
  -DAMReX_INSTALL=ON -DAMReX_SPACEDIM=3 -DAMReX_PRECISION=DOUBLE \
  -DAMReX_MPI=ON -DAMReX_MPI_THREAD_MULTIPLE=OFF -DAMReX_OMP=OFF \
  -DAMReX_EB=ON -DAMReX_FORTRAN=OFF -DAMReX_BUILD_SHARED_LIBS=OFF \
  -DAMReX_GPU_BACKEND=CUDA -DAMReX_CUDA_ARCH="$CUDA_ARCH" \
  -DAMReX_FASTMATH=OFF -DAMReX_CUDA_FASTMATH=OFF -DAMReX_ENABLE_TESTS=OFF
cmake --build build/dependencies/amrex-cuda --parallel 4
cmake --install build/dependencies/amrex-cuda

cmake -S . -B build/cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DMPI_C_COMPILER=mpicc -DMPI_CXX_COMPILER=mpicxx \
  -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" \
  -DCMAKE_CUDA_HOST_COMPILER="$(command -v g++)" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DAMReX_ROOT="$PWD/build/dependencies/install-cuda" \
  -DLBM_ENABLE_CUDA=ON -DBUILD_TESTING=ON \
  -DLBM_BUILD_SHOWCASES=ON -DLBM_TEST_OPENMP=OFF \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build/cuda --parallel 4
```

Use pinned staging, `amrex.use_gpu_aware_mpi=0`. A rank must receive the intended visible device; multi-rank GPU launch and affinity are environment-specific. Do not assume that a successful host consumer test establishes device assignment or GPU kernel correctness. The retained MIG performance launcher has a narrower documented [contract](../tests/performance/README.md); it is not a universal GPU launcher.

## Running and selecting tests

All showcase commands are run from the repository root. Their executables are placed in the top-level binary directory, for example `build/cpu/lbm_uniform_flow`, not under `build/cpu/showcases/`. Case-local build directories are also possible, but configure the same root: `cmake -S . -B showcases/uniform_flow/build ...`.

For CPU MPI, set ranks, threads and binding explicitly. With Open MPI, a typical non-hybrid launch is:

```bash
OMP_NUM_THREADS=1 mpirun -np 2 \
  --map-by core --bind-to core --rank-by slot --report-bindings \
  ./build/cpu/lbm_uniform_flow \
  showcases/uniform_flow/inputs/inputs.d2q9_guo_velocity_io
```

Use a compatible launcher syntax for other MPI implementations. Do not oversubscribe ranks × threads. Configure, build and run within an allocated compute environment when required by your system.

Inspect registration before running tests; plain `ctest` includes long numerical work:

```bash
ctest --test-dir build/cpu -N
ctest --test-dir build/cpu -L unit --output-on-failure -j1
```

Use one CTest driver per build tree. See [tests](../tests/README.md) for labels, scratch paths and source identity.

## Ordinary source archives

An ordinary archive contains project source and notices, but no Git history and no bundled AMReX checkout. Supply an external AMReX installation satisfying the same base/patch/configuration contract, then use the application configure command above with its actual `AMReX_ROOT`. Clone-only submodule/preparation commands are not required when that package already exists.

Ordinary builds can record an unknown project identity. Strict AMR numerical runners require the public commit corresponding to the archive. Obtain that full 40-character lowercase SHA from the downloaded revision's public metadata; never use an internal development SHA or an unrelated parent repository.

After assigning the verified value to `PUBLIC_COMMIT`:

```bash
export LBM_EXPECTED_GIT_COMMIT="$PUBLIC_COMMIT"
cmake -S . -B build/cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DAMReX_ROOT=/absolute/path/to/compatible-amrex-install \
  -DBUILD_TESTING=ON -DLBM_BUILD_SHOWCASES=ON \
  -DLBM_TEST_OPENMP=OFF \
  -DLBM_EXPECTED_GIT_COMMIT="$PUBLIC_COMMIT"
```

Keep the compiler/MPI selections consistent with the external package. The CMake cache and environment values must agree if both are supplied. In a clone, an explicit expected SHA must match the source repository HEAD. Missing/invalid/mismatched identity makes strict runners return `2`. CMake propagates the resolved identity to CTest; direct manual launches must receive the needed environment explicitly.

Record the actual compiler/MPI/CUDA versions, AMReX base SHA, patch SHA256, configure options and installation prefix with results. A base SHA alone does not identify a patched installation.
