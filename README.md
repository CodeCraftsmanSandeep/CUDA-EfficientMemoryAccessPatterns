# CUDA-Efficient Memory Access Patterns

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/username/repository)
[![CUDA Version](https://img.shields.io/badge/CUDA-sm_75-blue.svg)](https://developer.nvidia.com/cuda-zone)

This repository presents state-of-the-art memory access patterns tailored for CUDA applications. It extends the widely adopted grid-stride access paradigm with innovative techniques that enhance memory coalescence, spatial locality, and load balancing, while minimizing warp divergence.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Compilation and Execution](#compilation-and-execution)
- [Benchmark Results](#benchmark-results)
- [Usage and Extensions](#usage-and-extensions)
- [License](#license)
- [Contact](#contact)

---

## Overview

Efficient global memory access is crucial in GPU computing. Our implementation extends traditional grid-stride loops by incorporating block-stride and warp-stride techniques to achieve:
- **Maximized memory coalescing**
- **Enhanced spatial locality for each CUDA thread**
- **Even workload distribution and efficient load balancing**
- **Minimal warp divergence (at most one divergence per warp)**

These improvements directly translate into superior performance for reduction operations and can be readily applied to other HPC kernels.

---

## Key Features

| Feature                                | Description                                                                                         |
|----------------------------------------|-----------------------------------------------------------------------------------------------------|
| **Coalesced Memory Access**            | Optimizes memory transactions for maximum throughput.                                             |
| **Improved Spatial Locality**          | Enhances data access patterns compared to conventional grid-stride methods.                         |
| **Efficient Load Balancing**           | Ensures even distribution of work across threads to eliminate idle cycles.                         |
| **Minimal Warp Divergence**            | Limits divergence to a maximum of one per warp, preserving parallel efficiency.                     |

---

## Repository Structure

```plaintext
CUDA-Efficient-Memory-Access-Patterns/
├── ourImplementations/
│   │── efficientStridePatterns/
│   │   ├── blockStrideTreeReduction.cu
│   │   ├── blockStrideTreeReduction.out
│   │   ├── gridStrideTreeReduction.cu
│   │   ├── gridStrideTreeReduction.out
│   │   ├── warpStrideTreeReduction.cu
│   │   ├── warpStrideTreeReduction.out
│   │   ├── warpStrideTreeReductionUnrolled4.cu
│   │   ├── warpStrideTreeReductionUnrolled4.out
│   │   ├── warpStrideTreeReductionUnrolled16.cu
│   │   └── warpStrideTreeReductionUnrolled16.out
│   └── src.cu
├── thrustReduction/
│   ├── thrustReduction.cu
│   └── thrustReduction.out
└── README.md
```

---

## Compilation and Execution

### Building the Code

To compile the implementations, use the NVIDIA CUDA Compiler (`nvcc`). For example, to compile the block-stride tree reduction, run:

```bash
nvcc -arch=sm_75 ourImplementations/efficientStridePatterns/blockStrideTreeReduction.cu ourImplementations/src.cu -o ourImplementations/efficientStridePatterns/blockStrideTreeReduction.out
```

> **Note:** Replace `sm_75` with the appropriate compute capability for your GPU if necessary.

### Running the Executable

After compilation, you can run the executable as follows:

```bash
./ourImplementations/efficientStridePatterns/blockStrideTreeReduction.out < input > output
```

---

## Benchmark Results

Below is a summary of benchmark performance comparing our optimized implementations with traditional grid-stride loops and Thrust-based reductions:

| Implementation                           | Speedup vs. Grid-Stride | Speedup vs. Thrust Reduction |
|------------------------------------------|-------------------------:|-----------------------------:|
| **Warp-Strided Reduction**               |            **+50%**      |            **+18%**          |
| Block-Strided Reduction                  |            +25%         |            _              |
| Traditional Grid-Stride Reduction        |             Baseline     |            Baseline          |

*Results obtained on NVIDIA GPU (Compute Capability 7.5) using large input sizes typical of HPC applications.*

---

## Usage and Extensions

Researchers and developers are encouraged to:
- **Examine and modify the code:** Explore the implementations in the `ourImplementations/efficientStridePatterns/` directory.
- **Integrate with other projects:** The modular design allows easy incorporation into broader HPC frameworks.
- **Contribute improvements:** Submit pull requests or open issues to enhance functionality or extend support to additional access patterns.

Comprehensive documentation is provided within the repository to facilitate reproduction and extension of our results.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For questions, collaborations, or further information, please contact:
- **Name:** Chekkala Sandeep Reddy
- **Email:** sandeep.chekkala.wh@gmail.com
- **GitHub:** https://github.com/CodeCraftsmanSandeep

---

*We welcome your contributions to further advance the field of GPU-accelerated high-performance computing!*
