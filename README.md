# EfficientReductionTechniques

This repository explores a novel technique for load balancing and optimizing runtime checks in reduction problems. The method aims to minimize conditional checks while maintaining effective load distribution, leading to improved performance in parallel computing environments. The technique is generalized for a variety of reduction problems.

## Key Features:
- Load balancing for efficient computation.
- Minimization of runtime conditional checks.
- Generalized for multiple reduction problem types.
- Performance improvements over traditional approaches.

## Possible pitfalls while running the code

1) Make sure the values are within the bounds of datatype used.  
   a) For example while using cudaMalloc(), ```n*sizeof(T)``` is used, but ```n*sizeof(T)``` should be able to fit in ```size_t``` datatype.
2) Next point
  
