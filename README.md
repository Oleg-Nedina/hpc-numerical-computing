##  Project 1: Spectral Clustering on Social Networks
* **Goal:** Identify communities within a social network graph (351+ nodes) using **Spectral Clustering**.
* **Math:** Computation of the **Fiedler Vector** (eigenvector of the second smallest Laplacian eigenvalue) to partition the graph.
* **Tech:** Iterative solvers for eigenvalue problems, sparse matrix manipulation in `Eigen`.
* **Result:** Successfully separated tightly connected components (communities) by analyzing the graph Laplacian spectrum.

## Project 2: Image Filtering & Denoising
* **Goal:** Apply convolution filters (Smoothing, Sharpening, Sobel Edge Detection) to greyscale images.
* **Math:** Solving linear systems ($Ax=b$) to reverse blur/noise effects.
* **Tech:** Matrix-vector multiplication optimization, Preconditioned iterative solvers.

## 🛠 Technologies
* **C++ / C**
* **Eigen Library** for linear algebra.
* **LIS Library** for iterative solvers.
* **MPI / OpenMP** for parallelization strategies(not fully utilized).
