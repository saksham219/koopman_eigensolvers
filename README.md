Code for the master thesis- <strong>Fast Eigensolvers for Koopman Operator
Approximation.</strong>

The following classes contain the code for extending eigenfunctions and Koopman eigensolver algorithms for linear and non-linear systems:

- MatrixEigSolver: contains algorithms for finding eigenpairs of a general real matrix
- Linear2dSystem: methods for linear discrete system.
- Linear2dSystemContinuous: methods for linear continuous system.
- KoopmanEigenSolvers: methods for EDMD analysis and extending eigenfunctions.
- KoopmanEigenSolversDMD: methods for DMD analysis and extending eigenfunctions.
- NonLinear2dSystem: methods for non-linear system.

The following jupyter notebooks contain the analysis and plots:

- `extend_linear_system_eigenfunctions_DMD.ipynb`: Analysis for discrete linear system using DMD.
- `extend_linear_system_eigenfunctions.ipynb`: Analysis for discrete linear system using EDMD.
- `linear_system_continuous.ipynb`: Analysis for continuous linear system.
- `new_nonlln.ipynb`: Analysis for non-linear system.
- `non_linear_from_linear_system.ipynb`: Analysis for non-linear system transformed using linear system.

### Dependencies

- Datafold: 1.1.6
- Numpy: 1.21.6
- Scipy: 1.10.0
