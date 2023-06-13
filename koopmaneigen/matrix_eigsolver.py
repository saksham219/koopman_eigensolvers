import numpy as np
import copy

class MatrixEigSolver:

    def __init__(self, A):
        self.A = A


    def power_iteration(self, A, max_iterations = 100000, tolerance=1e-11):
        """Run power iteration on a matrix

        Args:
            A: numpy matrix
            max_iterations (int, optional): _description_. Defaults to 100000.
            tolerance (_type_, optional): _description_. Defaults to 1e-11.

        Returns:
            _type_: eigenvector and eigenvalue pair
        """


        eig_old = 0
        x = np.random.rand(A.shape[1])
        x = x/np.linalg.norm(x)
        for i in range(max_iterations):
            y = np.dot(A, x)
            norm = np.linalg.norm(y)
            x = y / norm 
            eigenvalue = np.dot(np.dot(x, A), x.T)
            if np.abs(eig_old - eigenvalue) < tolerance:
                return x, eigenvalue
            else:
                eig_old = eigenvalue
        
        print("max iter reached", eig_old - eigenvalue)
        return x, eigenvalue

    def power_iteration_with_deflation_asymm(self, num_eigen, num_iterations=100000, 
                                            tolerance=1e-8):
        """Run power iteration with deflation for an asymmetric matrix

        Args:
            num_eigen (_type_): number of eigenpairs to compute
            num_iterations (_type_): number of iterations for power method. Defaults to 100000.
            tolerance (_type_, optional): _description_. Tolerance for power method Defaults to 1e-8.

        Returns:
            _type_: _description_
        """
        eigs = []
        A  = copy.deepcopy(self.A)
        for i in range(num_eigen):
            # print(f"eigenvalue {i+1}")
            # get right eigenvector
            v, l = self.power_iteration(A, num_iterations, tolerance)

            # get left eigenvector
            f, l_d = self.power_iteration(A.T, num_iterations, tolerance)
            
            v = v.reshape(v.shape[0],1)
            f = f.reshape(f.shape[0],1)
            
            # normalize left eigenvector
            f_norm = f/(f.T@v)
            
            # print("f^Tv: ", f_norm.T@v) 
            # print("---------")

            eigs.append((v, l))

            # deflate matrix using left eigenvector
            A = A - l * v @ f_norm.T
        
        return eigs