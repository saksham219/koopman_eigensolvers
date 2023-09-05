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
            eigenvalue = np.dot(np.dot(A, x), x.T)
            if np.abs(eig_old - eigenvalue) < tolerance:
                print("iterations: ", i)
                return x, eigenvalue
            else:
                eig_old = eigenvalue
        
        print("max iter reached in power method", eig_old - eigenvalue)
        return x, eigenvalue

    
    # method based on arnoldi iteration
    def power_iteration_complex_eigs(self, A, max_iterations = 10000, tolerance=1e-10):

        n = A.shape[0]
        m = 2 # arnoldi krylow subspace dimension
        # x = np.random.rand(n) + 1j* np.random.rand(n)
        x = np.random.rand(n)
        x = x/np.linalg.norm(x)
        
        # run a few steps of power iteration
        # how to choose this max_iter?
        assert 1000<max_iterations
        x, _ = self.power_iteration(A, max_iterations=500)
        
        h = np.zeros((m+1, m), dtype=complex)
        V = np.zeros((n, m+1), dtype=complex)
    
        eig_old = np.inf
        for iter_ in range(max_iterations):
            V[:,0] = x

            for j in range(2):
                w = A@V[:,j]

                # modified gram-schmidt
                for i in range(j+1):
                    h[i,j] = V[:,i].conj().T @ w # a.conj().T @ b not equal to b.conj().T @a
                    w = w - h[i,j]*V[:,i]

                h[j+1,j] = np.linalg.norm(w)
                if (h[j+1,j] ==0):
                    break

                V[:,j+1] = w/h[j+1,j]

            # find eigenvalues and eigenvectors of 2x2hessenburg matrix h explicitly
            lambda_1 = 0.5 * ((h[0,0] + h[1,1]) + np.sqrt((h[0,0] + h[1,1])**2 - 4*(h[0,0] * h[1,1] - h[0,1] * h[1,0])))
            lambda_2 = 0.5 * ((h[0,0] + h[1,1]) - np.sqrt((h[0,0] + h[1,1])**2 - 4*(h[0,0] * h[1,1] - h[0,1] * h[1,0])))

            l = np.array([lambda_1, lambda_2])
            # get largest eigenvalue
            lambda_1 = l[np.argmax(np.abs([lambda_1, lambda_2]))]

            # get eigenvector corresponding to the largest eigenvalue
            eigvec = np.array([1, h[1,0]/(lambda_1 - h[1,1]) ])
            # eigvec = np.array([h[0,1], lambda_1  - h[0,0]])
            # eigvec = np.array([lambda_1 - h[1,1], h[1,0]])

            eigvec = eigvec/np.linalg.norm(eigvec)
           
            # get eigenvector of original matrix
            x = V[:,0:2] @ eigvec
            x = x/np.linalg.norm(x)

            if np.abs(eig_old - lambda_1) < tolerance:
                print(f"tol achieved in {iter_} iterations")
                return (x, lambda_1)
            else:
                eig_old = lambda_1
            
        print("max iter reached in above method")
        return x, lambda_1



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

    def power_iteration_with_deflation_asymm_complex(self, num_eigen, num_iterations=100000, 
                                            tolerance=1e-8):
        """Run power iteration with deflation for a real asymmetric matrix that might have 
        complex eigenvalues (occuring in conjugate pairs)

        Args:
            num_eigen (_type_): number of eigenpairs to compute
            num_iterations (_type_): number of iterations for power method. Defaults to 100000.
            tolerance (_type_, optional): _description_. Tolerance for power method Defaults to 1e-8.

        Returns:
            _type_: _description_
        """
        eigs = []
        left_eigs = []
        A  = copy.deepcopy(self.A)

        eig_iter = 0

        while eig_iter < num_eigen:
            # print(f"eigenvalue {i+1}")
            # get right eigenvector
            print(f"eigenvalue {eig_iter+1}")
            v, l = self.power_iteration_complex_eigs(A, num_iterations, tolerance)      
            
            # get left eigenvector
            f, l_d = self.power_iteration_complex_eigs(A.T, num_iterations, tolerance)

            # in the case where eigval is complex, both l and l_d should have the same imaginary part
            if (np.abs(l.imag) > 1e-5) and (np.imag(l) + np.imag(l_d) < 1e-6):
                v = v.conj()
                l = l.conj()

            v = v.reshape(v.shape[0],1)
            f = f.reshape(f.shape[0],1)

            # normalize left eigenvector
            f_norm = f/(f.T@v)
            # f_norm = f/(f.conj().T@v)

            eigs.append((v, l)) # why add the eigenvectors here after normalization?
            left_eigs.append((f, l_d))

            # deflate matrix using left eigenvector
            A = A - l * v @ f_norm.T
            
            print("lambda: ", l)
            
            # deflate with conjugate eigenvalue if eigenvalue found has suffieciently large complex part
            if np.abs(l.imag) > 1e-6:
            
                l_c = l.conj()
                f_c = f.conj()
                v_c = v.conj()
                
                eigs.append((v_c, l_c))
                left_eigs.append((f_c, l_d.conj()))
                f_norm_c = f_c/(f_c.T@ v_c)
                
                A = A - l_c * v_c @ f_norm_c.T
                
                num_eigen -=1
            eig_iter +=1
            print("---------------------------------")
        
        return eigs, left_eigs


