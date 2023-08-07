import numpy as np
import pandas as pd
from datafold.pcfold import TSCDataFrame
import matplotlib.pyplot as plt

class Linear2dSystem:

    def __init__(self, A: np.ndarray, eig_i, eig_j, eigenfunction_i, eigenfunction_j) -> None:
        """_summary_

        Args:
            A (np.ndarray): 2d numpy matrix
            eig_i (_type_): first eigenvalue 
            eig_j (_type_): second eigenvalue
            eigenfunction_i (_type_): function handle for first koopman eigenfunction
            eigenfunction_j (_type_): function handle for second koopman eigenfunction
        """  
             
        assert A.shape == (2,2)
        self.A = A
        self.eig_i = eig_i
        self.eig_j = eig_j
        self.eigenfunction_i = eigenfunction_i
        self.eigenfunction_j = eigenfunction_j

    def sample_system(self, initial_conditions):
        """sample 2d system for one time step given array of initial conditions

        Args:
            initial_conditions (_type_): _description_
        """
        time_series_dfs = []

        for ic in initial_conditions:
            solution = self.A @ ic

            solution = pd.DataFrame(
                data= np.array([ic, solution]),
                index=[0,1],
                columns=["x1", "x2"],
            )

            time_series_dfs.append(solution)

        tsc_data = TSCDataFrame.from_frame_list(time_series_dfs)
        return tsc_data

    def generate_trajectory(self, initial_condition, n_steps = 100):
        """Generate a trajectory with given number of steps and initial condition

        Args:
            initial_condition (_type_): 2d numpy array
            n_steps (int, optional): _description_. Defaults to 100.

        Returns:
            _type_: _description_
        """
        df = [initial_condition.flatten()]
        solution = initial_condition.flatten()
        for s in np.arange(0, n_steps):
            solution = self.A@ solution
            df.append(solution)

        df = pd.DataFrame(df, columns=["x1", "x2"])
        return df

    def generate_eigenfunction(self, x, y, m = 1, n = 1):
        """generate explicit koopman eigenfunctions

        Args:
            x (_type_): x1 component
            y (_type_): x2 component
            m (int, optional): exponent for first eigenfunction is raised. Defaults to 1.
            n (int, optional): exponent for second eigenfunction is raised. Defaults to 1.

        Returns:
            _type_: _description_
        """    
        p = 1
        for i in range(m):
            p = p * self.eigenfunction_i(x,y)
        
        for i in range(n):
            p = p * self.eigenfunction_j(x,y)
        
        return p
    
    def get_sorted_eigvalues(self, max_exponent_sum=4):
        """get sorted koopman eigenvalues where eigenvalues are powers of explicit eigenvalues and their products
        
        Args:
            max_exponent_sum (int, optional): maximum sum of powers of eigenvalues. Defaults to 3.
        """

        eig_dict = {}
        for m in range(max_exponent_sum+1):
            for n in range(max_exponent_sum+1):
                if m+n>max_exponent_sum:
                    continue
                if m==0 and n==0:
                    continue
                eig = (self.eig_i)**m * (self.eig_j) ** n
                eig_dict[(m,n)] = eig
            
        sorted_eig = sorted(eig_dict, key=eig_dict.get, reverse=True)
        sorted_eig
        return sorted_eig
                     


    def plot_eigenfunction_contour(self, x, y,  m, n, ax, normalize=True):
        """Plot a contour map of a given powers of eigenfunction

        Args:
            x: x-range of grid
            y: y-range of grid
            m (_type_): exponent of first eigenfunction
            n (_type_): exponent of second eigenfunction
            ax (_type_): _description_
            normalize: normalize grid values to be between -1 and 1

        Returns:
            _type_: _description_
        """
        X, Y = np.meshgrid(x, y)
        Z = self.generate_eigenfunction(X, Y, m,n)
        if normalize:
            Z = Z/np.max(np.abs(Z))

        h = ax.contourf(X, Y, Z, cmap='RdYlBu_r')
        plt.colorbar(h) 


    def trajectory_error_power(self, koopman_eigen, x, p, eigenvector_index=0):        
        """Compute trajectory error for extended eigenfunction


        Args:
            koopman_eigen (KoopmanEigen): object of koopman eigen
            x (_type_): values of grid coordinates
            p (_type_): power
            eigenvector_index (int, optional): Eigenvector index to use for extending eigenfunction. Defaults to 0.
        """
        assert x.shape[1] == 2

        Z_l = koopman_eigen.extend_eigenfunctions(x, eigenvector_indexes=[eigenvector_index, 0], pow_i=p, pow_j=0, normalize=False)

        Z_l_t = koopman_eigen.extend_eigenfunctions((self.A@(x.T)).T, eigenvector_indexes=[eigenvector_index, 0], pow_i=p, pow_j=0, normalize=False)

        eig_c = koopman_eigen.left_koopman_eigvals[eigenvector_index]**p
        traj_error = np.linalg.norm(Z_l_t - eig_c*Z_l)/np.sqrt(x.shape[0])
        return traj_error

    def trajectory_bound_const(self, koopman_eigen, x, p, eigenval_index=0):
        """Compute constant of trajectory error bound

        Args:
            koopman_eigen (_type_): _description_
            x (_type_): _description_
            p (_type_): _description_
            eigenval_index (int, optional): _description_. Defaults to 0.
        """

        def x_error(z, p, eigvalue):
            z = z.reshape(1,z.shape[0])
    
            phi_z = koopman_eigen.dict_transform(z)
            phi_Az = koopman_eigen.dict_transform((self.A@z.T).T)
    
            a = np.linalg.norm(phi_Az - eigvalue *phi_z)            
            s = 0
            for j in range(0, p):
                s =+ (np.linalg.norm(phi_Az)**(p-1-j)) * (np.linalg.norm(phi_z)**j) * (eigvalue**j)
            
            if p == 1:
                return a
                
            else:
                return a*s

        eigvalue = koopman_eigen.left_koopman_eigvals[eigenval_index]

        c = np.linalg.norm(np.apply_along_axis(lambda z:x_error(z, p, eigvalue), 1, x))
        
        c = c/np.sqrt(x.shape[0])
        return c

   
