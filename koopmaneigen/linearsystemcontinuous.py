import numpy as np
import pandas as pd
from datafold.pcfold import TSCDataFrame
import matplotlib.pyplot as plt
import scipy.linalg as ln

class Linear2dSystemContinuous:

    def __init__(self, A: np.ndarray, eig_i, eig_j, t_sample, eigenfunction_i, eigenfunction_j) -> None:
        """_summary_

        Args:
            A (np.ndarray): 2d numpy matrix
            eig_i (_type_): first eigenvalue of matrix 
            eig_j (_type_): second eigenvalue of matrix
            t_sample (float): sampling interval
            eigenfunction_i (_type_): function handle for first koopman eigenfunction
            eigenfunction_j (_type_): function handle for second koopman eigenfunction
        """  
             
        assert A.shape == (2,2)
        # assert np.isclose(V@V_inv, np.eye(2)).all()
        # assert np.isclose(V_inv@V, np.eye(2)).all()
        # assert np.isclose(A@V, V@np.diag([eig_i, eig_j])).all()

        self.A = A
        self.eig_i = eig_i
        self.eig_j = eig_j
        self.t_sample = t_sample
        self.eigenfunction_i = eigenfunction_i
        self.eigenfunction_j = eigenfunction_j

    def sample_system(self, initial_conditions):
        """sample 2d system for one time step given array of initial conditions

        Args:
            initial_conditions (_type_): _description_
        """
        time_series_dfs = []

        for ic in initial_conditions:
            # solution = self.V @ np.diag([np.exp(self.eig_i * t_sample), np.exp(self.eig_j*t_sample)]) @ self.V_inv @ ic
            solution = ln.expm(self.A*self.t_sample) @ ic 
            solution = pd.DataFrame(
                data= np.array([ic, solution]),
                index=[0,self.t_sample],
                columns=["x1", "x2"],
            )

            time_series_dfs.append(solution)

        tsc_data = TSCDataFrame.from_frame_list(time_series_dfs)
        return tsc_data

    def generate_trajectory(self, initial_condition, t_eval):
        """Generate a trajectory with given number of steps and initial condition

        Args:
            initial_condition (_type_): 2d numpy array
            t_eval (np.array): _description_. time array.

        Returns:
            _type_: _description_
        """
        df = [initial_condition.flatten()]
        
        # solution = initial_condition.flatten()
        # if t_eval[0] == 0:
        for t in t_eval[1:]:
            solution = ln.expm(self.A * t) @ initial_condition
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


    def trajectory_error_power(self, koopman_eigen, x, p, eigenvector_index=0, delv_norm=None):  
        """Compute trajectory error for extended eigenfunction


        Args:
            koopman_eigen (KoopmanEigen): object of koopman eigen
            x (_type_): values of grid coordinates
            p (_type_): power
            eigenvector_index (int, optional): Eigenvector index to use for extending eigenfunction. Defaults to 0.
            delv_norm (float, optional): norm of error vector added to left eigenvectors. Defaults to None
        """
        assert x.shape[1] == 2
        
        if not delv_norm:
            Z_l = koopman_eigen.extend_eigenfunctions(x, eigenvector_indexes=[eigenvector_index, 0], pow_i=p, pow_j=0, normalize=False)

            Z_l_t = koopman_eigen.extend_eigenfunctions((ln.expm(self.A*self.t_sample) @ (x.T)).T, eigenvector_indexes=[eigenvector_index, 0], pow_i=p, pow_j=0, normalize=False)

        if delv_norm:
            delv = np.random.rand(2)
            delv = (delv/np.linalg.norm(delv)) * delv_norm
            print("using delvnorm: ",np.linalg.norm(delv))
            
            Z_l = koopman_eigen.extend_eigenfunctions_delv(x, delv=delv, eigenvector_indexes=[eigenvector_index, 0], pow_i=p, pow_j=0, normalize=False)

            Z_l_t = koopman_eigen.extend_eigenfunctions_delv((ln.expm(self.A*self.t_sample) @ (x.T)).T, delv=delv, eigenvector_indexes=[eigenvector_index, 0], pow_i=p, pow_j=0, normalize=False)


        eig_c = koopman_eigen.left_koopman_eigvals[eigenvector_index]**p
        traj_error = np.linalg.norm(Z_l_t - eig_c*Z_l)/np.sqrt(x.shape[0])
        return traj_error

    def trajectory_bound_const(self, koopman_eigen, x, p, eigenval_index=0, delv_norm=None):
        """Compute constant of trajectory error bound

        Args:
            koopman_eigen (_type_): _description_
            x (_type_): _description_
            p (_type_): _description_
            eigenval_index (int, optional): _description_. Defaults to 0.
            delv_norm (float, optional): norm of error vector added to left eigenvectors. Defaults to None
        """

        def x_error(z, p, eigvalue):
            z = z.reshape(1,z.shape[0])
    
            phi_z = koopman_eigen.dict_transform(z)
            phi_Az = koopman_eigen.dict_transform((ln.expm(self.A*self.t_sample) @ (z.T)).T)

            assert phi_z.shape[0] == 1
            assert phi_Az.shape[0] == 1

            a = np.linalg.norm(phi_Az - eigvalue *phi_z)            
            s = 0
            for j in range(0, p):
                s += (np.linalg.norm(phi_Az)**(p-1-j)) * (np.linalg.norm(phi_z)**j) * (eigvalue**j)

            return a*s

        eigvalue = koopman_eigen.left_koopman_eigvals[eigenval_index]

        c = np.linalg.norm(np.apply_along_axis(lambda z:x_error(z, p, eigvalue), 1, x))
        
        c = c/np.sqrt(x.shape[0])

        if delv_norm:
            c = delv_norm * c

        return c

    def euler_method(self, ic, t, h=0.1):

        t_eval = np.arange(0, t, h)
        
        # Explicit Euler Method
        s = np.zeros((len(t_eval), ic.shape[0]))
        s[0,:] = ic

        for i in range(0, len(t_eval) - 1):
            s[i + 1,:] = s[i, :] + h*(self.A@(s[i, :]))

        return s[-1,:]

   
