import numpy as np
import pandas as pd
from datafold.pcfold import TSCDataFrame
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class NonLinear2dSystem:

    def __init__(self, F:function) -> None:
        """_summary_

        Args:
            F (function): Function handle for non linear system vector field
        """  
        # assert F.shape == (2,2)
        self.F = F

    def sample_system(self, initial_conditions, t_sample):
        """sample 2d system for one time step given array of initial conditions

        Args:
            initial_conditions (_type_): _description_
            t_sample:
        """
        time_series_dfs = []

        assert initial_conditions.ndim == 2
        assert initial_conditions.shape[1] == 2

        time_series_dfs = []

        for ic in initial_conditions:
            # get first two values
            solution = solve_ivp(
                self.F, t_span=(t_sample[0], t_sample[-1]), y0=ic, t_eval=t_sample
            )

            solution = pd.DataFrame(
                data=solution["y"][:,0:2].T,
                index=solution["t"][0:2],
                columns=["x1", "x2"],
            )
            
            time_series_dfs.append(solution)

        return TSCDataFrame.from_frame_list(time_series_dfs)

    def generate_trajectory(self, initial_condition, t_eval):
        """Generate a trajectory with given number of steps and initial condition

        Args:
            initial_condition (_type_): 2d numpy array
            t_eval (np.array, optional): _description_. Time steps

        Returns:
            _type_: _description_
        """
        return solve_ivp(
            self.F, t_span=(t_eval[0], t_eval[-1]), y0=initial_condition, t_eval=t_eval
        )["y"].T

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

            Z_l_t = koopman_eigen.extend_eigenfunctions((self.A@(x.T)).T, eigenvector_indexes=[eigenvector_index, 0], pow_i=p, pow_j=0, normalize=False)

        if delv_norm:
            delv = np.random.rand(2)
            delv = (delv/np.linalg.norm(delv)) * delv_norm
            print("using delvnorm: ",np.linalg.norm(delv))
            
            Z_l = koopman_eigen.extend_eigenfunctions_delv(x, delv=delv, eigenvector_indexes=[eigenvector_index, 0], pow_i=p, pow_j=0, normalize=False)

            Z_l_t = koopman_eigen.extend_eigenfunctions_delv((self.A@(x.T)).T, delv=delv, eigenvector_indexes=[eigenvector_index, 0], pow_i=p, pow_j=0, normalize=False)


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
    
            x = x.reshape(1,x.shape[0])
    
            phi_x = koopman_eigen.dict_transform(x)
            Tx_flat = np.apply_along_axis(lambda ic: solve_ivp(limit_cycle, t_span=(t_eval[0],t_eval[-1]), y0=ic, t_eval = t_eval)["y"][:,1],1, x)
            
            phi_Tx = koopman_eigen.dict_transform(Tx_flat)
            
            a = np.linalg.norm(phi_Tx - eigvalue *phi_x)
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

   
