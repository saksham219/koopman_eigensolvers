import numpy as np
import pandas as pd
from datafold.pcfold import TSCDataFrame
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class NonLinear2dSystem:
    def __init__(
        self,
        F,
        explicit_sol=None,
        eig_i=None,
        eig_j=None,
        eigenfunction_i=None,
        eigenfunction_j=None,
    ) -> None:
        """_summary_

        Args:
            F (function): Function handle for non linear system vector field
            explicit_sol (function): Function handle for explicit solution of the system (Takes three inputs -x1,x2,t)
            eig_i (_type_): first eigenvalue of matrix
            eig_j (_type_): second eigenvalue of matrix
            eigenfunction_i (_type_): function handle for first koopman eigenfunction
            eigenfunction_j (_type_): function handle for second koopman eigenfunction
        """
        # assert F.shape == (2,2)
        self.F = F
        self.explicit_sol = explicit_sol
        self.eig_i = eig_i
        self.eig_j = eig_j
        self.eigenfunction_i = eigenfunction_i
        self.eigenfunction_j = eigenfunction_j

    def sample_system(self, initial_conditions, t_sample):
        """sample 2d system for one time step given array of initial conditions

        Args:
            initial_conditions (_type_): _description_
            t_sample: array or single time value
        """
        time_series_dfs = []

        assert initial_conditions.ndim == 2
        assert initial_conditions.shape[1] == 2

        time_series_dfs = []

        for ic in initial_conditions:
            if self.explicit_sol:
                assert t_sample.__class__ == np.float64 or t_sample.__class__ == float
                solution = self.explicit_sol(*ic, t_sample)
                solution = pd.DataFrame(
                    data=np.array([ic, solution]),
                    index=[0, t_sample],
                    columns=["x1", "x2"],
                )
            else:
                solution = solve_ivp(
                    self.F, t_span=(t_sample[0], t_sample[-1]), y0=ic, t_eval=t_sample
                )
                solution = pd.DataFrame(
                    data=solution["y"][:, 0:2].T,
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

        if self.explicit_sol:
            df = [initial_condition.flatten()]
            # solution = initial_condition.flatten()
            # if t_eval[0] == 0:
            for t in t_eval[1:]:
                solution = self.explicit_sol(*initial_condition, t)
                df.append(solution)

            df = pd.DataFrame(df, columns=["x1", "x2"])
            return df

        return solve_ivp(
            self.F, t_span=(t_eval[0], t_eval[-1]), y0=initial_condition, t_eval=t_eval
        )["y"].T

    def trajectory_error_power_eigvec(
        self,
        x,
        koopman_eigen,
        eigvec: np.array,
        eig: float,
        p: int,
        t_eval: np.array,
        integration_method="RK45",
    ):
        """Get trajectory error for eigenpair with exponent p

        Args:
            x (_type_): _description_
            koopman_eigen (_type_): _description_
            eigvec (np.array): _description_
            eig (float): _description_
            p (int): _description_
            t_eval (np.array): _description_
            integration_method (str, optional): _description_. Defaults to "RK45".
        """
        eigfunc = koopman_eigen.eigenfunction_left(eigvec)

        if integration_method == "euler":
            raise Exception("not implemented")
            t_sample = t_eval[1]
            Tx = np.apply_along_axis(
                lambda x: euler_method(x, t_sample, h=0.0001), 1, x
            )
        else:
            Tx = np.apply_along_axis(
                lambda ic: solve_ivp(
                    self.F,
                    t_span=(t_eval[0], t_eval[-1]),
                    y0=ic,
                    t_eval=t_eval,
                    method=integration_method,
                )["y"][:, 1],
                1,
                x,
            )

        traj_error = np.linalg.norm(
            (eigfunc(Tx) ** p) - (eig**p) * (eigfunc(x)) ** p
        ) / np.sqrt(x.shape[0])
        return traj_error ** (1 / p)

    def trajectory_error_power(
        self, koopman_eigen, x, p, eigenvector_index=0, delv_norm=None
    ):
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
            Z_l = koopman_eigen.extend_eigenfunctions(
                x,
                eigenvector_indexes=[eigenvector_index, 0],
                pow_i=p,
                pow_j=0,
                normalize=False,
            )

            Z_l_t = koopman_eigen.extend_eigenfunctions(
                (self.A @ (x.T)).T,
                eigenvector_indexes=[eigenvector_index, 0],
                pow_i=p,
                pow_j=0,
                normalize=False,
            )

        if delv_norm:
            delv = np.random.rand(2)
            delv = (delv / np.linalg.norm(delv)) * delv_norm
            print("using delvnorm: ", np.linalg.norm(delv))

            Z_l = koopman_eigen.extend_eigenfunctions_delv(
                x,
                delv=delv,
                eigenvector_indexes=[eigenvector_index, 0],
                pow_i=p,
                pow_j=0,
                normalize=False,
            )

            Z_l_t = koopman_eigen.extend_eigenfunctions_delv(
                (self.A @ (x.T)).T,
                delv=delv,
                eigenvector_indexes=[eigenvector_index, 0],
                pow_i=p,
                pow_j=0,
                normalize=False,
            )

        eig_c = koopman_eigen.left_koopman_eigvals[eigenvector_index] ** p
        traj_error = np.linalg.norm(Z_l_t - eig_c * Z_l) / np.sqrt(x.shape[0])
        return traj_error

    def trajectory_bound_const(
        self, koopman_eigen, x, p, eigenval_index=0, delv_norm=None
    ):
        """Compute constant of trajectory error bound

        Args:
            koopman_eigen (_type_): _description_
            x (_type_): _description_
            p (_type_): _description_
            eigenval_index (int, optional): _description_. Defaults to 0.
            delv_norm (float, optional): norm of error vector added to left eigenvectors. Defaults to None
        """

        def x_error(z, p, eigvalue):
            z = z.reshape(1, z.shape[0])

            x = x.reshape(1, x.shape[0])

            phi_x = koopman_eigen.dict_transform(x)
            Tx_flat = np.apply_along_axis(
                lambda ic: solve_ivp(
                    limit_cycle, t_span=(t_eval[0], t_eval[-1]), y0=ic, t_eval=t_eval
                )["y"][:, 1],
                1,
                x,
            )

            phi_Tx = koopman_eigen.dict_transform(Tx_flat)

            a = np.linalg.norm(phi_Tx - eigvalue * phi_x)
            assert phi_z.shape[0] == 1
            assert phi_Az.shape[0] == 1

            a = np.linalg.norm(phi_Az - eigvalue * phi_z)
            s = 0
            for j in range(0, p):
                s += (
                    (np.linalg.norm(phi_Az) ** (p - 1 - j))
                    * (np.linalg.norm(phi_z) ** j)
                    * (eigvalue**j)
                )

            return a * s

        eigvalue = koopman_eigen.left_koopman_eigvals[eigenval_index]

        c = np.linalg.norm(np.apply_along_axis(lambda z: x_error(z, p, eigvalue), 1, x))

        c = c / np.sqrt(x.shape[0])

        if delv_norm:
            c = delv_norm * c

        return c

    def get_sorted_eigvalues(self, max_exponent_sum=4):
        """get sorted koopman eigenvalues where eigenvalues are powers of explicit eigenvalues and their products

        Args:
            max_exponent_sum (int, optional): maximum sum of powers of eigenvalues. Defaults to 3.
        """

        eig_dict = {}
        for m in range(max_exponent_sum + 1):
            for n in range(max_exponent_sum + 1):
                if m + n > max_exponent_sum:
                    continue
                if m == 0 and n == 0:
                    continue
                eig = (self.eig_i) ** m * (self.eig_j) ** n
                eig_dict[(m, n)] = eig

        sorted_eig = sorted(eig_dict, key=eig_dict.get, reverse=True)
        return sorted_eig

    def c_error(self, a, b):
        """function to check if one vector is a constant multiple of another

        Args:
            a (_type_): np array
            b (_type_): np array

        Returns:
            _type_: _description_
        """
        return np.abs(
            (a.conj().T @ b) * (a.conj().T @ b) - (a.conj().T @ a) * (b.conj().T @ b)
        )

    def euler_method(self, ic, t, h=0.1):
        """_summary_

        Args:
            ic (_type_): _description_
            t (_type_): _description_
            h (float, optional): _description_. Defaults to 0.1.

        Returns:
            _type_: _description_
        """

        t_eval = np.arange(0, t, h)
        # Explicit Euler Method

        assert ic.ndim == 1
        s = np.zeros((len(t_eval), ic.shape[0]))

        s[0, :] = ic
        for i in range(0, len(t_eval) - 1):
            s[i + 1, :] = s[i, :] + h * (self.A @ (s[i, :].T))

        return s[-1, :]

    def get_euler_eps_G(self, x, t_eval, integration_method="RK45"):
        if integration_method == "euler":
            raise Exception("not implemented")
            t_sample = t_eval[1]
            Tx_flat = np.apply_along_axis(
                lambda x: self.euler_method(x, t_sample, h=0.0001), 1, x
            )

        else:
            Tx_flat = np.apply_along_axis(
                lambda ic: solve_ivp(
                    self.F,
                    t_span=(t_eval[0], t_eval[-1]),
                    y0=ic,
                    t_eval=t_eval,
                    method=integration_method,
                )["y"][:, 1],
                1,
                x,
            )

        t_sample = t_eval[-1] / (t_eval.shape[0] - 1)

        Tx_exact = self.explicit_sol(x[:, 0], x[:, 1], t_sample).T

        assert Tx_exact.shape == Tx_flat.shape

        epsilon_x = Tx_flat - Tx_exact
        epsilon_G = np.apply_along_axis(lambda _: np.linalg.norm(_), 1, epsilon_x).max()
        #     epsilon_G = np.linalg.norm(epsilon_x)/np.sqrt(epsilon_x.shape[0])
        return epsilon_G

    def trajectory_bound_integration(self, koopman_eigen, x, p, epsilon_G, eigvalue):
        if "polynomial" in koopman_eigen.edmd.named_steps:
            J = koopman_eigen.calculate_jacobian_handle()
        else:
            J = None

        M = np.apply_along_axis(
            lambda z: np.linalg.norm(z), 1, koopman_eigen.dict_transform(x)
        ).max()

        #     M_F = np.apply_along_axis(lambda x: np.linalg.norm(x), 1, (self.A@(x.T)).T).max()

        #     bound_1 = (eigvalue * M  + L* epsilon_G)**p - (eigvalue * M)**p
        #     bound_2 = (L**p) * ((M_F  + epsilon_G)**p - (M_F)**p)

        jac_singular_max = np.apply_along_axis(
            lambda _: koopman_eigen.compute_singular_max(J, _), 1, x
        ).max()
        #     jac_singular_max = np.apply_along_axis(lambda _: np.linalg.norm(L(*_)), 1, x).max()
        #     print("jac_singular_max", jac_singular_max)
        bound_3 = (eigvalue * M + jac_singular_max * epsilon_G) ** p - (
            eigvalue * M
        ) ** p

        return bound_3 ** (1 / p)

    def trajectory_bound_given_epsilon(self, koopman_eigen, x, p, epsilon, eigvalue):
        assert x.shape[1] == 2

        if "polynomial" in koopman_eigen.edmd.named_steps:
            J = koopman_eigen.calculate_jacobian_handle()
        else:
            J = None
        M = np.apply_along_axis(
            lambda z: np.linalg.norm(z), 1, koopman_eigen.dict_transform(x)
        ).max()

        jac_singular_max = np.apply_along_axis(
            lambda y: koopman_eigen.compute_singular_max(J, y), 1, x
        ).max()
        #     print("jac_singular_max", jac_singular_max)

        return (1 / jac_singular_max) * (
            (epsilon**p + (eigvalue * M) ** p) ** (1 / p) - eigvalue * M
        )
