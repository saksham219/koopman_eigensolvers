import numpy as np
import pandas as pd
from datafold.pcfold import TSCDataFrame

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
    
    def get_sorted_eigvalues(self, max_exponent_sum=3):
        """get sorted koopman eigenvalues where eigenvalues are powers of explicit eigenvalues and their products
        
        Args:
            max_exponent_sum (int, optional): maximum sum of powers of eigenvalues. Defaults to 3.
        """

        eig_dict = {}
        for m in range(4):
            for n in range(4):
                if m+n>3:
                    continue
                if m==0 and n==0:
                    continue
                eig = (self.eig_i)**m * (self.eig_j) ** n
                eig_dict[(m,n)] = eig
            
        sorted_eig = sorted(eig_dict, key=eig_dict.get, reverse=True)
        sorted_eig
        return sorted_eig
                     

