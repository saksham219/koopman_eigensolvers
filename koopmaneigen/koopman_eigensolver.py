import numpy as np
import pandas as pd
from datafold.appfold.edmd import EDMD
from datafold.utils.general import sort_eigenpairs
from datafold.pcfold import TSCDataFrame

class KoopmanEigenSolvers:
    def __init__(self, edmd: EDMD, tsc_data: TSCDataFrame):
        """_summary_

        Args:
            edmd (EDMD): edmd object
            tsc_data (TSCDataFrame): time series dataframe
        """        
    
        self.edmd = edmd
        self.tsc_data = tsc_data
        dictionary_eval = self.edmd.dict_steps[0][1].transform(tsc_data)
        dictionary_eval = self.edmd._attach_id_state(X=tsc_data, X_dict=dictionary_eval) #include id states
        self.koopman_matrix = self.edmd.dmd_model._compute_koopman_matrix(dictionary_eval)
        
        # calculate right eigenvectors
        self.right_koopman_eigvals, self.right_koopman_eigvecs = sort_eigenpairs(*np.linalg.eig(self.koopman_matrix))

        # calculate left eigenvectors
        self.left_koopman_eigvals, self.left_koopman_eigvecs = sort_eigenpairs(*np.linalg.eig(self.koopman_matrix.T))

    def dict_transform(self, x: np.ndarray):
        """Transform an array according to the edmd dictionary basis

        Args:
            x (np.ndarray): _description_
        """
        df = pd.DataFrame(x)
        df.columns = self.tsc_data.columns
        df.columns.name = "feature"

        dictionary_eval = self.edmd.dict_steps[0][1].transform(df)
        dictionary_eval = self.edmd._attach_id_state(X=df, X_dict=dictionary_eval).to_numpy()
        return dictionary_eval


    def eigenfunction_left(self, left_eigenvector):
        """Generate eigenfunction of the Koopman operator using a left eigenvector

        Args:
            left_eigenvector (np.ndarray or np.array): _description_
        """
        return lambda x: self.dict_transform(x) @ left_eigenvector
        

    def extend_eigenfunctions(self, x:np.ndarray, eigenvector_indexes = [0,1], pow_i = 1, pow_j = 1 ):
        """Generate a new set of eigenfunctions for the koopman operator using two left eigenvectors of the koopman matrix

        Args:
            pow_i (_type_): power to which computed eigenfunction is raised
            pow_j (_type_): power to which computed eigenfunction is raised
            x (np.ndarray): _description_
            eigenvector_indexes (list, optional): indexes of eigenvectors to be used to give extended eigenfunctions. Defaults to [0,1].
        """        

        assert len(eigenvector_indexes) == 2

        dictionary_eval = self.dict_transform(x)

        v_1_phi = (dictionary_eval @ self.left_koopman_eigvecs[:,eigenvector_indexes[0]])
        v_2_phi = (dictionary_eval @ self.left_koopman_eigvecs[:,eigenvector_indexes[1]])
       
        eigfunc_extended = (v_1_phi ** pow_i) * (v_2_phi ** pow_j)      
        eigfunc_extended = eigfunc_extended/np.max(np.abs(eigfunc_extended))

        return eigfunc_extended

            








