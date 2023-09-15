from xmlrpc.client import boolean
import numpy as np
import pandas as pd
from datafold.appfold.edmd import EDMD
from datafold.utils.general import sort_eigenpairs
from datafold.pcfold import TSCDataFrame
import sympy as sym

class KoopmanEigenSolvers:
    def __init__(self, edmd: EDMD, tsc_data: TSCDataFrame, include_id_state: boolean=True):
        """_summary_

        Args:
            edmd (EDMD): edmd object
            tsc_data (TSCDataFrame): time series dataframe
        """        
    
        self.edmd = edmd
        self.tsc_data = tsc_data
        self.include_id_state = include_id_state
        dictionary_eval = self.edmd.dict_steps[0][1].transform(tsc_data)
        if include_id_state:
            dictionary_eval = self.edmd._attach_id_state(X=tsc_data, X_dict=dictionary_eval) #include id states
        # normalize here?

        self.koopman_matrix = self.edmd.dmd_model._compute_koopman_matrix(dictionary_eval)
        
        # calculate right eigenvectors
        self.right_koopman_eigvals, self.right_koopman_eigvecs = sort_eigenpairs(*np.linalg.eig(self.koopman_matrix))

        # calculate left eigenvectors
        self.left_koopman_eigvals, self.left_koopman_eigvecs = sort_eigenpairs(*np.linalg.eig(self.koopman_matrix.conj().T))

    def dict_transform(self, x: np.ndarray):
        """Transform an array according to the edmd dictionary basis

        Args:
            x (np.ndarray): _description_
        """
        df = pd.DataFrame(x)
        df.columns = self.tsc_data.columns
        df.columns.name = "feature"

        dictionary_eval = self.edmd.dict_steps[0][1].transform(df)
        if self.include_id_state:
            dictionary_eval = self.edmd._attach_id_state(X=df, X_dict=dictionary_eval).to_numpy()
        else:
            dictionary_eval = dictionary_eval.to_numpy()
        return dictionary_eval

    def normalize_dict_transform(self, dictionary_eval):
        dict_eval_norm = np.apply_along_axis(lambda x:x/np.linalg.norm(x), 1, np.array(dictionary_eval))
        return dict_eval_norm

    def eigenfunction_left(self, left_eigenvector):
        """Generate eigenfunction of the Koopman operator using a left eigenvector

        Args:
            left_eigenvector (np.ndarray or np.array): _description_
        """
        # return lambda x: self.normalize_dict_transform(self.dict_transform(x)) @ left_eigenvector
        return lambda x: self.dict_transform(x) @ left_eigenvector
        

    def extend_eigenfunctions(self, x:np.ndarray, eigenvector_indexes = [0,1], pow_i = 1, pow_j = 1, normalize=True ):
        """Generate a new set of eigenfunctions for the koopman operator using two left eigenvectors of the koopman matrix

        Args:
            pow_i (_type_): power to which computed eigenfunction is raised
            pow_j (_type_): power to which computed eigenfunction is raised
            x (np.ndarray): _description_
            eigenvector_indexes (list, optional): indexes of eigenvectors to be used to give extended eigenfunctions. Defaults to [0,1].
        """        

        assert len(eigenvector_indexes) == 2

        dictionary_eval = self.dict_transform(x)
        # dictionary_eval = self.normalize_dict_transform(dictionary_eval)

        v_1_phi = (dictionary_eval @ self.left_koopman_eigvecs[:,eigenvector_indexes[0]])
        v_2_phi = (dictionary_eval @ self.left_koopman_eigvecs[:,eigenvector_indexes[1]])
       
        eigfunc_extended = (v_1_phi ** pow_i) * (v_2_phi ** pow_j)      
        if normalize:
            eigfunc_extended = eigfunc_extended/np.max(np.abs(eigfunc_extended))

        return eigfunc_extended


    def eigenfunction_right(self, x:np.ndarray, right_eigvecs=None):        
        """Get the eigenfunctions using the right eigenvector matrix

        Args:
            x (np.ndarray): _description_
            right_eigvecs (_type_, optional): matrix of right eigenvectors to use . Defaults to None.

        Returns:
            _type_: _description_
        """        
        
        if right_eigvecs == None:
            right_eigvecs = self.right_koopman_eigvecs

        dictionary_eval = self.dict_transform(x)
        eigfunc_matrix = np.linalg.lstsq(right_eigvecs, dictionary_eval.T, rcond=1e-8)[0]
        return eigfunc_matrix.T

    def reconstruct_observable(self, x:np.ndarray, eigfunc_matrix:np.ndarray):
        """Reconstruct an observable using eigenfucions

        Args:
            x (np.ndarray): observable values
            eigfunc_matrix (np.ndarray): _description_
        """
        coeff = np.linalg.lstsq(eigfunc_matrix , x, rcond=1e-8)[0]
        
        recons_obs = eigfunc_matrix@coeff
        return recons_obs


    def extended_eigfunction_matrix(self, x:np.ndarray, eigvector_idx = [0,1], 
                                    exp_pairs=[(0,1), (1,0), (1,1)]):
        """Get extended set of eigenfunctions using two left eigenvectors and their powers

        Args:
            x (np.ndarray): _description_
            left_eigvec_i (_type_): _description_
            left_eigvec_j (_type_): _description_
            exp_pairs (_type_): _description_

        Returns:
            _type_: _description_
        """        

        eigfunc_matrix = np.empty((0, x.shape[0])) 

        assert len(eigvector_idx) == 2

        for (i,j) in exp_pairs:
                if i + j ==0:
                    continue
                    
                new_eigfunc = self.extend_eigenfunctions(x, eigenvector_indexes=eigvector_idx, 
                                                            pow_i=i, pow_j=j)
                eigfunc_matrix = np.vstack((eigfunc_matrix, new_eigfunc))

        eigfunc_matrix = eigfunc_matrix.T
        return eigfunc_matrix

    def calculate_jacobian(self, x):
        if "polynomial" in self.edmd.named_steps:
             raise Exception("for polynomial basis use calculate_jacobian_handle")

        
        centers = np.array(self.edmd.named_steps["rbf"].centers_)

        eps = self.edmd.named_steps["rbf"].kernel.epsilon

        J = np.apply_along_axis(lambda c: np.array([np.exp( (-1/(2*eps)) * (np.linalg.norm(x-c)**2)) *(-1/eps) * (x[0]-c[0]), 
                                    np.exp( (-1/(2*eps)) * (np.linalg.norm(x-c)**2)) *(-1/eps) * (x[1]-c[1])]), 1, centers).T
        return J


    def calculate_jacobian_handle(self):

        if "rbf" in self.edmd.named_steps:
            raise Exception("rbf doesn't use jacobian handle. use calculate_jacobian")
            # calculate jacobian for radial basis functions
            pass

        x1 = sym.Symbol("x1")
        x2 = sym.Symbol("x2")
        max_degree = self.edmd.named_steps["polynomial"].degree

        poly_list = []
        jac_list = []
        for m in range(max_degree+1):
            for n in range(max_degree+1):
                if m ==0 and n==0:
                    continue
                if m+n>max_degree:
                    continue
                poly = (x1**m ) * (x2**n)

                poly_list.append(poly)
                jac_list.append([sym.diff(poly, x1),sym.diff(poly, x2)])
                
        # L = sym.lambdify([x1,x2], [j for i in jac_list for j in i])
        J = sym.lambdify([x1,x2],jac_list)
        return J

    def compute_singular_max(self, J=None, x=None):
        # y = np.array([x[0]*np.exp(t_sample),(x[1] - (x[0]**2)/3)* np.exp(-t_sample) + (x[0]**2)/3 * np.exp(2*t_sample)]).T

        if "rbf" in self.edmd.named_steps:
            J_arr = self.calculate_jacobian(x)

        # if J is a function handle
        else:
            J_arr = np.array(J(*x)).T

        eigs = np.linalg.eig(J_arr.T@J_arr)[0]

        return np.sqrt(np.max(eigs))






