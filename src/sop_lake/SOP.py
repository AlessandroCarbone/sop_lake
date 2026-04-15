import numpy                                    as np
from scipy              import linalg           as LA
from .utils              import check_selfadjoint, closest_hermitian, closest_pos_semidef, is_pos_semidef

def adapt_residues(Gamma_list, p_type0, p_type):
    """
    Adapt the residues to a new type of representation.
    
    Parameters
    ----------
    Gamma_list : list
        List of residue matrices
    p_type0 : str
        Original representation type ('std' or 'sqrt')
    p_type : str
        Target representation type ('std' or 'sqrt')
    
    Returns
    -------
    list
        List of residue matrices in the target representation
    """
    if p_type0 == p_type:
        return Gamma_list
    if p_type0 == "std" and p_type == "sqrt":
        return [LA.sqrtm(Gamma) for Gamma in Gamma_list]
    if p_type0 == "sqrt" and p_type == "std":
        return [Gamma @ Gamma for Gamma in Gamma_list]
    raise ValueError(
        f"Incompatible representation types: '{p_type0}' -> '{p_type}'. "
        f"Valid types are 'std' and 'sqrt'."
    )

def mat_to_params(mat):
    params = []
    mat_flat = mat.flatten()
    for el in mat_flat:
        params.append(el.real)
        params.append(el.imag)
    return params

def SOP_to_params(Gamma_list,sigma_list):
    """Function to extract the parameters from the residues and the poles of the SOP.
    N.B. Here, if we have the actual matrix elements of the residues or the square root of the residues, we have to flatten the matrices in the same way.
    Gamma_list : list of the residues
    sigma_list : list of the pole
    """
    params = []
    for Gamma in Gamma_list:
        params += mat_to_params(Gamma)
    for sigma in sigma_list:
        if sigma.imag == 0.:
            params.append(sigma.real)
            params.append(0.)
        else:
            params.append(sigma.real)
            params.append(sigma.imag)
    return params

def params_to_SOP(popt,M):
    """ Function to extract the residues and the poles from the parameters of the SOP, given their number. 
    N.B. We assume they are already correctly adapted to the p_type representation we want.
    """
    num_params = [popt[i] for i in np.arange(len(popt)-2*M)]
    den_params = [popt[i] for i in np.arange(len(popt)-2*M,len(popt))]
    
    sigma_list = [complex(den_params[i*2],den_params[i*2+1]) for i in range(M)]
    dim    = int(np.sqrt(len(num_params)/(2*M)))
    Gamma_list = [np.zeros((dim,dim),dtype=np.complex128) for i in range(M)]
    for k in range(M):
        for i in range(dim):
            for j in range(dim):
                ind_real = 2*(dim**2)*k + i*2*dim + j*2
                ind_imag = 2*(dim**2)*k + i*2*dim + j*2 + 1
                #print(ind_real,ind_imag)
                Gamma_list[k][i,j] = complex(num_params[ind_real],num_params[ind_imag])
    Gamma_list = Gamma_list
    return Gamma_list,sigma_list

class SOP:
    def __init__(self, Gamma_list, sigma_list, p_type="std"):
        """ SOP representation given with the lists of residues and poles
        Gamma_list : List of residues 
        sigma_list : List of poles
        """
        self.p_type = p_type
        self.Gamma_list = Gamma_list
        self.sigma_list = sigma_list
        self.num_poles  = len(Gamma_list)
        self.dim        = len(Gamma_list[0])

    @classmethod
    def from_params(cls, params, num_poles,p_type="std"):
        """
        Create a SOP object from flattened parameters.
        
        Parameters
        ----------
        params : list or array
            Flattened parameters containing residues and poles
        num_poles : int
            Number of poles in the SOP representation
        
        Returns
        -------
        SOP
            A new SOP object initialized from the parameters
        """
        Gamma_list, sigma_list = params_to_SOP(params, num_poles)     
        return cls(Gamma_list, sigma_list, p_type)

    def to_dict(self):
        """ Convert the SOP object to a dictionary representation. """
        Gamma_dict = {"real": [Gamma.real.tolist() for Gamma in self.Gamma_list],
                      "imag": [Gamma.imag.tolist() for Gamma in self.Gamma_list]}
        sigma_dict = {"real": [sigma.real for sigma in self.sigma_list],
                      "imag": [sigma.imag for sigma in self.sigma_list]}
        SOP_dict = {
            "Gamma_list": Gamma_dict,
            "sigma_list": sigma_dict,
            "p_type": self.p_type
        }
        return SOP_dict

    def get_params(self):
        return SOP_to_params(self.Gamma_list,self.sigma_list)
    
    def evaluate(self,w_list):
        """
        Evaluate the SOP function on a given frequency list.
        """
        Gamma_vec = np.asarray(self.Gamma_list, dtype=np.complex128)
        sigma_vec = np.asarray(self.sigma_list, dtype=np.complex128)
        w_vec     = np.asarray(w_list, dtype=np.complex128)

        denom_vec = w_vec[:, None] - sigma_vec[None, :]  # shape (n_freqs, n_poles)

        if self.p_type == "std":
            if Gamma_vec.ndim == 1:                     # Scalar residues
                terms = Gamma_vec[None, :] / denom_vec  # shape (n_freqs, n_poles)
                SOP_vec = np.sum(terms, axis=1)
            else:                                       # Matrix residues        
                terms = Gamma_vec[None, :, :, :] / denom_vec[:, :, None, None]  # shape (n_freqs, n_poles, d, d)
                SOP_vec = np.sum(terms, axis=1)                                # shape (n_freqs, d, d)
        elif self.p_type == "sqrt":
            if Gamma_vec.ndim == 1:                         # Scalar residues
                Gprod_vec = Gamma_vec * Gamma_vec
                terms = Gprod_vec[None, :] / denom_vec
                SOP_vec = np.sum(terms, axis=1)
            else:                                           # Matrix residues    
                Gprod_vec = np.matmul(Gamma_vec, Gamma_vec)                     # shape (n_poles, d, d)
                terms = Gprod_vec[None, :, :, :] / denom_vec[:, :, None, None]  # (n_freqs, n_poles, d, d)
                SOP_vec = np.sum(terms, axis=1)                                # shape (n_freqs, d, d)
        else:
            raise ValueError(f"Unknown p_type: {self.p_type}")
        return SOP_vec
    
    def sort(self):
        """ Sorting the residues and poles in increasing order of the real part of the poles."""
        ind_list = [el[0] for el in sorted(enumerate(self.sigma_list),key=lambda x:x[1].real)]
        Gamma_list_sort = [self.Gamma_list[ind] for ind in ind_list]
        sigma_list_sort = [self.sigma_list[ind] for ind in ind_list]
        self.Gamma_list = Gamma_list_sort
        self.sigma_list = sigma_list_sort

    def is_odd(self):
        if self.num_poles % 2 != 0:
            return ValueError('Error - The number of poles must be even!')
        else:
            return np.allclose(self.Gamma_list[:int(self.num_poles / 2)],self.Gamma_list[int(self.num_poles / 2):][::-1]) and np.allclose(self.sigma_list[:int(self.num_poles / 2)],[-sigma for sigma in self.sigma_list[int(self.num_poles / 2):][::-1]])

    def change_residues_type(self,p_type_new):
        """ Adapting the residues to the new type of representation."""
        self.Gamma_list = adapt_residues(self.Gamma_list,self.p_type,p_type_new)
    
    def make_residues_hermitian(self):
        """ Making the residues Hermitian by replacing them with the closest Hermitian matrix."""
        Gamma_list = self.Gamma_list
        if False in [check_selfadjoint(Gamma) for Gamma in Gamma_list]:
            Gamma_list_new = [closest_hermitian(Gamma) if check_selfadjoint(Gamma) == False else Gamma for Gamma in Gamma_list]
            self.Gamma_list = Gamma_list_new
    
    def make_residues_pos_semidef(self):
        """ Making the residues positive semi-definite by replacing them with the closest positive semi-definite matrix."""
        Gamma_list = self.Gamma_list
        if self.p_type != "std":
            raise ValueError('Error - The residues can be made positive semi-definite only for the "std" representation!')
        else:
            Gamma_list_new = [closest_pos_semidef(Gamma) if is_pos_semidef(Gamma) == False else Gamma for Gamma in Gamma_list]
        self.Gamma_list = Gamma_list_new

    def make_residues_diagonal_real(self):
        ntot = self.Gamma_list[0].shape[0]
        Gamma_list = self.Gamma_list
        for k in range(len(Gamma_list)):
            for i in range(ntot):
                Gamma_list[k][i,i] = complex(Gamma_list[k][i,i].real,0.)
        self.Gamma_list = Gamma_list

    def make_poles_real(self):
        sigma_list = self.sigma_list
        if not np.allclose(np.array(sigma_list), np.array(sigma_list).real):
            raise ValueError('Error - Poles have significant imaginary parts and cannot be made real!')
        self.sigma_list = list(np.array(sigma_list).real)
    
    def make_residues_real_diagonal(self):
        Gamma_list = self.Gamma_list
        ntot       = Gamma_list[0].shape[0]
        self.Gamma_list = [Gamma[0,0].real * np.identity(ntot,dtype=np.complex128) for Gamma in Gamma_list]
    
def antisymm_SOP(half_Gamma_list, half_sigma_list):
    """ Creating a list of parameters from the first half-list where the residues are equal and symmetric, the poles are the opposite and symmetric."""
    Gamma_list_new =  half_Gamma_list + half_Gamma_list[::-1]
    sigma_list_new =  half_sigma_list + [-sigma for sigma in half_sigma_list[::-1]]
    return Gamma_list_new, sigma_list_new