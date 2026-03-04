import numpy as np
import scipy
from scipy.sparse                                       import csc_matrix, csr_matrix, kron, vstack
import scipy.linalg                                     as LA
import scipy.sparse.linalg as SLA
try:
    import qiskit_nature
    from qiskit_nature.second_q.operators import FermionicOp
except ImportError:
    qiskit_nature = None
    FermionicOp   = type(None)
from itertools                                          import combinations
from functools                                          import reduce
from .utils                                              import check_selfadjoint, diagonalize_Hamiltonian, FermionicOp_to_matrix

def G_operator(H,w,sparse=False):
    if isinstance(H,FermionicOp):
        ntot   = H.register_length
        H_mat  = FermionicOp_to_matrix(H,sparse=sparse)
        dim    = 2**ntot
    elif isinstance(H,np.ndarray):
        n     = len(H)
        H_mat = H
        dim   = n
    elif isinstance(H,scipy.sparse.csc.csc_matrix) or isinstance(H,scipy.sparse.csr.csr_matrix) or isinstance(H,scipy.sparse.coo.coo_matrix):
        n     = H.shape[0]
        H_mat = H
        dim   = n
    else:
        raise ValueError('Error - H type must be FermioniOp, np.ndarray or csc_matrix')
    
    if sparse == False:
        Gop  = LA.inv(w*np.identity(dim) - H_mat)
    else:
        wI  = csc_matrix(w*np.identity(dim))
        Gop = SLA.inv(wI - H_mat)
    return Gop

def number_operator(ntot):
    """ This function returns the number operator as a FermionicOp.
    ntot : Number of states on which the system is described 
    """
    N_terms = []
    for i in range(ntot):
        label = "+_"+str(i)+" -_"+str(i)
        N_terms.append({label: 1.})
    N_op = sum(FermionicOp(N_term,ntot) for N_term in N_terms)
    return N_op

def SD_states(ntot,Np,sparse=False,efficient=False):
    """ This function returns the Slater determinant (SD) states as a list of vectors from the total number of 
        states ntot and the number of particles Np. 
    ntot : Total number of states
    Np   : Number of particles
    sparse: Each state is represented as a sparse vector if sparse=True, otherwise as a dense np.array
    efficient: If True, the sparse implementation uses a more efficient way to create the list of SD states as a csr_matrix instead of a simple list
    """
    if sparse == False:
        # Dense implementation - Returns a list of np.arrays
        basis = [ [complex(1.,0.),complex(0.,0.)], [complex(0.,0.),complex(1.,0.)] ]
        
        if Np==0:
            v = []
            v.append(np.zeros(2**ntot,dtype=np.complex128))
            v[0][0] = complex(1.,0)
        else:
            # Combinations of ntot elements in Np (number of particles) places
            comb = combinations(np.arange(ntot),Np)
            comb = list(comb)
            
            # Creating a binary list to use the kronecker product: 1 stands for v1 and v0 for v0 
            comb_list = [np.zeros(ntot,dtype=int) for el in comb]
            for i,el in enumerate(comb):
                for ind in range(Np):
                    comb_list[i][el[ind]] = 1
          
            v = [reduce(np.kron,[basis[ind] for ind in el]) for el in comb_list]
    else:
        # Sparse implementation - Returns a list of csc_matrix elements where the first row is the state we want
        basis = [csc_matrix([complex(1.,0.),complex(0.,0.)]), csc_matrix([complex(0.,0.),complex(1.,0.)])]
        
        if Np==0:
            v = []
            v.append(csc_matrix(([1.],([0],[0])),shape=(1,2**ntot),dtype=np.complex128))
        else:
            # Combinations of ntot elements in Np (number of particles) places
            comb = combinations(np.arange(ntot),Np)
            comb = list(comb)
            
            # Creating a binary list to use the kronecker product: 1 stands for v1 and v0 for v0 
            comb_list = [np.zeros(ntot,dtype=int) for el in comb]
            for i,el in enumerate(comb):
                for ind in range(Np):
                    comb_list[i][el[ind]] = 1
            
            # v = [reduce(kron,[basis[ind] for ind in el]) for el in comb_list]
            index_list = [sum(bit << (ntot - 1 - i) for i, bit in enumerate(el)) for el in comb_list]
            Ncomb = len(comb_list)
            data = np.ones(Ncomb, dtype=np.complex128)
            rows = index_list
            cols = np.arange(Ncomb)

            v_eff = csr_matrix((data, (rows, cols)), shape=(2**ntot, Ncomb))
            v     = [v_eff[:,k].T for k in range(v_eff.shape[1])]
    return v_eff if efficient == True and sparse == True else v

def operator_SD(op,Np,sparse=False):
    """ Computes the matrix associated with the operator where the basis is given by the SD of Np particles in
    ntot states
    op [FermionicOp] : Generic operator
    Np [int]         : Number of particles considered
    sparse [bool]    : If True the output matrix is in sparse format
    """
    
    if isinstance(op,FermionicOp):
        ntot   = op.register_length
        op_mat = FermionicOp_to_matrix(op,sparse=True)
    elif isinstance(op,np.ndarray):
        ntot   = int(np.log2(len(op)))
        op_mat = csc_matrix(op)
    elif isinstance(op,scipy.sparse.csc.csc_matrix) or isinstance(op,scipy.sparse.csr.csr_matrix):
        ntot   = int(np.log2(op.shape[0]))
        op_mat = op
    else:
        raise ValueError('Error - op type must be FermionicOp, np.ndarray or sparse matrix')
    
    SD_vec = SD_states(ntot,Np,sparse=True,efficient=True) 
    # Ncomb  = len(SD_vec)
    # SD_mat = chunked_vstack([normalize_sparse(el) for el in SD_vec]) if Ncomb > 10000 else vstack(SD_vec)     # If SD_states with efficient == False
    SD_mat = SD_vec.transpose().tocsr()
    # print("          Slater determinants basis for N = {}, Np = {} evaluated".format(ntot,Np))
    op_SD = SD_mat.conj() @ op_mat @ SD_mat.T

    return op_SD.toarray() if sparse == False else op_SD
    
def canonical_orthonormalization(psir_list, psil_list):
    """ We perform the so called canonical orthogonalization in the language of quantum chemists. See Eq. (3.169) in Szabo and Ostlund, Modern Quantum Chemistry.
    N.B The left eigenvectors are given already with the conjugation!
    """
    if len(psir_list) != len(psil_list):
        raise ValueError('Error - Left and right eigenvector lists must have the same length!')
    dim = len(psir_list)
    Ul = np.vstack(psil_list)
    Ur = np.vstack(psir_list).T
    S = Ul @ Ur
    #print(S.round(5))
    _,V = LA.eig(S)
    Ul = LA.inv(V) @ Ul
    Ur = Ur @ V
    s = Ul @ Ur
    #print(s.round(5))
    psil_list = [Ul[i,:] / s[i,i] for i in range(dim)]   # Normalization in the left eigenvector!
    psir_list = [Ur[:,i] for i in range(dim)]
    # print("\n","CANONICAL ORTHONORMALIZATION")
    # print([np.dot(psil_list[i],psir_list[i]) for i in range(dim)],"\n")
    return psir_list, psil_list

def gs_subspace(H,max_gs_deg=7,sparse=True,self_adj_check=None):
    """" This function finds the ground state subspace of a given Hamiltonian H. It returns the list of left and right eigenvectors and the corresponding eigenvalue.
    H          : Hamiltonian
    max_gs_deg : Maximum degeneracy findable (number of eigenvalues to look for in the sparse diagonalization)
    sparse     : If True, the ground state is evaluated using sparse diagonalization
    self_adj_check : If True, the self-adjointness of H is assumed
    """
    if isinstance(H,FermionicOp):
        H = FermionicOp_to_matrix(H,sparse=True)
    elif isinstance(H,np.ndarray): 
        H = csc_matrix(H)
    d = H.shape[0]
    sparse_gs = False if d < 10 else sparse               # If the matrix has dimension smaller than 10, we use the dense diagonalization!
    
    self_adj = self_adj_check if self_adj_check is not None else check_selfadjoint(H,sparse=True)
    k_max    = min(d - 1,max_gs_deg)
    if sparse_gs == True:
        if self_adj == False:
            Er,Ur = SLA.eigs(H,k=k_max,which='SR')                           
            El,Ul = SLA.eigs(H.T.conj(),k=k_max,which='SR')
        else:
            Er,Ur = SLA.eigsh(H,k=k_max,which='SA')
            El,Ul = SLA.eigsh(H.T.conj(),k=k_max,which='SA') 

        # Ordering the eigenvalues and eigenvectors 
        ind_Er = [el[0] for el in sorted(enumerate(Er),key=lambda x:x[1].real)]
        ind_El = [el[0] for el in sorted(enumerate(El),key=lambda x:x[1].real)] 
        E0     = Er[ind_Er[0]]

        # Building matrix of left and right eigenvectors of the g.s. in the degenerate case
        gs_deg  = np.count_nonzero(Er.round(5) == E0.round(5))
        psi0r_list = [Ur[:,ind_Er[gs_ind]] for gs_ind in range(gs_deg)]            
        psi0l_list = [Ul[:,ind_El[gs_ind]].conj() for gs_ind in range(gs_deg)]                      
        
        # Orthonormalization of the g.s. subspace 
        # This is needed because the left and right eigenvectors have been calculated independently, so their overlap matrix is not the identity
        # N.B. We cannot use the Gramm-Schmidt decomposition here because the left and right eigenvectors have to be orthonormalized together!
        psi0r_list, psi0l_list = canonical_orthonormalization(psi0r_list, psi0l_list)
    else:
        if self_adj == False:
            Er,Ur  = LA.eig(H.todense())                                           
            ind_Er = [el[0] for el in sorted(enumerate(Er),key=lambda x:x[1].real)]
            Ul = LA.inv(Ur)                                                             
            E0 = Er[ind_Er[0]]
            
            gs_deg  = np.count_nonzero(Er.round(5) == E0.round(5))
            psi0r_list = [Ur[:,ind_Er[gs_ind]] for gs_ind in range(gs_deg)]            
            psi0l_list = [Ul[ind_Er[gs_ind],:] for gs_ind in range(gs_deg)] 
        else:
            E,U = LA.eigh(H.todense()) 
            E0  = E[0]
        
            gs_deg  = np.count_nonzero(E.round(5) == E0.round(5))
            psi0r_list = [U[:,gs_ind] for gs_ind in range(gs_deg)]
            psi0l_list = np.array(psi0r_list).conj()
    return E0, psi0r_list, psi0l_list

def diagonalize_Fock_Hamiltonian(H,num_subspaces):
    """ This function return the eigenvalues and the right/left eigenvectors of a given Hamiltonian H assuming that H lives in the Fock space. The diagonalization is performed
    subspace by subspace for computational efficiency since the full-fermionic Fock space is the direct sum of the subspaces with fixed number of particles.
    """
    if isinstance(H,FermionicOp):
        H = FermionicOp_to_matrix(H,sparse=True)
    elif isinstance(H,np.ndarray): 
        H = csc_matrix(H)
    d = H.shape[0]
    
    E_full_list, psir_full_list, psil_full_list = [], [], []
    for Np_try in range(0,num_subspaces + 1):
        H_subspace = operator_SD(H,Np_try)
        E_list, psir_list, psil_list = diagonalize_Hamiltonian(H_subspace)
        
        # From vector in Np_try-space to vector in full Fock space
        SD_vec  = SD_states(num_subspaces,Np_try) 
        psir_list2, psil_list2 = [np.zeros(2**num_subspaces,dtype=np.complex128) for i in range(len(psir_list))], [np.zeros(2**num_subspaces,dtype=np.complex128) for i in range(len(psil_list))]
        for i in range(len(psir_list)):    
            psir_list2[i] = sum(np.dot(psir_list[i][j],SD_vec[j]) for j in range(len(SD_vec)))   
            psil_list2[i] = sum(np.dot(psil_list[i][j],SD_vec[j]) for j in range(len(SD_vec))) 
        E_full_list += list(E_list)
        psir_full_list += psir_list2
        psil_full_list += psil_list2
    return E_full_list, psir_full_list, psil_full_list

def statistical_weights(E_list,beta):
    Z = sum(np.exp(-beta*E) for E in E_list)
    weights = [np.exp(-beta*E) / Z for E in E_list]
    return weights

def Np_from_A_ia(G_ia_list,w_list):
    """" Evaluation of the number of particles from the GF evaluated on the imaginary axis. See, e.g., eq. (23) of paper from Von Barth, PRB 54, 12 (1996). 
    The latter equation works also for all mu, but the effect must be accounted in the list of GF values given as input, i.e. using G(-iw + mu). G(-iw + mu) can be
    both in the real or in the k space!
    N.B. If the integration is made with np.trapz the list of frequencies correspond to the domain (0,+inf), ideally.
    G_ia_list : List of values of the GF evaluated on the imaginary axis frequencies, - iw + mu
    w_list  : List of frequencies 
    """
    ntot     = len(G_ia_list[0])
    int_list = [np.trace(G_ia_list[iw]).real for iw in range(len(w_list))]
    #int_func = interp1d(w_list, int_list, kind='cubic', bounds_error=False, fill_value=0.)

    # First and second term of the integral
    int1 = ntot / 2
    #int2 = integrate.quad(lambda w: int_func(w),0.,np.inf)[0] / np.pi                # Based on function which interpolates the integrand
    int2 = np.trapz(int_list,w_list) / np.pi                                          # Only if frequencies go from 0 to infinity (ideally)
    Np   = int1 + int2
    return Np

def continued_fraction(coeff1_list,coeff2_list,w):
    """ This function evaluates the continued fraction given the two lists of coefficients and the frequency list.
    coeff1_list : List of coefficients appearing at the numerator of the continued fraction
    coeff2_list : List of coefficients appearing at the denominator of the continued fraction
    w           : Frequency where the continued fraction has to be evaluated
    """
    dim = len(coeff1_list)
    if dim == 0:
        raise ValueError("Error - The continued fraction has dimension 0!")
    else:
        cf_el = 0.
        cf_el_list = []
        for i in range(1,dim+1):
            numerator = 1. if i == dim else coeff2_list[-i]
            cf_el = numerator / (w - coeff1_list[-i] - cf_el)
            cf_el_list.append(cf_el)
        cf = cf_el_list[-1]
        return cf

def compute_avg_GF(C_list, Z_list, w_list, self_adj_check=True):
    """
    Computes the average Green's function via sum over poles (SOP), averaged over multiple ground states.

    Parameters:
        C_list: array-like, shape (n_gs, n_poles) - residues of the Green's function
        Z_list: array-like, shape (n_poles,) or dictionary - poles of the Green's function or Lanczos coefficients
        w_list: array-like, shape (n_freqs,) - frequency grid

    Returns:
        SOP_list: np.ndarray of shape (n_freqs,)
    """
    if isinstance(Z_list,dict) == False:
        ntot  = C_list[0][0].shape[0]
        C_vec = np.asarray(C_list, dtype=np.complex128)             # (n_gs, n_poles)
        Z_vec = np.asarray(Z_list, dtype=np.complex128)             # (n_poles,)
        w_vec = np.asarray(w_list, dtype=np.complex128)             # (n_freqs,)
        
        denom_vec = w_vec[:, None] - Z_vec[None, :]                 # Compute: (n_freqs, n_poles): C[:, None, :] shape: (n_gs, 1, n_poles), denom[None, :, :] shape: (1, n_freqs, n_poles)
        GF = C_vec[:, None, :, :, :] / denom_vec[:, :, None, None]  # Result shape: (n_gs, n_freqs, n_poles)
        GF_sum   = np.sum(GF, axis=2)                               # Sum over poles: (n_gs, n_freqs)       
        SOP_list = np.mean(GF_sum, axis=0)                          # Average over ground states: (n_freqs,)    
    else:
        E0, ntot = Z_list["gs_energy"], Z_list["ntot"]
        if self_adj_check == True:
            gs_deg = len(C_list)
            SOP_list = [np.zeros((ntot,ntot),dtype=np.complex128) for w in w_list]
            for gs_ind in range(gs_deg):
                for i in range(ntot):
                    for j in range(ntot):
                        for iw,w in enumerate(w_list):
                            num1, num2 = C_list[gs_ind][0][i,j], C_list[gs_ind][1][j,i]
                            # num1, num2 = 1., 1.
                            idx = str(i) + str(j)
                            SOP_list[iw][i,j] += num1 * continued_fraction(*Z_list["lanczos_coeff"][gs_ind][idx][0],w + E0) - num2 * continued_fraction(*Z_list["lanczos_coeff"][gs_ind][idx][1],-w + E0)
            SOP_list = np.array([SOP / gs_deg for SOP in SOP_list]) 
        else:
            raise NotImplementedError("Error - The non-self-adjoint case is not implemented yet.")
    return SOP_list

def DOS_diff(G_list,G_list2):
    """ Sum of squared differences of the DOS functions of two different Green's functions normalized on the number of points in the frequency grid
    G_list  : First Green's function evaluated on a frequency grid
    G_list2 : Second Green's function evaluated on a frequency grid
    """
    if len(G_list) != len(G_list2):
        raise ValueError("The two Green's functions must be evaluated on the same frequency grid")
    DOS_diff_list = [(np.trace(np.abs(G_list[iw].imag)) - np.trace(np.abs(G_list2[iw].imag))) / np.pi for iw in range(len(G_list))]        # DOS(w) = Tr{ A(w) } = sum_i { | Im{G_ii(w)} | / π }
    diff          = sum(DOS_diff**2 for DOS_diff in DOS_diff_list) / len(G_list)                                                        
    return diff

def calc_g(v_emb_list,SigmaA_list,hA_1,w_list,mu):
    ntot   = len(v_emb_list[0])
    g_list = [LA.inv((w + mu)*np.identity(ntot) - hA_1 - v_emb_list[iw] - SigmaA_list[iw]) for iw,w in enumerate(w_list)]
    return g_list