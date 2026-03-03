import os, sys
import numpy            as np
import scipy
import scipy.linalg     as LA
from scipy.sparse       import csc_matrix, isspmatrix
from sympy              import Matrix
import qiskit_nature
from qiskit_nature.second_q.operators                   import FermionicOp
from qiskit_nature.second_q.mappers                     import JordanWignerMapper

def check_selfadjoint(M,rtol=1e-05,atol=1e-08,sparse=False):
    """This function check is he matrix is self-adjoint in case it's sparse or not
    M      : Matrix
    rtol   : Relative tolerance parameter
    atol   : Absolute tolerance parameter
    sparse : If the matrix is sparse must be put equal to True, otherwise to False
    """
    if sparse==True:
        return np.allclose(M.A, M.conj().T.A, rtol=rtol, atol=atol)
    else:
        return np.allclose(M, np.conj(M).T, rtol=rtol, atol=atol)

def FermionicOp_to_matrix(op,sparse=False):
    """Convert a FermionicOp to a matrix in the full-fermionic Fock space 
    op     : FermionicOp
    sparse : If the matrix is sparse must be put equal to True, otherwise to False
    """
    if isinstance(op,qiskit_nature.second_q.operators.FermionicOp) == False:
        raise ValueError('Error - op type must be Qiskit FermionicOp')
    jw_op = JordanWignerMapper().map(op)                                      # Jordan-Wigner mapper of Qiskit to do the transformation to a matrix
    # Here we have to correct the order of the Pauli operators in order to have the right mapping from qubit states to vector states in the Fock space: 0000 --> [1,0,0,0] , 1000 --> [0,1,0,0], ...
    for pauli in jw_op.paulis:
        pauli.x = pauli.x[::-1]
        pauli.z = pauli.z[::-1]
    mat = jw_op.to_matrix(sparse=sparse)
    return mat

def exp_value(op,psil,psir):
    if isinstance(op,qiskit_nature.second_q.operators.FermionicOp):
        if isinstance(psil,scipy.sparse.csc.csc_matrix) or isinstance(psil,scipy.sparse.csr.csr_matrix):
            op_mat  = FermionicOp_to_matrix(op,sparse=True)
            exp_val = ( psil.conj() @ op_mat @ psir.T ) / ( psil.conj() @ psir.T )[0,0]
            exp_val = exp_val.todense()[0,0]
        else:
            op_mat  = FermionicOp_to_matrix(op,sparse=False)
            exp_val = ( psil.conj() @ op_mat @ psir ) / ( psil.conj() @ psir )
    elif isinstance(op,np.ndarray):
        op_mat  = op
        exp_val = ( psil.conj() @ op_mat @ psir ) / ( psil.conj() @ psir )
    elif isinstance(op,scipy.sparse.csc.csc_matrix) or isinstance(op,scipy.sparse.csr.csr_matrix):
        op_mat  = op
        if isinstance(psil,scipy.sparse.csc.csc_matrix) or isinstance(psil,scipy.sparse.csr.csr_matrix) or isinstance(psil,scipy.sparse.bsr_matrix):
            exp_val = ( psil.conj() @ op_mat @ psir.T ) / ( psil.conj() @ psir.T )[0,0]
            exp_val = exp_val.todense()[0,0]
        else:
            exp_val = ( psil.conj() @ op_mat @ psir ) / ( psil.conj() @ psir )
    else:
        raise ValueError('Error - op type must be FermionicOp, np.ndarray or sparse matrix')
    
    return exp_val

def Matsubara_freq(n, beta):
    return (2*n + 1) * np.pi / beta

def is_pos_semidef(A):
    if check_selfadjoint(A) == True:
        return np.all(LA.eigh(A)[0] >= 0.)
    else:
        return "Not Hermitian"

def closest_hermitian(A):
    """ Function to find the closest Hermitian matrix to A.
    A : matrix
    """
    return 0.5 * (A + A.T.conj())

def closest_pos_semidef(A):
    """ Function to find the closest positive semidefinite matrix to A using the Frobenius norm. See: https://doi.org/10.1016/0024-3795(88)90223-6 .
    A : matrix
    """
    if check_selfadjoint(A) == False:
        raise ValueError('A is not Hermitian')
    else:
        Y = 0.5 * (A + A.T)
        λ_list, U = LA.eigh(Y)
        λp_list   = [λ if λ > 0. else 0. for λ in λ_list]
        Dp = np.diag(λp_list)
        A2 = U @ Dp @ LA.inv(U)
        return A2
    
def check_diagonalizable(A):
    """This function checks if a matrix is diagonalizable by checking if algebraic and geometric multiplicities are the same
    A      : Hamiltonian or matrix
    """
    if isinstance(A,qiskit_nature.second_q.operators.FermionicOp):
        A_mat = FermionicOp_to_matrix(A)
    elif isinstance(A,np.ndarray):
        A_mat = A
    elif isinstance(A,scipy.sparse.csc.csc_matrix) or isinstance(A,scipy.sparse.csr.csr_matrix):
        A_mat = A.todense()
    else:
        raise ValueError('Error - op type must be FermionicOp, np.ndarray or sparse matrix')
    
    eigvals  = Matrix(A_mat).eigenvals()
    alg_mult = sum(el for el in list(eigvals.values()))
    geo_mult = np.linalg.matrix_rank(A_mat)
    return alg_mult == geo_mult

def RMSE(ex_list,pred_list):
    """"" Function which estimates the Root Mean Square Error (RMSE) between two lists of values.
    ex_list  : List of exact values
    pred_list: List of predicted values
    """
    MSE  = np.square(np.subtract(ex_list,pred_list)).mean()
    RMSE = np.sqrt(MSE)
    return RMSE

def order_of_magnitude(number):
    return int(np.floor(np.log10(number)))

def diagonalize_Hamiltonian(H):
    """ This function return the eigenvalues and the right/left eigenvectors of a given Hamiltonian H.
    """
    if isinstance(H,qiskit_nature.second_q.operators.FermionicOp):
        H = FermionicOp_to_matrix(H,sparse=True)
    elif isinstance(H,np.ndarray): 
        H = csc_matrix(H)

    self_adj = check_selfadjoint(H,sparse=True)
    if self_adj == False:
        Er,Ur  = LA.eig(H.todense())                                           
        ind_Er = [el[0] for el in sorted(enumerate(Er),key=lambda x: x[1].real)]
        E         = [Er[ind_Er[i]] for i in range(len(Er))]
        psir_list = [Ur[:,ind_Er[i]] for i in range(len(Er))]
        Ur        = np.array(psir_list).T
    else:
        E,U = LA.eigh(H.todense())
        Ur  = U
        psir_list = [U[:,i] for i in range(U.shape[1])]
    Ul = LA.inv(Ur)
    psil_list = [Ul[i,:].conj() for i in range(Ul.shape[0])]
    return E, psir_list, psil_list

def gramm_schmidt(v_list):
    basis = []
    for v in v_list:
        v_dag = v.conj().T
        w = v - sum( np.dot(v_dag @ b, b)  for b in basis )  
        basis.append(w / np.linalg.norm(w))
    return np.array(basis)

def save_list(w_list, complex_list, file):
    """ Function to save a list of complex numbers in a file
    complex_list : List of complex numbers to be saved
    w_list       : List of frequencies corresponding to the complex numbers
    file         : File where the complex numbers will be saved
    """
    with open(file, "w") as file:
        print("# Frequency // Real part // Imaginary part", file=file)
        for iw,w in enumerate(w_list):
            val = complex_list[iw]
            print("{:<18} {:<18} {:<18}".format(w,val.real,val.imag),file=file)
        print("# End of file", file=file)

def remove_input_files(path_to_dir):
    for filename in os.listdir(path_to_dir):
        if filename.endswith("input.txt"):
            os.remove(filename)

def pruning_sparse_zeros(A,threshold=1e-10):
    A.eliminate_zeros() 
    # print("Before removing zeros: sparsity = ",1 - A.nnz / (A.shape[0] * A.shape[1]))
    A.data[np.abs(A.data) < threshold] = 0
    A.eliminate_zeros()
    # print("\tAfter removing zeros: sparsity = ",1 - A.nnz / (A.shape[0] * A.shape[1]))
    return A

def to_scalar_if_sparse(object):
    if isspmatrix(object):
        return object.A[0,0]
    else:
        return object
    
def check_sparsity(A):
    """ This function checks the sparsity of a matrix given n sparse form.
    A: Hamiltonian sparse
    """
    if isspmatrix(A):
        sparsity = 1 - A.nnz / (A.shape[0] * A.shape[1])
    else:
        sparsity = 0.0
    return sparsity
class hidden_prints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def site_spin_label(i):
    """ Converts the numerical index used in coding to (site, spin) label
    i : index
    """
    r = i%2
    q = (i - r)/2
    site = int(q + 1)
    spin = r
    spin_symbol   = {0 : r'\uparrow', 1 : r'\downarrow'}
    site_spin_lab = str(site)+spin_symbol[spin]
    return site_spin_lab

def Np_from_A_isa(G_isa_list,w_list):
    """" Evaluation of the number of particles from the GF evaluated on the imaginary semi-axis. See, e.g., eq. (23) of paper from Von Barth, PRB 54, 12 (1996). 
    The latter equation works also for all mu, but the effect must be accounted in the list of GF values given as input, i.e. using G(-iw + mu). G(-iw + mu) can be
    both in the real or in the k space!
    N.B. If the integration is made with np.trapz the list of frequencies correspond to the domain (0,+inf), ideally.
    G_isa_list : List of values of the GF evaluated on the imaginary axis frequencies, - iw + mu
    w_list  : List of frequencies 
    """
    ntot     = len(G_isa_list[0])
    int_list = [np.trace(G_isa_list[iw]).real for iw in range(len(w_list))]
    #int_func = interp1d(w_list, int_list, kind='cubic', bounds_error=False, fill_value=0.)

    # First and second term of the integral
    int1 = ntot / 2
    #int2 = integrate.quad(lambda w: int_func(w),0.,np.inf)[0] / np.pi                # Based on function which interpolates the integrand
    int2 = np.trapz(int_list,w_list) / np.pi                                          # Only if frequencies go from 0 to infinity (ideally)
    Np   = int1 + int2
    return Np

def numerical_grad(f,x0,delta):
    grad = []
    for i in range(len(x0)):
        x1 = x0.copy()
        x2 = x0.copy()
        x1[i] += delta
        x2[i] -= delta
        grad.append((f(x1) - f(x2)) / (2 * delta))
    return np.array(grad)

