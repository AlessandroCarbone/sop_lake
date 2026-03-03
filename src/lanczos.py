import numpy                        as np
try:
    import qiskit_nature
    from qiskit_nature.second_q.operators import FermionicOp
except ImportError:
    qiskit_nature = None
    FermionicOp   = type(None)
from scipy.sparse                                       import csc_matrix, csr_matrix, kron, vstack
import scipy.linalg                  as LA
import scipy.sparse.linalg           as SLA

from utils                                              import to_scalar_if_sparse, FermionicOp_to_matrix

def biorthogonalize_vector(phi,chi,phi_list,chi_list):
    if len(phi_list) != len(chi_list):
        raise ValueError('Error - Left and right vector lists must have the same length')
    dim = len(phi_list)
    phi_new = phi - sum(to_scalar_if_sparse(chi_list[j] @ phi) * phi_list[j] for j in range(dim))
    chi_new = chi - sum(to_scalar_if_sparse(chi @ phi_list[j]) * chi_list[j] for j in range(dim))
    return phi_new, chi_new

def full_biorthogonalization(phi_list,chi_list):
    """ This function biorthogonalizes two sets of vectors phi_list and chi_list using the modified Gram-Schmidt algorithm.
    phi_list: list of ket vectors
    chi_list: list of bra vectors (to be given as the transposed conjugate of the actual ket vectors)
    """ 
    dim = len(phi_list)
    phi_list2, chi_list2 = [], []
    for i in range(dim):
        phi2, chi2 = biorthogonalize_vector(phi_list[i], chi_list[i], phi_list2, chi_list2)
        phi_list2.append(phi2)
        chi_list2.append(chi2)
    return phi_list2, chi_list2

def lanczos_basis(phi0,H,dim):
    """ This function builds the Lanczos basis starting from an initial vector phi0.
    phi0 : Initial vector as a ket
    dim  : Dimension of the Lanczos basis to be built
    """
    if isinstance(H,FermionicOp):
        H = FermionicOp_to_matrix(H,sparse=True)
    elif isinstance(H,np.ndarray): 
        H = csc_matrix(H)
    phi_list, a_list, b_list = [],[], []
    phi0 = phi0 / LA.norm(phi0)
    phi_list.append(phi0)
    a0, b0 = phi0.conj() @ H @ phi0, 1.
    a_list.append(a0)
    b_list.append(b0)
    phi = H @ phi0 - a0 * phi0                            # First element of the Lanczos basis  
    for i in range(2,dim + 2):
        b = LA.norm(phi)
        if b < 1e-14:
            break
        phi = phi / b
        b_list.append(b)
        phi, _ = biorthogonalize_vector(phi,phi.conj(),phi_list,[el.conj() for el in phi_list])
        norm_phi = LA.norm(phi)
        if norm_phi < 1e-14:
            break
        phi = phi / norm_phi
        phi_list.append(phi)          
        a = phi.conj() @ H @ phi
        a_list.append(a)
        phi_new = H @ phi_list[i-1] - a * phi_list[i-1] - b * phi_list[i-2]
        phi = phi_new

    # Testing Lanczos basis orthogonality and tridiagonalization of H
    # U = np.vstack(phi_list).T
    # prod_U = U.conj().T @ U
    # print("\n","U^{-1} @ U")
    # for i in range(10):
    #     for j in range(10):
    #         if i != j and np.abs(prod_U[i,j]) > 1e-6:
    #             print("({},{}): {}".format(i,j,prod_U[i,j]))
    # tridiag_H = U.conj().T @ H @ U
    # print("\n","Tridiagonal matrix U^{-1} H U")
    # for i in range(5):
    #     for j in range(5):
    #         if i != j and (j != i - 1 and j != i + 1) and np.abs(tridiag_H[i,j]) > 1e-6:
    #             print("({},{}): {}".format(i,j,tridiag_H[i,j]))
    a_list, b_list = np.array(a_list).real.tolist(), np.array(b_list).real.tolist()
    return a_list, b_list

def bilanczos_basis(phi0,chi0,H,dim):
    """ This function builds the Lanczos basis starting from two initial vectors phi0 and chi0 to evaluate the Green's function between them.
    phi0 : Initial vector for the ket side (right)
    chi0 : Initial vector for the bra side (left) to be given as bra, i.e. as the transposed conjugate of the actual ket vector!
    H    : Hamiltonian
    dim  : Dimension of the Lanczos basis to be built
    """
    if isinstance(H,FermionicOp):
        H = FermionicOp_to_matrix(H,sparse=True)
    elif isinstance(H,np.ndarray): 
        H = csc_matrix(H)
    phi_list, chi_list, alpha_list, beta_list, gamma_list = [], [], [], [], []
    omega0 = to_scalar_if_sparse(chi0 @ phi0)
    if np.abs(omega0) < 1e-14:
        return [], [], []
    phi0, chi0 = phi0 / np.sqrt(omega0), chi0 / np.sqrt(omega0)
    phi_list.append(phi0)
    chi_list.append(chi0)
    alpha0, beta0, gamma0 = to_scalar_if_sparse(chi0 @ H @ phi0), 1., 1.             # b0 and gamma0 should be set to 0 to be consistent with the algorithm, but here are equal to 1 to be used in the continued_fraction function
    alpha_list.append(alpha0)
    beta_list.append(beta0)
    gamma_list.append(gamma0)
    phi = H @ phi0 - alpha0 * phi0                              # First ket element of the Lanczos basis
    chi = chi0 @ H - alpha0 * chi0                              # First bra element of the Lanczos basis
    for i in range(2,dim + 2):
        omega = to_scalar_if_sparse(chi @ phi)
        if np.abs(omega) < 1e-14:
            # print(f"Converged at iteration {i}, omega = 0")
            break
        beta  = np.sqrt(np.abs(omega.conj()))
        gamma = omega / beta
        phi, chi = phi / gamma, chi / beta
        beta_list.append(beta)
        gamma_list.append(gamma)
        phi, chi = biorthogonalize_vector(phi, chi, phi_list, chi_list)
        omega = to_scalar_if_sparse(chi @ phi)
        if np.abs(omega) < 1e-14:
            break
        beta  = np.sqrt(np.abs(omega.conj()))
        gamma = omega / beta
        phi, chi = phi / gamma, chi / beta
        phi_list.append(phi)
        chi_list.append(chi)
        alpha = to_scalar_if_sparse(chi @ H @ phi)
        alpha_list.append(alpha)
        phi_new = H @ phi_list[i-1] - alpha_list[i-1] * phi_list[i-1] - beta_list[i-1] * phi_list[i-2]
        chi_new = chi_list[i-1] @ H - alpha_list[i-1] * chi_list[i-1] - gamma_list[i-1] * chi_list[i-2]
        phi, chi = phi_new, chi_new

    # Testing Lanczos basis: basis orthogonality and tridiagonalization of H
    # V, U = np.vstack(chi_list), np.vstack(phi_list).T
    # prod_U = V @ U
    # print("\n","V @ U")
    # for i in range(10):
    #     for j in range(10):
    #         if i != j and np.abs(prod_U[i,j]) > 1e-6:
    #             print("({},{}): {}".format(i,j,prod_U[i,j]))
    # print("\n","Tridiagonal matrix V @ H @ U")
    # tridiag_H = V @ H @ U
    # for i in range(10):
    #     for j in range(10):
    #         if i != j and (j != i - 1 and j != i + 1) and np.abs(tridiag_H[i,j]) > 1e-6:
    #             print("({},{}): {}".format(i,j,tridiag_H[i,j]))
    alpha_list, beta_list, gamma_list = np.array(alpha_list).real.tolist(), np.array(beta_list).real.tolist(), np.array(gamma_list).real.tolist()
    return alpha_list, beta_list, gamma_list

def lanczos_to_SOP(alpha_list,beta_list,gamma_list=None):
    """ This function converts the bi-Lanczos coefficients to the residues and poles to be used in the evaluation of the Green's function via SOP representation. If we have Lanczos
    coefficients, beta coefficients are equal to the gamma ones. Since the continued fraction is obtained by taking: (omega - H_Krylov)^(-1)_{1,1}, we have to diagonalize the Krylov
    Hamiltonian.
    """
    if len(alpha_list) == 0:
        return [], []
    dim = len(alpha_list)
    gamma_list = beta_list if gamma_list is None else gamma_list
    H_Krylov = np.zeros((dim,dim),dtype=np.complex128)
    for i in range(dim):
        for j in range(dim):
            if i == j:
                H_Krylov[i,j] = alpha_list[i]
            elif j == i + 1:
                H_Krylov[i,j] = beta_list[i+1]
            elif j == i - 1:
                H_Krylov[i,j] = gamma_list[i]
    E,Ur = LA.eig(H_Krylov)                                                                 # N.B. Remember the eigenvalues are not sorted!
    psir_list = [Ur[:,i] for i in range(Ur.shape[1])]
    Ul        = LA.inv(Ur)
    psil_list = [Ul[i,:] for i in range(Ul.shape[0])]                                       # Remember that the left eigenvectors are rows of Ul, and are already conjugated
    res_list  = [np.outer(psir_list[i],psil_list[i])[0,0] for i in range(len(E))]            # Residues as projected on the first element
    pol_list  = E
    return res_list, pol_list

def lanczos_to_SOP_GF(C_list,Z_dict):
    """ This function converts the bi-Lanczos coefficients to the residues and poles to be used in the evaluation of the Green's function via SOP representation. 
    Z_dict: must be adapted to be the list with residues and poles
    """
    gs_deg = len(C_list)
    ntot   = Z_dict["ntot"]
    E0     = Z_dict["gs_energy"]
    res_list, pol_list = [], []
    for gs_ind in range(gs_deg):
        for i in range(ntot):
            for j in range(ntot):
                mat = np.zeros((ntot,ntot),dtype=np.complex128)
                mat[i,j] = 1.
                num1, num2 = C_list[gs_ind][0][i,j], C_list[gs_ind][1][j,i]
                idx = str(i) + str(j)
                resp_list, polp_list = Z_dict["lanczos_coeff"][gs_ind][idx][0]
                resm_list, polm_list = Z_dict["lanczos_coeff"][gs_ind][idx][1]
                res_list += [num1 * resp * mat / gs_deg for resp in resp_list]
                pol_list += [polp - E0 for polp in polp_list]
                res_list += [num2 * resm * mat / gs_deg for resm in resm_list]
                pol_list += [-polm + E0 for polm in polm_list]
    res_list = [res_list]
    return res_list, pol_list
