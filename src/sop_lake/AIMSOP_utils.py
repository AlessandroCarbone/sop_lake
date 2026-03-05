import numpy        as np
import scipy.linalg as LA

from .utils import check_selfadjoint


def AIMSOP_matrix(hA,Gamma_list,sigma_list,p_type="std"):
    """ This function returns the AIMSOP matrix given the residues and poles
    hA : fragment Hamiltonian in its matrix representation
    Gamma_list : List of residues
    sigma_list : List of poles
    """
    num_poles = len(sigma_list)
    ntot = len(Gamma_list[0])
    d_AIM = ntot * (num_poles + 1)
    
    h_AIM = np.zeros((d_AIM,d_AIM),dtype=np.complex128)
    for i in range(ntot):
        for j in range(ntot):
            h_AIM[i,j] = hA[i,j]

    Gamma_sqrt_list = Gamma_list if p_type == "sqrt" else [LA.sqrtm(Gamma) for Gamma in Gamma_list]
    for k in range(num_poles):
        for i in range(ntot):
            h_AIM[ntot*(k+1)+i,ntot*(k+1)+i] = sigma_list[k]
            for j in range(ntot):
                h_AIM[i,ntot*(k+1)+j] = Gamma_sqrt_list[k][i,j]
                h_AIM[ntot*(k+1)+i,j] = Gamma_sqrt_list[k][i,j]
    return h_AIM

def reversed_AIMSOP(G_SOP):
    """ This function takes care of the reversed algorithmic inversion: given G as a SOP, returns (w - G^{-1}(w)) as a SOP and an additional constant matrix. SOP means a list of poles and a list of residues
    as usual. We follow the procedure described by Andrea Ferretti in the proof on the "well-behaved"-ness of the Dyson equation. See A. Ferretti, T. Chiarotti, and N. Marzari, Phys. Rev. B 110, 045149 (2024).
    This function can be easily generalized when we want to evaluate w + mu - h0 - SOP_2(w) - G^{-1}(w), for instance in the evaluation of the self-energy in DMFT immediately after having retrieved the impurity GF.
    G_SOP : SOP object as defined in SOP.py
    """
    zero_thr = 1e-5
    num_poles = G_SOP.num_poles
    ntot = G_SOP.dim
    A_list, Z_list = G_SOP.Gamma_list, G_SOP.sigma_list                                                     # List of residues and poles of the input GF
    if np.array(Z_list).imag.any() != 0:
        raise ValueError('Error - The poles must be real!')
    Omega  = np.sqrt(max([np.abs(Z)**2 for Z in Z_list]) + 1e-1)                                            # Constant scalar used in the proof by Andrea Ferretti
    h0_F   = sum(-Z_list[k] * A_list[k] for k in range(num_poles))                                          # h0 to be used in the AIM-SOP on the F function defined in the proof   
    Gamma_F_list = [A_list[k] * (Omega**2 - Z_list[k]**2) for k in range(num_poles)]
    hAIM_F = AIMSOP_matrix(h0_F,Gamma_F_list,Z_list,p_type=G_SOP.p_type)
    if check_selfadjoint(hAIM_F) == True:
        Omega_list, U = LA.eigh(hAIM_F)                                                                     # Omega_list = list of poles of the final SOP object
        B_list = [np.outer(U[:,i],U[:,i].conj())[:ntot,:ntot] for i in range(len(Omega_list))]
        const_term = sum(Omega_list[i] * B_list[i] for i in range(len(Omega_list)))                         # Evaluated using the division of the polynomia, keeping also +/- Omega eigenvalues and corresponding eigenvectors
        # inds_to_del = []                                                                                    # Indices to remove in the Omega_list
        # for el in Omega_list:
        #     if np.abs(el - Omega) < zero_thr:
        #         inds_to_del.append(np.where(Omega_list == el)[0][0])
        #     elif np.abs(el + Omega) < zero_thr:
        #         inds_to_del.append(np.where(Omega_list == el)[0][0])
        # Omega_list = np.delete(Omega_list,inds_to_del)                                                      # Deleting the eigenvalues that are too close to Omega or -Omega
        # B_list     = [B for i,B in enumerate(B_list) if i not in inds_to_del]                               # Deleting the corresponding eigenvectors
        B2_list    = [B_list[i] * (Omega_list[i]**2 - Omega**2) for i in range(len(Omega_list))]            # List of residues evaluated using the division of the polynomia
        return const_term, B2_list, Omega_list
    else:
        raise ValueError('Error - The AIMSOP matrix is not self-adjoint!')