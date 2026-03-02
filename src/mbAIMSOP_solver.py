import numpy                         as np
import qiskit_nature
from qiskit_nature.second_q.operators                   import FermionicOp
from scipy.sparse                                       import csc_matrix, csr_matrix, kron, vstack, hstack
import scipy.linalg                  as LA
import scipy.sparse.linalg           as SLA
import logging
from itertools                                          import combinations

from SOP                                                import SOP
from lanczos                                            import bilanczos_basis, lanczos_to_SOP, lanczos_to_SOP_GF
from utils                                              import exp_value, check_selfadjoint, pruning_sparse_zeros, FermionicOp_to_matrix
from mb_utils                                           import number_operator, gs_subspace, operator_SD, SD_states, diagonalize_Fock_Hamiltonian, statistical_weights

logger = logging.getLogger(__name__)
class solver:
    """ Solver for the DMFT cycle based on the exact diagonalization of the AIM Hamiltonian H_AIM. It returns
    the local Green's function, g.
    H_AIM :  Many-body AIMSOP Hamiltonian
    ntot   : Number of states in the fragment
    M      : Number of residues/poles
    eta    : Broadening to be considered to evaluate the poles of the Green's function g
    mu     : Chemical potential - To be added: H_AIM - mu*N_00
    beta   : Inverse temperature - If None, zero-temperature case is considered
    sparse_gs      : If True, the ground state is evaluated using sparse diagonalization
    input_matrices : List of input matrices containing the number operator on the fragment A in the AIM-SOP Fock space, the number operator in the full AIM-SOP Fock space and the creators/annihilators on the fragment in the AIM-SOP Fock space
    """
    def __init__(self, H_AIM, ntot, num_poles, eta, mu, beta=None, sparse_gs=True, input_matrices=None):
        self.H_AIM = H_AIM
        self.ntot  = ntot
        self.num_poles = num_poles
        self.eta = eta
        self.mu  = mu
        self.beta = beta
        self.sparse_gs  = sparse_gs
        self.input_matrices = input_matrices if input_matrices is not None else [None, None, None]
        d_AIM = ntot * (num_poles + 1)
        self.d_AIM = d_AIM

    def get_operator_lists(self):
        input_matrices, ntot, num_poles, d_AIM = self.input_matrices, self.ntot, self.num_poles, self.d_AIM
        # Check if cached matrices are valid: they must exist, have shape attribute, and have the correct dimension for current d_AIM
        cached_valid = (input_matrices != [None,None,None] and 
                       input_matrices[0] is not None and 
                       hasattr(input_matrices[0], 'shape') and 
                       input_matrices[0].shape[0] == 2**d_AIM)
        if not cached_valid:
            logger.info("\t\tInitializing many-body AIM-SOP solver WITHOUT input matrices in the Fock space")
            # Identity on AIM fictitious bath
            one_vec    = np.array([1]*(2**(ntot*num_poles)))
            I_AIM_bath = csc_matrix((one_vec,(np.arange(2**(ntot*num_poles)),np.arange(2**(ntot*num_poles)))),shape=(2**(ntot*num_poles),2**(ntot*num_poles)),dtype=np.complex128)

            # Number operator on the fragment in the AIMSOP Fock space
            NA_op  = number_operator(ntot)
            NA_mat = FermionicOp_to_matrix(NA_op,sparse=True)
            NA_AIM = kron(NA_mat,I_AIM_bath)
            
            # Number operator on the AIMSOP Fock space
            N_AIM_op = number_operator(d_AIM)
        
            # Set of creators and annihilators on the fragment
            cid_mat_list = [FermionicOp_to_matrix(FermionicOp({"+_"+str(i): 1.},d_AIM),sparse=True) for i in range(ntot)]
            ci_mat_list  = [FermionicOp_to_matrix(FermionicOp({"-_"+str(i): 1.},d_AIM),sparse=True) for i in range(ntot)]
            fermionic_op_list = [cid_mat_list,ci_mat_list]

            # Saving the matrices inside the list of inputs
            self.input_matrices = [NA_AIM, N_AIM_op, fermionic_op_list]
        else:
            logger.info("\t\tInitializing many-body AIM-SOP solver WITH input matrices in the Fock space")
            NA_AIM   = input_matrices[0]
            N_AIM_op = input_matrices[1]
            cid_mat_list, ci_mat_list = input_matrices[2]
        return NA_AIM, N_AIM_op, cid_mat_list, ci_mat_list
    
    def get_gs_subspace(self, max_gs_deg=7, method="std",self_adj_check=None):
        """ This function finds the ground state subspace of a given Hamiltonian H_AIM. It returns the list of left and right eigenvectors and the corresponding eigenvalue.
        max_gs_deg : Maximum degeneracy findable (number of eigenvalues to look for in the sparse diagonalization)
        method     : Method to be used for the ground state calculation. It can be "std" (ground state calculation from the full matrix H_AIM - mu*N_A), or "subspaces" (ground state calculation from all the subspaces w/ Np_try particles).
        """
        mu, d_AIM, sparse_gs, H_AIM = self.mu, self.d_AIM, self.sparse_gs, self.H_AIM
        NA_AIM = self.input_matrices[0]
        if method == "std":
            if isinstance(H_AIM,qiskit_nature.second_q.operators.FermionicOp):
                H_AIM = FermionicOp_to_matrix(H_AIM,sparse=True)
            elif isinstance(H_AIM,np.ndarray): 
                H_AIM = csc_matrix(H_AIM)
            H_AIM_mu = H_AIM - mu * NA_AIM             # Sparse matrix of the full gran-canonical problem to study on the AIM Fock space
            
            # Ground state enrgy and right/left eigenvectors of H_AIM - mu*N_00 (No fixed number of particles!)
            max_gs_deg = 5                                                                          # Maximum degeneracy findable in the sparse diagonalization
            # Method 1 - Ground state calculation from the full matrix H_AIM - mu*N_00
            E0_AIM, psi0_AIMr_list, psi0_AIMl_list = gs_subspace(H_AIM_mu,max_gs_deg,sparse=sparse_gs,self_adj_check=self_adj_check)   
            gs_deg = len(psi0_AIMr_list)
        elif method == "subspaces":
            E0_AIM_list, gsr_tot_list, gsl_tot_list = [], [], []
            for Np_try in range(1,d_AIM + 1):
                E0_AIM, gs_AIMr_list, gs_AIMl_list = gs_subspace(operator_SD(H_AIM_mu,Np_try),max_gs_deg,sparse=sparse_gs,self_adj_check=self_adj_check)
                E0_AIM_list.append(E0_AIM)
                gsr_tot_list.append(gs_AIMr_list)
                gsl_tot_list.append(gs_AIMl_list)
            ind_gs_list = [el[0] for el in sorted(enumerate(E0_AIM_list),key=lambda x:x[1].real)]
            E0_AIM, gs_AIMr_list, gs_AIMl_list = E0_AIM_list[ind_gs_list[0]], gsr_tot_list[ind_gs_list[0]], gsl_tot_list[ind_gs_list[0]]
            Np_gs       = ind_gs_list[0] + 1
            gs_deg = len(gs_AIMr_list)

            # From vector in Np_gs-space to vector in full Fock space
            SD_vec  = SD_states(d_AIM,Np_gs) 
            psi0_AIMr_list, psi0_AIMl_list = [np.zeros(2**d_AIM,dtype=np.complex128) for i in range(gs_deg)], [np.zeros(2**d_AIM,dtype=np.complex128) for i in range(gs_deg)]
            for i in range(gs_deg):    
                psi0_AIMr_list[i] = sum(np.dot(gs_AIMr_list[i][j],SD_vec[j]) for j in range(len(SD_vec)))   
                psi0_AIMl_list[i] = sum(np.dot(gs_AIMl_list[i][j],SD_vec[j]) for j in range(len(SD_vec)))
        logger.info("\t\tG.s. energy E0_AIM = {} w/ degeneracy deg = {}".format(np.round(E0_AIM,5),gs_deg))
        return E0_AIM, psi0_AIMr_list, psi0_AIMl_list
    
    def diagonalize_Hamiltonian(self):
        mu, d_AIM = self.mu, self.d_AIM
        NA_AIM = self.get_operator_lists()[0]
        if isinstance(H_AIM,qiskit_nature.second_q.operators.FermionicOp):
            H_AIM = FermionicOp_to_matrix(H_AIM,sparse=True)
        elif isinstance(H_AIM,np.ndarray): 
            H_AIM = csc_matrix(H_AIM) 

        H_AIM_mu = H_AIM - mu * NA_AIM             # Sparse matrix of the full gran-canonical problem to study on the AIM Fock space
        H_AIM_mu = H_AIM_mu.todense()
        E_full_list, psir_full_list, psil_full_list = diagonalize_Fock_Hamiltonian(H_AIM_mu,d_AIM)
        return E_full_list, psir_full_list, psil_full_list
    
    def get_Gimp(self, max_gs_deg=7, gs_search="std",method="std",self_adj_check=None):
        """ This function returns the local impurity Green's function G_imp.
        gs_search  : method to evaluate the ground state of the AIM-SOP Hamiltonian. It can be "std" (from the full matrix H_AIM - mu*N_A), or "subspaces" (from all the subspaces)
        method     : method to evaluate the Green's function. It can be "std" (from the dense diagonalization of the AIM Hamiltonian in the (Np_gs +/- 1)-particle subspaces), or "lanczos" (from the bi-Lanczos algorithm)
        self_adj_check : If True/False, it forces the check of the self-adjointness of H_AIM, which relies on dense, computationally demanding operations. If None, the check is performed 
        """
        NA_AIM, N_AIM_op, cid_mat_list, ci_mat_list = self.get_operator_lists()
        beta = self.beta
        E0_AIM, psi0_AIMr_list, psi0_AIMl_list = self.get_gs_subspace(max_gs_deg, method=gs_search, self_adj_check=self_adj_check)
        # Finite temperature case (to be implemented!)
        # if beta != None:
        #     E_full_list, psir_full_list, psil_full_list = self.diagonalize_Hamiltonian()
        #     E0_AIM = min(E_full_list)
        #     psi0_AIMr_list = [psir_full_list[i] for i in range(len(E_full_list)) if np.isclose(E_full_list[i],E0_AIM)]
        #     psi0_AIMl_list = [psil_full_list[i] for i in range(len(E_full_list)) if np.isclose(E_full_list[i],E0_AIM)]
        #     gs_deg   = len(psi0_AIMr_list)
        #     wgt_list = statistical_weights(E_full_list,beta)[0] if beta != None else 1.     # Statistical weight to use when beta != None (here, zero-temperature case with Matsubara frequencies)

        gs_deg = len(psi0_AIMr_list)
        ntot, d_AIM, eta, num_poles = self.ntot, self.d_AIM, self.eta, self.num_poles
        vecl = csc_matrix(psi0_AIMl_list[0].conj().round(16))
        vecr = csc_matrix(psi0_AIMr_list[0].round(16))
        vecl, vecr = pruning_sparse_zeros(vecl), pruning_sparse_zeros(vecr)
        Np_gs = int(np.abs(exp_value(N_AIM_op,vecl,vecr)).round(3))                         # Number of particles in the g.s.
        
        H_AIM_mu = self.H_AIM - self.mu * NA_AIM

        d_AIM_max   = 20                      # Maximum size of the AIM system to perform the dense diagonalization of the subspaces
        dim_sub_max = 5000                    # Maximum size of the dense matrix to use dense diagonalization of the subspaces
        k_max       = 100                     # Maximum number of eigenvalues/vectors to be found with sparse diagonalization of the subspaces
        if num_poles > 8:
            dim_p = len(list(combinations(np.arange(d_AIM),Np_gs + 1))) if 0 < Np_gs < d_AIM else None
            dim_m = len(list(combinations(np.arange(d_AIM),Np_gs - 1))) if 0 < Np_gs <= d_AIM else None
        else:
            dim_p, dim_m = None, None
        cond_lanczos = True if method == "lanczos" else ((d_AIM >= d_AIM_max) and ((dim_p != None and dim_p >= dim_sub_max) or (dim_m != None and dim_m >= dim_sub_max)))
        if method != "lanczos" and cond_lanczos == True:
            logger.info("\t\tMaximum number of sites and matrix dimension for dense diagonalization: {} - {}".format(d_AIM_max,dim_sub_max))
            logger.info("\t\tDimension of the (Np_gs +/- 1)-particle subspaces: {} - {}".format(dim_p,dim_m))

        C_AIM_list = []
        if cond_lanczos == True:
            Z_AIM_list = {"gs_energy": E0_AIM, "ntot": ntot, "lanczos_coeff": []}
            logger.info("\t\t...Starting Lanczos-based evaluation of the Green's function with k_max = {}...".format(k_max))
        else:
            logger.info("\t\t...Starting dense ED-based evaluation of the Green's function...")
            # Standard (canonical) basis of Slater determinants in the (Np_gs +/- 1)-Fock supspaces
            SD_vecp = SD_states(d_AIM,Np_gs + 1,sparse=True,efficient=True)
            SD_vecm = SD_states(d_AIM,Np_gs - 1,sparse=True,efficient=True)
            SD_matp = SD_vecp.transpose().tocsr()
            SD_matm = SD_vecm.transpose().tocsr()
            
        for gs_ind in range(gs_deg):
            psi0_AIMr = psi0_AIMr_list[gs_ind]
            psi0_AIMl = psi0_AIMl_list[gs_ind]
            
            # Approximation to machine precision (10^{-16}) to make psi0_AIM sparse
            psi0_AIMr = csc_matrix(psi0_AIMr.round(16))
            psi0_AIMl = csr_matrix(psi0_AIMl.round(16))
            # ort_val = np.abs((psi0_AIMl @ psi0_AIMr.T)[0,0])**2
            # logger.info("\t\tFidelity left and right g.s. : ",ort_val) 

            # Set of left and right vectors from creating/annihilating particles on the gs, i.e. vml = < psi0_AIM^L | c_i^+ , vpl = < psi0_AIM^L | c_i , vpr = c_i^+ | psi0_AIM^R > , vmr = c_i | psi0_AIM^R >
            vpr_list = [cid_mat @ psi0_AIMr.T for cid_mat in cid_mat_list]              
            vpl_list = [psi0_AIMl @ ci_mat for ci_mat in ci_mat_list]
            vmr_list = [ci_mat  @ psi0_AIMr.T for ci_mat in ci_mat_list]                
            vml_list = [psi0_AIMl @ cid_mat for cid_mat in cid_mat_list] 
            Vpr = hstack(vpr_list, format='csc') 
            Vpl = vstack(vpl_list, format='csr')                                                         
            Vmr = hstack(vmr_list, format='csc')
            Vml = vstack(vml_list, format='csr')
            if num_poles >= 6:                          # Removing elements which are equal to zero according to a fixed threshold
                logger.info("\t\tSparsity of vectors after creation/annihilation on the g.s.")
                Vpr = pruning_sparse_zeros(Vpr)
                Vpl = pruning_sparse_zeros(Vpl)
                Vmr = pruning_sparse_zeros(Vmr)
                Vml = pruning_sparse_zeros(Vml)

            if cond_lanczos == True:        # Lanczos-based evaluation of the Green's function
                C_AIM = [Vpl @ Vpr, Vml @ Vmr]
                Z_AIM = {}
                if self_adj_check == True:
                    for i in range(ntot):
                        for j in range(ntot):
                            alphap_list, betap_list, gammap_list = bilanczos_basis(Vpr[:,j],Vpl[i,:],H_AIM_mu,k_max)  # N.B. Here bilanczos_basis supports sparse entries, but the functions is approx. 10x slower than the dense version. Why? 
                            alpham_list, betam_list, gammam_list = bilanczos_basis(Vmr[:,i],Vml[j,:],H_AIM_mu,k_max)
                            idx = str(i) + str(j)
                            resp_list, polp_list = lanczos_to_SOP(alphap_list,betap_list,gammap_list)
                            resm_list, polm_list = lanczos_to_SOP(alpham_list,betam_list,gammam_list)
                            Z_AIM[idx] = [[resp_list, polp_list], [resm_list, polm_list]]
                else:
                    raise NotImplementedError("Error - Lanczos-based evaluation of the Green's function not implemented for non-self-adjoint Hamiltonians")
                Z_AIM_list["lanczos_coeff"].append(Z_AIM)
            else:               # Dense diagonalization of the (Np_gs +/- 1)-particle subspaces
                if Np_gs == 0:
                    raise TypeError("Number of particles must be different from 0 in the ground state")
                elif Np_gs > 0 and Np_gs < d_AIM:
                    if gs_ind == 0:
                        # Dense diagonalization
                        # (Np_gs +/- 1)-particle AIM Hamiltonians + right/left eigenstates
                        H_AIM_mup = operator_SD(H_AIM_mu,Np_gs + 1)
                        H_AIM_mum = operator_SD(H_AIM_mu,Np_gs - 1)

                        self_adj_p = self_adj_check if self_adj_check is not None else check_selfadjoint(H_AIM_mup)           # Assuming that the submatrices are Hermitian (True) or the opposite according to self_adj_check
                        self_adj_m = self_adj_check if self_adj_check is not None else check_selfadjoint(H_AIM_mum)
                        E_p,U_pr = LA.eig(H_AIM_mup) if self_adj_p == False else LA.eigh(H_AIM_mup)
                        E_m,U_mr = LA.eig(H_AIM_mum) if self_adj_m == False else LA.eigh(H_AIM_mum)
                        logger.info("\t\tDiagonalization of the (Np_gs +/- 1)-particle subspaces completed")
                        U_pl = LA.inv(U_pr)
                        U_ml = LA.inv(U_mr)

                        # Poles of H_AIM
                        Z_AIM_list  = [-(E0_AIM - E_p[k] + complex(0,eta)) for k in range(len(E_p))]
                        Z_AIM_list += [E0_AIM - E_m[k] + complex(0,eta) for k in range(len(E_m))]
                        logger.info("\t\tPoles of the Green's function evaluated")
                    
                    # Efficient projection of the right/left eigenvectors on the standars basis via sparse products
                    vpr_mat = SD_matp.conj() @ Vpr                                                              # Right vectors in (Np_gs + 1)-particle Fock subspace  
                    vmr_mat = SD_matm.conj() @ Vmr                                                              # Right vectors in (Np_gs - 1)-particle Fock subspace   
                    vpl_mat = Vpl @ SD_matp.T                                                                   # Left vectors in (Np_gs + 1)-particle Fock subspace             
                    vml_mat = Vml @ SD_matm.T                                                                   # Left vectors in (Np_gs - 1)-particle Fock subspace

                    # Residues of H_AIM
                    # Calculating the Dyson amplitudes - If we take the right Dyson amplitude for Np_gs + 1 each element j,k represents the product < psi0_AIM | c_j^+ | U[:,k] >
                    dys_amp_pr = vpl_mat @ U_pr
                    dys_amp_pl = U_pl    @ vpr_mat
                    dys_amp_mr = vml_mat @ U_mr
                    dys_amp_ml = U_ml    @ vmr_mat
                    # Kronecker product on columns of dys_amp_pr and dys_amp_pl > Putting matrix in one vector sorted following vertical direction (1st column, 2nd column, ...) > Reshaping in len(E_p) matrices with dim. (ntot,ntot)
                    C_AIMp = np.einsum('ij,kj->ikj', dys_amp_pr, dys_amp_pl.T).ravel(order='F').reshape(len(E_p),ntot,ntot)
                    C_AIMm = np.einsum('ij,kj->ikj', dys_amp_mr, dys_amp_ml.T).ravel(order='F').reshape (len(E_m),ntot,ntot)
                    C_AIM  = list(C_AIMp) + list(C_AIMm)
                elif Np_gs == d_AIM:
                    if gs_ind == 0:
                        # Dense diagonalization
                        # (Np_gs +/- 1)-particle AIM Hamiltonians + right/left eigenstates
                        H_AIM_mum = operator_SD(H_AIM_mu,Np_gs - 1)
                        self_adj_m = self_adj_check if self_adj_check is not None else check_selfadjoint(H_AIM_mum)
                        E_m,U_mr = LA.eig(H_AIM_mum) if self_adj_m == False else LA.eigh(H_AIM_mum)
                        logger.info("\t\tDiagonalization of the (Np_gs - 1)-particle subspace completed")
                        U_ml = LA.inv(U_mr)

                        # Poles of H_AIM
                        Z_AIM_list = [E0_AIM - E_m[k] + complex(0,eta) for k in range(len(E_m))]
                        logger.info("\t\tPoles of the Green's function evaluated")

                    # Projection of the right/left eigenvectors on the standard basis
                    vmr_mat = SD_matm.conj() @ Vmr                                                              # Right vectors in (Np_gs - 1)-particle Fock subspace   
                    vml_mat = Vml @ SD_matm.T                                                                   # Left vectors in (Np_gs - 1)-particle Fock subspace

                    # Residues of H_AIM
                    # Calculating the Dyson amplitudes - If we take the right Dyson amplitude for Np_gs + 1 each element j,k represents the product < psi0_AIM | c_j^+ | U[:,k] >
                    dys_amp_mr = vml_mat @ U_mr
                    dys_amp_ml = U_ml    @ vmr_mat
                    # Kronecker product on columns of dys_amp_pr and dys_amp_pl > Putting matrix in one vector sorted following
                    # vertical direction (1st column, 2nd column, ...) > Reshaping in len(E_p) matrices with dim. (ntot,ntot)
                    C_AIMm = np.einsum('ij,kj->ikj', dys_amp_mr, dys_amp_ml.T).ravel(order='F').reshape(len(E_m),ntot,ntot)
                    C_AIM  = list(C_AIMm)
                else:
                    raise TypeError("Number of particles exceeds the dimension of the AIM system")
            logger.info("\t\tResidues (ind = {}) of the Green's function evaluated".format(gs_ind))
            C_AIM_list.append(C_AIM)            # Adding residues to the list  
        if cond_lanczos == True:
            C_AIM_list, Z_AIM_list = lanczos_to_SOP_GF(C_AIM_list,Z_AIM_list)
        return C_AIM_list, Z_AIM_list

    def make_Gimp_SOP(self, C_AIM_list, Z_AIM_list):
        """ This function constructs transforms the residues and poles of the Green's function G_imp into a SOP-class object.
        """
        res_Gimp_list = [sum(C_AIM_list[gs_ind][k] for gs_ind in range(len(C_AIM_list))) / len(C_AIM_list) for k in range(len(Z_AIM_list))]
        pol_Gimp_list = Z_AIM_list
        return SOP(res_Gimp_list, pol_Gimp_list)