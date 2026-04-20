import time, logging
import numpy                as np
import scipy
import scipy.linalg         as LA
try:
    import qiskit_nature
    from qiskit_nature.second_q.operators import FermionicOp
except ImportError:
    qiskit_nature = None
    FermionicOp   = type(None)
from scipy.sparse                                       import csc_matrix, kron
from .utils                                              import Matsubara_freq, is_pos_semidef, closest_pos_semidef, check_selfadjoint
from .mb_utils                                           import number_operator, compute_avg_GF, calc_g, FermionicOp_to_matrix
from .SOP                                                import adapt_residues, SOP
from .AIMSOP_utils                                       import reversed_AIMSOP, AIMSOP_matrix

logger = logging.getLogger(__name__)

def create_mbAIMSOP_Hamiltonian(HA,SOP,input_matrices=[None,None,None],dmft_sim=None):
    """ This function builds the many-body AIM Hamiltonian using the residues and the poles (Gamma_k and sigma_k) 
    from the SOP fitting
    HA         : Hamiltonian of the fragment A 
    Gamma_list : List of residues
    sigma_list : List of poles
    input_matrices : List of input matrices to be used for the Hamiltonian of the fragment A, the coupling and the bath terms of the AIM-SOP Hamiltonians
    method         : Method to be used to build the many-body AIM-SOP Hamiltonian: "qiskit", "quspin", or "openfermion"
    dmft_sim        : DMFT simulation class object (to update the input matrices if needed)
    """
    if isinstance(HA,FermionicOp):
        ntot    = HA.register_length
        HA_mat = FermionicOp_to_matrix(HA,sparse=True)
    elif isinstance(HA,np.ndarray):
        ntot    = int(np.log2(len(HA)))
        HA_mat = csc_matrix(HA)
    elif isinstance(HA,scipy.sparse.csc_matrix) or isinstance(HA,scipy.sparse.csr_matrix):
        ntot    = int(np.log2(HA.shape[0]))
        HA_mat = HA
    else:
        raise ValueError('Error - op type must be Qiskit FermionicOp or np.ndarray')
    
    # Retrieving residues and poles from the SOP object
    Gamma_list = SOP.Gamma_list                                                             # List of residues
    sigma_list = SOP.sigma_list                                                             # List of poles
    p_type     = SOP.p_type                                                                 # Type of representation of the residues
    M          = SOP.num_poles                                                              # Number of poles/residues
    
    input_matrices = dmft_sim.hamiltonian_matrices if dmft_sim is not None else input_matrices
    d_AIM = ntot*(M+1)                                                                      # Dimension of the many-body AIM-SOP Fock space
    if input_matrices == [None,None,None] or input_matrices[0] is None or not hasattr(input_matrices[0], 'shape') or input_matrices[0].shape[0] != 2**d_AIM:        # Creating the matrices to construct the many-body AIM-SOP Hamiltonian
        if input_matrices[0] is not None and hasattr(input_matrices[0], 'shape') and input_matrices[0].shape[0] != 2**d_AIM:
            logger.info("\t\tWARNING: The dimension of the input matrices is not consistent with the number of poles and residues!")
        t0 = time.time()
        # Identity on fictitious bath of AIM Fock space
        one_vec    = np.array([1]*(2**(ntot*M)))
        I_AIM_bath = csc_matrix((one_vec,(np.arange(2**(ntot*M)),np.arange(2**(ntot*M)))),shape=(2**(ntot*M),2**(ntot*M)),dtype=np.complex128)
        
        # Interacting fragment in the AIM Fock space
        logger.info("\t\t...Building the local interacting Hamiltonian in the AIM Fock space...")
        HA_AIM = kron(HA_mat,I_AIM_bath)
        t1 = time.time()

        # Coupling and fictitious bath terms in the AIM Fock space (via Qiskit)
        logger.info("\t\t...Building the coupling term in the AIM Fock space...")
        hc_AIM_list, hb_AIM_list = [[[] for k in range(M)] for ind in range(2)], [[] for k in range(M)]
        for k in range(M):
            for i in range(ntot):
                for j in range(ntot):
                    label_ij  = "+_"+str(i)+" -_"+str(ntot*(k+1) + j)
                    label_ij2 = "+_"+str(ntot*(k+1) + i)+" -_"+str(j)
                    op_ij     = FermionicOp({label_ij: 1.},d_AIM)
                    op_ij2    = FermionicOp({label_ij2: 1.},d_AIM)
                    hc_AIM_list[0][k].append(FermionicOp_to_matrix(op_ij,sparse=True))          # Fermionic operator for the coupling term corresponding to c_{i}^\dagger c_{j_k}
                    hc_AIM_list[1][k].append(FermionicOp_to_matrix(op_ij2,sparse=True))         # Fermionic operator for the coupling term corresponding to c_{i_k}^\dagger c_{j}
        logger.info("\t\t...Building the bath term in the AIM Fock space...")
        for k in range(M):
            for i in range(ntot):
                label = "+_"+str(ntot*(k+1) + i)+" -_"+str(ntot*(k+1) + i)
                op    = FermionicOp({label: 1.},d_AIM)
                hb_AIM_list[k].append(FermionicOp_to_matrix(op,sparse=True))                    # Fermionic operator for the bath term corresponding to c_{i_k}^\dagger c_{i_k}

        t2 = time.time()
        logger.info("\t\tTime for building: HA_AIM = {}, hc and hb = {}".format(t1 - t0,t2 - t1))

        # Saving the list of matrices inside the list of inputs
        input_matrices[0] = HA_AIM
        input_matrices[1] = hc_AIM_list
        input_matrices[2] = hb_AIM_list
        if dmft_sim is not None:
            dmft_sim.hamiltonian_matrices = input_matrices
    else:
        # Reading the matrices to construct the many-body AIM-SOP Hamiltonian
        HA_AIM      = input_matrices[0]
        hc_AIM_list = input_matrices[1]
        hb_AIM_list = input_matrices[2]
    
    if p_type == "std":
        # Square root of the matrices Gamma    
        check_pos_semidef = [is_pos_semidef(Gamma) for Gamma in Gamma_list]
        if False in check_pos_semidef:
            logger.info("\t\tWARNING: The residues are not positive semidefinite!")
            Gamma_list = [closest_pos_semidef(Gamma) for Gamma in Gamma_list]                              # Taking the closest positive semidefinite matrix to the original Hermitian residue to guarantee the positive semidefiniteness of the square root                 
        Gammasqrt_list = [LA.sqrtm(Gamma) for Gamma in Gamma_list]                 
    elif p_type == "sqrt":
        Gammasqrt_list = Gamma_list

    # Fictitious coupling in the AIM Fock space
    hc_AIM = 0.
    for k in range(M):
        for i in range(ntot):
            for j in range(ntot):
                hc_AIM += Gammasqrt_list[k][i,j] * hc_AIM_list[0][k][i*ntot + j]
                hc_AIM += Gammasqrt_list[k][i,j] * hc_AIM_list[1][k][i*ntot + j]

    # Fictitious bath in the AIM Fock space
    hb_AIM = 0.
    for k in range(M):
        for i in range(ntot):
            hb_AIM += sigma_list[k] * hb_AIM_list[k][i]
    
    # Interacting Hamiltonian in the AIM Fock space
    H_AIM = HA_AIM + hb_AIM + hc_AIM
    return H_AIM

def self_consistency_DMFT(SOP,Gimp_SOP,epsk_list,hA_1,w_list,mu,paramagnetic=False):
    """ This function performs the operation of self-consistency typical of a DMFT simulation. Specifically, this is the upfolding from the
    impurity Green's function to the local (lattice) Green's function and the self-energy."""
    ntot = SOP.dim
    Nk   = len(epsk_list)
    v_emb_fit_list = SOP.evaluate(w_list)
    
    # Finding new initial values: SigmaA, GA = G_loc
    Gimp_list      = Gimp_SOP.evaluate(w_list)                                                  # Impurity GF G_imp from the residues and poles returned by the solver
    IA_mat         = np.identity(ntot,dtype=np.complex128)
    if paramagnetic == True and np.allclose(hA_1,hA_1[0,0] * IA_mat):
        SigmaA_list = [(w + mu - hA_1[0,0] - v_emb_fit_list[iw][0,0] - pow(Gimp_list[iw][0,0],-1)) * IA_mat for iw,w in enumerate(w_list)]                # Self-energy from the impurity GF and non-interacting GF on the fragment
        Gloc_list   = [sum(1 / (w + mu - epsk_list[k] - SigmaA_list[iw][0,0]) for k in np.arange(Nk)) * IA_mat / Nk for iw,w in enumerate(w_list)]
    else:
        zero_list      = [np.zeros((ntot,ntot),dtype=np.complex128) for w in w_list]
        G0imp_fit_list = calc_g(v_emb_fit_list,zero_list,hA_1,w_list,mu)
        SigmaA_list = [LA.inv(G0imp_fit_list[iw]) - LA.inv(Gimp_list[iw]) for iw in range(len(w_list))]                                                # Self-energy from the impurity GF and non-interacting GF on the fragment
        Gloc_list     = [sum(LA.inv((w + mu - epsk_list[k])*IA_mat - SigmaA_list[iw]) for k in np.arange(Nk)) / Nk for iw,w in enumerate(w_list)]       # Lattice GF G_loc from the upfloding to the periodic lattice!\
    return SigmaA_list, Gloc_list

def linear_mixing_lists(list1,list2,alpha):
    if len(list1) != len(list2):
        raise ValueError("The two lists must have the same length for linear mixing.")
    else:
        if alpha <= 0 or alpha > 1:
            raise ValueError("alpha must be in the range (0, 1] for linear mixing.")
        else:
            list_new = [alpha * list1[k] + (1 - alpha) * list2[k] for k in range(len(list1))]
    return list_new

def frequency_axis(axis, eta_axis, num_pts=10000, matsubara_params={"beta": 1500, "Nw_max": 3000},w_edges=[-10,10]):
    """ Function which initializes the frequency grid for the an embedding cycle."""
    if axis == "imaginary":
        # w_list = [Matsubara_freq(n,beta_T) for n in range(0,5000)]
        w_list = [Matsubara_freq(n,matsubara_params["beta"]) for n in range(-matsubara_params['Nw_max'],matsubara_params['Nw_max'])]                                                  # Matsubara frequencies
        grid_text   = "(Matsubara freq. with Nw_max = " + str(matsubara_params['Nw_max']) + " and beta = " + str(matsubara_params['beta']) + ")"
        w_imag_list = [complex(0,w) for w in w_list]
        w_sim_list = w_imag_list
    else:
        w_list = np.linspace(w_edges[0],w_edges[1],num_pts).tolist()                                                           # Frequency grid 'standard'
        # w_list = np.concatenate((np.linspace(-10,-1,100), np.linspace(-1,1,100)[1:], np.linspace(1,10,100)[1:]))            # Frequency grid denser around 0
        grid_text = ""
        if axis == "erf":                                                                                                      
            w_erf_list  = [w + eta_axis * complex(0,scipy.special.erf(w)) for w in w_list]
            w_sim_list = w_erf_list
            grid_text = " w/ eta_erf = {}".format(eta_axis)
        elif axis == "shift":
            w_shift_list = [w - eta_axis * complex(0,1) for w in w_list]                                                    # Shifted frequency grid
            grid_text = " w/ shift = {}".format(eta_axis)
            w_sim_list = w_shift_list
        else:
            w_sim_list = w_list
    return w_list, w_sim_list

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

def get_SigmaA_SOP(Gimp_SOP,SOP_vemb,mu,hA_1):
    """ This function returns the self-energy Sigma_A as a SOP object, given the impurity GF and the embedding potential as SOP objects, and the non-interacting Hamiltonian of the fragment, using the hypotheses of DMFT on the local self-energy and Green's function. 
    This is done by performing the reversed algorithmic inversion to get the self-energy Sigma_A as a SOP object.
    """
    const_term, B2_list, Omega_list = reversed_AIMSOP(Gimp_SOP)                                                # Constant term and SOP associated with (w - G_imp^{-1})
    const_term_SigmaA = mu * np.identity(Gimp_SOP.dim,dtype=np.complex128) - hA_1 - const_term                 # Constant term of the self-energy Sigma_A                       
    res_list_SigmaA   = adapt_residues(SOP_vemb.Gamma_list,SOP_vemb.p_type,"std") + B2_list                    # Residues of the self-energy Sigma_A
    res_list_SigmaA   = [-1 * res for res in res_list_SigmaA]                                                  
    pol_list_SigmaA   = SOP_vemb.sigma_list + list(Omega_list)                                                 # Poles of the self-energy Sigma_A
    SigmaA_SOP = SOP(res_list_SigmaA, pol_list_SigmaA, p_type="std")
    return SigmaA_SOP, const_term_SigmaA

def get_Gloc_SOP(SOP_vemb,Gimp_SOP,mu,hA_1,epsk_list):
    """ This function return the local Green's function G_loc as a SOP object, given the embedding potential and the impurity GF as SOP objects, and the non-interacting Hamiltonian of the fragment using the local interaction hypothesis, as in DMFT. 
    This is done by performing the reversed algorithmic inversion to get the self-energy Sigma_A as a SOP object, and then evaluating the local Green's function G_loc from the self-energy via the Dyson equation.
    """
    ntot  = Gimp_SOP.dim
    I_loc = np.identity(ntot,dtype=np.complex128)
    SigmaA_SOP, const_term_SigmaA = get_SigmaA_SOP(Gimp_SOP,SOP_vemb,mu,hA_1)                                              # Self-energy Sigma_A as a SOP object and its constant term
    A_list, Z_list = [], []
    for ind,epsk in enumerate(epsk_list):
        h0_inv   = (epsk - mu) * I_loc + const_term_SigmaA
        hAIM_inv = AIMSOP_matrix(h0_inv,SigmaA_SOP.Gamma_list,SigmaA_SOP.sigma_list,p_type=SigmaA_SOP.p_type)
        if check_selfadjoint(hAIM_inv) == True:
            Z2_list, U = LA.eigh(hAIM_inv)                                                                                  # Eigenvalues (poles) and eigenvectors of the AIMSOP matrix
            C2_list = [np.outer(U[:,i],U[:,i].conj())[:ntot,:ntot] for i in range(len(Z2_list))]                            # Residues of the AIMSOP matrix
        else:
            Z2_list, Ur = LA.eig(hAIM_inv)
            Ul = LA.inv(Ur)
            C2_list = [np.outer(Ur[:,i],Ul[i,:])[:ntot,:ntot] for i in range(len(Z2_list))]                                 # N.B. No need to conjugate the left eigenvectors!
        A_list += C2_list
        Z_list += Z2_list.tolist()
    Gloc_SOP = SOP(A_list,Z_list,p_type="std")
    return Gloc_SOP

def get_new_vemb_SOP(SOP_vemb,Gimp_SOP,mu,hA_1,epsk_list):
    """ This function returns the new embedding potential as a SOP object, calculating the local Green's function and the self-energy as SOP objects, and the non-interacting Hamiltonian of the fragment, using the hypotheses
    of DMFT on the local self-energy and Green's function. This is done by performing the reversed algorithmic inversion to get the new embedding potential as a SOP object.
    """
    ntot  = Gimp_SOP.dim
    I_loc = np.identity(ntot,dtype=np.complex128)
    SigmaA_SOP, const_term_SigmaA = get_SigmaA_SOP(Gimp_SOP,SOP_vemb,mu,hA_1)                                           # Self-energy Sigma_A as a SOP object and its constant term
    Gloc_SOP = get_Gloc_SOP(SOP_vemb,Gimp_SOP,mu,hA_1,epsk_list)                                                        # Local Green's function G_loc as a SOP object
    const_term, B2_list, Omega_list = reversed_AIMSOP(Gloc_SOP)                                                         # Constant term and residues and poles of (omega - Gloc^{-1}) as a SOP object
    const_term_vemb = mu * I_loc - hA_1 - const_term_SigmaA - const_term                                                # Constant term of the new embedding potential
    res_list_vemb   = adapt_residues(SigmaA_SOP.Gamma_list,SigmaA_SOP.p_type,"std") + B2_list                           # Residues of the new embedding potential
    res_list_vemb   = [-1 * res for res in res_list_vemb]                                                  
    pol_list_vemb   = SigmaA_SOP.sigma_list + list(Omega_list)                                                          # Poles of the new embedding potential
    vemb_SOP_new    = SOP(res_list_vemb, pol_list_vemb, p_type="std")
    return vemb_SOP_new, const_term_vemb
    