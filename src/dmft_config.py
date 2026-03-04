import numpy            as np
import scipy
import yaml  # type: ignore[import-untyped]
import logging
from dataclasses        import dataclass, field
from typing             import Optional

from utils              import check_selfadjoint
from SOP                import SOP
from hubbard            import prepare_Hubbard_Hamiltonians
from embedding_utils    import frequency_axis, self_consistency_DMFT, linear_mixing_lists
from mb_utils           import operator_SD, G_operator
from data_io            import read_dmft_data, read_vemb_data, read_conv_history

logger = logging.getLogger(__name__)
@dataclass
class Hubbard_system_config:
    size : int              # Total number of sites in the lattice
    t    : float            # Hopping parameter
    U    : float            # On-site interaction
    sizeA: int              # Number of sites in the fragment (impurity), referred to as "A" in the code
    Np   : float            # Particle density per site
    bc   : int              # Boundary conditions: 1 for open, 0 for closed
    Np_tot  : float = field(init=False)
    epsk_list: list = field(init=False, default_factory=list)

    def __post_init__(self):
        self.Np_tot    = self.Np * self.size
        self.epsk_list = self.calculate_epsk_list()
        self.ntot      = 2 * self.sizeA                 # Total dimension of the impurity problem (spin included)

    def calculate_epsk_list(self):
        """ Function to calculate the dispersion relation for a 1D Hubbard system with nearest-neighbour hopping and given boundary conditions (either chain or ring)."""
        Nk = self.size / self.sizeA
        return [- 2 * self.t * np.cos(2 * np.pi * k / Nk) for k in range(int(Nk))]
    
    def get_Hubbard_Hamiltonian(self):
        return prepare_Hubbard_Hamiltonians(self.t, self.U, self.size, self.sizeA, bc=self.bc)
    
    def get_single_particle_hA(self):
        hA = self.get_Hubbard_Hamiltonian()[2]
        hA_1 = operator_SD(hA,1) if hA != 0. else np.zeros((2 * self.sizeA,2 * self.sizeA),dtype=np.complex128)
        return hA_1

@dataclass
class input_config:
    input_case : str = "from_file"   # "from_file", "non_int", "fixed_params" are the options
    config_file: Optional[str] = None
    dmft_file  : Optional[str] = None
    vemb_file  : Optional[str] = None
    conv_file  : Optional[str] = None
    # p0_index   : Optional[int] = None

@ dataclass
class embedding_config:
    max_iter  : int = 300
    num_poles : int = 4
    mu_fixed  : Optional[float] = None  # If not None, the fixed chemical potential during the DMFT cycle to the input value
    p_type    : str = "sqrt"            # Type of the residues of the embedding potential: "std" or "sqrt"
    axis      : str = "imaginary"        # Axis for the mixing/fit of the embedding potential: "erf", "shift" or "imaginary"
    eta_axis  : float = 0.5 if axis != "imaginary" else 0.0
    num_pts   : int = 10000             # Number of grid points for the erf axis and imaginary semi-axis (used for the integral) in the DMFT cycle
    beta_T    : float = 1500.           # Inverse temperature for the Matsubara frequencies - N.B. 2 * pi / beta gives the spacing between two consecutive frequencies
    Nw_max    : int = 3000              # Maximum number of Matsubara frequencies
    w_edges   : list = field(default_factory=lambda: [-10, 10])         # Edges of the frequency grid for the real axis 
    sparse_gs : bool = True             # If True, the solver will search the ground state of the many-body AIM-SOP Hamiltonian with a sparse method
    gs_search : str = "std"             # Method to search the g.s. in the solver: "std" for standard diagonalization, "subspaces" for diagonalization in each subspace with fixed number of particles
    solver_method : str = "std"         # Method to evaluate G_imp in the solver: "std" for standard diagonalization, "lanczos" for bi-Lanczos algorithm
    matsubara_params : dict = field(init=False)
    
    def __post_init__(self):
        self.matsubara_params = {
            "beta": self.beta_T,
            "Nw_max": self.Nw_max
        }          
@dataclass
class optimization_config:
    mixing_method  : str = "linear"         # Mixing method for the self-consistency: "linear" (the only implemented method so far)
    alpha          : float = 0.5            # Linear mixing parameter for the self-consistency
    opt_method     : str = "scipy_CG"         # Optimization method to minimize the cost function for the dynamics of poles: "anal_SD", "scipy_CG", "custom_CG" -N.B. "custom_CG" is based on the interpolation of delta
    opt_params     : list = field(default_factory=lambda: [100, 1e-5])  # Parameters for the optimization method: for "scipy_CG", "custom_CG" or "anal_SD", [max_iter, tol]]
    initial_mixing : bool = True            # If True, the initial guess for the embedding potential is mixed with the previous one, otherwise it is not
    complex_poles  : bool = False           # Complex poles for the dynamics of poles of the embedding potential
    herm_residues  : bool = True            # If True, the residues (or the corresponding square root if p_type = "sqrt") of the embedding potential are imposed hermitian
    fixed_residues : bool = False           # Fixed residues for the dynamics of poles of the embedding potential
    odd_spectrum   : bool = False           # If True, the spectrum is an function, i.e. the poles are antisymmetric with respect to the axis origin and the residues symmetric
    paramagnetic   : bool = True            # If True, the embedding potential is a list of scalars, i.e. we assume a spin up-down symmetry in the system related to the paramagnetic behaviour
    interp_method  : str = "sampling"       # Method used to interpolate the delta parameter in the analytic CG minimization: "sampling" or "parabola"
    print_interp   : bool = False           # If True, the interpolation of the delta parameter is saved in ./figures folder
    p0_start       : str = "always"         # "never" if we never want to use the previous paramters, "always" for the opposite, "no_first_iter" if we don't want to use the previous parameters at the first iterations
    bounds         : dict = field(init=False)
    thr_diff_prev  : float = 1e-8                       # Thresholds to accept convergence for the local difference and the difference between two consecutive iterations
    # thr_diff_cost  : float = 1e-4                       # Threshold on the difference between 2 consecutive cost function values
    thr_stagnation : float = thr_diff_prev * 1e-1       # Threshold on the stagnation of the diff_prev to stop the self-consistent cycle
    RMSE_thr       : float = 0.5                        # Threshold on the RMSE of the embedding potential to stop the self-consistent cycle

    def __post_init__(self):
        self.bounds = {
            "complex_poles": self.complex_poles,
            "fixed_residues": self.fixed_residues,
            "odd_spectrum": self.odd_spectrum,
            "herm_residues": self.herm_residues,
            "paramagnetic": self.paramagnetic
        }   # Dictionary with the bounds for the residues and poles of the embedding potential during the fitting and/or the dynamics of poles
@dataclass
class sim_config:
    system: Hubbard_system_config
    input: input_config
    embedding: embedding_config
    optimization: optimization_config
        
    def get_input_variables(self):
        logger.info("Reading configuration of the DMFT cycle")
        input_case = self.input.input_case
        w_list, w_test_list = frequency_axis(
            axis=self.embedding.axis,
            eta_axis=self.embedding.eta_axis,
            num_pts=self.embedding.num_pts,
            matsubara_params={
                "beta": self.embedding.beta_T,
                "Nw_max": self.embedding.Nw_max
            },
            w_edges=self.embedding.w_edges
        )
        ntot = self.system.ntot
        perturb_matrix = np.ones((ntot,ntot),dtype=np.complex128)                                   # Perturbation matrix to add to the initial guess of residues and poles p0
        if input_case == "non_int":
            epsk_list = self.system.epsk_list
            Nk = len(epsk_list)
            mu0 = self.embedding.mu_fixed if self.embedding.mu_fixed is not None else 0.                                # Initial chemical potential of the cycle
            h   = self.system.get_Hubbard_Hamiltonian()[0]                                          # Non-interacting Hamiltonian of the system
            if h is not None:
                h_1 = operator_SD(h,1)                                                              # Single-particle matrix of h == first quantization of h
                Gloc_list0 = [G_operator(h_1,w + mu0)[:ntot,:ntot] for w in w_test_list]      # GF on the test axis
            else:
                Gloc_list0 = [sum(1 / (w - epsk_list[k]) for k in np.arange(Nk))*np.identity(ntot,dtype=np.complex128) / Nk for w in w_test_list]  # GF on the test axis
            SigmaA_list0 = [np.zeros((ntot,ntot),dtype=np.complex128) for w in w_list]      # Zero self-energy on the real axis
            if self.embedding.num_poles == 4:         # Random initial set of parameters for the embedding potential
                Gamma_list0 = [perturb_matrix] * self.embedding.num_poles
                sigma_list0 = [-2, -0.1, 0.1, 2] if self.embedding.axis == "erf" else [-2,-1.,1.,2]
                SOP0 = SOP(Gamma_list0, sigma_list0, p_type=self.embedding.p_type)
            else:
                SOP0 = SOP(None,None,p_type=self.embedding.p_type)
            Gimp_SOP0 = None
        elif input_case == "from_file":
            input_info = load_sim_config(self.input.config_file)
            epsk_list0 = input_info.system.epsk_list
            Gimp_SOP0, Gloc_list0, SigmaA_list0 = read_dmft_data(self.input.dmft_file)[1:]
            _, SOP0       = read_vemb_data(self.input.vemb_file)
            conv_history0 = read_conv_history(self.input.conv_file)
            mu0 = self.embedding.mu_fixed if self.embedding.mu_fixed is not None else conv_history0["mu"][-1]  
            if [self.embedding.axis, self.embedding.eta_axis, self.embedding.num_pts, self.embedding.w_edges, self.system.size, self.embedding.beta_T] != [input_info.embedding.axis, input_info.embedding.eta_axis, input_info.embedding.num_pts, input_info.embedding.w_edges, input_info.system.size, input_info.embedding.beta_T]:
                logger.info("Adapting initial G_loc and Sigma to the new simulation (change in axis, grid points, or number of sites)")
                hA = prepare_Hubbard_Hamiltonians(input_info.system.t, input_info.system.U, input_info.system.size, input_info.system.sizeA, bc=input_info.system.bc)[2]
                hA_1 = operator_SD(hA,1) if hA != 0. else np.zeros((input_info.system.sizeA,input_info.system.sizeA),dtype=np.complex128)
                SigmaA_list0, Gloc_list0 = self_consistency_DMFT(SOP0,Gimp_SOP0,epsk_list0,hA_1,w_test_list,mu0,paramagnetic=input_info.optimization.paramagnetic)
        
        if SOP0.Gamma_list is not None:
            if self.optimization.odd_spectrum == True and SOP0.is_odd() == False: 
                raise ValueError("Initial residues and poles for the embedding potential don't return an odd spectrum") 
            if False in [check_selfadjoint(Gamma) for Gamma in SOP0.Gamma_list]:
                raise ValueError("Initial residues for the embedding potential are not hermitian")
    
        return Gimp_SOP0, Gloc_list0, SigmaA_list0, SOP0, mu0

def load_sim_config(yaml_file: str) -> sim_config:
    with open(yaml_file, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Ensure all numeric values in optimization config are floats
    opt_cfg = cfg["optimization"].copy()
    for key in ["alpha", "thr_diff_prev", "thr_stagnation", "RMSE_thr"]:
        if key in opt_cfg and opt_cfg[key] is not None:
            opt_cfg[key] = float(opt_cfg[key])
    
    return sim_config(
        system=Hubbard_system_config(**cfg["system"]),
        input=input_config(**cfg["input"]),
        embedding=embedding_config(**cfg["embedding"]),
        optimization=optimization_config(**opt_cfg)
    )