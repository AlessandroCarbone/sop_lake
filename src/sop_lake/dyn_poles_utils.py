import numpy                            as np
import scipy
from scipy           import linalg       as LA
from scipy.signal    import find_peaks, peak_prominences
from .utils           import check_selfadjoint, closest_hermitian, is_pos_semidef, closest_pos_semidef
from .SOP             import SOP, SOP_to_params, params_to_SOP

def set_fit_bounds(p0,M,dim,fixed_params_ind,mu=0.,eps=1e-3):
    """ This function sets the bounds in the fitting routine for the residues and the poles of the SOP. In general all poles have to be time ordered with respect to the chemical potential mu,
    and those which have to be fixed are set to move around a small range of +/- eps.
    """
    # Bounds for the poles of the fit - Poles have to be time ordered, i.e. real parts are locked before or after mu and imaginary parts are locked to be positive or negative
    max_imag_pole  = np.inf
    res_fit_bounds = [[-np.inf] * (2 * M * dim**2), [np.inf] * (2 * M * dim**2)]
    pol_fit_bounds = [[] for i in range(2)]
    for ind in range(M):
        el_real, el_imag = p0[2*(dim**2)*M + ind*2], p0[2*(dim**2)*M + ind*2 + 1]
        if ind in fixed_params_ind:
            down_real_lim = el_real - eps if el_real < mu else el_real
            up_real_lim   = el_real if el_real < mu else el_real + eps
            down_imag_lim = 0. if el_real < mu else -max_imag_pole
            up_imag_lim   = max_imag_pole if el_real < mu else 0.
        else:
            down_real_lim = -np.inf if el_real < mu else mu
            up_real_lim   = mu if el_real < mu else np.inf
            down_imag_lim = 0. if el_real < mu else -max_imag_pole
            up_imag_lim   = max_imag_pole if el_real < mu else 0.
        # Bounds for the real part of the pole
        pol_fit_bounds[0].append(down_real_lim)
        pol_fit_bounds[1].append(up_real_lim)
        # Bounds for the imaginary part of the pole
        pol_fit_bounds[0].append(down_imag_lim)
        pol_fit_bounds[1].append(up_imag_lim)
                
    fit_bounds = (tuple(res_fit_bounds[0] + pol_fit_bounds[0]),tuple(res_fit_bounds[1] + pol_fit_bounds[1]))
    return fit_bounds

def mat_list_forfit(mat_list,dim,wgrid_len):
    mat_wlist = [[] for i in range(dim**2)]
    for i in range(dim): 
        for j in range(dim):
            for ind in range(wgrid_len):
                mat_wlist[i*dim+j].append(mat_list[ind][i,j])

    mat_wlist_real, mat_wlist_imag = [], []
    for i in range(dim**2):
        mat_wlist_real = np.append(mat_wlist_real, [mat_wlist[i][k].real for k in range(wgrid_len)])
        mat_wlist_imag = np.append(mat_wlist_imag, [mat_wlist[i][k].imag for k in range(wgrid_len)])
    mat_wlist_forfit   = np.append(mat_wlist_real,mat_wlist_imag)
    return mat_wlist_forfit

def find_relevant_peaks(func_list,w_list,min_height=1e-5,cond="prominence"):
    """ This routine returns the most relavant peaks of a function according to a specific condition, for instance height or prominence.
    func_list  : List of the values of the function
    w_list     : Frequency grid for the function
    min_height : Minimum height of the peaks to be considered relevant
    cond       : Condition to select the peaks, either "height" or "prominence"
    """
    peaks, prop = find_peaks(func_list,min_height)                                                              # Peaks position                                                       
    wpeaks_list = [w_list[iw] for iw in peaks]                                                                  # List of the positions of the peaks
    # print("          Peaks positions : ",*wpeaks_list)
    
    # Sorting peaks by prominence or height
    if cond == "prominence":
        rel_peaks_list  = peak_prominences(func_list, peaks)[0]                                                 # Prominence of the peaks
    elif cond == "height":
        rel_peaks_list  = prop['peak_heights']                                                                  # Peaks height
    sort_list       = [el[0] for el in sorted(enumerate(rel_peaks_list),key=lambda x: x[1],reverse=True)]       # Here we sort the peaks by PROMINENCE/HEIGHT in decreasing order, so that the fitting takes in consideration the most relevant ones first
    rel_peaks_list  = [rel_peaks_list[ind] for ind in sort_list]                                                # Relevance according to which the peaks are sorted (height or prominence)       
    wpeaks_list     = [wpeaks_list[ind] for ind in sort_list]
    # print("         Peaks positions sorted by ",cond,": ",*wpeaks_list)
    return wpeaks_list, rel_peaks_list

def change_poles_params(p0,M,dim,sigma_new_list,complex_poles=False):
    """ Here we change the poles in a given list of parameters (residues and poles in a single vector) with a new list. The new list of poles could be longer or smaller 
    than the original one.
    """
    if any(el != el.real for el in sigma_new_list):
        raise ValueError('Error - w_new_list must be a list of real numbers')
    p0_new = p0
    for ind in range(min(M,len(sigma_new_list))):
        p0_new[2*M*(dim**2) + ind*2] = sigma_new_list[ind]                                                          # Changing the poles in the new ones
        if complex_poles == False:
            p0_new[2*M*(dim**2) + ind*2 + 1] = 0.                                                                   # Setting to zero the imaginary part of the poles
    return p0_new

def change_residues_params(p0,M,Gamma_new_list):
    """ Here we change the residues in a given list of parameters (residues and poles in a single vector) with a new list. The new list of residues could be longer or smaller 
    than the original one.
    """
    dim    = len(Gamma_new_list[0])
    p0_new = p0
    for k in range(min(M,len(Gamma_new_list))):
        Gamma = Gamma_new_list[k]
        for i in range(dim):
            for j in range(dim):
                p0_new[k*2*(dim**2) + i*2*dim + j*2]     = Gamma[i,j].real
                p0_new[k*2*(dim**2) + i*2*dim + j*2 + 1] = Gamma[i,j].imag
    return p0_new

def time_ordered_params(p0,dim,mu=0.):
    """ Forcing time ordering of the poles w.r.t. to the expected chemical potential, i.e. correcting the imaginary parts of the poles in the list of parameters
    """
    M = int(len(p0) / (2*(dim**2 + 1)))
    p0_new = p0
    for ind in range(M):
        el_real, el_imag = p0_new[2*(dim**2)*M + ind*2], p0_new[2*(dim**2)*M + ind*2 + 1]
        if el_imag != 0.: 
            p0_new[2*(dim**2)*M + ind*2 + 1] = np.abs(el_imag) if el_real < mu else -np.abs(el_imag)
    return p0_new

def complex_lin_lsq_mat(mat_list,sigma_list,w_list):
    """ This function performs a general linear least squares fit of a matrix list. This means that it returns the residues of the SOP representation when the poles are fixed. 
    See pag. 788 of "W. Press, et al., Numerical Recipes" for more details.
    N.B. Avoid applying this function when the list of frequencies is real (axis = "real")!
    vemb_list  : List of complex matrices to fit
    sigma_list : List of complex poles
    w_list     : List of complex frequency points
    """
    M, ntot = len(sigma_list), len(mat_list[0])
    a_list, b_list = [sigma.real for sigma in sigma_list], [sigma.imag for sigma in sigma_list]

    gk_p  = lambda w, ak: w.real - ak
    gk_2p = lambda w, bk: bk - w.imag
    gk    = lambda w, ak, bk: gk_p(w,ak)**2 + gk_2p(w,bk)**2

    H, N, O, P = np.zeros((M,M),dtype=np.float64), np.zeros((M,M),dtype=np.float64), np.zeros((M,M),dtype=np.float64), np.zeros((M,M),dtype=np.float64)
    for k in range(M):
        for l in range(M):
            H[k,l] = sum(gk_p(w,a_list[l])  * gk_p(w,a_list[k])  / (gk(w,a_list[l],b_list[l]) * gk(w,a_list[k],b_list[k])) for w in w_list)
            N[k,l] = sum(gk_2p(w,b_list[l]) * gk_p(w,a_list[k])  / (gk(w,a_list[l],b_list[l]) * gk(w,a_list[k],b_list[k])) for w in w_list)
            O[k,l] = sum(gk_2p(w,b_list[l]) * gk_2p(w,b_list[k]) / (gk(w,a_list[l],b_list[l]) * gk(w,a_list[k],b_list[k])) for w in w_list)
            P[k,l] = sum(gk_p(w,a_list[l])  * gk_2p(w,b_list[k]) / (gk(w,a_list[l],b_list[l]) * gk(w,a_list[k],b_list[k])) for w in w_list)
    Q = H + O  + (N - P) @ LA.inv(H + O) @ (N - P)

    gamma_list, theta_list = [], []
    for i in range(ntot):
        for j in range(ntot):
            gamma_list.append(np.zeros(M,dtype=np.float64))
            theta_list.append(np.zeros(M,dtype=np.float64))
            for k in range(M):
                gamma_list[-1][k] = sum(mat_list[iw][i,j].real * gk_p(w,a_list[k]) / gk(w,a_list[k],b_list[k]) + mat_list[iw][i,j].imag * gk_2p(w,b_list[k]) / gk(w,a_list[k],b_list[k]) for iw,w in enumerate(w_list))
                theta_list[-1][k] = sum(-mat_list[iw][i,j].real * gk_2p(w,b_list[k]) / gk(w,a_list[k],b_list[k]) + mat_list[iw][i,j].imag * gk_p(w,a_list[k]) / gk(w,a_list[k],b_list[k]) for iw,w in enumerate(w_list))
    
    A_list, B_list = [], []
    for i in range(ntot):
        for j in range(ntot):
            A_list.append(LA.inv(Q) @ (gamma_list[i*ntot + j] + (N - P) @ LA.inv(H + O) @ theta_list[i*ntot + j]))
            B_list.append(LA.inv(H + O) @ (theta_list[i*ntot + j] - (N - P) @ A_list[-1]))
    Gamma_list = [np.zeros((ntot,ntot),dtype=np.complex128) for k in range(M)]
    for k in range(M):
        for i in range(ntot):
            for j in range(ntot):
                Gamma_list[k][i,j] = complex(A_list[i*ntot + j][k], B_list[i*ntot + j][k])
    return Gamma_list

def set_initial_params(p0,M,w_list,mat_list,mu=0.,axis="imaginary",eta_axis=0.,complex_poles=False,p_type="std"):
    """ This function sets the initial guess for the parameters of the fitting routine. If p0 is not given, 
    the function looks for the position of the peaks of the spectral function, ~ Im[g(w)], to guess the parameters of the fitting
    p0          : Initial guess for the SOP parameters of the fitting routine, if None the function looks for the peaks of the spectral function to guess the parameters
    M           : Number of poles/residues, i.e., SOP parameters, to describe the embedding potential
    w_list      : List of frequencies on the simulation axis
    mat_list    : List of matrices to fit with the SOP representation, on the same frequency grid as w_list
    mu          : Chemical potential
    complex_poles : If True, the poles of the initial parameter are taken/kept complex, otherwise real
    """
    dim        = len(mat_list[0])                                                                                               
    p0_len     = len(p0) if p0 is not None else 0
    p0_len_exp = 2 * M * (dim**2 + 1)                                                                                   # Expected length of the list of parameters              
    Gamma_list0_guess = [np.identity(dim,dtype=np.complex128) for k in range(M)]                                        # This initial guess is correct for both cases of p_type
    sigma_list0_guess = np.linspace(w_list[0] / 2.,w_list[-1] / 2.,M)
    p0_guess = SOP_to_params(Gamma_list0_guess,sigma_list0_guess)                                                       # Initial guess from equally spaced poles and identity residues
    if axis == "erf":                                                                                                   
        w_test_list  = [w + eta_axis * complex(0,scipy.special.erf(w)) for w in w_list]
    elif axis == "imaginary":                                                                                                         
        w_test_list = [complex(0,w) for w in w_list]
    elif axis == "shift":
        w_test_list = [w - eta_axis * complex(0,1) for w in w_list]                                                      # Shifted frequency grid
    else:
        w_test_list = w_list
    
    func_list   = np.array([np.abs(mat_list[iw][0,0].imag) for iw,w in enumerate(w_list)])                              # Function to use to find the peaks, i.e. the poles: imaginary part of the list of matrices
    func_list2  = np.array([np.abs(mat_list[iw][0,0].real) for iw,w in enumerate(w_list)])                              # Function to use to find the peaks, i.e. the poles: real part of the list of matrices
    min_height, peaks_cond  = 1e-5, "prominence"                                                                        # Condition to find the peaks: minimal height and peak character
    # min_height  = max(find_peaks(func_list,0.01)[1]['peak_heights']) / 10.                                            # Alternative minimal height to detect a peak   
    wpeaks_list = []
    if p0 is None or p0_len != p0_len_exp:                                                                              # If no initial guess is given or the length of the list of parameters is not the expected one
        peaks_height_list  = find_peaks(func_list,min_height)[1]['peak_heights']                                                                                                                                  
        if len(peaks_height_list) != 0:
            wpeaks_list, rel_peaks_list = find_relevant_peaks(func_list,w_list,min_height=min_height,cond=peaks_cond)   # List of the relevant peaks of the spectral function, ~ Im[g(w)], to guess the parameters of the fitting
            p0_text = "\t\tInitial parameters from the position of prominent peaks" 
        else:
            peaks_height_list2 = find_peaks(func_list2,min_height)[1]['peak_heights']
            if len(peaks_height_list2) != 0:
                wpeaks_list, rel_peaks_list = find_relevant_peaks(func_list2,w_list,min_height=min_height,cond=peaks_cond)  # List of the relevant peaks of the spectral function, ~ Re[g(w)], to guess the parameters of the fitting
                p0_text = "\t\tInitial parameters from the position of prominent peaks"
            else:
                p0_new = p0_guess                                                                                           # If no peaks are found, we use the guess list 
                p0_text = "\t\tInitial parameters from random guess (equally spaced poles and identity residues)" 
    else:                                                                                            
        p0_new  = p0
        p0_text = "\t\tInitial parameters from input data"
    
    herm_text, pos_semidef_text = "", ""
    if len(wpeaks_list) != 0:                                                                                       
        sigma_list0 = [wpeaks_list[iw] for iw in range(min(M,len(wpeaks_list)))]
        Gamma_list0 = complex_lin_lsq_mat(mat_list,sigma_list0,w_test_list) if axis != "shift" else [rel_peaks_list[k] / max(rel_peaks_list) * np.identity(dim,dtype=np.complex128) for k in range(len(sigma_list0))]    # Residues from the linear least squares fit
        if len(sigma_list0) < M:                                                                                                            # Completing the basis of residues and poles (if necessary) until reaching M elements
            sigma_list0 = sigma_list0 + list(np.linspace(-0.09, 0.1, M - len(sigma_list0)))                                                  # Add poles equally spaced around 0                  
            Gamma_list0 = Gamma_list0 + [np.zeros((dim,dim),dtype=np.complex128) for k in range(M - len(Gamma_list0))]
        if False in [check_selfadjoint(Gamma) for Gamma in Gamma_list0]:
            herm_text = " - imposed hermiticity"
            Gamma_list0 = [closest_hermitian(Gamma) if check_selfadjoint(Gamma) == False else Gamma for Gamma in Gamma_list0]               # Substituting each Gamma w/ the closest Hermitian matrix
        if False in [is_pos_semidef(Gamma) for Gamma in Gamma_list0]:            
            pos_semidef_text = " - imposed pos. semi-definiteness"                      
            Gamma_list0 = [closest_pos_semidef(Gamma) if is_pos_semidef(Gamma) == False else Gamma for Gamma in Gamma_list0]                # Taking the closest positive semidefinite matrix of the previous matrix if necessary                                                        
        Gamma_list0 = [LA.sqrtm(Gamma) for Gamma in Gamma_list0] if p_type == "sqrt" else Gamma_list0                   # Square root of the residues
        p0_new = SOP_to_params(Gamma_list0,sigma_list0)                                                                 # Parameters from the residues and the poles
    if complex_poles == False:
        Gamma_list0, sigma_list0 = params_to_SOP(p0_new,M)
        sigma_list0 = sigma_list0.real if np.allclose(np.array(sigma_list0),np.array(sigma_list0).real) == False else sigma_list0
        p0_new = SOP_to_params(Gamma_list0,sigma_list0)                                                           
    else:
        p0_new = time_ordered_params(p0_new,dim,mu=mu)                                                                  # Forcing time ordering of the poles in the parameters
    SOP_new = SOP.from_params(p0_new,M,p_type=p_type)                                                                   # SOP representation of the parameters
    SOP_new.sort()
    Gamma_list_new, sigma_list_new = SOP_new.Gamma_list, SOP_new.sigma_list                                             # Sorting the residues and the poles in the parameters
    p0_new = SOP_to_params(Gamma_list_new,sigma_list_new)                                                               # Parameters from the residues and the poles
    print(p0_text,", p_type = ",p_type,herm_text,pos_semidef_text)
    print("\t\tInitial poles: ",*sigma_list_new)
    print("\t\tPoles weights: ",*[np.abs(np.trace(Gamma).real) for Gamma in Gamma_list_new],"\n")
    return p0_new

def std_to_sqrt_params(p,M):
    """ This function transforms the parameters of the SOP from standard to square root representation. """
    Gamma_list, sigma_list = params_to_SOP(p,M)
    if False in [is_pos_semidef(Gamma) for Gamma in Gamma_list]:
        print("\t\tWARNING: The residues are not positive semidefinite!")
        Gamma_list = [closest_pos_semidef(Gamma) for Gamma in Gamma_list]                              # Taking the closest positive semidefinite matrix to the original Hermitian residue to guarantee the positive semidefiniteness of the square root                 
    Gammasqrt_list = [LA.sqrtm(Gamma) for Gamma in Gamma_list]
    p_new = SOP_to_params(Gammasqrt_list,sigma_list)
    return p_new

def grad_to_params(grad):
    """" This function extracts the parameters (residues and poles) from the gradient chi2.chi2grad of Tommaso's routines.
    grad : Gradient of the cost function
    """
    p = []
    for el in grad:
        p.append(el.real)
        p.append(el.imag)
    return np.array(p)

def conj_params(p):
    """ This function returns the conjugate of the parameters p."""
    p_conj = p
    for i in range(len(p)):
        if i % 2 == 1:
            p_conj[i] = -p_conj[i]
    return p_conj

def real_params_to_complex(p):
    """ This function turns the list of real parameters in a list of complex ones.
    """
    p_complex = []
    for i in range(len(p)):
        if i % 2 == 0:
            p_complex.append(complex(p[i],p[i+1]))
    return np.array(p_complex)

def complex_params_to_real(p):
    """ This function turns the list of complex parameters in a list of real ones.
    """
    p_real = []
    for i in range(len(p)):
        p_real.append(p[i].real)
        p_real.append(p[i].imag)
    return np.array(p_real)