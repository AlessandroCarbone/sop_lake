import numpy         as np
import scipy.linalg  as LA
from .dyn_poles_utils import find_relevant_peaks
from .SOP             import SOP, antisymm_SOP, params_to_SOP, SOP_to_params
import time

def weights_cost_func(func_list,w_list,min_height=0.5,cond="prominence",peak_dist=0.1):
    """ This function returns a list of weights for the cost function in order to improve fitting around specific frequencies.
    func_list : Function list used to find the frequencies where the weights are needed, i.e. its peaks
    """
    wpeaks_list, peak_prom_list = find_relevant_peaks(func_list,w_list,min_height=min_height,cond=cond)     
    weight_list = []
    for iw,w in enumerate(w_list):
        for iwpeak,wpeak in enumerate(wpeaks_list):
            if np.abs(w - wpeak) < peak_dist:
                weight_list.append(peak_prom_list[iwpeak] * 20)                         # Adding weight proportional to the prominence of the closest peak! 
                break
        if len(weight_list) != iw + 1:
            weight_list.append(1.)
    # weight_list = [100. if -zero_tol < w < zero_tol else 1. for w in w_list]          # Weights around the Fermi level
    # weight_list = [10 * gaussian(w,mu=0.,sigma=0.5) for w in w_list]                  # Weights around the Fermi level with a gaussian distribution
    return weight_list

def cost_func_vemb(v_emb_list,w_list,SOP,weight_list=None,func_type="chi2"):
    """ Cost function to minimize with flattened parameters in SciPy function.
    weight_list : List of weights for the cost function in order to improve fitting around specific frequencies
    method      : Method to compute the cost function, either "chi2" or "imag_chi2" (takes only the imaginary part of the chi2 cost function)                                                                                              
    """
    M, ntot = SOP.num_poles, len(v_emb_list[0])                                   # Number of poles/residues + dimension of the matrices
    weight_list = np.ones(len(w_list)) if weight_list == None else weight_list      # Weights for the cost function
    if func_type == "chi2":
        cost = sum(weight_list[iw] * LA.norm(v_emb_list[iw] - SOP.evaluate([w])[0])**2 for iw,w in enumerate(w_list))
        return cost
    elif func_type == "imag_chi2":
        if SOP.p_type == "std":
            Gamma_list, sigma_list = SOP.Gamma_list, SOP.sigma_list
            A_list, B_list = [Gamma.real for Gamma in Gamma_list], [Gamma.imag for Gamma in Gamma_list]
            a_list, b_list = [sigma.real for sigma in sigma_list], [sigma.imag for sigma in sigma_list]
            # Auxiliary functions to compute cost function
            g = lambda w, ak, bk: (w.real - ak)**2 + (bk - w.imag)**2
            h = lambda w, ak, bk, Akij, Bkij: Akij * (bk - w.imag) + Bkij * (w.real - ak)
            F_list = [np.zeros((ntot,ntot),dtype=np.float64) for w in w_list]
            for iw,w in enumerate(w_list):
                for i in range(ntot):
                    for j in range(ntot):
                        F_list[iw][i,j] = v_emb_list[iw][i,j].imag - sum(h(w,a_list[k],b_list[k],A_list[k][i,j],B_list[k][i,j]) / g(w,a_list[k],b_list[k]) for k in range(M))
            cost = sum(weight_list[iw] * F_list[iw][i,j]**2 for iw in range(len(w_list)) for i in range(ntot) for j in range(ntot))
            return cost
        else:
            raise ValueError('Error - func_type = "imag_chi2" requires p_type = "std"')

def grad_cost_func_vemb(v_emb_list,w_list,SOP,bounds={"complex_poles": False, "fixed_residues": False},weight_list=None,func_type="chi2"):         
    """" This function computes the analytical gradient of the cost function for the fitting of the embedding potential v_emb(w) with a SOP representation.
    v_emb_list  : List of values of the embedding potential v_emb(w)
    w_list      : List of frequencies on the test axis on which the simulation is performed
    Gamma_list  : List of matrices of residues
    sigma_list  : List of poles
    weight_list : List of weights for the cost function in order to improve fitting around specific frequencies 
    bounds      : Dictionary containing the bounds for the parameters
    """
    if SOP.p_type != "std":
        raise ValueError('Error - SOP.p_type must be equal to "std"')
    complex_poles  = bounds["complex_poles"] if "complex_poles" in bounds.keys() else False     # Boolean to specify if the poles are complex or not
    fixed_residues = bounds["fixed_residues"] if "fixed_residues" in bounds.keys() else False   # Boolean to specify if the residues are fixed during the optimization or not
    fixed_poles    = bounds["fixed_poles"] if "fixed_poles" in bounds.keys() else False         # Boolean to specify if the poles are fixed during the optimization or not
    odd_spectrum   = bounds["odd_spectrum"] if "odd_spectrum" in bounds.keys() else False       # Boolean to specify if the spectrum is odd, i.e. with particle-hole symmetry, or not
    herm_residues  = bounds["herm_residues"] if "herm_residues" in bounds.keys() else False     # Boolean to specify to calculate the residues as hermitian, i.e. using only the upper triangular matrix of the residues

    t0   = time.time()
    M    = SOP.num_poles                                   # Number of poles/residues
    ntot = len(v_emb_list[0])                             # Dimension of the matrices
    weight_list    = np.ones(len(w_list)) if weight_list == None else weight_list

    # Auxiliary functions used to compute the derivative of the cost function
    f = lambda w, ak, bk, Akij, Bkij: Akij * (w.real - ak) - Bkij * (bk - w.imag)
    g = lambda w, ak, bk: (w.real - ak)**2 + (bk - w.imag)**2
    h = lambda w, ak, bk, Akij, Bkij: Akij * (bk - w.imag) + Bkij * (w.real - ak)

    A_list, B_list = [Gamma.real for Gamma in SOP.Gamma_list], [Gamma.imag for Gamma in SOP.Gamma_list]
    a_list = [sigma.real for sigma in SOP.sigma_list]
    b_list = [sigma.imag for sigma in SOP.sigma_list] if complex_poles == True else np.zeros(M)
    E_list, F_list = [np.zeros((ntot,ntot),dtype=np.float64) for w in w_list], [np.zeros((ntot,ntot),dtype=np.float64) for w in w_list]
    for iw,w in enumerate(w_list):
        for i in range(ntot):
            for j in range(ntot):
                if func_type == "chi2":
                    E_list[iw][i,j] = v_emb_list[iw][i,j].real - sum(f(w,a_list[k],b_list[k],A_list[k][i,j],B_list[k][i,j]) / g(w,a_list[k],b_list[k]) for k in range(M)) 
                F_list[iw][i,j] = v_emb_list[iw][i,j].imag - sum(h(w,a_list[k],b_list[k],A_list[k][i,j],B_list[k][i,j]) / g(w,a_list[k],b_list[k]) for k in range(M))
    t1 = time.time()

    if fixed_residues == False:
        grad_res_list = []
        for k in range(M):
            ak, bk = a_list[k], b_list[k]
            for i in range(ntot):
                for j in range(ntot):
                    if j >= i or herm_residues == False:
                        grad_res_real = sum(weight_list[iw] * (-2 * E_list[iw][i,j] * (w.real - ak)  / g(w,ak,bk) - 2 * F_list[iw][i,j] * (bk - w.imag) / g(w,ak,bk)) for iw,w in enumerate(w_list))
                        grad_res_imag = sum(weight_list[iw] * (-2 * E_list[iw][i,j] * (-bk + w.imag) / g(w,ak,bk) - 2 * F_list[iw][i,j] * (w.real - ak) / g(w,ak,bk)) for iw,w in enumerate(w_list))
                        grad_res_list.append(grad_res_real)
                        if j == i and herm_residues == True:
                            grad_res_list.append(0.)
                        else:
                            grad_res_list.append(grad_res_imag)
                    else:
                        grad_res_list.append(grad_res_list[2 * k * ntot**2 + 2 * (j * ntot + i)])          
                        grad_res_list.append(-grad_res_list[2 * k * ntot**2 + 2 * (j * ntot + i) + 1])
    else:
        grad_res_list = [0.]*(2 * M * (ntot**2))
    t2 = time.time()

    grad_poles_list = [0.]*(2*M)
    if fixed_poles == False:
        for k in range(M):
            ak, bk = a_list[k], b_list[k]
            for iw,w in enumerate(w_list):
                gwk = g(w,ak,bk)
                for i in range(ntot):
                    for j in range(ntot):
                        Akij, Bkij = A_list[k][i,j], B_list[k][i,j]
                        dE_a = - (- Akij * gwk + 2 * (w.real - ak) * f(w,ak,bk,Akij,Bkij)) / gwk**2
                        dF_a = - (- Bkij * gwk + 2 * (w.real - ak) * h(w,ak,bk,Akij,Bkij)) / gwk**2
                        grad_poles_list[2*k] += weight_list[iw] * (2 * E_list[iw][i,j] * dE_a + 2 * F_list[iw][i,j] * dF_a)
                        if complex_poles == True:
                            dE_b = - (- Bkij * gwk - 2 * (bk - w.imag) * f(w,ak,bk,Akij,Bkij)) / gwk**2 
                            dF_b = - (  Akij * gwk - 2 * (bk - w.imag) * h(w,ak,bk,Akij,Bkij)) / gwk**2
                            grad_poles_list[2*k + 1] += weight_list[iw] * (2 * E_list[iw][i,j] * dE_b + 2 * F_list[iw][i,j] * dF_b)
                        else:
                            grad_poles_list[2*k + 1] == 0
    grad_list = np.array(grad_res_list + grad_poles_list)
    if odd_spectrum == True:                                                                        # If the spectrum is odd, we need to symmetrize the residues and poles in order to conserve the symmetries properties
        grad_res_list, grad_poles_list = grad_res_list[:M * ntot**2], grad_poles_list[:M]
        grad_list = np.array(grad_res_list + grad_poles_list)
        grad_list = np.array(SOP_to_params(*antisymm_SOP(*params_to_SOP(grad_list,int(M / 2)))))
    t = time.time()
    # print("     Times to evaluate the gradient: ",t1 - t0,t2 - t1, t -t2," - Total time: ",t - t0)
    return grad_list

def grad_cost_func_vemb_sqrt(v_emb_list,w_list,SOP,bounds={"complex_poles": False, "fixed_residues": False},weight_list=None,func_type="chi2"):
    if SOP.p_type != "sqrt":
        raise ValueError('Error - SOP.p_type must be equal to "sqrt"')
    complex_poles  = bounds["complex_poles"] if "complex_poles" in bounds.keys() else False     # Boolean to specify if the poles are complex or not
    fixed_residues = bounds["fixed_residues"] if "fixed_residues" in bounds.keys() else False   # Boolean to specify if the residues are fixed during the optimization or not
    fixed_poles    = bounds["fixed_poles"] if "fixed_poles" in bounds.keys() else False         # Boolean to specify if the poles are fixed during the optimization or not
    odd_spectrum   = bounds["odd_spectrum"] if "odd_spectrum" in bounds.keys() else False       # Boolean to specify if the spectrum is odd, i.e. with particle-hole symmetry, or not
    herm_residues  = bounds["herm_residues"] if "herm_residues" in bounds.keys() else False     # Boolean to specify to calculate the residues as hermitian, i.e. using only the upper triangular matrix of the residues

    num_poles = SOP.M                                          # Number of poles/residues
    ntot      = len(v_emb_list[0])                             # Dimension of the matrices
    weight_list    = np.ones(len(w_list)) if weight_list == None else weight_list

    # Auxiliary functions used to compute the derivative of the cost function
    f = lambda w, ak, bk, Akij, Bkij: Akij * (w.real - ak) - Bkij * (bk - w.imag)
    g = lambda w, ak, bk: (w.real - ak)**2 + (bk - w.imag)**2
    h = lambda w, ak, bk, Akij, Bkij: Akij * (bk - w.imag) + Bkij * (w.real - ak)

    M_list, N_list = [Gamma_sqrt.real for Gamma_sqrt in SOP.Gamma_list], [Gamma_sqrt.imag for Gamma_sqrt in SOP.Gamma_list]
    A_list, B_list = [M_list[k] @ M_list[k] - N_list[k] @ N_list[k] for k in range(num_poles)], [M_list[k] @ N_list[k] + N_list[k] @ M_list[k] for k in range(num_poles)]
    a_list = [sigma.real for sigma in SOP.sigma_list]
    b_list = [sigma.imag for sigma in SOP.sigma_list] if complex_poles == True else np.zeros(num_poles)
    E_list, F_list = [np.zeros((ntot,ntot),dtype=np.float64) for w in w_list], [np.zeros((ntot,ntot),dtype=np.float64) for w in w_list]
    for iw,w in enumerate(w_list):
        for i in range(ntot):
            for j in range(ntot):
                if func_type == "chi2":
                    E_list[iw][i,j] = v_emb_list[iw][i,j].real - sum(f(w,a_list[k],b_list[k],A_list[k][i,j],B_list[k][i,j]) / g(w,a_list[k],b_list[k]) for k in range(num_poles)) 
                F_list[iw][i,j] = v_emb_list[iw][i,j].imag - sum(h(w,a_list[k],b_list[k],A_list[k][i,j],B_list[k][i,j]) / g(w,a_list[k],b_list[k]) for k in range(num_poles))
    
    if fixed_residues == False:
        grad_res_sqrt_list = []
        for k in range(num_poles):
            ak, bk = a_list[k], b_list[k]
            dAlm_Mij = lambda l, m, i, j: int(l == i) * M_list[k][j,m] + int(m == j) * M_list[k][l,i]
            dBlm_Mij = lambda l, m, i ,j: int(l == i) * N_list[k][j,m] + int(m == j) * N_list[k][l,i]
            dAlm_Nij = lambda l, m, i, j: -dBlm_Mij(l,m,i,j)
            dBlm_Nij = lambda l, m, i, j: dAlm_Mij(l,m,i,j)
            df_M = lambda w, l, m, i, j: (w.real - ak) * dAlm_Mij(l,m,i,j) - (bk - w.imag) * dBlm_Mij(l,m,i,j)
            dh_M = lambda w, l, m, i, j: (bk - w.imag) * dAlm_Mij(l,m,i,j) + (w.real - ak) * dBlm_Mij(l,m,i,j)
            df_N = lambda w, l, m, i, j: (w.real - ak) * dAlm_Nij(l,m,i,j) - (bk - w.imag) * dBlm_Nij(l,m,i,j)
            dh_N = lambda w, l, m, i, j: (bk - w.imag) * dAlm_Nij(l,m,i,j) + (w.real - ak) * dBlm_Nij(l,m,i,j)
            for i in range(ntot):
                for j in range(ntot):
                    if j >= i or herm_residues == False:
                        grad_res_sqrt_real = sum(weight_list[iw] * (-2 * E_list[iw][i,j] * df_M(w,l,m,i,j) / g(w,ak,bk) - 2 * F_list[iw][i,j] * dh_M(w,l,m,i,j) / g(w,ak,bk)) for iw,w in enumerate(w_list) for l in range(ntot) for m in range(ntot))
                        grad_res_sqrt_imag = sum(weight_list[iw] * (-2 * E_list[iw][i,j] * df_N(w,l,m,i,j) / g(w,ak,bk) - 2 * F_list[iw][i,j] * dh_N(w,l,m,i,j) / g(w,ak,bk)) for iw,w in enumerate(w_list) for l in range(ntot) for m in range(ntot))
                        grad_res_sqrt_list.append(grad_res_sqrt_real)
                        if j == i and herm_residues == True:
                            grad_res_sqrt_list.append(0.)
                        else:
                            grad_res_sqrt_list.append(grad_res_sqrt_imag)
                    else:
                        grad_res_sqrt_list.append(grad_res_sqrt_list[2 * k * ntot**2 + 2 * (j * ntot + i)])
                        grad_res_sqrt_list.append(-grad_res_sqrt_list[2 * k * ntot**2 + 2 * (j * ntot + i) + 1])
    else:
        grad_res_sqrt_list = [0.]*(2 * num_poles * (ntot**2))

    grad_poles_list = [0.]*(2*num_poles)
    if fixed_poles == False:
        for k in range(num_poles):
            ak, bk = a_list[k], b_list[k]
            for iw,w in enumerate(w_list):
                gwk = g(w,ak,bk)
                for i in range(ntot):
                    for j in range(ntot):
                        Akij, Bkij = A_list[k][i,j], B_list[k][i,j]
                        dE_a = - (- Akij * gwk + 2 * (w.real - ak) * f(w,ak,bk,Akij,Bkij)) / gwk**2
                        dF_a = - (- Bkij * gwk + 2 * (w.real - ak) * h(w,ak,bk,Akij,Bkij)) / gwk**2
                        grad_poles_list[2*k] += weight_list[iw] * (2 * E_list[iw][i,j] * dE_a + 2 * F_list[iw][i,j] * dF_a)
                        if complex_poles == True:
                            dE_b = - (- Bkij * gwk - 2 * (bk - w.imag) * f(w,ak,bk,Akij,Bkij)) / gwk**2 
                            dF_b = - (  Akij * gwk - 2 * (bk - w.imag) * h(w,ak,bk,Akij,Bkij)) / gwk**2
                            grad_poles_list[2*k + 1] += weight_list[iw] * (2 * E_list[iw][i,j] * dE_b + 2 * F_list[iw][i,j] * dF_b)
                        else:
                            grad_poles_list[2*k + 1] == 0.
    grad_list = np.array(grad_res_sqrt_list + grad_poles_list)

    if odd_spectrum == True:                                                                        # If the spectrum is odd, we need to symmetrize the residues and poles in order to conserve the symmetries properties
        grad_res_sqrt_list, grad_poles_list = grad_res_sqrt_list[:num_poles * ntot**2], grad_poles_list[:num_poles]
        grad_list = np.array(grad_res_sqrt_list + grad_poles_list)
        grad_list = np.array(SOP_to_params(*antisymm_SOP(*params_to_SOP(grad_list,int(num_poles / 2)))))
    return grad_list