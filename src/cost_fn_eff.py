import numpy         as np
import scipy.linalg  as LA

def cost_func_scalar(v_emb_list,w_list,Gamma_list,sigma_list,p_type="sqrt"):
    num_poles  = len(sigma_list)
    res_list = [Gamma**2 for Gamma in Gamma_list] if p_type == "sqrt" else Gamma_list
    cost = sum(np.abs(v_emb_list[iw] - sum(res_list[k] / (w - sigma_list[k]) for k in range(num_poles)))**2 for iw,w in enumerate(w_list))
    return cost

def grad_cost_func_scalar_sqrt(v_emb_list,w_list,Gamma_list,sigma_list,bounds={"odd_spectrum": False}):
    odd_spectrum = bounds["odd_spectrum"] if "odd_spectrum" in bounds else False
    num_poles = len(sigma_list)

    # Auxiliary functions used to compute the derivative of the cost function
    f = lambda w, ak, Mk: Mk**2 * (w.real - ak)
    g = lambda w, ak: (w.real - ak)**2 + w.imag**2
    h = lambda w, Mk: - Mk**2 * w.imag
    df_ak = lambda Mk: - Mk**2
    dg_ak = lambda w, ak: - 2 * (w.real - ak)
    E_list = [v_emb_list[iw].real - sum(f(w,sigma_list[k].real,Gamma_list[k].real) / g(w,sigma_list[k].real) for k in range(num_poles)) for iw,w in enumerate(w_list)]
    F_list = [v_emb_list[iw].imag - sum(h(w,Gamma_list[k].real) / g(w,sigma_list[k].real) for k in range(num_poles)) for iw,w in enumerate(w_list)]

    grad_res_simpl_list = []
    for k in range(num_poles):
        ak, Mk = sigma_list[k].real, Gamma_list[k].real
        grad_res = sum(2 * E_list[iw]  * (- 2 * Mk * (w.real - ak) / g(w,ak)) + 2 * F_list[iw] * (2 * Mk * w.imag / g(w,ak)) for iw,w in enumerate(w_list))
        grad_res_simpl_list.append(grad_res)
        if odd_spectrum == True and k + 1 == int(num_poles / 2):
            grad_res_simpl_list = grad_res_simpl_list + grad_res_simpl_list[::-1]
            break 
    
    grad_pol_simpl_list = []
    for k in range(num_poles):
        ak, Mk = sigma_list[k].real, Gamma_list[k].real
        grad_pol = sum(- 2 * E_list[iw] * (df_ak(Mk) * g(w,ak) - f(w,ak,Mk) * dg_ak(w,ak)) / g(w,ak)**2 - 2 * F_list[iw] * (- h(w,Mk) * dg_ak(w,ak)) / g(w,ak)**2 for iw,w in enumerate(w_list))
        grad_pol_simpl_list.append(grad_pol)
        if odd_spectrum == True and k + 1 == int(num_poles / 2):
            grad_pol_simpl_list = grad_pol_simpl_list + [-grad_pol for grad_pol in grad_pol_simpl_list[::-1]]
            break

    grad_simpl_list = np.array(grad_res_simpl_list + grad_pol_simpl_list)
    return grad_simpl_list

def convert_scalar_to_matrix_params(scalar_list,num_poles):
    if len(scalar_list) != 2 * num_poles:
        raise ValueError(f"Expected {2 * num_poles} scalar parameters, got {len(scalar_list)}")
    scalar_res_list, scalar_pol_list = scalar_list[:num_poles], scalar_list[num_poles:]
    basis_mat_res, basis_mat_pol = [1., 0., 0., 0., 0., 0., 1., 0.], [1., 0.]                             # Assuming the residues are hermitian and the poles are real, we can use these basis vectors for the gradient of matrices
    mat_res_list, mat_pol_list = [], []
    for k in range(num_poles):
        mat_res = [scalar_res_list[k] * el for el in basis_mat_res]
        mat_pol = [scalar_pol_list[k] * el for el in basis_mat_pol]
        mat_res_list += mat_res
        mat_pol_list += mat_pol
    mat_list = np.array(mat_res_list + mat_pol_list)
    return mat_list