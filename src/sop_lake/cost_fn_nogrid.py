import numpy as np

def cost_func_scalar_nogrid(Gammav_list,sigmav_list,Gamma_list,sigma_list,p_type="sqrt",eta=0.1):
    """ This function computes the cost function for scalar residues and poles without using a frequency grid.
    It uses a Lorentzian broadening with width eta to approximate the delta functions.

    eta: broadening parameter necessary to guarantee the convergence of the cost function
    """
    Gamma_list2 = [Gamma**2 for Gamma in Gamma_list] if p_type == "sqrt" else Gamma_list
    res_list = Gammav_list + [-Gamma for Gamma in Gamma_list2]
    pol_list = sigmav_list + sigma_list
    num_poles  = len(pol_list)
    cost = complex(0,2 * np.pi) * sum((res_list[i] * res_list[j]) / (pol_list[j] - pol_list[i] + complex(0,2 * eta)) for j in range(num_poles) for i in range(num_poles))
    return cost.real

def grad_cost_func_scalar_nogrid(Gammav_list,sigmav_list,Gamma_list,sigma_list,bounds={"odd_spectrum": False},p_type="sqrt",eta=0.1):       # To correct!
    """ This function computes the gradient of the cost function for scalar residues and poles without using a frequency grid.
    
    p_type : "std" or "sqrt" indicates the parametrization used for the residues/poles of the discretized embedding potential
    eta    : broadening parameter necessary to guarantee the convergence of the cost function
    """
    odd_spectrum = bounds["odd_spectrum"] if "odd_spectrum" in bounds else False
    num_poles = len(sigma_list)

    grad_res_list, grad_pol_list = [], []
    for k in range(num_poles):
        sigma, Gamma = sigma_list[k].real, Gamma_list[k].real
        if p_type == "sqrt":
            raise NotImplementedError("Gradient computation for 'sqrt' parametrization is not implemented yet.")
        elif p_type == "std":
            grad_res  = - 8 * np.pi * eta * sum(Gammav_list[l].real / ((sigmav_list[l].real - sigma)**2 + 4 * eta**2) for l in range(len(sigmav_list)))
            grad_res += + 8 * np.pi * eta * sum(Gamma_list[l].real / ((sigma_list[l].real - sigma)**2 + 4 * eta**2) for l in range(len(sigma_list)))
            grad_pol  = - 16 * np.pi * eta * sum((Gammav_list[l].real * Gamma * (sigmav_list[l].real - sigma)) / ((sigmav_list[l].real - sigma)**2 + 4 * eta**2)**2 for l in range(len(sigmav_list)))
            grad_pol += + 16 * np.pi * eta * sum((Gamma_list[l].real * Gamma * (sigma_list[l].real - sigma)) / (((sigma_list[l].real - sigma)**2 - 4 * eta**2)**2 + 16 * eta**2 * (sigma_list[l].real - sigma)**2) for l in range(len(sigma_list)))
        grad_res_list.append(grad_res)
        grad_pol_list.append(grad_pol)
        if odd_spectrum == True and k + 1 == int(num_poles / 2):
            grad_res_list = grad_res_list + grad_res_list[::-1]
            grad_pol_list = grad_pol_list + [-grad for grad in grad_pol_list[::-1]]
            break
    grad_list = np.array(grad_res_list + grad_pol_list)
    return grad_list