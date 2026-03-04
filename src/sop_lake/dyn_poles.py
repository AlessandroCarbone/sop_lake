import numpy                as np
import scipy.linalg         as LA
import matplotlib.pyplot    as plt
import scipy
from scipy.interpolate      import interp1d
from scipy.optimize         import minimize_scalar, minimize

from .SOP                    import SOP, antisymm_SOP, adapt_residues, SOP_to_params, params_to_SOP
from .utils                  import RMSE, check_selfadjoint, closest_hermitian
from .dyn_poles_utils        import set_initial_params
from .cost_fn                import cost_func_vemb, grad_cost_func_vemb, grad_cost_func_vemb_sqrt
from .cost_fn_eff            import cost_func_scalar, grad_cost_func_scalar_sqrt, convert_scalar_to_matrix_params

def compute_cost_function(vemb_list,w_list,SOP,weight_list=None,func_type="chi2",paramagnetic=False):
    """ This function returns the cost function of the embedding potential.
    method : Method to compute the cost function, "alessandro" or "tommaso" - N.B. They return the same results, but belong to 2 different codes
    p_type : Type of parameters - "std" if standard residues and poles, "sqrt" if sqrt of the residues and poles
    """
    if paramagnetic == False:
        cost = cost_func_vemb(vemb_list,w_list,SOP,weight_list=weight_list,func_type=func_type)
    else:
        v_emb00_list = np.array(vemb_list)[:,0,0]
        Gamma00_list = np.array(SOP.Gamma_list)[:,0,0]
        if np.allclose(np.array(Gamma00_list),np.array(Gamma00_list).real) == False:
            raise ValueError("Gamma00_list should contain real numbers only.")
        if np.allclose(np.array(SOP.sigma_list),np.array(SOP.sigma_list).real) == False:
            raise ValueError("sigma_list should contain real numbers only.")
        cost = cost_func_scalar(v_emb00_list,w_list,Gamma00_list,SOP.sigma_list,p_type=SOP.p_type)
    return cost

def compute_grad_cost_function(vemb_list,w_test_list,SOP,bounds={"complex_poles": False, "fixed_residues": False, "paramagnetic": False},weight_list=None,func_type="chi2"):
    """ This function returns the gradient of the cost function of the embedding potential.
    method : Method to compute the cost function, "alessandro" or "tommaso" - N.B. They belong to 2 different codes and they return different results
    p_type : Type of parameters - "std" if standard residues and poles, "sqrt" if sqrt of the residues and poles
    """
    paramagnetic = bounds["paramagnetic"] if "paramagnetic" in bounds.keys() else False
    if SOP.p_type == "std":
        grad = grad_cost_func_vemb(vemb_list,w_test_list,SOP,bounds=bounds,weight_list=weight_list,func_type=func_type)
    elif SOP.p_type == "sqrt":
        if paramagnetic == False:
            grad = grad_cost_func_vemb_sqrt(vemb_list,w_test_list,SOP,bounds=bounds,weight_list=weight_list,func_type=func_type)
        else:
            v_emb00_list = np.array(vemb_list)[:,0,0]
            Gamma00_list = np.array(SOP.Gamma_list)[:,0,0]
            if np.allclose(np.array(Gamma00_list),np.array(Gamma00_list).real) == False:
                raise ValueError("Gamma00_list should contain real numbers only.")
            if np.allclose(np.array(SOP.sigma_list),np.array(SOP.sigma_list).real) == False:
                raise ValueError("sigma_list should contain real numbers only.")
            grad_scalar = grad_cost_func_scalar_sqrt(v_emb00_list,w_test_list,Gamma00_list,SOP.sigma_list,bounds=bounds)
            grad        = convert_scalar_to_matrix_params(grad_scalar,SOP.num_poles)
    return grad

def parabola_x_list(x0,fact=0.9,method="proximity",initial_bracket=[None, None], num_inc=0, borders=[1e-12,1.]):
    """ This function returns a list of x values to perform a parabolic interpolation close to x0.
    x0            : Value around which the parabola x-list is centered
    fact          : Factor to multiply x0 to get the x values of the list in the method "proximity"
    method        : Method to generate the x-list
    initial_bracket : Initial bracket for the parabolic interpolation
    num_bins      : Number of bins to enlarge the bracket until the borders
    num_inc       : Number of increments of the bracket
    borders       : Borders of the interval where the parabola can be defined
    """
    if method == "proximity":       
        x_list = np.array([x0 - fact * x0, x0, x0 + fact * x0]) 
        # x_list = np.array([x0 - fact * 10**order_of_magnitude(x0), x0, x0 + fact * 10**order_of_magnitude(x0)])    
    elif method == "increasing":
        # oom      = order_of_magnitude(x0)
        num_bins = 10
        if initial_bracket[0] != None and initial_bracket[1] != None and num_inc < num_bins:                           
            x0_left, x0_right = initial_bracket[0], initial_bracket[1]
            d_left, d_right   = np.abs(x0_left - borders[0]), np.abs(x0_right - borders[1])          # Distance of the initial bracket from the borders
            x_left, x_right = x0_left  - (num_inc / num_bins) * d_left, x0_right + (num_inc / num_bins) * d_left
            x_list  = np.array([x_left, x0, x_right])   
        else:
            print("     ERROR - Initial bracket not defined")        
    return x_list

def anal_parabola_vertex(a,b,c,fa,fb,fc,thr=1e-20):
    """ This function returns the vertex of the parabola defined by the 3 points (a,fa), (b,fb), (c,fc).
    """
    r = (b - a) * (fb - fc)
    q = (b - c) * (fb - fa)
    u = b - ((b - c) * q - (b - a) * r) / (2.0 * np.maximum(np.abs(q - r), thr) * np.sign(q - r))
    return u

def bracket_parabola_vertex(a0,b0,f,p,h):
    """  This routine searches in the downhill direction (defined by the function as evaluated at the initial points) and returns 3 new points that bracket a minimum of the function.
    See Chapter 10.1 of "Press, Teukolsky, et al., Numerical Recipes" for more details.
    a0, b0 : Initial points to define the bracket
    f    : Function to minimize
    p    : Parameters of the function
    h    : Direction of the downhill
    """
    GOLD, GLIMIT, TINY = 1.618034, 100.0, 1.0e-20   # Here GOLD is the default ratio by which successive intervals are magnified and GLIMIT is the maximum magnification allowed for a parabolic-fit step
    a, b   = a0, b0
    fa, fb = f(p + a * h), f(p + b * h)             # Values of the function at the parameters moved by a and b in the direction given by h
    if fb > fa:                                     # Switching the points so that we can go downhill in the direction from a to b
        a, b = b, a
        fa, fb = fb, fa
    c = b + GOLD * (b - a)                          # First guess for c
    fc = f(p + c * h)
    while fb > fc:                                  # Keep moving in the downhill direction until the minimum is bracketed
        u = anal_parabola_vertex(a,b,c,fa,fb,fc,thr=TINY)         # u by parabolic extrapolation from a, b, c
        ulim = b + GLIMIT * (c - b)
        if (b - u) * (u - c) > 0.0:                 # If u is between b and c
            fu = f(p + u * h)
            if fu < fc:
                a, fa, b, fb = b, fb, u, fu
            elif fu > fb:
                c, fc = u, fu
            u = c + GOLD * (c - b)
            fu = f(p + u * h)
        elif (c - u) * (u - ulim) > 0.0:
            fu = f(p + u * h)
            if fu < fc:
                b, c, u = c, u, u + GOLD * (u - c)
                fb, fc, fu = fc, fu, f(p + u * h)
        elif (u - ulim) * (ulim - c) >= 0.0:      # Limit parabolic u to maximum u value
            u = ulim
            fu = f(p + u * h)
        else:
            u = c + GOLD * (c - b)
            fu = f(p + u * h)
        a, b, c    = b, c, u
        fa, fb, fc = fb, fc, fu
    
    delta_list, f_list = [a, b, c], [fa, fb, fc]
    ind_list   = [el[0] for el in sorted(enumerate(delta_list),key=lambda x:x[1])]                   # Sorting a,b,c in increasing order
    delta_list, f_list = [delta_list[ind] for ind in ind_list], [f_list[ind] for ind in ind_list]
    return np.array(delta_list), np.array(f_list)

def parabola_ratio(x_list,f_list,x_res,x_guess,borders=[-1.,1.]):
    """ This function returns a parameter to evaluate the accuracy of the estimate of the vertex of a parabola. 
    It shows how good is the parabolic approximation.
    x_list : List of x values
    f_list : List of f values
    x_res  : Estimate of the parabola vertex
    x_guess: Guess for the parabola vertex
    borders: Borders of the interval where the parabola vertex can be defined
    """
    x_best = x_list[np.argmin(f_list)]                                                                # Interpolation of the cost function
    x_ref = x_best if borders[0] <= x_best <= borders[1] else x_guess                                 # Guess for the minimum of the cost function
    x_min = x_res if borders[0] <= x_res <= borders[1] and x_res != 0. else x_ref                     # Extracting the learning rate that minimizes the cost function fit
    parab_ratio = np.abs(x_res / x_min)                                                               # Ratio showing how good is the parabolic approximation
    return parab_ratio, x_min                                                                

def parabola_check_line_search(x_min,cost_func,grad_cost_func,p,h):
    """ Here we calculate a ratio which allows to see how much parabolic is the cost function f we are minimizing. If the function is a perfect parabola, the ratio should be 1.
    Basically at the vertex position x_min, the tangent straight line g(x) and f(0) should be equidistant from f(x_min), i.e. |f(x_min) - g(x_min)| = |f(x_min) - f(0)|. Here f(0)
    is the cost function evaluated at the learning rate delta = 0, i.e. cost(p).
    """
    f0 = cost_func(p)                   
    der_f0 = np.dot(grad_cost_func(p),h)                # f'(0) is the directional derivative of the cost function at delta = 0, i.e. f'(0) = grad_cost_func(p) * h
    fmin = cost_func(p + x_min * h)
    g_min = f0 + der_f0 * x_min                         # Tangent of the parabola in x = 0  evaluated at x = x_min
    diff1 = np.abs(fmin - g_min)
    diff2 = np.abs(fmin - f0)
    if diff2 != 0.:
        ratio = diff1 / diff2
    else:
        ratio = 0.
    return ratio

def parabola_vertex(x_list,f_list,method="anal",x_guess=None,RMSE_thr=0.5):
    """ This function uses a quadratica approximation to find the vertex of a parabola, either via an analytical formula or interpolation.
    method : Method to find the vertex of the parabola, "anal" or "interp"
    """
    params = {}
    if method == "anal":
        x_res = anal_parabola_vertex(*x_list,*f_list)
    elif method == "interp":
        f_fit  = interp1d(x_list,f_list,kind='quadratic', fill_value='extrapolate')                    # Interpolation of the cost function
        RMSE_parab = RMSE(f_list,f_fit(x_list))                                                        # RMSE of the parabolic fit
        res        = minimize_scalar(f_fit)                                                            # Finding the minimum of the cost function from the fit function                    
        x_res = res.x                                                             
        if RMSE_parab > RMSE_thr:
            print("          Interpolation not accurate - RMSE = ",RMSE_parab)
        params["RMSE"], params["f_fit"] = RMSE_parab, f_fit
    parab_ratio, x_min = parabola_ratio(x_list,f_list,x_res,x_guess)                                    # Returns best final approximation of the parabola vertex and ratio showing how good is the parabolic approximation
    params["parab_ratio"] = parab_ratio
    return x_min, params

def find_delta_in_bracket(a,b,p,h,f,num_pts=1000):
    """ This function explores an interval and finds the point that minimizes the function f in the line search.
    """
    def phi(delta):
        return f(p + delta * h)
    delta_interval = np.linspace(a,b,num_pts)                           
    phi_delta_list = [phi(delta) for delta in delta_interval]
    delta_min = delta_interval[np.argmin(phi_delta_list)]
    return delta_min
    
def plot_interpolation(x_list,x_best,p,dir,f,y_list=None,f_fit=None,path_to_file="figures",indices={"n_cycle": None, "n_iter": None}):
    n_cycle = indices["n_cycle"] if indices["n_cycle"] != None else 0
    n_iter  = indices["n_iter"] if indices["n_iter"] != None else 0  
    plt.figure()
    x_list2    = list(x_list) + [x_best]
    x_fit_list = np.linspace(min(x_list2),max(x_list2),500)
    plt.plot(x_fit_list,[f(p + x_fit * dir) for x_fit in x_fit_list],label="Cost func.",color="black")
    y_list = [f(p + x * dir) for x in x_list] if y_list is None else y_list
    plt.scatter(x_list,y_list,label="Data for interp.",color="red")
    if f_fit != None:
        plt.plot(x_fit_list,f_fit(x_fit_list),label="Fit",color="blue")
    plt.scatter(x_best,f(p + x_best * dir),label="Final delta",color="green")
    plt.legend()
    plt.savefig(path_to_file+"/Cost_func_interp_"+str(n_cycle)+"_"+str(n_iter)+".pdf",bbox_inches='tight')
    plt.close()

def line_search_nicola(delta,f,p,h,borders=[1e-8,1],thr=1e-12,interp_method="sampling"):
    """ This function performs the line search using the bracketing method (from "Numerical Recipes" book) and delta selection using either the analytical parabola vertex or the sampling method.
    interp_method : "parabola" (analytical parabola vertex) or "sampling" (sampling the cost function in the bracket)
    """
    params = {}
    delta_list = np.array([0., delta, 2. * delta])
    f_list     = [f(p + el * h) for el in delta_list]
    if np.abs(f_list[1] - f_list[0]) < thr and np.abs(f_list[2] - f_list[1]) < thr:
        delta = delta_list[1]
        interp_case = "no_change"
    else:
        if interp_method == "parabola":
            delta       = anal_parabola_vertex(*delta_list,*f_list)                                         # Analytical quadratic interpolation
            interp_case = "vertex"
        elif interp_method == "sampling":
            delta_grid  = list(np.linspace(borders[0],delta_list[1],50)) + list(np.linspace(delta_list[1],delta_list[2],10))[1:] 
            delta       = min(delta_grid, key=lambda el: f(p + el * h))                                     # Finding the minimum of the cost function from the fit function
            interp_case = "min"
        else:
            raise ValueError("interp_method should be either 'parabola' or 'sampling'")
        if np.abs(delta) == np.inf or delta == np.nan:
            delta = delta_list[1]
            interp_case = "no_change"
    f_delta = f(p + delta * h)                                                                              # Value of the cost function at the interpolated delta 
    in_out_text = "out of the" if delta < borders[0] or delta > borders[1] else "in the"
    print("              First guess: delta = {}, cost fn = {} - ".format(delta,f_delta),in_out_text," borders with interp_case = {}".format(interp_case))
    if f_delta > f_list[0] or delta < borders[0] or delta > borders[1]:
        if f_list[1] < f_delta or f_list[2] < f_delta:
            delta = delta_list[1] if f_list[1] < f_list[2] else delta_list[2]
            interp_case = "list"
        else:
            if delta_list[1] / 10. < borders[0]:
                delta = delta_list[1]
                interp_case = "no_change"
            else:
                delta = delta_list[1] / 10.
                interp_case = "div"
    print("                    delta = {},  norm_p = {}, norm_h = {}, interp_case = {}".format(delta,LA.norm(p),LA.norm(h),interp_case))
    params["delta_list"], params["f_list"] = delta_list, f_list
    params["interp_case"] = interp_case
    return delta, params

def line_search_alessandro(delta,f,p,h,delta_max=10.,delta_thr=1e-8,method="min_bracket",parabola_method="anal"):
    """ This function performs the line search usin the bracketing method (from "Numerical Recipes" book) and delta selection of Alessandro.
    method : Method to find the minimum of the cost function, either "min_bracket" (expensive way which finds the minimum in the bracket) or "vertex" (takes the vertex of the parabola fitting the 3 points given in the bracket)
    """
    params = {}
    # a0, b0 = delta, delta + 2 * delta                                                                                       # Bracket to find the minimum of the cost function according to Nicola Marzari  
    # delta_list = [a0, (a0 + b0) / 2., b0]
    # f_list     = [f(p + delta * h) for delta in delta_list]                                                                 # Function evaluated at each point reached from the step
    a0, b0 = delta - 0.5 * delta, delta + 0.5 * delta                                                                       # Bracket to find the minimum of the cost function
    delta_list, f_list = bracket_parabola_vertex(a0,b0,f,p,h)                                                               # Bracketing the minimum of the cost function
    
    if method == "min_bracket":
        delta = find_delta_in_bracket(min(delta_list),max(delta_list),p,h,f)                                                # Finding a point which minimizes f in an interval close to the vertex of the parabola
    elif method == "vertex":
        x_min_fit = delta_list[np.argmin(f_list)]
        if x_min_fit > delta_max or x_min_fit < -delta_max:                                                                 # If the bracketing doesn't work properly, use the more rudimental function on the surrounding
            delta_list = parabola_x_list(delta)
            if np.abs(delta_list[0] - delta_list[1]) < delta_thr or np.abs(delta_list[1] - delta_list[2]) < delta_thr:
                delta_list = list(set(delta_list))                                                                          # Removing duplicates
                while len(delta_list) != 3:
                    fact = 0.1
                    delta_list += [delta_list[-1] + fact * delta_list[-1]]
                delta_list = np.array(sorted(delta_list))
            f_list = [f(p + el * h) for el in delta_list]                                                                   # Function evaluated at each point reached from the step
        delta, parab_params = parabola_vertex(delta_list,f_list,method=parabola_method,x_guess=delta)                       # Quadratic interpolation
        if np.any([el < f(p + delta * h) for el in f_list]):
            delta_bracket = find_delta_in_bracket(min(delta_list),max(delta_list),p,h,f)                                    # Finding a point which minimizes f in an interval close to the vertex of the parabola
            delta         = delta_bracket
        params.update(parab_params)
    params["delta_list"], params["f_list"] = delta_list, f_list
    return delta, params

def CG_minimization(f,grad_f,p0,maxiter=1000,delta0=None,grad_thr=1e-8,cost_thr=1e-4,method="interp_delta",maxiter_direct_der=50,print_interp=False,path_to_file="figures",n_cycle=None,reset_cond=False,interp_method="sampling"):
    """ Conjugate gradient minimization algorithm in the Hestenes-Stiefel formulation where only the gradient of the cost function to minimize is needed.
    """
    n_reset     = 20                                                                    # Number of iterations after which resetting the delta and the direction
    delta_reset = 1e-5                                                                  # Value of the delta to reset the direction
    delta_min, delta_max = 1e-11, 10.                                                   # Minimum and maximum values of delta to consider the interpolation successful            
    borders   = [delta_min, delta_max]                                                  # Borders of the interval where the parabola vertex can be defined
    parabola_method = "anal"                                                            # Method to find the vertex of the parabola                                 
    cond_stop  = False                                                                  # Necessary condition to stop the minimization when the cost is going down in the last iterations
    cond_stop2 = False                                                                  # Another condition to stop the minimization when the difference between two consecutive cost functions is below the threshold cost_thr                     
    p = p0
    h = -grad_f(p)
    g = -grad_f(p)
    delta = delta0 if delta0 != None else delta_reset

    def cost_diff(cost,cost0):                                                         # Function to calculate the difference of the cost functions in two consecutive iterations - To be used to check the convergence of the CG minimization
        return np.abs(cost - cost0)

    params = {}
    p_list, cost_list, grad_cost_list, delta_best_list, parab_ratio_list, interp_case_list = [], [], [], [], [], []
    print("          Analytic CG line minimization: ",method)
    print("          interp_method = {}".format(interp_method))
    print("          Initial values: cost fn = {}, norm grad cost fn = {}".format(f(p),LA.norm(g)),"\n")
    count_interp_resets = 0
    for n_iter in range(maxiter):
        print("              n_iter = {}".format(n_iter))
        # Checking norm of the direction is not already at the minimum of the convergence
        norm_h = LA.norm(h)
        if norm_h <= grad_thr and cond_stop == True:
            print("          Line minimization in CG stopped at iter. {} - Gradient below the threshold".format(n_iter))
            break

        if reset_cond == True and ((n_iter != 0 and n_iter % n_reset == 0)): # Resetting the delta and the direction after n_reset iterations - Other condition: (n_iter > 5 and cost_list[-1] >= cost_list[-2])
            delta = delta0 if delta0 != None else delta_reset                                                        
            h = -grad_f(p)
            g = -grad_f(p)
            p_new = p + delta * h                                                                                               # SD step to reset the set of parameters
            parab_ratio_list.append(0.)
            delta_best_list.append(delta)
            print("              Resetting delta to {} and direction".format(delta))
        else:                                                                                                                   # Line search part of the CG minimization
            if method == "interp_delta":                                                     
                # delta, ls_params   = line_search_alessandro(delta,f,p,h,delta_max=delta_max,parabola_method=parabola_method)    # Alessandro's interpolation                       
                # delta_list, f_list = ls_params["delta_list"], ls_params["f_list"]
                delta, ls_params = line_search_nicola(delta,f,p,h,borders=borders,interp_method=interp_method)                  # Nicola's interpolation
                delta_list, f_list, interp_case = ls_params["delta_list"], ls_params["f_list"], ls_params["interp_case"]
                interp_case_list.append(interp_case)                                                                            # Appending the interpolation case to the list
                if len(interp_case_list) > 2 and interp_case_list[-1] == "no_change" and interp_case_list[-2] == "no_change":                                 # Resetting delta if there's no change in delta twice in a row                                                                                  # If the interpolation didn't work, reset delta and direction
                    delta = delta_reset
                    h = -grad_f(p)
                    g = -grad_f(p)
                    p_new = p + delta * h
                    count_interp_resets += 1
                    print("              Resetting delta to {} and direction".format(delta))
                parab_ratio = parabola_check_line_search(delta,f,grad_f,p,h)                                                    # Checking the parabolic approximation of the interpolation
                p_new = p + delta * h                                                                                           # SD with the best delta

                # Plotting the interpolation of the cost function
                if print_interp == True:
                    f_fit = None                                                                                                # Or: ls_params["f_fit"] if line_search_alessandro is used
                    plot_interpolation(delta_list,delta,p,h,f,y_list=f_list,f_fit=f_fit,path_to_file=path_to_file,indices={"n_cycle": n_cycle, "n_iter": n_iter})   
            
            elif method == "fixed_delta":                                                   # Keeping fixed the delta is not a line minimization
                norm_h = LA.norm(h)
                if norm_h <= grad_thr:
                    print("               Parameters update w/ fixed delta in CG stopped at iter. {} - Gradient below the threshold".format(n_iter))
                    break
                p_new = p + delta * h                                                      # Line minimization with gradient descent for the best delta
        
        f_p    = f(p_new)
        grad_p = grad_f(p_new)
        g_new     = - grad_p
        norm_grad = LA.norm(g_new)
        print("              cost fn = {}, norm grad. cost fn = {}".format(f_p,norm_grad))
        if method == "interp_delta":
            print("              parab_ratio = {}".format(parab_ratio),"\n")
        # Stopping condition related to the value of the cost function
        if f_p > 1e5 or (n_iter > 0 and f_p > 2 * cost_list[-1]):
            # Removing the last set of parameters, and the corresponding associated values
            if len(p_list) == 0:
                p_list.append(p0)
                delta_best_list.append(0.)
                parab_ratio_list.append(0.)
                cost_list.append(f(p0))
                grad_cost_list.append(grad_f(p0))
            elif len(p_list) > 1:
                p_list.pop()
                delta_best_list.pop()
                parab_ratio_list.pop()
                cost_list.pop()
                grad_cost_list.pop()
            print("               CG minimization stopped at iter. {} - Cost function diverging in iter. {}".format(n_iter - 1,n_iter))
            break
        p_list.append(p_new)
        delta_best_list.append(delta)
        parab_ratio_list.append(parab_ratio)
        cost_list.append(f_p)
        grad_cost_list.append(grad_p)
        
        # Stopping conditions of the CG minimization related to the convergence behaviour
        cond_stop  = all(cost_list[-1] <= el for el in [cost_list[-2],cost_list[-3],cost_list[-4]]) if len(cost_list) > 3 else False
        cond_stop2 = all(cost_diff(cost_list[-i],cost_list[-i-1]) <= cost_thr for i in range(1,4)) if len(cost_list) > 3 else False
        if norm_grad <= grad_thr and cond_stop == True:  
            print("               CG minimization stopped at iter. {} - Gradient below the threshold".format(n_iter))
            break
        if cond_stop2 == True:
            if cond_stop == True:
                print("               CG minimization stopped at iter. {} - Cost function in a local minimum".format(n_iter))
                break
            else:
                print("               CG minimization stopped at iter. {} - Cost function stagnating".format(n_iter))
                break
        gamma = np.dot(g_new - g,g_new) / np.dot(g,g)                                      # Polak-Ribiere update parameter
        # gamma = np.dot(g_new - g,g_new) / np.dot(g_new - g,h)                              # Hestenes-Stiefel update parameter
        gamma = max(gamma,0.)                                                              # Polak-Ribiere + (Nocedal and Wright, Numerical Optimization) - Meant to guarantee the minimization!
        h_new = g_new + gamma * h                                                          # Conjugate gradient
        h, g, p = h_new, g_new, p_new
    
    params["n_fin_iter"]       = len(p_list)
    params["p_list"]           = p_list
    params["cost_list"]        = cost_list
    params["grad_cost_list"]   = grad_cost_list
    params["delta_best_list"]  = delta_best_list if delta_best_list != [] else [delta0]
    params["parab_ratio_list"] = parab_ratio_list
    return p_list[-1], params

def step_SD_dyn_poles(vemb_list,w_list,SOP0,delta=0.1,thr=1e-6,bounds={"complex_poles": False, "fixed_residues": False},weight_list=None,func_type="chi2"):   
    """ This function performs one step of steepest descent to minimize the cost function associated with the embedding potential.
    This function is part of the algorithm for the dynamics of poles.
    """
    M    = SOP0.num_poles
    p0   = SOP0.get_params()
    grad = compute_grad_cost_function(vemb_list,w_list,SOP0,bounds=bounds,weight_list=weight_list,func_type=func_type)
    grad = grad if LA.norm(grad) > thr else np.zeros(len(grad),dtype=np.float64)
    p    = p0 - delta * grad
    Gamma_list, sigma_list = params_to_SOP(p,M)
    SOP1 = SOP(Gamma_list,sigma_list,p_type=SOP0.p_type)
    return SOP1

def constrain_SOP(SOP,n_cycle=None,bounds={"complex_poles":False,"fixed_residues": False, "odd_spectrum": True, "herm_residues": True}):
    """ Adaptation of an input SOP to some physical constraints applying to residues and poles.
    """
    complex_poles = bounds["complex_poles"] if "complex_poles" in bounds.keys() else False                                                                                              
    odd_spectrum  = bounds["odd_spectrum"] if "odd_spectrum" in bounds.keys() else False
    herm_residues = bounds["herm_residues"] if "herm_residues" in bounds.keys() else True                                        
    paramagnetic  = bounds["paramagnetic"] if "paramagnetic" in bounds.keys() else False
    
    M    = SOP.num_poles
    ntot = SOP.dim
    count_zero_weights = sum(1 for Gamma in SOP.Gamma_list if np.abs(np.trace(Gamma).real) < 1e-8)                                  # Counting the number of weights equal to zero associated with the poles                                                                                   
    if herm_residues == True:
        print("          Forcing hermiticity of the residues if necessary")
        SOP.make_residues_hermitian()                                                                                               # Substituting each Gamma w/ the closest Hermitian matrix
        if n_cycle is not None and n_cycle == 0:
            print("          Fixing imaginary part of the diagonal of the residues to zero if necessary")
            SOP.make_residues_diagonal_real()
    if count_zero_weights > 0:
        print("          Substituting null residues with 0.01 * (matrix of ones)")
        perturb_matrix = np.ones((ntot,ntot),dtype=np.complex128)                                                                   # Perturbation matrix to add to the residues with zero weight            
        Gamma_list2 = [Gamma if np.abs(np.trace(Gamma).real) > 1e-8 else 0.01 * perturb_matrix for Gamma in SOP.Gamma_list]            # Removing residues with zero weight and putting multiples of the perturbation matrix
        SOP.Gamma_list = Gamma_list2
    if paramagnetic == True:
        print("          Adapting residues and poles to paramagnetic case")
        SOP.make_poles_real()
        SOP.make_residues_real_diagonal()
    SOP.sort()                                                                                                                     # Sorting the residues and the poles in the parameters
    if odd_spectrum == True:
        if SOP.is_odd() == False:                                                                                                  # Making the spectrum symmetric if the spectrum is an odd function, e.g. half-filling case of the Hubbard model
            print("          Forcing odd symmetry on initial parameters")
            SOP.Gamma_list, SOP.sigma_list = antisymm_SOP(SOP.Gamma_list[:int(M / 2)],SOP.sigma_list[:int(M / 2)])
    return SOP

def dyn_poles(vemb_list,w_test_list,method,num_poles,SOP0,opt_params=[1,0.001],mu=0.,n_cycle=None,bounds={"complex_poles":False,"fixed_residues": False, "odd_spectrum": True, "herm_residues": True},axis="erf",weight_list=None,func_type="chi2",interp_method="sampling",print_interp=False):
    """" This function implements the dynamics of the poles, i.e. moves the poles in order to minimize a cost function which takes the difference between the embedding potential and its
    representation as a SOP. 
    vemb_list    : List of values of the embedding potential v_emb(w)
    w_test_list   : Frequencies grid on the simulation axis
    method        : Method to minimize the cost function (scipy_CG, num_least_squares, custom_CG, anal_SD)
    num_poles     : Number of poles to describe the embedding potential
    SOP0          : Initial set of parameters describing the embedding potential as a SOP 
    opt_params    : List of parameters for the optimization method, e.g. [maxiter, eps] for the scipy_CG method
    mu            : Chemical potential
    p0            : Initial guess for the parameters
    n_cycle       : Number of the DMFT cycle
    axis          : Axis where the poles are moved, "erf" or "imaginary"
    grad_method   : Method to compute the gradient of the cost function and the step of the SD
    func_type     : Type of the cost function, "chi2" or "imag_chi2"
    interp_method : Method to interpolate the cost function, "sampling" or "parabola"
    """
    print("DMFT iteration {}".format(n_cycle))   
    complex_poles = bounds["complex_poles"] if "complex_poles" in bounds.keys() else False                                                                                              
    odd_spectrum  = bounds["odd_spectrum"] if "odd_spectrum" in bounds.keys() else False
    herm_residues = bounds["herm_residues"] if "herm_residues" in bounds.keys() else True                                        
    paramagnetic  = bounds["paramagnetic"] if "paramagnetic" in bounds.keys() else False

    ntot    = SOP0.dim
    p_type  = SOP0.p_type                                                                                                           # Dimension of the matrices
    w_list  = np.array(w_test_list).imag if axis == "imaginary" else np.array(w_test_list).real                                     # List of the frequencies on the real axis
    eta_axis = np.abs(w_test_list[0].imag / scipy.special.erf(w_list[0])) if axis == "erf" and w_list[0] != 0. else 0.
    p0 = None if SOP0.Gamma_list == None or SOP0.sigma_list == None else SOP0.get_params()                                      # Initial set of parameters
    p0_new   = set_initial_params(p0,num_poles,w_list,vemb_list,mu=mu,axis=axis,eta_axis=eta_axis,complex_poles=complex_poles,p_type=p_type)
    if (p0 is not None and len(p0) == len(p0_new) and np.allclose(p0,p0_new)) and (n_cycle is not None and n_cycle > 2):                                       # This is to avoid the case when wrong parameters initialized at the beginning of the DMFT cycle           
        SOP0 = SOP.from_params(p0_new,num_poles,p_type=p_type)
        p0 = p0_new
    else:         
        SOP0 = constrain_SOP(SOP.from_params(p0_new,num_poles,p_type=p_type),n_cycle=n_cycle,bounds=bounds)                                                                                                                  # The updates of residues and poles via CG or other optimizations should preserve the properties which are reinforced here below
        p0 = SOP0.get_params()  
        res_list0 = adapt_residues(SOP0.Gamma_list,p_type,"std")
        print("               Corrected initial poles: ",*SOP0.sigma_list)
        print("               Corrected initial residues: ",*np.ravel(res_list0),"\n")

    # RMSE of the initial set of parameters
    vemb_fit_list0 = SOP0.evaluate(w_test_list)
    RMSE_real_list0 = [RMSE(np.array(vemb_list).real[:,i,i],np.array(vemb_fit_list0).real[:,i,i]) for i in range(ntot)]
    RMSE_imag_list0 = [RMSE(np.array(vemb_list).imag[:,i,i],np.array(vemb_fit_list0).imag[:,i,i]) for i in range(ntot)]
    RMSE_list0      = RMSE_real_list0 + RMSE_imag_list0
    print("          RMSE of the initial set of parameters describing the embedding potential")
    print("               RMSE real on ",axis," axis in each site: ",*RMSE_real_list0)
    print("               RMSE imag on ",axis," axis in each site: ",*RMSE_imag_list0,"\n")

    # Cost function to minimize with flattened parameters and corresponding gradient
    cost_func      = lambda p: compute_cost_function(vemb_list,w_test_list,SOP.from_params(p,num_poles,p_type=p_type),weight_list=weight_list,func_type=func_type,paramagnetic=paramagnetic)        # Cost function from Alessandro's routines
    grad_cost_func = lambda p: compute_grad_cost_function(vemb_list,w_test_list,SOP.from_params(p,num_poles,p_type=p_type),bounds=bounds,weight_list=weight_list,func_type=func_type)               # Gradient of the cost function from Alessandro's routines 
   
    ##########################################
    # Dynamics of poles using SciPy routines #
    ##########################################
    params = {}                                                                                 # Dictionary to store additional parameters of the optimization
    if method == "scipy_CG":                                                                      # Numerical CONJUGATE GRADIENT method
        p_list = []
        def save_CG_params(x):
            p_list.append(x)
        jac = grad_cost_func                                                                    # Jacobian is not needed for the numerical CG - Otherwise: grad_cost_func
        maxiter, eps = opt_params[0], opt_params[1]                                             # Maximum number of iterations and absolute step for the optimization routine
        result = minimize(cost_func, p0, method='CG', jac=jac, options={'maxiter': maxiter, 'eps': eps}, callback=save_CG_params)   # Conjugate gradient minimization
        p      = result.x
        params["p_list"] = p_list
        print("          Optimization parameters of numerical CG (max num. CG steps = {}): eps = {}, num. CG steps = {}".format(maxiter,eps,len(params["p_list"])))
        print("     Final values: cost fn = {}, norm grad. cost fn = {}".format(cost_func(p),LA.norm(grad_cost_func(p))),"\n")

    ###################################################
    # Dynamics of poles using our analytical gradient #
    ###################################################

    elif method =="custom_CG":                                                                          # Analytical CONJUGATE GRADIENT method
        maxiter, delta0   = opt_params                                                                  # Maximum number of iterations and input step factor in the optimization
        if all(el < 0.1 for el in RMSE_list0):
            delta0 = 1e-7                                                                               # If the RMSE of the initial set of parameters is small, use a smaller delta to start the optimization                                
        linmin_method = "interp_delta"                                                                  # Line minimization method and
        reset_cond = False                                                                              # Reset CG every n_reset iterations
        p, params_CG  = CG_minimization(cost_func,grad_cost_func,p0,maxiter=maxiter,delta0=delta0,method=linmin_method,print_interp=print_interp,n_cycle=n_cycle,reset_cond=reset_cond,interp_method=interp_method)   # Conjugate gradient minimization
        delta_best_list = params_CG["delta_best_list"]
        params.update(params_CG)
        print("          Optimization parameters of analytical CG (max num. CG steps = {}): delta_list = {}, num. CG steps = {}".format(maxiter,delta_best_list,len(params["p_list"])))
    
    elif method == "anal_SD":                                                                                               # Analytical STEEPEST DESCENT with parameters N_iter and delta
        N_iter, delta = opt_params                                                                                          # Number of steps in the gradient descent and step factor in gradient descent                                                                     
        SOP_in = SOP0
        p_list = []
        for ind in range(N_iter):
            SOP_out = step_SD_dyn_poles(vemb_list,w_test_list,SOP_in,delta=delta,bounds=bounds,func_type=func_type)           # Steepest descent with analytical gradient
            p = SOP_out.get_params()                                                                 #                         # Parameters from the residues and poles
            p_list.append(p)
            SOP_in = SOP_out
        params["p_list"] = p_list
        print("          Optimization parameters of analytical SD: N_iter = {}, delta = {}".format(N_iter,delta))
        print("     Final values: cost fn = {}, norm grad. cost fn = {}".format(cost_func(p),LA.norm(grad_cost_func(p))),"\n")
    
    Gamma_list, sigma_list = params_to_SOP(p,num_poles)
    SOP_final = SOP(Gamma_list,sigma_list,p_type=p_type)
    res_list = adapt_residues(Gamma_list,p_type,"std")
    print("               Final poles: ",*sigma_list)
    print("               Final residues: ",*np.ravel(res_list),"\n")
    
    vemb_fit_list = SOP_final.evaluate(w_test_list)
    RMSE_real_list = [RMSE(np.array(vemb_list).real[:,i,i],np.array(vemb_fit_list).real[:,i,i]) for i in range(ntot)]
    RMSE_imag_list = [RMSE(np.array(vemb_list).imag[:,i,i],np.array(vemb_fit_list).imag[:,i,i]) for i in range(ntot)]
    print("     RMSE of the final set of parameters describing the embedding potential")
    print("          RMSE real on ",axis," axis in each site: ",*RMSE_real_list)
    print("          RMSE imag on ",axis," axis in each site: ",*RMSE_imag_list,"\n")
    return SOP_final, params