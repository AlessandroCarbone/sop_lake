import numpy            as np
import scipy
from scipy              import linalg           as LA
import yaml  # type: ignore[import-untyped]
from typing             import Any
import json
import time
import logging
import matplotlib.pyplot as plt
from contextlib         import redirect_stdout
from .SOP                import SOP, antisymm_SOP, adapt_residues
from .utils              import RMSE, check_sparsity, Np_from_A_isa
from .embedding_utils    import frequency_axis, create_mbAIMSOP_Hamiltonian, self_consistency_DMFT, linear_mixing_lists, DOS_diff
from .dmft_config        import sim_config, load_sim_config
from .dyn_poles          import compute_cost_function, compute_grad_cost_function, dyn_poles, params_to_SOP
from .mbAIMSOP_solver    import solver
from .data_io            import mat_list_to_dict, array_to_dict, plot_dmft_results, plot_convergence

logger = logging.getLogger(__name__)
class dmft_simulation:
    def __init__(self, config: sim_config):
        self.config = config
        self.hA_1 = config.system.get_single_particle_hA()
        self.hA, self.HA = config.system.get_Hubbard_Hamiltonian()[2:4]
        
        # Frequency grids
        self.w_list, self.w_sim_list = frequency_axis(
            axis=config.embedding.axis,
            eta_axis=config.embedding.eta_axis,
            num_pts=config.embedding.num_pts,
            matsubara_params={
                "beta": config.embedding.beta_T,
                "Nw_max": config.embedding.Nw_max
            }
        )
        w_tan_list      = np.tan(np.linspace(0,np.pi / 2,10000)[1:-1])                      # List of imaginary frequencies to be placed on the imaginary semi-axis - Tangent grid
        w_isa_list      = [-complex(0,w) for w in w_tan_list]                               # List of imaginary frequencies for the calculation of the number of particles on the imaginary semi-axis (isa)
        self.w_tan_list = w_tan_list
        self.w_isa_list = w_isa_list

        # Main self-consistent variables
        Gimp_SOP0, Gloc_list0, SigmaA_list0, SOP0, mu0 = config.get_input_variables()
        self.Gloc_list   = Gloc_list0
        self.SigmaA_list = SigmaA_list0
        self.Gimp_SOP    = Gimp_SOP0
        self.vemb_list   = None
        self.mu  = mu0
        self.SOP = SOP0
        self.SigmaA_isa_list = None
        self.Gloc_isa_list   = None
        self.vemb_isa_list   = None
        self.SOP_prev        = None
        self.Gimp_SOP_prev   = None

        # Convergence history and optimization data
        self.conv_history: dict[str, list] = {
            "iter": [],
            "mu": [],
            "diff_loc": [],
            "diff_prev": [],
            "diff_stagn": [],
            "cost": [],
            "grad_cost": []
        }
        self.opt_data: dict[str, Any] = {}

        self.hamiltonian_matrices = [None, None, None]                                      # Cached matrices for building auxiliary Hamiltonian (HA_AIM, hc_AIM_list, hb_AIM_list)
        self.solver_matrices = [None, None, None]                                           # Cached matrices for solver operators (NA_AIM, N_AIM_op, fermionic_op_list)

    def update_conv_history(self,num_iter, mu, diff_loc, diff_prev, diff_stagn, cost, grad_cost_dict):
        self.conv_history["iter"].append(num_iter)
        self.conv_history["mu"].append(mu)
        self.conv_history["diff_loc"].append(diff_loc)
        self.conv_history["diff_prev"].append(diff_prev)
        self.conv_history["diff_stagn"].append(diff_stagn)
        self.conv_history["cost"].append(cost)
        self.conv_history["grad_cost"].append(grad_cost_dict)

    def dmft_step(self):
        """ This function performs the DMFT step of the self-consistent cycle, i.e. fits the embedding potential, uses the solver and returns the new embedding potential for the next iteration."""
        logger.info("\tStart DMFT step {}".format(len(self.conv_history["iter"])))
        self_adj_check = True if self.config.optimization.complex_poles == False and self.config.optimization.herm_residues == True else False
        num_poles = self.config.embedding.num_poles
        ntot = self.hA_1.shape[0]

        # Dynamics of poles and residues of the embedding potential 
        t0 = time.time()
        logger.info("\tStart dynamics of poles of the embedding potential on the {} axis".format(self.config.embedding.axis))
        interp_text = ", interp_method = {}".format(self.config.optimization.interp_method) if self.config.optimization.opt_method == "custom_CG" else ""
        with open("opt_log.txt","a") as opt_log_file:               # Alternative to avoid printing this file - "with hidden_prints():"
            with redirect_stdout(opt_log_file):
                SOP_new, params = dyn_poles(self.vemb_list,self.w_sim_list,self.config.optimization.opt_method,self.config.embedding.num_poles,self.SOP,opt_params=self.config.optimization.opt_params,mu=self.mu,n_cycle=len(self.conv_history["iter"]),bounds=self.config.optimization.bounds,axis=self.config.embedding.axis,interp_method=self.config.optimization.interp_method,print_interp=self.config.optimization.print_interp)                
        num_min_steps = len(params["p_list"]) if self.config.optimization.opt_method == "scipy_CG" or "custom_CG" else self.config.optimization.opt_params[0]
        logger.info("\topt_method = {}, num. minimization steps = {} {}".format(self.config.optimization.opt_method,num_min_steps,interp_text))
        t1 = time.time()
        logger.info("\tEnd dynamics of poles of the embedding potential on the {} axis. Time elapsed: {:.2f} seconds".format(self.config.embedding.axis,t1 - t0))

        # Post-processing of the SOP
        pp_actions = []
        if self.config.embedding.p_type == "std":
            SOP_new.make_residues_hermitian()
            SOP_new.make_residues_pos_semidef()
            pp_actions.append("Making residues hermitian and positive semi-definite")
        if self.config.embedding.p_type == "sqrt" and self.config.optimization.odd_spectrum == False and self.config.optimization.herm_residues == False:
            if SOP_new.is_odd() == False:
                Gamma_list_new, sigma_list_new = antisymm_SOP(SOP_new.Gamma_list[:int(num_poles / 2)],SOP_new.sigma_list[:int(num_poles / 2)])                              # Adjusting the residues and poles to be odd
                SOP_new.Gamma_list = Gamma_list_new
                SOP_new.sigma_list = sigma_list_new
            SOP_new.make_residues_hermitian()
            pp_actions.append("Making residues hermitian and antisymmetric spectrum")
        if self.config.optimization.complex_poles == False:
            SOP_new.make_poles_real()
            pp_actions.append("Making poles real")
        SOP_new.sort()
        
        if pp_actions:
            logger.info("\tPost-processing of the SOP: " + ", ".join(pp_actions))
            cost_pp      = compute_cost_function(self.vemb_list,self.w_sim_list,SOP_new,paramagnetic=self.config.optimization.paramagnetic)                                   # Cost function value from residues and poles on the main test axis
            grad_cost_pp = compute_grad_cost_function(self.vemb_list,self.w_sim_list,SOP_new,bounds=self.config.optimization.bounds)                                          # Gradient of the cost function from residues and poles on the main test axis
            p_pp         = SOP_new.get_params()
        if self.config.optimization.opt_method == "custom_CG":
            params["p_list"].append(p_pp)
            params["cost_list"].append(cost_pp)
            params["grad_cost_list"].append(grad_cost_pp)
        elif self.config.optimization.opt_method == "scipy_CG":
            cost_list = [compute_cost_function(self.vemb_list,self.w_sim_list,SOP(*params_to_SOP(p,num_poles),self.SOP.p_type),paramagnetic=self.config.optimization.paramagnetic) for p in params["p_list"]]
            grad_cost_list = [compute_grad_cost_function(self.vemb_list,self.w_sim_list,SOP(*params_to_SOP(p,num_poles),self.SOP.p_type),bounds=self.config.optimization.bounds) for p in params["p_list"]]
            params["cost_list"]      = cost_list
            params["grad_cost_list"] = grad_cost_list
        params_dict = params.copy()
        params_dict["p_list"] = [p.tolist() for p in params["p_list"]]
        params_dict["grad_cost_list"] = [array_to_dict(grad_cost) for grad_cost in params["grad_cost_list"]]
        self.opt_data[len(self.conv_history["iter"])] = params_dict
        t2 = time.time()
        logger.info("\tPost-processing of the SOP. Time elapsed: {:.2f} seconds".format(t2 - t1))

        # Printing the residues and poles
        logger.info("\tNumber of poles/residues: {}".format(SOP_new.num_poles))
        logger.info("\tPoles: %s",SOP_new.sigma_list)
        res_list = adapt_residues(SOP_new.Gamma_list,SOP_new.p_type,"std")
        logger.info("\tResidues (in 'std' representation): %s",np.ravel(res_list))

        # Printing the accuracy of the fitting
        vemb_fit_list  = SOP_new.evaluate(self.w_sim_list)
        RMSE_real_list = [RMSE(np.array(self.vemb_list).real[:,i,i],np.array(vemb_fit_list).real[:,i,i]) for i in range(ntot)]
        RMSE_imag_list = [RMSE(np.array(self.vemb_list).imag[:,i,i],np.array(vemb_fit_list).imag[:,i,i]) for i in range(ntot)]
        logger.info("\tRMSE real on {} axis in each site: {}".format(self.config.embedding.axis,RMSE_real_list))
        logger.info("\tRMSE imag on {} axis in each site: {}".format(self.config.embedding.axis,RMSE_imag_list))
        if any(val > self.config.optimization.RMSE_thr for val in RMSE_real_list + RMSE_imag_list):                                         
            logger.warning("\tRMSE in fitting procedure is bigger than {}".format(self.config.optimization.RMSE_thr))
        t3 = time.time()
        
        # Many-body solver to get new local Green's function and self-energy
        logger.info("\tStart many-body AIM-SOP solver")
        H_aux = create_mbAIMSOP_Hamiltonian(self.HA,SOP_new,dmft_sim=self)
        logger.info("\t\tH_aux sparsity: {}".format(check_sparsity(H_aux)))

        eta_solver = 0.                                     # Small imaginary part added to the real frequencies if outside the DMFT cycle 
        # Always use fresh solver matrices to avoid dimension mismatches when num_poles changes
        imp_solver = solver(H_aux,ntot,num_poles,eta_solver,self.mu,sparse_gs=self.config.embedding.sparse_gs,input_matrices=self.solver_matrices)
        C_list, Z_list = imp_solver.get_Gimp(gs_search=self.config.embedding.gs_search,method=self.config.embedding.solver_method,self_adj_check=self_adj_check)
        Gimp_SOP       = imp_solver.make_Gimp_SOP(C_list,Z_list)
        self.solver_matrices = imp_solver.input_matrices        # Save computed matrices for reuse in next DMFT iterations
        t4 = time.time()
        logger.info("\tEnd many-body AIM-SOP solver. Time elapsed: {:.2f} seconds".format(t4 - t3))

        # Here should we add a check on impurity GF poles? See old code to have an idea.

        # Self-consistency step
        logger.info("\tStart self-consistency step")
        SigmaA_list, Gloc_list         = self_consistency_DMFT(SOP_new,Gimp_SOP,self.config.system.epsk_list,self.hA_1,self.w_sim_list,self.mu,paramagnetic=self.config.optimization.paramagnetic)
        SigmaA_isa_list, Gloc_isa_list = self_consistency_DMFT(SOP_new,Gimp_SOP,self.config.system.epsk_list,self.hA_1,self.w_isa_list,self.mu,paramagnetic=self.config.optimization.paramagnetic)
        t5 = time.time()
        logger.info("\tEnd self-consistency step. Time elapsed: {:.2f} seconds".format(t5 - t4))

        # Define new embedding potentials for the next iteration
        IA_mat        = np.identity(ntot,dtype=np.complex128)
        vemb_list     = [(w + self.mu) * IA_mat - self.hA_1 - ( LA.inv(Gloc_list[iw]) + SigmaA_list[iw] ) for iw,w in enumerate(self.w_sim_list)]
        vemb_isa_list = [(w + self.mu) * IA_mat - self.hA_1 - ( LA.inv(Gloc_isa_list[iw]) + SigmaA_isa_list[iw] ) for iw,w in enumerate(self.w_isa_list)]       
        t6 = time.time()
        logger.info("\tEnd DMFT step. Total time elapsed: {:.2f} seconds".format(t6 - t0))

        self.SOP_prev      = self.SOP
        self.Gimp_SOP_prev = self.Gimp_SOP
        self.SOP         = SOP_new
        self.Gimp_SOP    = Gimp_SOP
        self.Gloc_list   = Gloc_list
        self.SigmaA_list = SigmaA_list
        self.vemb_list   = vemb_list
        self.SigmaA_isa_list = SigmaA_isa_list
        self.Gloc_isa_list   = Gloc_isa_list
        self.vemb_isa_list   = vemb_isa_list

    def run(self,output_file_names={"conv": "conv_output.json", "dmft": "dmft_output.json", "vemb": "vemb_output.json", "opt": "opt_output.json"}):
        """ This function runs the DMFT simulation according to the input configuration."""
        self.write_params()
        # Load config from file if provided, otherwise use current config
        input_info = load_sim_config(self.config.input.config_file) if self.config.input.config_file is not None else self.config
        self.write_input_info(input_info)
        
        Nk     = len(self.config.system.epsk_list)
        ntot   = self.hA_1.shape[0]
        IA_mat = np.identity(ntot,dtype=np.complex128)                                                                                                      # Identity matrix in the single-particle space of the fragment
        vemb_list0 = [(w + self.mu) * IA_mat - self.hA_1 - ( LA.inv(self.Gloc_list[iw]) + self.SigmaA_list[iw] ) for iw,w in enumerate(self.w_sim_list)]
        # Linear mixing the initial embedding potential if we have previous data and a high number of sites
        if (self.config.input.input_case == "from_file" or self.config.input.input_case == "fixed_p0") and input_info.system.size > 100 and self.config.optimization.initial_mixing == True:                    # If we have previous data and a high number of sites, we use them to initialize the cycle
            alpha_mix0 = input_info.optimization.alpha if input_info.optimization.alpha != 1. else 0.5
            logger.warning("Linear mixing with alpha = {} of the embedding potential using the previous set of residues and poles".format(alpha_mix0))
            vemb_fit_list0 = self.SOP.evaluate(self.w_sim_list)
            vemb_list0     = linear_mixing_lists(vemb_list0,vemb_fit_list0,alpha_mix0)                                                                      # Anderson linear mixing of the initial embedding potential on the test axis
        self.vemb_list  = vemb_list0

        t0 = time.time()
        for n_iter in range(self.config.embedding.max_iter):
            logger.info("\n===============================")
            logger.info("DMFT iteration {}".format(n_iter))
            # Updating chemical potential
            # if n_iter != 0 and self.config.embedding.mu_fixed is not None:
                # Here we need to add the routine to adjust the chemical potential. See adjust_mu_ia function in the old code.
                # mu, vemb_isa_list = ... (to complete)
                # self.mu, self.vemb_isa_list = mu, vemb_isa_list
                # Gloc_list     = [sum( LA.inv((w + self.mu - self.config.system.epsk_list[k]) * IA_mat - self.SigmaA_list[iw]) for k in np.arange(Nk)) / Nk for iw,w in enumerate(self.w_sim_list)]          
                # Gloc_isa_list = [sum( LA.inv((w_isa + self.mu - self.config.system.epsk_list[k]) * IA_mat - self.SigmaA_isa_list[iw]) for k in np.arange(Nk)) / Nk for iw,w_isa in enumerate(self.w_isa_list)]          
                # vemb_list     = [(w + self.mu) * IA_mat - self.hA_1 - ( LA.inv(self.Gloc_list[iw]) + self.SigmaA_list[iw] ) for iw,w in enumerate(self.w_sim_list)]
                # self.Gloc_list, self.Gloc_isa_list, self.vemb_list = Gloc_list, Gloc_isa_list, vemb_list
            logger.info("\tChemical potential mu: {:.6f}".format(self.mu))
            if n_iter > 0:
                Np_iter = Np_from_A_isa(self.Gloc_isa_list,self.w_tan_list)
                logger.info("\tNumber of particles: {:.6f}".format(Np_iter))

            if self.config.optimization.mixing_method == "linear":
                self.dmft_step()
                t1 = time.time()
                # Linear mixing of the embedding potential on the main axis and the imaginary semi-axis
                vemb_fit_list = self.SOP.evaluate(self.w_sim_list)
                vemb_list_new = linear_mixing_lists(self.vemb_list,vemb_fit_list,self.config.optimization.alpha)
                self.vemb_list = vemb_list_new
                vemb_fit_isa_list = self.SOP.evaluate(self.w_isa_list)
                vemb_isa_list_new = linear_mixing_lists(self.vemb_isa_list,vemb_fit_isa_list,self.config.optimization.alpha)
                self.vemb_isa_list = vemb_isa_list_new
                t2 = time.time()
                logger.info("\tLinear mixing of the embedding potential. Time elapsed: {:.2f} seconds".format(t2 - t1))
            else:
                raise ValueError("Mixing method {} not recognized.".format(self.config.embedding.mixing_method))

            # Updating convergence history
            diff_prev  = DOS_diff(self.Gimp_SOP.evaluate(self.w_sim_list),self.Gimp_SOP_prev.evaluate(self.w_sim_list)) if n_iter > 0 else 0.            # Previous difference: two consecutive impurity GFs 
            diff_loc   = DOS_diff(self.Gloc_list,self.Gimp_SOP.evaluate(self.w_sim_list))                                                                # Local difference: current local GF and impurity GF              
            diff_stagn = np.abs(diff_prev - self.conv_history["diff_prev"][-1]) if n_iter > 1 else 0.               # Stagnation difference: difference between the last two differences of the impurity DOS
            final_cost = self.opt_data[n_iter]["cost_list"][-1] if len(self.opt_data[n_iter]["cost_list"]) > 0 else None
            final_grad = self.opt_data[n_iter]["grad_cost_list"][-1] if len(self.opt_data[n_iter]["grad_cost_list"]) > 0 else None
            self.update_conv_history(n_iter,self.mu,diff_loc,diff_prev,diff_stagn,final_cost,final_grad)
            if n_iter > 0:
                logger.info("Diff. 2 consecutive imp. DOS [{} axis]: {}".format(self.config.embedding.axis,diff_prev))
            logger.info("Diff. local and imp. DOS [{} axis]: {}".format(self.config.embedding.axis,diff_loc))
            
            # Saving data until this iteration
            self.save_conv_history(output_file_names["conv"])
            self.save_Gimp_Gloc_SigmaA(output_file_names["dmft"])
            self.save_vemb_SOP(output_file_names["vemb"])
            self.save_optimization_data(output_file_names["opt"])
            x_bracket = [-self.config.system.U, self.config.system.U] if self.config.embedding.axis == "erf" else [-5., 5.]
            self.save_plots(x_bracket=x_bracket)            # By default, it only saves the first elements of the diagonals of the dynamical functions

            # Establishing convergence or stagnation of the self-consistent cycle
            if n_iter > 4 and all(diff <= self.config.optimization.thr_diff_prev for diff in self.conv_history["diff_prev"][-3:]):                    # Checking if self-consistency is reached (last 4 elements below the threshold)
                logger.info("Self-consistency reached!")
                break
            if n_iter > 4 and all(diff <= self.config.optimization.thr_stagnation for diff in self.conv_history["diff_stagn"][-10:]):               # Checking if the difference between the last 10 elements in diff_prev is below the threshold
                logger.info("Stagnation reached!")
                break
            if n_iter == self.config.embedding.max_iter:
                logger.info("Maximum number of iterations reached!")

    def write_input_info(self, input_info):
        logger.info("\nINPUT - input_case = {}".format(self.config.input.input_case))
        if self.config.input.input_case == "from_file":
            logger.info("\nSystem: U = {} | mu = {} | N = {} | num_poles = {}".format(input_info.system.U,self.mu,input_info.system.size,input_info.embedding.num_poles))
            axis_text = " | beta_T = {} | Nw_max = {}".format(input_info.embedding.matsubara_params["beta"],input_info.embedding.matsubara_params["Nw_max"]) if input_info.embedding.axis == "imaginary" else " | eta_axis = {}".format(input_info.embedding.eta_axis)
            logger.info("        axis = {} | p_type = {} {}".format(input_info.embedding.axis,input_info.embedding.p_type,axis_text))
        logger.info("\nInitial poles: %s",self.SOP.sigma_list)
        logger.info("\nInitial residues: %s",adapt_residues(self.SOP.Gamma_list,input_info.embedding.p_type,"std"))
        logger.info("\n========================================\n")
    
    def write_params(self):
        logger.info("\nSystem: U = {} | N = {} | num_poles = {} | mu_fixed = {}".format(self.config.system.U,self.config.system.size,self.config.embedding.num_poles,self.config.embedding.mu_fixed))
        axis_text = " | beta_T = {} | Nw_max = {}".format(self.config.embedding.matsubara_params["beta"],self.config.embedding.matsubara_params["Nw_max"]) if self.config.embedding.axis == "imaginary" else " | eta_axis = {} | num_pts = {} | w_edges = {}".format(self.config.embedding.eta_axis, self.config.embedding.num_pts, self.config.embedding.w_edges)
        logger.info("\n- axis = {} | p_type = {} {}".format(self.config.embedding.axis,self.config.embedding.p_type,axis_text))
        logger.info("\n- max_iter = {} | p_type = {} | mixing_method = {} with alpha = {} | solver_method = {}".format(self.config.embedding.max_iter,self.config.embedding.p_type,self.config.optimization.mixing_method,self.config.optimization.alpha,self.config.embedding.solver_method))
        logger.info("\n- opt_method = {} with opt_params = {}".format(self.config.optimization.opt_method,self.config.optimization.opt_params))
        logger.info("\n- Bounds on residues/poles: {}".format(self.config.optimization.bounds))
        logger.info("\n========================================")
    
    def save_conv_history(self,file):
        with open(file, "w") as f:
            json.dump(self.conv_history, f, indent=2)

    def save_Gimp_Gloc_SigmaA(self,file):
        w_list, Gimp_SOP, Gloc_list, SigmaA_list = self.w_list, self.Gimp_SOP, self.Gloc_list, self.SigmaA_list
        Gimp_SOP_dict = Gimp_SOP.to_dict()
        Gloc_dict = mat_list_to_dict(Gloc_list)
        SigmaA_dict = mat_list_to_dict(SigmaA_list)
        data_dict = {
            "w_list": w_list,
            "Gimp_SOP": Gimp_SOP_dict,
            "Gloc_list": Gloc_dict,
            "SigmaA_list": SigmaA_dict
        }
        with open(file, "w") as f:
            json.dump(data_dict, f, indent=2)
    
    def save_vemb_SOP(self, file):
        vemb_list, SOP = self.vemb_list, self.SOP
        vemb_dict = mat_list_to_dict(vemb_list)
        SOP_dict = SOP.to_dict()
        data_dict = {}
        data_dict["vemb_list"] = vemb_dict
        data_dict.update(SOP_dict)
        with open(file, "w") as f:
            json.dump(data_dict, f, indent=2)
    
    def save_optimization_data(self, file):
        with open(file, "w") as f:
            json.dump(self.opt_data, f, indent=2)

    def save_plots(self,path_to_file="figures",indices=[0,0], x_bracket=[-5,5]): 
        """ This function saves plots of the local GF, impurity GF and the embedding potential. 
        """
        n_iter = self.conv_history["iter"][-1] if len(self.conv_history["iter"]) > 0 else "try"
        i, j = indices[0], indices[1]                                                                       # Matrix indices
        Gimp_list = self.Gimp_SOP.evaluate(self.w_sim_list)                                                 
        vemb_fit_list  = self.SOP.evaluate(self.w_sim_list)
        Glocij_list = [np.array(self.Gloc_list).real[:,i,j],np.array(self.Gloc_list).imag[:,i,j]]           # Local GF
        Gimpij_list = [np.array(Gimp_list).real[:,i,j],np.array(Gimp_list).imag[:,i,j]]                     # Impurity GF
        SigmaAij_list = [np.array(self.SigmaA_list).real[:,i,j],np.array(self.SigmaA_list).imag[:,i,j]]     # Self-energy
        vembij_list   = [np.array(self.vemb_list).real[:,i,j],np.array(self.vemb_list).imag[:,i,j]]         # v_emb 
        vembij_fit_list = [np.array(vemb_fit_list).real[:,i,j],np.array(vemb_fit_list).imag[:,i,j]]         # v_emb fit 
        diff_loc_list, diff_prev_list = self.conv_history["diff_loc"], self.conv_history["diff_prev"]       # Lists used to evaluate the convergence of the algorithm
        
        plot_dmft_results(self.w_list, indices, Gimpij_list, Glocij_list, SigmaAij_list, vembij_list, vembij_fit_list, n_iter, path_to_file, self.config, x_bracket)
        # Convergence plots
        if (isinstance(n_iter, int) and n_iter > 1) or n_iter == "try":
            plot_convergence(diff_loc_list, diff_prev_list, n_iter, path_to_file)

       