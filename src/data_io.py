import numpy                        as np
import json, os
import matplotlib
matplotlib.use('Agg')                       # Use a non-interactive backend for plotting
import matplotlib.pyplot            as plt
matplotlib.rcParams['text.usetex'] = True

from SOP            import SOP

def read_dmft_data(file: str):
    with open(file, "r") as f:
        data_dict = json.load(f)
    w_list = data_dict["w_list"]
    Gimp_SOP_dict = data_dict["Gimp_SOP"]
    C_real, C_imag = Gimp_SOP_dict["Gamma_list"]["real"], Gimp_SOP_dict["Gamma_list"]["imag"]
    Z_real, Z_imag = Gimp_SOP_dict["sigma_list"]["real"], Gimp_SOP_dict["sigma_list"]["imag"]
    C_list = [np.array(C_real[iw]) + 1j * np.array(C_imag[iw]) for iw in range(len(C_real))]
    Z_list = [Z_real[iw] + 1j * Z_imag[iw] for iw in range(len(Z_real))]
    Gimp_SOP  = SOP(C_list, Z_list)
    Gloc_real = data_dict["Gloc_list"]["real"]
    Gloc_imag = data_dict["Gloc_list"]["imag"]
    SigmaA_real = data_dict["SigmaA_list"]["real"]
    SigmaA_imag = data_dict["SigmaA_list"]["imag"]
    Gloc_list = [np.array(Gloc_real[iw]) + 1j * np.array(Gloc_imag[iw]) for iw in range(len(w_list))]
    SigmaA_list = [np.array(SigmaA_real[iw]) + 1j * np.array(SigmaA_imag[iw]) for iw in range(len(w_list))]
    return w_list, Gimp_SOP, Gloc_list, SigmaA_list

def read_vemb_data(file: str):
    with open(file, "r") as f:
        data_dict = json.load(f)
    vemb_real = data_dict["vemb_list"]["real"]
    vemb_imag = data_dict["vemb_list"]["imag"]
    Gamma_real = data_dict["Gamma_list"]["real"]
    Gamma_imag = data_dict["Gamma_list"]["imag"]
    sigma_real = data_dict["sigma_list"]["real"]
    sigma_imag = data_dict["sigma_list"]["imag"]
    p_type = data_dict["p_type"]
    vemb_list = [np.array(vemb_real[iw]) + 1j * np.array(vemb_imag[iw]) for iw in range(len(vemb_real))]
    Gamma_list = [np.array(Gamma_real[iw]) + 1j * np.array(Gamma_imag[iw]) for iw in range(len(Gamma_real))]
    sigma_list = [sigma_real[iw] + 1j * sigma_imag[iw] for iw in range(len(sigma_real))]
    SOP_vemb = SOP(Gamma_list, sigma_list, p_type=p_type)
    return vemb_list, SOP_vemb

def read_conv_history(file: str):
    with open(file, "r") as f:
        conv_history = json.load(f)
    return conv_history

def array_to_dict(array):
    """ Converts a numpy array to a dictionary with "real" and "imag" keys for JSON serialization."""
    return {"real": array.real.tolist(), "imag": array.imag.tolist()}

def mat_list_to_dict(mat_list):
    """ Converts a list of matrices (np.ndarray) to a dictionary with "real" and "imag" keys for JSON serialization."""
    real_list = [mat.real.tolist() for mat in mat_list]
    imag_list = [mat.imag.tolist() for mat in mat_list]
    return {"real": real_list, "imag": imag_list}

def clean_folder(fig_dir="figures"):
    """ Remove all files in the specified folder and any files in the current directory that match certain prefixes or suffixes. Typicall, only
    output files and printed figures."""
    for filename in os.listdir(fig_dir):
        if os.path.isfile(os.path.join(fig_dir, filename)):
            os.remove(os.path.join(fig_dir, filename))
    prefix_list = []
    suffix_list = ["output.txt","output.json","log.txt"]
    for filename in os.listdir('.'):
        # for prefix in prefix_list:
        #     if filename.startswith(prefix):
        #         os.remove(filename)
        if filename in os.listdir('.'):
            for suffix in suffix_list:
                if filename.endswith(suffix):
                    os.remove(filename)

def plot_dmft_results(w_list, indices, Gimpij_list, Glocij_list, SigmaAij_list, vembij_list, vembij_fit_list, n_iter, path_to_file,config, x_bracket=[-5,5]):
    # Color palettes
    palette         = ["#822D8A","#D8BFD8"]
    vik_red_palette = ["#044F88","#4F93B5","#902B05","#C26F40"]

    # Plot tools
    i, j            = indices
    ylabels         = [r"Re", r"Im"]
    labels          = [r"$i={}$".format(i) , r"$j={}$".format(j)]
    title           = r"Local and impurity GFs - $n_\text{{iter}} = {}$".format(n_iter)
    title2          = r"Embedding potential $v_\text{{emb}}$ - $n_\text{{iter}} = {}$".format(n_iter)
    subtitle        = "axis = {}".format(config.embedding.axis) 
    titles          = [title,title2]
    file_name       = "plots_{}_{}{}.pdf".format(n_iter,i,j)
    
    # Plotting the local and impurity GFs
    fig, ax = plt.subplots(2, 2, sharex='col', figsize=(18,9))
    fig.subplots_adjust(hspace=0)
    for k in range(2):
        # GF plots
        ax[k,0].plot(w_list,Gimpij_list[k],label=r"$G_\text{imp}$",linewidth=3,linestyle="-",color=vik_red_palette[2])
        ax[k,0].plot(w_list,Glocij_list[k],label=r"$G_\text{loc}$",linewidth=3,linestyle="--",color=vik_red_palette[3])
        ax[k,0].plot(w_list,SigmaAij_list[k],label=r"$\Sigma_A$",linewidth=3,linestyle="-",color=vik_red_palette[1])

        # Embedding potential plots
        ax[k,1].plot(w_list,vembij_list[k],label="Data",linewidth=3,linestyle="-",color=palette[0])
        ax[k,1].plot(w_list,vembij_fit_list[k],label="Fit",linewidth=3,linestyle="-",color=palette[1])
    
        # Plot customization
        if k==0:
            for l in range(2):
                ax[k,l].set_title(titles[l]+"\n"+subtitle,fontsize=18)
                # ax[k,l].yaxis.set_ticks(ax[k,l].get_yticks())
                # ax[k,l].set_yticklabels(np.round(ax[k,l].get_yticks().tolist(),3),fontsize=18)
                ax[k,l].grid()
        elif k==1:
            for l in range(2):
                ax[k,l].set_xlabel(r"$\omega$",size=18)
                # ax[k,l].xaxis.set_ticks(ax[k,l].get_xticks())
                # ax[k,l].set_xticklabels(np.round(ax[k,l].get_xticks().tolist(),3),fontsize=18)
                ax[k,l].set_xlim(x_bracket[0],x_bracket[1])
                ax[k,l].legend(title=labels[0],title_fontsize=18,fontsize=18,frameon=True,loc=1)
                # ax[k,l].yaxis.set_ticks(ax[k,l].get_yticks())
                # ax[k,l].set_yticklabels(np.round(ax[k,l].get_yticks().tolist(),10),fontsize=18)
                ax[k,l].grid()
        for l in range(2):
            ax[k,l].set_ylabel(ylabels[k],size=18)
    plt.savefig(path_to_file+"/"+file_name,bbox_inches='tight')        
    plt.close(fig)

def plot_convergence(diff_loc_list, diff_prev_list, n_iter, path_to_file):
    vik_red_palette = ["#044F88","#4F93B5","#902B05","#C26F40"]     # Color palette
    fig, ax1 = plt.subplots(figsize=(9,6)) 
    x_list   = np.arange(1,len(diff_prev_list))
    ax1.plot(x_list,diff_loc_list[1:],label="Local diff.",linewidth=3,color=vik_red_palette[0])
    ax1.plot(x_list,diff_prev_list[1:],label="Prev diff.",linewidth=3,color=vik_red_palette[1])
    ax1.set_title(r"Convergence of the DMFT cycle - $n_{{iter}} = {}$".format(n_iter),size=18)
    ax1.set_xlabel(r"DMFT iter. $k$",size=18)
    ax1.set_yscale("log")
    ax1.legend(loc=1)
    plt.savefig(path_to_file+"/convergence_DMFT.pdf",bbox_inches='tight')
    plt.close(fig)