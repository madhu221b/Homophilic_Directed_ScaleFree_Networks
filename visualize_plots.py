import os
import pickle as pkl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

T = 30
hMM_list, hmm_list = np.arange(0,1.1,0.1), np.arange(0,1.1,0.1)

def get_line_plot_t_vs_frac_bet(path,homophily_list):
    fig, ax = plt.subplots( nrows=1, ncols=1 ) 

    dict_files = [file_name for file_name in os.listdir(path) if "frac" in file_name]
    for dict_path in dict_files:
        dict_path = os.path.join(path,dict_path)
        with open(dict_path,"rb") as f:
            dict_ = pkl.load(f)
        
        hMM, hmm = dict_path.split("hMM")[-1].split("-")[0], dict_path.split("hmm")[-1].split("_")[0]
        label = "hMM:{} , hmm:{}".format(hMM,hmm)
        if "{},{}".format(hMM,hmm) in homophily_list:
            bet = {t:round(val["bet"] ,5) for t,val in dict_.items()}
            dict_ = {t:round((val["min"]-val["maj"]) ,2) for t,val in dict_.items()}
            dict_,bet = dict(sorted(dict_.items())), dict(sorted(bet.items()))
            # ax.plot(dict_.keys(), dict_.values(),marker="o",label=label)
            ax.plot(bet.keys(), bet.values(),marker="o",label=label)
    
    # idxs = [_ for _ in range(len(dict_))]
    # ax.set_xticks(idxs)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Avg betweenness of minority nodes")
    ax.legend(loc = "lower right",bbox_to_anchor=(0.8,0.3))
    fig.savefig('plots/time_vs_frac_trial_cnw.png',bbox_inches='tight')   # save the figure to file
    plt.close(fig)    # close the figure window

def get_line_plot_t_vs_frac(path):
    fig, ax = plt.subplots( nrows=1, ncols=1 ) 

    dict_files = [file_name for file_name in os.listdir(path) if "frac" in file_name]
    for dict_path in dict_files:
        dict_path = os.path.join(path,dict_path)
        with open(dict_path,"rb") as f:
            dict_ = pkl.load(f)
        
        hMM, hmm = dict_path.split("hMM")[-1].split("-")[0], dict_path.split("hmm")[-1].split("_")[0]
        label = "hMM:{} , hmm:{}".format(hMM,hmm)
        if hmm == "0.5" and hMM == "0.5": continue
        dict_ = {t:round((val["min"]-val["maj"]) ,2) for t,val in dict_.items()}
        dict_ = dict(sorted(dict_.items()))
        ax.plot(dict_.keys(), dict_.values(),marker="o",label=label)
        # ax.plot(dict_.keys(), dict_.values(),marker="-",label=label)
    
    # idxs = [_ for _ in range(len(dict_))]
    # ax.set_xticks(idxs)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Difference (frac minority - frac majority)")
    ax.legend(loc = "lower right",bbox_to_anchor=(0.7,0))
    fig.savefig('plots/time_vs_frac.png',bbox_inches='tight')   # save the figure to file
    plt.close(fig)    # close the figure window


def get_grid_frac(files):
    grid = np.zeros((len(hmm_list),len(hMM_list)))
    for dict_path in files:
        hMM, hmm = dict_path.split("hMM")[-1].split("-")[0], dict_path.split("hmm")[-1].split("_")[0]
        hMM_idx, hmm_idx = int(float(hMM)*10), int(float(hmm)*10)
        print("hMM: {}, hmm: {}".format(hMM, hmm))
        with open(dict_path, 'rb') as f:
             dict_ = pkl.load(f)
        grid[hmm_idx][hMM_idx] = round(dict_[T-1]["min"] - dict_[T-1]["maj"],2)
    return grid

def get_heatmap_frac(file_path,model=""):  

    plot_directory = "../plots/heatmap/frac"
    if not os.path.exists(plot_directory): os.makedirs(plot_directory)

    all_files = os.listdir(file_path)
    frac_files = [os.path.join(file_path,file_name) for file_name in all_files if "frac" in file_name]
    grid = get_grid_frac(frac_files)
  
    heatmap = grid.T

    hmm_ticks = [np.round(hmm,2) for hmm in hmm_list]
    hMM_ticks = [np.round(hMM,2) for hMM in hMM_list]
    vmin, vmax = -0.3, 0.3
    ax = sns.heatmap(heatmap, cmap=plt.cm.coolwarm,xticklabels=hmm_ticks,yticklabels=hMM_ticks,vmin=vmin,vmax=vmax)
    ax.invert_yaxis()
    ax.set_xlabel("Homophily for Minority Class")
    ax.set_ylabel("Homophily for Majority Class")
    fig = ax.get_figure()
    fig.savefig(plot_directory+"/frac_{}.png".format(model))




if __name__ == "__main__":
    path = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/cnw_p_1.0_q_1.0"
    # get_line_plot_t_vs_frac(path)
    # "0.2,0.2", "0.8,0.8", "0.2,0.8", "0.8,0.2"
    get_line_plot_t_vs_frac_bet(path, ["0.2,0.8", "0.8,0.2"])

   # get_heatmap_frac(path, "fw")