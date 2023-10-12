"""
Plotting Action Steering

10/13/2023
_______________

Summary: the script plots the results obtained after processing the experiments for action steering.
	 These are the results in Fig. 9 and 10 of the manuscript.

DISCLAIMER: THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""

####^^^^ Imports and settings
##^^ generic
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import tikzplotlib

##^^ importing project functions
from utils_experiment_list_action_steering import *

##^^ plotting settings

tikzsave = True
plotsave = True
plotvisual = False

for agent in list_agents:
    print(f"= = = = = = = = = = = = = = = = =\nAnalyzing agent: {agent}\n= = = = = = = = = = = = = = = = =")

    ##^^ Choose strategy in exp_strategy by selecting the index:
    ## 0: baseline
    ## 1: max_reward
    ## 2: max_reward_obs_20
    ## 3: min_reward
    ## 4: min_reward_obs_20
    ## 5: imp_bitrate
    ## 6: imp_bitrate_obs_20
    ## 7: baseline_obs_20

    baseline_strategy = list(exp_configuration.keys())[0] #choose among indexes 0/7
    ar_strategy = list(exp_configuration.keys())[5] #choose among indexes 1-6

    print(f'- Comparing strategies: "{ar_strategy}" with "{baseline_strategy}"')

    #^ repeat for each slice
    for slice_id in range(0,3):
        print(f'- Analysis of slice {slice_id}')

        #^ pick the corresopnding file from the results
        filepath = "../results/action-steering/data-for-cdfs/"

        #^ now pick files with corresponding KPIs:
        baseline_exp_id = exp_configuration[baseline_strategy][0]
        ar_exp_id = exp_configuration[ar_strategy][0]

        #^ getting the file names
        f_baseline_tx_brate = filepath+baseline_strategy+"/"+agent+"-exp"+str(baseline_exp_id[agent])+"_SL"+str(slice_id)+"_tx_brate.npy"
        f_baseline_tx_pkts = filepath+baseline_strategy+"/"+agent+"-exp"+str(baseline_exp_id[agent])+"_SL"+str(slice_id)+"_tx_pkts.npy"
        f_baseline_dl_buffer = filepath+baseline_strategy+"/"+agent+"-exp"+str(baseline_exp_id[agent])+"_SL"+str(slice_id)+"_dl_buffer.npy"

        f_ar_tx_brate = filepath+ar_strategy+"/"+agent+"-exp"+str(ar_exp_id[agent])+"_SL"+str(slice_id)+"_tx_brate.npy"
        f_ar_tx_pkts = filepath+ar_strategy+"/"+agent+"-exp"+str(ar_exp_id[agent])+"_SL"+str(slice_id)+"_tx_pkts.npy"
        f_ar_dl_buffer = filepath+ar_strategy+"/"+agent+"-exp"+str(ar_exp_id[agent])+"_SL"+str(slice_id)+"_dl_buffer.npy"

        #^ arrays to store data
        baseline_avg_tx_brate = []
        baseline_avg_tx_pkts = []
        baseline_avg_dl_buffer = []

        ar_avg_tx_brate = []
        ar_avg_tx_pkts = []
        ar_avg_dl_buffer = []

        with open(f_baseline_tx_brate, 'rb') as a, open(f_baseline_tx_pkts, 'rb') as b, open(f_baseline_dl_buffer, 'rb') as c:
            baseline_avg_tx_brate = np.load(a)
            baseline_avg_tx_pkts = np.load(b)
            baseline_avg_dl_buffer = np.load(c)
        
        for f in [a,b,c]:
            f.close()

        with open(f_ar_tx_brate, 'rb') as a, open(f_ar_tx_pkts, 'rb') as b, open(f_ar_dl_buffer, 'rb') as c:
            ar_avg_tx_brate = np.load(a)
            ar_avg_tx_pkts = np.load(b)
            ar_avg_dl_buffer = np.load(c)
        
        for f in [a,b,c]:
            f.close()

        # sort the data:
        baseline_avg_tx_brate_sorted = np.sort(np.array(baseline_avg_tx_brate))
        ar_avg_tx_brate_sorted = np.sort(np.array(ar_avg_tx_brate))

        baseline_avg_tx_pkts_sorted = np.sort(np.array(baseline_avg_tx_pkts))
        ar_avg_tx_pkts_sorted = np.sort(np.array(ar_avg_tx_pkts))

        baseline_avg_dl_buffer_sorted = np.sort(np.array(baseline_avg_dl_buffer))
        ar_avg_dl_buffer_sorted = np.sort(np.array(ar_avg_dl_buffer))

        # calculate the proportional values of samples
        p_c_tx_brate = 1. * np.arange(len(baseline_avg_tx_brate)) / (len(baseline_avg_tx_brate) - 1)
        p_nc_tx_brate = 1. * np.arange(len(ar_avg_tx_brate)) / (len(ar_avg_tx_brate) - 1)

        p_c_tx_pkts = 1. * np.arange(len(baseline_avg_tx_pkts)) / (len(baseline_avg_tx_pkts) - 1)
        p_nc_tx_pkts = 1. * np.arange(len(ar_avg_tx_pkts)) / (len(ar_avg_tx_pkts) - 1)

        p_c_dl_buffer = 1. * np.arange(len(baseline_avg_dl_buffer)) / (len(baseline_avg_dl_buffer) - 1)
        p_nc_dl_buffer = 1. * np.arange(len(ar_avg_dl_buffer)) / (len(ar_avg_dl_buffer) - 1)

        #^ plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))

        ax1.plot(baseline_avg_tx_brate_sorted,p_c_tx_brate,label="No action change", color="darkorchid")
        ax1.plot(ar_avg_tx_brate_sorted,p_nc_tx_brate,label=ar_strategy, color="darkgreen")
        ax1.set_xlabel('TX BRATE',fontsize=12)
        ax1.set_ylabel("CDF",fontsize=12)
        ax1.legend()

        ax2.plot(baseline_avg_tx_pkts_sorted,p_c_tx_pkts,label="No action change", color="darkorchid")
        ax2.plot(ar_avg_tx_pkts_sorted,p_nc_tx_pkts,label=ar_strategy, color="darkgreen")
        ax2.set_xlabel('TX PKTS',fontsize=12)
        ax2.set_ylabel("CDF",fontsize=12)
        ax2.legend()

        ax3.plot(baseline_avg_dl_buffer_sorted,p_c_dl_buffer,label="No action change", color="darkorchid")
        ax3.plot(ar_avg_dl_buffer_sorted,p_nc_dl_buffer,label=ar_strategy, color="darkgreen")
        ax3.set_xlabel('DL BUFFER',fontsize=12)
        ax3.set_ylabel("CDF",fontsize=12)
        ax3.legend()

        #^ exporting/saving plots in different formats or simply visualize
        outdir = "../results/action-steering/plot-kpis/"+ar_strategy+"_vs_"+baseline_strategy+"/"
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        filename = outdir + "plot_gains_"+agent+"_exp"+"_SL"+str(slice_id)
        file_ext_fig = filename + ".png"
        file_ext_tex = filename + ".tex"
        # print(filename)
        if tikzsave:
            tikzplotlib.save(file_ext_tex)
        if plotsave:
            plt.savefig(file_ext_fig)
        if plotvisual:
            plt.show()

        plt.close()