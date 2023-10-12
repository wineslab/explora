"""
SHAP analysis and plots

10/12/2023
_______________
    
Summary: The script provides functions to allow visualization of the experiments
         with key parameters colors according to SHAP relevance

DISCLAIMER: THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING 
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""

####^^ IMPORTS

##^^ matplotlib
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

##^^ importing generic project functions
from utils_generic_functions import *


def getSHAPExplanationsPerMetric(shap_values):
    """
    Gets SHAP values per output of the autoencoder, i.e., the input of the DRL agent
    
    Parameters
    _______________
    
    shap_values: the explanations

    Output
    _______________
    shap_oae_a_col: output autoencoder, metric a
    shap_oae_b_col: output autoencoder, metric b
    shap_oae_c_col: output autoencoder, metric c

    """
    shap_oae_a_col= np.array([item[0] for item in shap_values])
    shap_oae_b_col= np.array([item[1] for item in shap_values])
    shap_oae_c_col= np.array([item[2] for item in shap_values])

    return shap_oae_a_col,shap_oae_b_col,shap_oae_c_col


def plotExplanations(exp_drl_shap_values,data_slice,slice_id,exp,output_path_suffix,Plot=True,SavetoFile=False):
    """
    Plots the values of the experiment over time and colors them according to SHAP explanations

    """

    ##^^ splitting data_structure
    prbs = data_slice[0]
    output_drl_sched = data_slice[1]
    output_autoencoder = data_slice[2]
    
    ##^^ picking up the individual metrics of the autoencoder
    slice_outae_a = []
    slice_outae_b = []
    slice_outae_c = []

    for k in range (len(output_autoencoder)):
        ou_ae=unstack(output_autoencoder[k])
        for j in range(0, len(ou_ae)):
            if j == 0:
                slice_outae_a.append(ou_ae[j].item())
            if j == 1:
                slice_outae_b.append(ou_ae[j].item())
            if j == 2:
                slice_outae_c.append(ou_ae[j].item())

    slice_outae_a = np.array(slice_outae_a)
    slice_outae_b = np.array(slice_outae_b)
    slice_outae_c = np.array(slice_outae_c)

    ##^^ value of samples
    x=np.array(range(0,len(output_drl_sched)))   

    ##^^ picking up all the individual shap values
    shap_oae_a_col,shap_oae_b_col,shap_oae_c_col = getSHAPExplanationsPerMetric(exp_drl_shap_values)

    curr_sl_prbs = []
    curr_sl_sched = []

    for j in range(len(prbs)):
        curr_sl_prbs.append(prbs[j][slice_id])
        curr_sl_sched.append(output_drl_sched[j][slice_id])
    curr_sl_sched = np.array(curr_sl_sched)
    curr_sl_prbs = np.array(curr_sl_prbs)

    # print(len(x),len(prbs),len(output_drl_sched),len(output_autoencoder),len(shap_oae_a_col),len(slice_outae_a))
    # print(x.shape,curr_sl_prbs.shape,curr_sl_sched.shape,slice_outae_a.shape,shap_oae_a_col.shape)

    ##^^ SHAP-based plot
    titlename="Experiment "+str(exp)+" Slice ID "+str(slice_id)
    fig, axs = plt.subplots(5, sharex=True)
    fig.suptitle(titlename)

    axs[0].scatter(x, curr_sl_sched)
    axs[1].scatter(x, curr_sl_prbs)
    axs[2].scatter(x, slice_outae_a, c=shap_oae_a_col, cmap = 'coolwarm')
    axs[3].scatter(x, slice_outae_b, c=shap_oae_b_col, cmap = 'coolwarm')
    axs[4].scatter(x, slice_outae_c, c=shap_oae_c_col, cmap = 'coolwarm')

    axs[0].set(ylabel='Sched. Pol.')
    axs[0].set_ylim([0, 2])
    axs[1].set(ylabel='PRBs')
    axs[1].yaxis.set_label_position('right')
    axs[2].set(ylabel='O_AE_0')
    axs[3].set(ylabel='O_AE_1')
    axs[3].yaxis.set_label_position('right')
    axs[4].set(ylabel='O_AE_2')

    plt.xlabel("Samples")
    plt.tight_layout()

    if SavetoFile:
        outdir="../results/motivation-results/exp-"+str(exp)+"/"
        pathExist = os.path.exists(outdir)
        if not pathExist:
            os.makedirs(outdir)
        outfile=outdir+"sl-"+str(slice_id)+output_path_suffix+".png"
        plt.savefig(outfile,dpi=300)

    if Plot:
        plt.show()
    plt.close()
