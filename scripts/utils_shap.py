"""
SHAP 

10/12/2023
_______________
    
Summary: The script provides functions to perform SHAP on the DRL agents

DISCLAIMER: THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING 
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""

####^^ IMPORTS

##^^ importing generic project functions
from utils_generic_functions import *

##^^ matplotlib and seaborn
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import seaborn as sns
import tikzplotlib

##^^ relevant to shap
import xgboost
import shap

##^^ distances
from scipy.spatial import distance
from scipy.stats import wasserstein_distance


##^^ settings
np.set_printoptions(precision=3)

##^^ structures

class policyChangeStruct:
    __slots__ = ["prevPRBval", "prevPRBshap", "outAEAval", "outAEAshap","outAEBval", "outAEBshap","outAECval", "outAECshap"]
    def __init__(self,prevPRBval,prevPRBshap,outAEAval,outAEAshap,outAEBval,outAEBshap,outAECval,outAECshap):
        self.prevPRBval=prevPRBval
        self.prevPRBshap=prevPRBshap
        self.outAEAval=outAEAval
        self.outAEAshap=outAEAshap
        self.outAEBval=outAEBval
        self.outAEBshap=outAEBshap
        self.outAECval=outAECval
        self.outAECshap=outAECshap
    def __repr__(self):
        return f"PRBs (val shap):{self.prevPRBval} {self.prevPRBshap}\
                \nOU_AE_A (val shap): {self.outAEAval} {self.outAEAshap}\
                \nOU_AE_B (val shap):{self.outAEBval} {self.outAEBshap}\
                \nOU_AE_C (val shap):{self.outAECval} {self.outAECshap}"


####^^ Specific Functions

def buildMatrixInputOutputDRLAgent(slice_id,output_autoencoder,prbs,output_drl_sched):
    """
    Prepares the data for being processed by the Decision Tree and SHAP
    
    Parameters
    _______________
    
    output_autoencoder: result to be analyzed -> 1x3 
    prbs: PRBs allocated in the previous round
    output_drl_sched: policy that the DRL agent chooses

    Output
    _______________

    X: all the inputs of the DRL agent zipped
    y: the policy that is the output of the DRL agent
    slice_prevprbs: the array with the allocation in the previous slot of PRBs (useful for plots)
    slice_outae_a: array with metric a
    slice_outae_b: array with metric b
    slice_outae_c: array with metric c 

    """

    ##^^setting the metrics from the autoencoder right
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


    # print(f'{len(prbs)} {len(slice_outae_a)} {len(slice_outae_b)} {len(slice_outae_c)}')

    #^ retrieving the past prb allocation
    slice_prevprbs=[]
    
    #^ creating the output for the decision tree
    y=[]
    for k in range(1,len(prbs)):
        slice_prevprbs.append(prbs[k-1][slice_id])
        y.append(output_drl_sched[k][slice_id])

    #^ creating the input for the decision tree
    X=list(zip(slice_prevprbs, slice_outae_a[1:], slice_outae_b[1:], slice_outae_c[1:]))

    # print(X)

    X=np.array(X,dtype="object")

    # print(np.cov(X.T))

    # and mean centered
    X.mean(0)

    return X,y,slice_prevprbs,slice_outae_a,slice_outae_b,slice_outae_c

def buildMatrixInputOutputDRLAgentOnlyOutputAE(slice_id,output_autoencoder,prbs,output_drl_sched):
    """
    Prepares the data for being processed by the Decision Tree and SHAP without considering the
    previous PRBs as part of the process
    
    Parameters
    _______________
    
    output_autoencoder: result to be analyzed -> 1x3 
    prbs: PRBs allocated in the previous round
    output_drl_sched: policy that the DRL agent chooses

    Output
    _______________

    X: all the inputs of the DRL agent zipped
    y: the policy that is the output of the DRL agent
    slice_prevprbs: the array with the allocation in the previous slot of PRBs (useful for plots)
    slice_outae_a: array with metric a
    slice_outae_b: array with metric b
    slice_outae_c: array with metric c 

    """

    ##^^setting the metrics from the autoencoder right
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


    # print(f'{len(prbs)} {len(slice_outae_a)} {len(slice_outae_b)} {len(slice_outae_c)}')

    #^ retrieving the past prb allocation
    slice_prevprbs=[]
    
    #^ creating the output for the decision tree
    y=[]
    for k in range(1,len(prbs)):
        slice_prevprbs.append(prbs[k-1][slice_id])
        y.append(output_drl_sched[k][slice_id])

    #^ creating the input for the decision tree
    X=list(zip(slice_outae_a[1:], slice_outae_b[1:], slice_outae_c[1:]))

    # print(X)

    X=np.array(X,dtype="object")

    # print(np.cov(X.T))

    # and mean centered
    X.mean(0)

    return X,y,slice_prevprbs,slice_outae_a,slice_outae_b,slice_outae_c

def printMatrixInputOutputDRLAgent(X,y):
    """
    Prints the data that will be processed by the Decision Tree and SHAP
    
    Parameters
    _______________
    
    X: the input of the DRL agent zipped (PRBs, ae_metric_a, ae_metric_b, ae_metric_c)
    y: the output of the DRL agent, the policy choosen (0,1,2)

    Output
    _______________


    """
    print("---------------------")
    for k in range(len(X)):
        print(f'{X[k]} | {y[k]}')
    print("---------------------")

def explainDRLAgent(X,y,model,Plot=True):
    """
    Performs the correlation analysis
    
    Parameters
    _______________
    
    X: the input of the DRL agent zipped (PRBs, ae_metric_a, ae_metric_b, ae_metric_c)
    y: the output of the DRL agent, the policy choosen (0,1,2)
    Plot: if "True" creates the plot

    Output
    _______________

    shap_values: this tells how much relevant one metric was

    """
    # creating the DMatrix with XGBoost
    Xd = xgboost.DMatrix(X, label=y)
    # make sure the SHAP values add up to marginal predictions
    pred = model.predict(Xd, output_margin=True)
    # print("Model error =", np.linalg.norm(y-model.predict(Xd)))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xd)
    # make sure the SHAP values add up to marginal predictions => lower than 1e-05 is ok
    # mg_pred = np.abs(shap_values.sum(1) + explainer.expected_value - pred).max()
    # print(f'Marginal prediction value (<1e-05): {mg_pred}')

    if Plot:
        ##^^ using xgboost built-in plot to show feature importance
        xgboost.plot_importance(model)
        plt.close()

        ##^^ Beeswarm plot
        ## see: https://towardsdatascience.com/introduction-to-shap-with-python-d27edc23c454
        shap.summary_plot(shap_values, X, feature_names=['PRBs_IN', 'O_AE_0', 'O_AE_1', 'O_AE_2'])
        plt.close()

        # ##^^ Decision plot
        # ## see: https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/decision_plot.html
        # expected_value = explainer.expected_value
        # shap_array = explainer.shap_values(X)

        # shap.decision_plot(expected_value, shap_array[100:140],feature_names=['PRBs_IN', 'O_AE_0', 'O_AE_1', 'O_AE_2'])

    return shap_values

def getSHAPExplanations(shap_values):
    """
    Plots the values over time and colors them according to SHAP explanations
    
    Parameters
    _______________
    
    shap_values: the explanations

    Output
    _______________
    shap_prbs_col: PRBs
    shap_oae_a_col: output autoencoder, metric a
    shap_oae_b_col: output autoencoder, metric b
    shap_oae_c_col: output autoencoder, metric c

    """
    shap_prbs_col= [item[0] for item in shap_values]
    shap_oae_a_col= [item[1] for item in shap_values]
    shap_oae_b_col= [item[2] for item in shap_values]
    shap_oae_c_col= [item[3] for item in shap_values]

    return shap_prbs_col,shap_oae_a_col,shap_oae_b_col,shap_oae_c_col

def getSHAPExplanationsOnlyOutputAE(shap_values):
    """
    Plots the values over time and colors them according to SHAP explanations
    without considering the previous PRBs as part of the process
    
    Parameters
    _______________
    
    shap_values: the explanations

    Output
    _______________
    shap_oae_a_col: output autoencoder, metric a
    shap_oae_b_col: output autoencoder, metric b
    shap_oae_c_col: output autoencoder, metric c

    """
    shap_oae_a_col= [item[0] for item in shap_values]
    shap_oae_b_col= [item[1] for item in shap_values]
    shap_oae_c_col= [item[2] for item in shap_values]

    return shap_oae_a_col,shap_oae_b_col,shap_oae_c_col


def printSHAPExplanations(y,shap_values,slice_prevprbs,slice_outae_a,slice_outae_b,slice_outae_c,exp,slice_id,zoom_range,ZoomIn=True):
    """
    Plots the values over time and colors them according to SHAP explanations
    
    Parameters
    _______________
    
    y: the output of the DRL agent, the policy choosen (0,1,2)
    shap_values: the explanations
    slice_prevprbs: the array with the allocation in the previous slot of PRBs (useful for plots)
    slice_outae_a: array with metric a
    slice_outae_b: array with metric b
    slice_outae_c: array with metric c
    exp: current experiment
    slice_id: current slice id in use
    ZoomIn: allows to zoom in and make a smaller plot

    Output
    _______________


    """
    x=list(range(0,len(y))) # recall that takes already out the first value that is "the start"

    ##^^ picking up all the individual shap values
    shap_prbs_col, shap_oae_a_col,shap_oae_b_col,shap_oae_c_col = getSHAPExplanations(shap_values)
    

    if ZoomIn:
        shap_prbs_col= shap_prbs_col[zoom_range]
        shap_oae_a_col= shap_oae_a_col[zoom_range]
        shap_oae_b_col= shap_oae_b_col[zoom_range]
        shap_oae_c_col= shap_oae_c_col[zoom_range]

        plot_prevprbs = slice_prevprbs[zoom_range]
        plot_outae_a = slice_outae_a[zoom_range]
        plot_outae_b = slice_outae_b[zoom_range]
        plot_outae_c =  slice_outae_c[zoom_range]
        x=x[zoom_range]
        y=y[zoom_range]

        # print(f'TIMESTEP SCHED_POLICY || VAL_PREV_PRBs|SHAP_PREV_PRBs VAL_OUAE_A|SHAP_OUAE_A VAL_OUAE_B|SHAP_OUAE_B VAL_OUAE_C|SHAP_OUAE_C')
        # for k in (range(len(x))):
        #     print(f'{x[k]} {y[k]} || {plot_prevprbs[k]}|{shap_prbs_col[k]} {plot_outae_a[k]}|{shap_oae_a_col[k]} {plot_outae_b[k]}|{shap_oae_b_col[k]} {plot_outae_c[k]}|{shap_oae_c_col[k]}')
    else:
        ##^^ defining the actual data
        plot_prevprbs = slice_prevprbs
        plot_outae_a = slice_outae_a[1:]
        plot_outae_b = slice_outae_b[1:]
        plot_outae_c =  slice_outae_c[1:]

    print(f'TIMESTEP SCHED_POLICY || VAL_PREV_PRBs|SHAP_PREV_PRBs VAL_OUAE_A|SHAP_OUAE_A VAL_OUAE_B|SHAP_OUAE_B VAL_OUAE_C|SHAP_OUAE_C')
    for k in (range(len(x))):
        print(f'{x[k]} {y[k]} || {plot_prevprbs[k]}|{shap_prbs_col[k]} {plot_outae_a[k]}|{shap_oae_a_col[k]} {plot_outae_b[k]}|{shap_oae_b_col[k]} {plot_outae_c[k]}|{shap_oae_c_col[k]}')

def printSHAPExplanationsOnlyOutputAE(y,shap_values,slice_prevprbs,slice_outae_a,slice_outae_b,slice_outae_c,exp,slice_id,zoom_range,ZoomIn=True):
    """
    Plots the values over time and colors them according to SHAP explanations
    without considering the previous PRBs as part of the process
    Explanations come from the decision trees
    
    Parameters
    _______________
    
    y: the output of the DRL agent, the policy choosen (0,1,2)
    shap_values: the explanations
    slice_prevprbs: the array with the allocation in the previous slot of PRBs (useful for plots)
    slice_outae_a: array with metric a
    slice_outae_b: array with metric b
    slice_outae_c: array with metric c
    exp: current experiment
    slice_id: current slice id in use
    ZoomIn: allows to zoom in and make a smaller plot

    Output
    _______________


    """
    x=list(range(0,len(y))) # recall that takes already out the first value that is "the start"

    ##^^ picking up all the individual shap values
    shap_oae_a_col,shap_oae_b_col,shap_oae_c_col = getSHAPExplanationsOnlyOutputAE(shap_values)
    

    if ZoomIn:
        shap_oae_a_col= shap_oae_a_col[zoom_range]
        shap_oae_b_col= shap_oae_b_col[zoom_range]
        shap_oae_c_col= shap_oae_c_col[zoom_range]

        plot_prevprbs = slice_prevprbs[zoom_range]
        plot_outae_a = slice_outae_a[zoom_range]
        plot_outae_b = slice_outae_b[zoom_range]
        plot_outae_c =  slice_outae_c[zoom_range]
        x=x[zoom_range]
        y=y[zoom_range]

        # print(f'TIMESTEP SCHED_POLICY || VAL_PREV_PRBs|SHAP_PREV_PRBs VAL_OUAE_A|SHAP_OUAE_A VAL_OUAE_B|SHAP_OUAE_B VAL_OUAE_C|SHAP_OUAE_C')
        # for k in (range(len(x))):
        #     print(f'{x[k]} {y[k]} || {plot_prevprbs[k]}|{shap_prbs_col[k]} {plot_outae_a[k]}|{shap_oae_a_col[k]} {plot_outae_b[k]}|{shap_oae_b_col[k]} {plot_outae_c[k]}|{shap_oae_c_col[k]}')
    else:
        ##^^ defining the actual data
        plot_prevprbs = slice_prevprbs
        plot_outae_a = slice_outae_a[1:]
        plot_outae_b = slice_outae_b[1:]
        plot_outae_c =  slice_outae_c[1:]

    print(f'TIMESTEP SCHED_POLICY || VAL_PREV_PRBs|SHAP_PREV_PRBs VAL_OUAE_A|SHAP_OUAE_A VAL_OUAE_B|SHAP_OUAE_B VAL_OUAE_C|SHAP_OUAE_C')
    for k in (range(len(x))):
        print(f'{x[k]} {y[k]} || {plot_prevprbs[k]} - {plot_outae_a[k]}|{shap_oae_a_col[k]} {plot_outae_b[k]}|{shap_oae_b_col[k]} {plot_outae_c[k]}|{shap_oae_c_col[k]}')

def plotExplanations(y,shap_values,slice_prevprbs,slice_outae_a,slice_outae_b,slice_outae_c,exp,slice_id,zoom_range,Plot=True,SavetoFile=False,ZoomIn=True):
    """
    Plots the values over time and colors them according to SHAP explanations
    
    Parameters
    _______________
    
    y: the output of the DRL agent, the policy choosen (0,1,2)
    shap_values: the explanations
    slice_prevprbs: the array with the allocation in the previous slot of PRBs (useful for plots)
    slice_outae_a: array with metric a
    slice_outae_b: array with metric b
    slice_outae_c: array with metric c
    exp: current experiment
    slice_id: current slice id in use
    Plot: shows the plots
    SavetoFile: saves to files all the plots
    ZoomIn: allows to zoom in and make a smaller plot

    Output
    _______________


    """
    x=list(range(0,len(y))) # recall that takes already out the first value that is "the start"

    ##^^ picking up all the individual shap values
    shap_prbs_col, shap_oae_a_col,shap_oae_b_col,shap_oae_c_col = getSHAPExplanations(shap_values)
    
    if ZoomIn:
        ## Note: sometimes when zooming there's one issue, i.e., no high relevant features are found thus the colors are only those of nonrelevant features
        shap_prbs_col= shap_prbs_col[zoom_range]
        shap_oae_a_col= shap_oae_a_col[zoom_range]
        shap_oae_b_col= shap_oae_b_col[zoom_range]
        shap_oae_c_col= shap_oae_c_col[zoom_range]

        plot_prevprbs = slice_prevprbs[zoom_range]
        plot_outae_a = slice_outae_a[zoom_range]
        plot_outae_b = slice_outae_b[zoom_range]
        plot_outae_c =  slice_outae_c[zoom_range]
        x=x[zoom_range]
        y=y[zoom_range]
    else:
        ##^^ defining the actual data (recall that there's one element less for PRBs as we take that of t-1)
        plot_prevprbs = slice_prevprbs
        plot_outae_a = slice_outae_a[1:]
        plot_outae_b = slice_outae_b[1:]
        plot_outae_c =  slice_outae_c[1:]

    # print(f'{len(plot_prevprbs)} {len(plot_outae_a)} {len(plot_outae_b)} {len(plot_outae_c)} | {len(shap_prbs_col)} {len(shap_oae_a_col)} {len(shap_oae_b_col)} {len(shap_oae_c_col)}')

    ##^^ PLOT TYPE - one slice, all the variables together

    ## SHAP-based plot
    titlename="Experiment "+str(exp)+" Slice ID "+str(slice_id)
    fig, axs = plt.subplots(5, sharex=True)
    fig.suptitle(titlename)

    # norm = plt.Normalize(vmin=-1, vmax=1) # define color scala between -1 and +1 (as SHAP values)

    axs[0].scatter(x, y)
    axs[1].scatter(x, plot_prevprbs, c=shap_prbs_col, cmap = 'coolwarm')
    axs[2].scatter(x, plot_outae_a, c=shap_oae_a_col, cmap = 'coolwarm')
    axs[3].scatter(x, plot_outae_b, c=shap_oae_b_col, cmap = 'coolwarm')
    axs[4].scatter(x, plot_outae_c, c=shap_oae_c_col, cmap = 'coolwarm')

    axs[0].set(ylabel='Sched. Pol.')
    axs[1].set(ylabel='Past PRBs')
    axs[1].yaxis.set_label_position('right')
    axs[2].set(ylabel='O_AE_0')
    axs[3].set(ylabel='O_AE_1')
    axs[3].yaxis.set_label_position('right')
    axs[4].set(ylabel='O_AE_2')

    plt.xlabel("Samples")
    plt.tight_layout()
    
    if SavetoFile:
        outdir="../results/exp-"+str(exp)+"/"
        pathExist = os.path.exists(outdir)
        if not pathExist:
            os.makedirs(outdir)
        if ZoomIn:
            outfile=outdir+"sl-"+str(slice_id)+"-dtall-zoom"+str(zoom_range)[6:-7]+".png"
        else:  
            outfile=outdir+"sl-"+str(slice_id)+"-dtall.png"
        plt.savefig(outfile,dpi=300)

    if Plot:
        plt.show()
    plt.close()

def plotExplanationsOnlyOutputAE(y,shap_values,slice_prevprbs,slice_outae_a,slice_outae_b,slice_outae_c,exp,slice_id,zoom_range,Plot=True,SavetoFile=False,ZoomIn=True):
    """
    Plots the values over time and colors them according to SHAP explanations
    without considering the previous PRBs as part of the process
    Explanations come from the decision trees

    Parameters
    _______________
    
    y: the output of the DRL agent, the policy choosen (0,1,2)
    shap_values: the explanations
    slice_prevprbs: the array with the allocation in the previous slot of PRBs (useful for plots)
    slice_outae_a: array with metric a
    slice_outae_b: array with metric b
    slice_outae_c: array with metric c
    exp: current experiment
    slice_id: current slice id in use
    Plot: shows the plots
    SavetoFile: saves to files all the plots
    ZoomIn: allows to zoom in and make a smaller plot

    Output
    _______________


    """
    x=list(range(0,len(y))) # recall that takes already out the first value that is "the start"

    ##^^ picking up all the individual shap values
    shap_oae_a_col,shap_oae_b_col,shap_oae_c_col = getSHAPExplanationsOnlyOutputAE(shap_values)
    
    if ZoomIn:
        ## Note: sometimes when zooming there's one issue, i.e., no high relevant features are found thus the colors are only those of nonrelevant features
        shap_oae_a_col= shap_oae_a_col[zoom_range]
        shap_oae_b_col= shap_oae_b_col[zoom_range]
        shap_oae_c_col= shap_oae_c_col[zoom_range]

        ## reshaping to get rid of the first value
        slice_outae_a = slice_outae_a[1:]
        slice_outae_b = slice_outae_b[1:]
        slice_outae_c = slice_outae_c[1:]

        plot_prevprbs = slice_prevprbs[zoom_range]
        plot_outae_a = slice_outae_a[zoom_range]
        plot_outae_b = slice_outae_b[zoom_range]
        plot_outae_c =  slice_outae_c[zoom_range]
        x=x[zoom_range]
        y=y[zoom_range]
    else:
        ##^^ defining the actual data (recall that there's one element less for PRBs as we take that of t-1)
        plot_prevprbs = slice_prevprbs
        plot_outae_a = slice_outae_a[1:]
        plot_outae_b = slice_outae_b[1:]
        plot_outae_c =  slice_outae_c[1:]

    # print(f'{len(plot_prevprbs)} {len(plot_outae_a)} {len(plot_outae_b)} {len(plot_outae_c)} | {len(shap_prbs_col)} {len(shap_oae_a_col)} {len(shap_oae_b_col)} {len(shap_oae_c_col)}')

    ##^^ PLOT TYPE - one slice, all the variables together

    ## SHAP-based plot
    titlename="Experiment "+str(exp)+" Slice ID "+str(slice_id)
    fig, axs = plt.subplots(5, sharex=True)
    fig.suptitle(titlename)

    # norm = plt.Normalize(vmin=-1, vmax=1) # define color scala between -1 and +1 (as SHAP values)

    axs[0].scatter(x, y)
    axs[1].scatter(x, plot_prevprbs)
    axs[2].scatter(x, plot_outae_a, c=shap_oae_a_col, cmap = 'coolwarm', norm=norm)
    axs[3].scatter(x, plot_outae_b, c=shap_oae_b_col, cmap = 'coolwarm', norm=norm)
    axs[4].scatter(x, plot_outae_c, c=shap_oae_c_col, cmap = 'coolwarm', norm=norm)

    axs[0].set(ylabel='Sched. Pol.')
    axs[1].set(ylabel='Past PRBs')
    axs[1].yaxis.set_label_position('right')
    axs[2].set(ylabel='O_AE_0')
    axs[3].set(ylabel='O_AE_1')
    axs[3].yaxis.set_label_position('right')
    axs[4].set(ylabel='O_AE_2')

    plt.xlabel("Samples")
    plt.tight_layout()
    
    if SavetoFile:
        outdir="../results/exp-"+str(exp)+"/"
        pathExist = os.path.exists(outdir)
        if not pathExist:
            os.makedirs(outdir)
        if ZoomIn:
            outfile=outdir+"sl-"+str(slice_id)+"-dtoutae-zoom"+str(zoom_range)[6:-7]+".png"
        else:  
            outfile=outdir+"sl-"+str(slice_id)+"-dtoutae.png"
        plt.savefig(outfile,dpi=300)

    if Plot:
        plt.show()
    plt.close()

def plotExplanationsDRLAgentOnlyOutputAE(y,shap_values,slice_prevprbs,slice_outae_a,slice_outae_b,slice_outae_c,exp,slice_id,zoom_range,Plot=True,SavetoFile=False,ZoomIn=True):
    """
    Plots the values over time and colors them according to SHAP explanations
    without considering the previous PRBs as part of the process
    Explanations come from the DRL agend directly
    
    Parameters
    _______________
    
    y: the output of the DRL agent, the policy choosen (0,1,2)
    shap_values: the explanations
    slice_prevprbs: the array with the allocation in the previous slot of PRBs (useful for plots)
    slice_outae_a: array with metric a
    slice_outae_b: array with metric b
    slice_outae_c: array with metric c
    exp: current experiment
    slice_id: current slice id in use
    Plot: shows the plots
    SavetoFile: saves to files all the plots
    ZoomIn: allows to zoom in and make a smaller plot

    Output
    _______________


    """
    x=list(range(0,len(y))) # recall that takes already out the first value that is "the start"

    if ZoomIn:
        ##^^ picking up all the individual shap values
        shap_oae_a_col = shap_values[1:,0]
        shap_oae_b_col = shap_values[1:,1]
        shap_oae_c_col = shap_values[1:,2]

        ## reshaping to get rid of the first value
        # slice_prevprbs = slice_prevprbs[1:]
        slice_outae_a = slice_outae_a[1:]
        slice_outae_b = slice_outae_b[1:]
        slice_outae_c = slice_outae_c[1:]

        # shap_oae_a_col = shap_oae_a_col[zoom_range]
        # shap_oae_b_col = shap_oae_b_col[zoom_range]
        # shap_oae_c_col =  shap_oae_c_col[zoom_range]

        plot_prevprbs = slice_prevprbs[zoom_range]
        plot_outae_a = slice_outae_a[zoom_range]
        plot_outae_b = slice_outae_b[zoom_range]
        plot_outae_c =  slice_outae_c[zoom_range]
        x=x[zoom_range]
        y=y[zoom_range]
    else:
        ##^^ picking up all the individual shap values
        shap_oae_a_col = shap_values[1:,0]
        shap_oae_b_col = shap_values[1:,1]
        shap_oae_c_col = shap_values[1:,2]

        ##^^ defining the actual data (recall that there's one element less for PRBs as we take that of t-1)
        plot_prevprbs = slice_prevprbs
        plot_outae_a = slice_outae_a[1:]
        plot_outae_b = slice_outae_b[1:]
        plot_outae_c =  slice_outae_c[1:]

    # print(f'{len(plot_prevprbs)} {len(plot_outae_a)} {len(plot_outae_b)} {len(plot_outae_c)} | {len(shap_oae_a_col)} {len(shap_oae_b_col)} {len(shap_oae_c_col)}')

    ##^^ PLOT TYPE - one slice, all the variables together

    ## SHAP-based plot
    titlename="Experiment "+str(exp)+" Slice ID "+str(slice_id)
    fig, axs = plt.subplots(5, sharex=True)
    fig.suptitle(titlename)

    # norm = #plt.Normalize(vmin=-1, vmax=1) # define color scala between -1 and +1 (as SHAP values)

    print(len(x),len(plot_outae_a),len(shap_oae_a_col))

    axs[0].scatter(x, y)
    axs[1].scatter(x, plot_prevprbs)
    # axs[2].scatter(x, plot_outae_a, c=shap_oae_a_col, cmap = 'coolwarm', norm=norm)
    # axs[3].scatter(x, plot_outae_b, c=shap_oae_b_col, cmap = 'coolwarm', norm=norm)
    # axs[4].scatter(x, plot_outae_c, c=shap_oae_c_col, cmap = 'coolwarm', norm=norm)
    axs[2].scatter(x, plot_outae_a, c=shap_oae_a_col, cmap = 'coolwarm')
    axs[3].scatter(x, plot_outae_b, c=shap_oae_b_col, cmap = 'coolwarm')
    axs[4].scatter(x, plot_outae_c, c=shap_oae_c_col, cmap = 'coolwarm')

    axs[0].set(ylabel='Sched. Pol.')
    axs[1].set(ylabel='PRBs')
    axs[1].yaxis.set_label_position('right')
    axs[2].set(ylabel='O_AE_0')
    axs[3].set(ylabel='O_AE_1')
    axs[3].yaxis.set_label_position('right')
    axs[4].set(ylabel='O_AE_2')

    plt.xlabel("Samples")
    plt.tight_layout()
    
    if SavetoFile:
        outdir="../results/exp-"+str(exp)+"/"
        pathExist = os.path.exists(outdir)
        if not pathExist:
            os.makedirs(outdir)
        if ZoomIn:
            outfile=outdir+"sl-"+str(slice_id)+"-drlagent-zoom"+str(zoom_range)[6:-7]+".png"
        else:  
            outfile=outdir+"sl-"+str(slice_id)+"-drlagent.png"
        plt.savefig(outfile,dpi=300)

    if Plot:
        # plt.show()

        outdir="../results/exp-"+str(exp)+"/"
        pathExist = os.path.exists(outdir)
        if not pathExist:
            os.makedirs(outdir)
        if ZoomIn:
            outfile=outdir+"sl-"+str(slice_id)+"-drlagent-zoom"+str(zoom_range)[6:-7]+".tex"
        else:  
            outfile=outdir+"sl-"+str(slice_id)+"-drlagent.tex"
        tikzplotlib.save(outfile)
    plt.close()

def plotCorrelationSHAPValue(shap_metric,value_metric,exp,slice_id,metric_id,print_matrix=False,reduce_labels=False):
    """
    Bins and plots the correlation analysis. Binning occurs with the Freeman-Diaconis rule (reference: https://stats.stackexchange.com/a/862)

    Parameters
    _______________
    
    shap_metric: shap values of a given metric (np.array)
    value_metric: actual values of a given metric (np.array)
    exp: number of experiment
    slice_id: index of the slice ID
    metric_id: current metric (e.g., previous PRBs or output autoencoder A)
    print_matrix: if "True" prints the correlation matrix to file in a specific folder
    reduce_labels: if "True" reduces the number of labels

    Output
    _______________
    none
    
    """
    # if metric_id==0:
    iqr_shap = float(abs(np.percentile(shap_metric, 25)-np.percentile(shap_metric, 75)))
    iqr_value = float(abs(np.percentile(value_metric, 25)-np.percentile(value_metric, 75)))

    # print(iqr_shap,iqr_value)

    h_shap = (2 * (iqr_shap) )/ pow(len(shap_metric),1/3)
    h_value = (2 * (iqr_value) )/ pow(len(value_metric),1/3)

    # print(h_shap,h_value)

    num_bin_shap = int(abs((max(shap_metric)-min(shap_metric)))/h_shap)
    num_bin_value = int(abs((max(value_metric)-min(value_metric)))/h_value)

    # print(num_bin_shap,num_bin_value)

    bins_shap=[]
    bins_value=[]

    for z in range(0,num_bin_shap):
        bins_shap.append(round(min(shap_metric)+z*h_shap,3))

    for z in range(0,num_bin_value):
        bins_value.append(round(min(value_metric)+z*h_value,3))

    print(f'Mean (wid | num): {h_shap} | {num_bin_shap}')
    print(f'Stdev (wid | num): {h_value} | {num_bin_value}')

    # matrix of correlation
    matrix_correlation=np.zeros((num_bin_shap+1, num_bin_value+1))

    from scipy.stats import binned_statistic

    digitized_shap = np.digitize(shap_metric, bins_shap)
    digitized_value = np.digitize(value_metric, bins_value)
    # digitized_shap = binned_statistic(shap_metric, shap_metric, bins=bins_shap)[2]
    # digitized_value = binned_statistic(value_metric, value_metric, bins=bins_value)[2]
    
    # filling-up the matrix
    for j in range(0,len(shap_metric)):
        # print(f'{j} {correlation_mean[j]} {digitized_shap[j]} -- {correlation_stdev[j]} {digitized_value[j]}')
        matrix_correlation[digitized_shap[j],digitized_value[j]]+=1

    # ## printing on file the matrix of correlation
    if print_matrix:
        filenameout="./var-files/corr_shapvalues_exp-"+str(exp)+"_sliceid-"+str(slice_id)+"_metricid-"+str(metric_id)+".dat"
        mat = np.matrix(matrix_correlation.T)
        with open(filenameout, 'w') as f_out:
            for line in mat:
                np.savetxt(f_out, line, fmt='%.2f')
        f_out.close()
    #     for r in range(0,len(digitized_shap)):
    #         for c in range(0,len(digitized_value)):
    #             print(f'{r} {c} {matrix_correlation[digitized_shap[r]][digitized_value[c]]}',file=f_out)
    #         print("",file=f_out)

    # plot
    fig, ax = plt.subplots()
    titlename="Experiment "+str(exp)+" Slice ID "+str(slice_id)+"_metricid-"+str(metric_id)
    plt.title(titlename)
    plt.xlabel("SHAP",fontsize=14)
    plt.xticks(range(0,len(bins_shap)), bins_shap, rotation ='vertical')
    plt.ylabel("Value",fontsize=14)
    plt.yticks(range(0,len(bins_value)), bins_value)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    # for mpl 3.3 and higher use
    cmap = cm.get_cmap("OrRd").copy()
    cmap.set_under(color='white') 

    im=ax.imshow(matrix_correlation.T, cmap=cmap,vmin=0.0000001)
    cbar=fig.colorbar(im, cax=cax, orientation='vertical')

    ## reduce the number of visible labels
    if reduce_labels:
        every_nth = 4
        for n, label in enumerate(ax.xaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)
        for n, label in enumerate(ax.yaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False) 
    plt.show()
    plt.close()


def findPolicyChange(y):
    """
    This function returns the time steps where there has been a policy change 
    
    Parameters
    _______________
    y: the policy that is the output of the DRL agent

    Output
    _______________
    pol_change_sl0: list of time instances when the policy has changed for slice 0
    pol_change_sl1: same as above, but for slice 1
    pol_change_sl2: same as above, but for slice 2
    
    """

    ##^^ aux structures
    prev_pol_sl0 = y[0][0]
    prev_pol_sl1 = y[0][1]
    prev_pol_sl2 = y[0][2]
    ##^^ return structures
    pol_change_sl0 = []
    pol_change_sl1 = []
    pol_change_sl2 = []

    for k in range(1,len(y)):
        if y[k][0] != prev_pol_sl0:
            pol_change_sl0.append(k)
        if y[k][1] != prev_pol_sl1:
            pol_change_sl1.append(k)
        if y[k][2] != prev_pol_sl2:
            pol_change_sl2.append(k)
    
    # print(pol_change_sl0,pol_change_sl1,pol_change_sl2)

    return pol_change_sl0, pol_change_sl1, pol_change_sl2

def buildPolicyChangeInfoAnalysis(timestep,obs_w,X,shap_values,y):
    """
    This function takes a look to the <window> of samples before a policy change 
    
    Parameters
    _______________

    timestep: time where the policy changes in at least one of the slices
    observation_window: number N of samples we pick to make the analysis (timestep - N)
    X: all the inputs of the DRL agent zipped
    shap_values: the explanations on the entire list of metrics

    Output
    _______________
    <policyChangeStruct> with 
    prevPRBs (value and shap)
    outAEA (value and shap)
    outAEB (value and shap)
    outAEC (value and shap)
    
    """
    # print(f'Time step of policy change: {timestep-1}')

    ##^^ structures to fill

    prevPRBval = []
    prevPRBshap = []
    outAEAval = []
    outAEAshap = []
    outAEBval = []
    outAEBshap = []
    outAECval = []
    outAECshap = []

    # for k in range(obs_w+1,0,-1):
    #     print(f'{timestep-k}: {y[timestep-k]} ',end='')
    # print()

    # looping from 
    for k in range(obs_w+1,0,-1):
        # print(f'{timestep-k}: {X[timestep-k]} ',end='')
        prevPRBval.append(X[timestep-k][0])
        outAEAval.append(X[timestep-k][1])
        outAEBval.append(X[timestep-k][2])
        outAECval.append(X[timestep-k][3])

        prevPRBshap.append(shap_values[timestep-k][0])
        outAEAshap.append(shap_values[timestep-k][1])
        outAEBshap.append(shap_values[timestep-k][2])
        outAECshap.append(shap_values[timestep-k][3])


    # print(prevPRBval,outAEAval,outAEBval,outAECval)
    # print(prevPRBshap,outAEAshap,outAEBshap,outAECshap)

    return policyChangeStruct(prevPRBval,prevPRBshap,outAEAval,outAEAshap,outAEBval,outAEBshap,outAECval,outAECshap)

def buildWassersteinDistanceMatrix(dt_shap_values,drl_shap_values,zoom_range):
    """
    This function compares the explanations obtained from the DRL agent directly and those obtained from the decision trees 
    
    Parameters
    _______________
    
    dt_shap_values: array with shap values of the decision tree
    drl_shap_values: array with shap values of the decision tree
    zoom_range

    Output
    _______________
    
    was_dist: np array
    
    """

    ##^^ picking up all the individual shap values
    dt_shap_oae_a,dt_shap_oae_b,dt_shap_oae_c = getSHAPExplanationsOnlyOutputAE(dt_shap_values)
    #^ transform into np arrays and normalize shap values in range [-1:1]
    dt_shap_oae_a = np.array(dt_shap_oae_a[zoom_range])
    dt_shap_oae_b = np.array(dt_shap_oae_b[zoom_range])
    dt_shap_oae_c = np.array(dt_shap_oae_c[zoom_range])

    drl_shap_oae_a = drl_shap_values[:,0]
    drl_shap_oae_b = drl_shap_values[:,1]
    drl_shap_oae_c = drl_shap_values[:,2]

    #^ sanitize possibe nan values
    drl_shap_oae_a = np.where(np.isnan(drl_shap_oae_a), 0.0, drl_shap_oae_a)
    drl_shap_oae_b = np.where(np.isnan(drl_shap_oae_b), 0.0, drl_shap_oae_b)
    drl_shap_oae_c = np.where(np.isnan(drl_shap_oae_c), 0.0, drl_shap_oae_c)

    drl_norm_oae_a = scalingMinusOnetoOne(drl_shap_oae_a)
    drl_norm_oae_b = scalingMinusOnetoOne(drl_shap_oae_b)
    drl_norm_oae_c = scalingMinusOnetoOne(drl_shap_oae_c)

    ##^ comparison
    # comparison_oae_a = [abs(i-j) for i,j in zip(dt_shap_oae_a,drl_norm_oae_a)]
    # comparison_oae_b = [abs(i-j) for i,j in zip(dt_shap_oae_b,drl_norm_oae_b)]
    # comparison_oae_c = [abs(i-j) for i,j in zip(dt_shap_oae_c,drl_norm_oae_c)]

    ##^ distance
    dst_oae_a = wasserstein_distance(dt_shap_oae_a,drl_norm_oae_a)
    dst_oae_b = wasserstein_distance(dt_shap_oae_b,drl_norm_oae_b)
    dst_oae_c = wasserstein_distance(dt_shap_oae_c,drl_norm_oae_c)

    was_dist = np.array([dst_oae_a,dst_oae_b,dst_oae_c])

    return was_dist

def plotWassersteinDistanceMatrix(was_matrix,outdir,outfilename):
    """
    This function compares the explanations obtained from the DRL agent directly and those obtained from the decision trees 
    
    Parameters
    _______________
    
    was_matrix: matrix with Wasserstein Distance (DT vs DRL) - [Rows (output ae: 0,1,2) | Cols: (slice id: 0,1,2)]

    Output
    _______________
    
    was_dist: np array
    
    """

    xticks = np.arange(0,n_outputs_ae)
    yticks = np.arange(0,n_outputs_ae)

    fig, ax = plt.subplots()
    cax = ax.matshow(was_matrix.T, cmap=plt.cm.Blues, vmin=0.0)
    # ax.set(xticks=range(0,len(n_outputs_ae)), xticklabels=xticks)
    # ax.set(yticks=range(0,len(n_outputs_ae)), yticklabels=yticks)
    ax.set_xlabel("Output AE",fontsize=12)
    ax.set_ylabel("Slice ID",fontsize=12)
    fig.colorbar(cax, ax=ax,  location = 'right',shrink=.99)

    outdir = outdir
    pathExist = os.path.exists(outdir)
    if not pathExist:
        os.makedirs(outdir)
    outfile=outdir+outfilename
    plt.savefig(outfile,dpi=300)
    plt.show()
    plt.close()

def printWassersteinDistanceMatrix(was_matrix):
    """
    This function prints the Wasserstein Distance Matrix
    
    Parameters
    _______________
    
    was_matrix: matrix with Wasserstein Distance (DT vs DRL) - [Rows (output ae: 0,1,2) | Cols: (slice id: 0,1,2)]

    Output
    _______________
    
    """

    print()
    print(f'---- Wasserstein Distance (DT vs DRL) ----')
    print(f'.... [Rows (output ae: 0,1,2) | Cols: (slice id: 0,1,2)] ....')
    for r in range(n_outputs_ae-1,-1,-1):
        for c in range(n_outputs_ae):
            print(was_matrix[r][c],end=' ')
        print()

def plotShapAndValues(shap_values,slice_outae_a,slice_outae_b,slice_outae_c,exp,slice_id,zoom_range):
    """
    Generates plots with SHAP values and actual metric values.
    
    Parameters
    _______________
    

    Output
    _______________


    """

    ##^^ picking up all the individual shap values
    shap_oae_a_col = shap_values[:,0]
    shap_oae_b_col = shap_values[:,1]
    shap_oae_c_col = shap_values[:,2]

    plot_outae_a = slice_outae_a[zoom_range]
    plot_outae_b = slice_outae_b[zoom_range]
    plot_outae_c =  slice_outae_c[zoom_range]

    ## SHAP-based plot
    titlename="Experiment "+str(exp)+" Slice ID "+str(slice_id)
    fig, axs = plt.subplots(3, sharex=True)
    fig.suptitle(titlename)

    print(len(plot_outae_a),len(shap_oae_a_col))

    axs[0].scatter(shap_oae_a_col, plot_outae_a)
    axs[1].scatter(shap_oae_b_col, plot_outae_b)
    axs[2].scatter(shap_oae_c_col, plot_outae_c)

    axs[0].set(ylabel='O_AE_0')
    axs[1].set(ylabel='O_AE_1')
    axs[1].yaxis.set_label_position('right')
    axs[2].set(ylabel='O_AE_2')

    plt.xlabel("SHAP_values")
    plt.tight_layout()
    plt.show()
