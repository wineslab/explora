"""
Analysis of SHAP explanations and DTs

10/12/2023
_______________
    
Summary: The script analyzes the explanations obtained for both the DRL agents
         offline with the two methods, SHAP and DT (via xgboost)

"""

####^^^^ Imports and settings
import xgboost
from sklearn.metrics import accuracy_score


##^^ project functions
from utils_shap_plotting import *
from utils_db_process import *

def regularizeShapes(output_ae, out_ae_list, prbs, output_drl_sched,len_drl_shap_val):
    """
    It may happen that there's a mismatch for the same experiment between 
    the size of the array of the SHAP values and the size of the array with actual values.

    This function aligns the two so that can be processed.

    """

    start_sample = len(output_ae)-len_drl_shap_val

    output_ae = output_ae[start_sample:]
    prbs = prbs[start_sample:]
    output_drl_sched = output_drl_sched[start_sample:]

    ou_ae_sl0 = out_ae_list[0][start_sample:]
    ou_ae_sl1 = out_ae_list[1][start_sample:]
    ou_ae_sl2 = out_ae_list[2][start_sample:]

    # print(len(ou_ae_sl0),len(ou_ae_sl1),len(ou_ae_sl2))
    out_ae_list =  [ou_ae_sl0,ou_ae_sl1,ou_ae_sl2]

    return output_ae, out_ae_list, prbs, output_drl_sched

def runExperiments(list_of_experiments, model_name, technique="SHAP"):
    """
    
    
    Parameters
    _______________
    
    list_of_experiments: self-explicative
    model_name: path of the DT pre-trained
    technique: could be "SHAP" or "DT"

    Output
    _______________

    None

    """

    ##^^ ANALYSIS for each experiment
    for exp in list_of_experiments:
        print(f'Current experiment under analysis: {exp} \n-------')

        dataset, datetime_arr, cut_th = processPklDataset(exp)

        # print(dataset)

        ##^^ Populating structures from the dataset
        output_ae, out_ae_list, prbs, output_drl_sched = populateInputOutputsDRLAgent(dataset, datetime_arr, cut_th)

        ##^^ Checking upon the technique:

        if technique == "DT":
            ##^^ loading the explanations from the single tree for all slices 
            xgb_model = pkl.load(open(model_name, "rb"))
            # print(f'Shape of: (1) DRL input: {np.array(output_ae).shape}; (2) DRL output: {np.array(output_drl_sched).shape}')

            #^ testing the DT trained offline on the experiments: need a DMatrix first, set input/output as np arrays and predict
            agent_input = np.array(output_ae)
            agent_output = np.array(output_drl_sched)
            Xd = xgboost.DMatrix(agent_input, label=agent_output) # xgboost needs DMatrix to make predictions
            agent_predict = xgb_model.predict(Xd)

            #^ model error (just euclidean distance, not fully accurate) and accuracy
            # print("Model error =", np.linalg.norm(agent_output-agent_predict))

            accuracy = accuracy_score(np.argmax(agent_output, axis=1), np.argmax(agent_predict, axis=1))
            print(f"Accuracy: {accuracy * 100:.2f}%")
            print()

        if technique == "SHAP":
            ##^^ Loading the SHAP explanations from the DRL agent
            drlagent_file = "../results/shap_explanations/shap-values/exp-"+str(exp)+"/explanation.npy"
            exp_drl_shap_values = np.load(drlagent_file)
            exp_drl_shap_values = np.squeeze(exp_drl_shap_values)# just reading this it will get a shape like (1, x, 9), hence squeeze to get (x, 9)
            #^ Our modified tool when feeds the SHAP Kernel Explainer does not output log odds, but rather sort of a distribution that always sums up to 0, 
            #  hence use a linear transformation to make it a prob. distribution
            normalizer = 1 / exp_drl_shap_values.sum()
            exp_drl_shap_values_a = exp_drl_shap_values * normalizer
            exp_drl_shap_values = exp_drl_shap_values_a
            # print(exp_drl_shap_values,exp_drl_shap_values.sum())

            if(len(output_ae)>len(exp_drl_shap_values)):
                # print("True")
                output_ae, out_ae_list, prbs, output_drl_sched = regularizeShapes(output_ae, out_ae_list, prbs, output_drl_sched,len(exp_drl_shap_values))

            ##^^ looping on the slices to plot a detailed view of the experiment with SHAP relevance values based coloring
            for slid, sl in enumerate(out_ae_list):
                print("-------------------------------------------------")
                print(f'---- Current Slice ID: {slid} ----')
                print("-------------------------------------------------")

                data_slice = [prbs,output_drl_sched,sl]
                drl_suffix = "-drl"
                plotExplanations(exp_drl_shap_values,data_slice,slid,exp,drl_suffix,Plot=plot_toggle,SavetoFile=save_to_file_toggle)


if __name__ == "__main__":

    ##^ choose the technique first
    technique = "SHAP" # "SHAP" or "DT"

    ##^^ list of experiments 
    #^ agent 1 - embb:
    list_exp_ag_embb = [46,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

    #^ agent 2 - urllc:
    list_exp_ag_urllc = [43,28,29,44,31,32,33,34,45,36,37,38,39,40,41,42]

    if technique == "SHAP":
        ##^^ define global variables
        plot_toggle = True
        save_to_file_toggle = True

        ##^^ RUN experiment choosing either agent embb or urllc
        ##   use None as second argument as there's no model to pass as for DTs (next)
        runExperiments(list_exp_ag_embb,None,technique) 
        # runExperiments(list_exp_ag_urllc,None,technique)

    if technique == "DT":
        ##^^ loading the DT trees trained on different agents
        dt_mod_ag_embb = "../results/pre-trained_dt_models/dt_agent_embb.sav"
        dt_mod_ag_urllc = "../results/pre-trained_dt_models/dt_agent_urllc.sav"

        ##^^ run experiment choosing either agent embb or urllc
        runExperiments(list_exp_ag_embb,dt_mod_ag_embb,technique)
        # runExperiments(list_exp_ag_urllc, dt_mod_ag_urllc,technique)

    print("-- Finished main --")
