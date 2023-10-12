"""
Offline training of one decision tree for all the slices
    
10/12/2023
_______________
    
Summary: This script trains one DT for the three slices to have a fair comparison 
	 between the explanations obtained from the the DRL agent with SHAP.

DISCLAIMER: THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING 
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""

##^^ Imports
import xgboost
from sklearn.metrics import accuracy_score

##^^ importing project functions
from utils_db_process import *

def createInputsfromDB(pkl_dataset_name):
    """
    Reads the db and generates np arrays with input and output of the agent
    Note that the input of the DRL agent is the output of the autoencoder
    
    Parameters
    _______________
    
    pkl_dataset_name: filename of the dataset to be used

    Output
    _______________
 
    drl_input: np array with inputs for the DT (the outputs of the autoencoder)
    drl_output: np array with the scheduling policy

    """
    ##^^ Read the dataset and the list of actions that are unknown in this dataset

    f = open(pkl_dataset_name, 'rb')
    dataset = pkl.load(f)

    prb_input, drl_input, drl_output = processDatasetAllSlices(dataset,ret_action=True) # to process only "output-ae <3x3> -> action"

    print("-- Load completed --")

    return drl_input, drl_output

def trainAndTestModel(drl_input,drl_output,test_set_perc):
    """
    Trains and tests the DT with coarse analysis on split train/test
    
    Parameters
    _______________
    
    drl_input: np array with inputs for the DT (the outputs of the autoencoder)
    drl_output: np array with the scheduling policy
    test_set_perc: scalar that is the percentage of test set [0.0-1.0]

    Output
    _______________

    target: np array with the ground truth
    pred: np array with the prediction of the DT
    error: np array with the error

    """
    ###^^^^ Starting training the DT for "output-ae <3x3> -> policy <3>"

    train_set_size = int(len(drl_input)*test_set_perc)
    test_set_size = int(len(drl_input)-train_set_size)

    input_train = drl_input[train_set_size:]
    output_train = drl_output[train_set_size:]

    input_test = drl_input[test_set_size:]
    output_test = drl_output[test_set_size:]

    print(f'{input_train.shape} {output_train.shape} | {input_test.shape} {output_test.shape}')

    ##^^ Train the decision tree
    # for parameters check:  https://xgboost.readthedocs.io/en/stable/parameter.html
    Xd = xgboost.DMatrix(input_train, label=output_train)
    Xt = xgboost.DMatrix(input_test, label=output_test)

    # for training parameters check: https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.train 
    model = xgboost.train({
        'eta':1, 'gamma': 0, 'max_depth':20, 'base_score': 0, "lambda": 0, "alpha": 0, 'scale_pos_weight': 1,
        'grow_policy': 'depthwise', 'sampling_method': 'gradient_based'
    }, Xd, 300)# 300 is the rounds of evaluation

    print(f'Best number of trees: {model.best_ntree_limit}')
    print("Model error =", np.linalg.norm(output_test-model.predict(Xt)))

    accuracy = accuracy_score(np.argmax(output_test, axis=1), np.argmax(model.predict(Xt), axis=1))
    print(f"Accuracy: {accuracy * 100:.2f}%")

def trainAndExportModel(drl_input, drl_output, suffix):
    """
    Trains the DT on the entire dataset and exports it
    
    Parameters
    _______________
    
    drl_input: np array with inputs for the DT (the outputs of the autoencoder)
    drl_output: np array with the scheduling policy
    suffix: type of agent for the filename

    Output
    _______________


    """

    #^^ Train the decision tree
    # to check parameters: https://xgboost.readthedocs.io/en/stable/parameter.html
    Xd = xgboost.DMatrix(drl_input, label=drl_output)

    # for training parameters check: https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.train 
    model = xgboost.train({
        'eta':1, 'gamma': 0, 'max_depth':25, 'base_score': 0, "lambda": 0, "alpha": 0, 'scale_pos_weight': 1,
        'grow_policy': 'depthwise', 'sampling_method': 'gradient_based'
    }, Xd, 300)# 300 is the rounds of evaluation

    ##^^ saving the model
    outdir="../results/pre-trained_dt_models/"
    pathExist = os.path.exists(outdir)
    if not pathExist:
        os.makedirs(outdir)
    model_name=outdir+"dt_agent_"+suffix+"-NEW"+".sav" #NOTE the NEW is just to avoid overriding
    print(model_name)
    pkl.dump(model, open(model_name, "wb"))

    print("-- Train and export complete --")


if __name__ == "__main__":
    
    agent = "urllc" # choose between "embb" and "urllc"

    if agent == "embb":
        pkl_dataset_name = "../data/offline-training/datasets/embb-dataset.pkl" # first agent: "embb"
    elif agent=="urllc":
        pkl_dataset_name = "../data/offline-training/datasets/urllc-dataset.pkl" # second agent: "urllc"
    else:
        print("Please choose among [embb, urllc].")

    ##^^ Process DB to pick input and output of the DRL agent

    drl_input, drl_output = createInputsfromDB(pkl_dataset_name)

    #^ this function is only to check the performance of the trained DT on part of the dataset
    test_set_perc = 0.2
    trainAndTestModel(drl_input, drl_output,test_set_perc)

    #^ this function is about storing the DT (trained now on the entire dataset)
    trainAndExportModel(drl_input, drl_output, agent)
