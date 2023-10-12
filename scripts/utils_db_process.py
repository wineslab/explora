"""
DB processing

10/12/2023
_______________
    
Summary: The script provides functions to process the pkl files 
         that are the generated with SCOPE
         
DISCLAIMER: THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING 
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""

####^^ IMPORTS

##^^ general
import pickle as pkl
from datetime import datetime, timedelta

##^^ importing generic project functions
from utils_generic_functions import *

####^^^^ Structures

#^ info on the format
metric_dict = {"dl_buffer": 1,
               "tx_brate": 2,
               "tx_pkts": 5,
               "ratio_req_granted": 3,
               "slice_id": 0,
               "slice_prb": 4}

class ueMetricInfo:
    
    __slots__ = ["slice_id", "dl_buffer", "tx_brate", "ratio_req_granted","slice_prb", "tx_pkts"]

    def __init__(self,slice_id,dl_buffer,tx_brate,ratio_req_granted,slice_prb,tx_pkts):
        self.slice_id = slice_id
        self.dl_buffer = dl_buffer
        self.tx_brate = tx_brate
        self.ratio_req_granted = ratio_req_granted
        self.slice_prb = slice_prb
        self.tx_pkts = tx_pkts

class ueKPI:
    
    __slots__ = ["dl_buffer", "tx_brate", "tx_pkts"]

    def __init__(self,dl_buffer,tx_brate,tx_pkts):
        self.dl_buffer = dl_buffer
        self.tx_brate = tx_brate
        self.tx_pkts = tx_pkts

class stateSliceKPI:
    
    __slots__ = ["avg_dl_buffer", "std_dl_buffer", "avg_tx_brate", "std_tx_brate", "avg_tx_pkts", "std_tx_pkts"]

    def __init__(self,avg_dl_buffer,std_dl_buffer,avg_tx_brate,std_tx_brate,avg_tx_pkts,std_tx_pkts):
        self.avg_dl_buffer = avg_dl_buffer
        self.std_dl_buffer = std_dl_buffer
        self.avg_tx_brate = avg_tx_brate
        self.std_tx_brate = std_tx_brate
        self.avg_tx_pkts = avg_tx_pkts
        self.std_tx_pkts = std_tx_pkts

####^^ Specific Functions

def processPklDataset(exp):
    """
    Prepares the data for being processed subsequently
    
    Parameters
    _______________
    
    exp: current experiment under analysis

    Output
    _______________

    dataset: self-explicative
    datetime_arr: list of timesteps after that users finally attached to the BS
    cut_th: returns the cut point after which samples are legit (see comments next)

    """

    ##^^ Reading the pickled dataset
    exp_dataset = "../data/motivation_main-results/exp"+str(exp)+ "/ai_log_dict.pkl"

    f = open(exp_dataset, 'rb')
    dataset = pkl.load(f)

    ##^^ Reading the timestamps and printing them into datetime format
    ##   ---
    ##   Recall that UEs initially have to attach, hence at the beginning the first samples
    ##   have to be discarded. Next, we check the time and use a threshold of 2 min to discard 
    ##   non useful samples
    datetime_arr = []
    for j in dataset.keys():
        # print(datetime.utcfromtimestamp(j/1000.0).strftime('%Y-%m-%d %H:%M:%S'))
        datetime_arr.append(datetime.utcfromtimestamp(j/1000.0))#.strftime('%Y-%m-%d %H:%M:%S'))

    time_threshold = timedelta(minutes=2, seconds=0)
    cut_th = 0 # this is telling at which point i
    for i in range(1,len(datetime_arr)):
        diff=datetime_arr[i]-datetime_arr[i-1]
        if diff > time_threshold:
            # print(f'Difference Exp {exp}: {diff} in {i}/{i-1}')
            cut_th = i
    # print(cut_th)
    datetime_arr=datetime_arr[cut_th+1:]

    return dataset, datetime_arr, cut_th

def populateInputOutputsDRLAgent(dataset,datetime_arr,cut_th):
    """
    
    
    Parameters
    _______________
    
    dataset: self-explicative
    datetime_arr: list of timesteps after that users finally attached to the BS
    cut_th: returns the cut point after which samples are legit (see comments next)

    Output
    _______________

    output_ae: output of the autoencoder (i.e., the input to the DRL agent)
    out_ae_list: above with slice granularity
    prbs: the slicing the agent selects
    output_drl_sched: the scheduling policy the agent selects

    """

    ##^^ general structures
    output_ae = []
    out_ae_list = []
    prbs = []
    output_drl_sched = []
    ##^^ creating the structures per slice
    ou_ae_sl0 = []
    ou_ae_sl1 = []
    ou_ae_sl2 = []

    for idx, item in enumerate(dataset.values()):
        if idx > cut_th:
            output_ae.append(item['AUTOENCODER_OUTPUT'])
            prbs.append(item['PRB'])
            output_drl_sched.append(item['SCHEDULING'])

    for j in range(len(datetime_arr)):
        ###^^^ Per slice 
        ##^^ populating the output of the autoencoder
        ou_ae_sl0.append(output_ae[j][0:3])
        ou_ae_sl1.append(output_ae[j][3:6])
        ou_ae_sl2.append(output_ae[j][6:9])

    out_ae_list =  [ou_ae_sl0,ou_ae_sl1,ou_ae_sl2]

    return np.array(output_ae), out_ae_list, np.array(prbs), np.array(output_drl_sched)

def processDatasetAllSlices(dataset,ret_action=False):
    """
    Prepares the data to train one decision tree for all the slices
    
    Parameters
    _______________
    
    dataset: processed dataset

    Output
    _______________

    drl_input = np array with inputs for all the slices 
    drl_output = np array with outputs for all the slices

    """
    ##^^ Populating temp structures from the dataset
    output_drl_sched = []
    output_drl_prb = []
    prev_action_sched = []
    prev_action_prb = []
    prbs = []
    slice_id = []
    output_ae = []

    for l in dataset:
        output_drl_sched.append(l['output_of_drl_sched'])
        output_drl_prb.append(l['output_of_drl_prb'])
        prev_action_sched.append(l['previous_action_sched'])
        prev_action_prb.append(l['previous_action_prb'])
        slice_id.append(l['slice_id'])
        output_ae.append(l['output_of_autoencoder'])

    # print(output_ae)

    ouae_slice_a = []
    ouae_slice_b = []
    ouae_slice_c = []
    prev_prb_slice_a = []
    prev_prb_slice_b = []
    prev_prb_slice_c = []
    oudrl_slice_a = []
    oudrl_slice_b = []
    oudrl_slice_c = []

    for k in range(len(output_drl_sched)):
        for curr_slice_id in range(3):
            if(slice_id[k]==0):
                ouae_slice_a.append(output_ae[k])
                prev_prb_slice_a.append(prev_action_prb[k])
                oudrl_slice_a.append(output_drl_sched[k])
            if(slice_id[k]==1):
                ouae_slice_b.append(output_ae[k])
                prev_prb_slice_b.append(prev_action_prb[k])
                oudrl_slice_b.append(output_drl_sched[k])
            if(slice_id[k]==2):
                ouae_slice_c.append(output_ae[k])
                prev_prb_slice_c.append(prev_action_prb[k])
                oudrl_slice_c.append(output_drl_sched[k])

    ouae_slice_a = np.squeeze(np.array(ouae_slice_a))
    ouae_slice_b = np.squeeze(np.array(ouae_slice_b))
    ouae_slice_c = np.squeeze(np.array(ouae_slice_c))

    ##^^ here we get arrays of shape (x,3)/(x,4) for the input of the DRL agent - depending on the type of agent
    drl_inp_slice_a = stackOutputAE(ouae_slice_a) #stackOutputAEPRBs(ouae_slice_a,prev_prb_slice_a) 
    drl_inp_slice_b = stackOutputAE(ouae_slice_b) #stackOutputAEPRBs(ouae_slice_b,prev_prb_slice_b)
    drl_inp_slice_c = stackOutputAE(ouae_slice_c) #stackOutputAEPRBs(ouae_slice_c,prev_prb_slice_c)

    drl_input = np.hstack((drl_inp_slice_a,drl_inp_slice_b,drl_inp_slice_c))# resulting array of shape (x,12)

    ##^^ prb in input
    prb_input = []
    for x,y,z in zip(prev_prb_slice_a,prev_prb_slice_b,prev_prb_slice_c):
        el = [x,y,z]
        prb_input.append(el)
    prb_input = np.array(prb_input)    

    ##^^ output of the agent
    drl_output = []
    for x,y,z in zip(oudrl_slice_a,oudrl_slice_b,oudrl_slice_c):
        el = [x,y,z]
        drl_output.append(el)
    drl_output = np.array(drl_output)

    # print(drl_input.shape,drl_output.shape)

    if ret_action == True:
        return prb_input, drl_input,drl_output
    else:
        return drl_input,drl_output
    

def readUEMetrics(exp):
    """
    This functions is useful for the core part of EXPLORA, namely the "analysis-explora.py" file
    
    Parameters
    _______________

    exp: current experiment

    Output
    _______________

    dict_ue_metrics: dictionary with timestep as key; value: the list of prb allocation the users report
    dict_sched_pol: dictionary with timestep as key; value: the scheduling policy enforced for that timestep

    Notes
    _______________

    Meaning of the parameters with an example

    metric_dict = {"slice_id": 0,
                   "dl_buffer [bytes]": 1,
                   "tx_brate downlink [Mbps]": 2,
                   "ratio_req_granted": 3,
                   "slice_prb": 4
                   "tx_pkts downlink": 5,
                   }
    ueMetricInfo is the structure that picks this up


    """

    ##^^ Variables
    dict_ue_metrics = {}
    dict_sched_pol = {}

    timestep = 0

    ##^^ Loading the user metrics from file
    file_metric_ues = "../data/motivation_main-results/exp"+str(exp)+ "/xapp_drl_sched_slicing_ric_26_agent.log"

    f = open(file_metric_ues, 'rb')

    ##^^ reading 
    for line in f:
        line = str(line.rstrip())
        ##^^ picking the PRB allocation and building the apposite structure
        substring_prb = "Received data:"
        if substring_prb in line:
            # print(line)
            timestep += 1
            string_split = line.split("data:")
            # print(f'{timestep} - {string_split[1][2:-2]} ', end = '')

            ##^^ retrieving only the actual values
            ueMetric_string = str(string_split[1][2:-2]).split("\\\\n")# [2:-2] to pick only the data of PRB allocation taking out string delimiters

            for el in ueMetric_string:
                # print (el)
                if "," in el:
                    ueMetric = str(el).split(",")

                    ue_metric_info = ueMetricInfo(int(ueMetric[0]),float(ueMetric[1]),float(ueMetric[2]),float(ueMetric[3]),float(ueMetric[4]),int(ueMetric[5]))

                    if timestep in dict_ue_metrics:
                        dict_ue_metrics[timestep].append(ue_metric_info)
                    else:
                        dict_ue_metrics[timestep]=[ue_metric_info]
        ##^^ picking the scheduling policy and building the apposite structure
        substring_sched = "Sending to DU"
        if substring_sched in line:
            # print(line)
            string_split = line.split("DU:")
            # print(f' --- {string_split[1][2:7]} \n')# [2:7] to pick only the data of the scheduling policy taking out string delimiters
            if timestep not in dict_sched_pol:
                dict_sched_pol[timestep]=string_split[1][2:7]

    return dict_ue_metrics, dict_sched_pol


def regularizeShapes(output_ae, out_ae_list, prbs, output_drl_sched, len_drl_shap_val):
    """
    
    
    Parameters
    _______________
    

    Output
    _______________


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


def extractTimingOnlineTraining(exp, exp_timing, exp_strategy):
    """
    This function reads the logs and takes out time of online training triggering and states learned
    
    Parameters
    _______________

    exp: current experiment (int)

    Output
    _______________

    online_training_times: dict with "start_times" and "stop_times" as keys, values are datetime objects
    action_info: dict with datetime objects as keys and list of actions as values
    metrics_info: dict with datetime objects as keys and list of actions as values
    reward_info: dict with datetime objects as keys and rewards as values

    Notes
    _______________


    """

    ##^^ Loading the log - careful when opening it

    logfile = "../data/action-steering/"+exp_timing+"/exp"+str(exp)+ "/log_train_agent_online.log"

    if not os.path.exists(logfile):
        # print("XAI: could not open the file 'log_train_agent_online' - trying with 'xapp_drl_sched_slicing_ric_26_agent'")
        logfile = "../data/action-steering/"+exp_timing+"/exp"+str(exp)+ "/xapp_drl_sched_slicing_ric_26_agent.log"

    try:
        f = open(logfile, 'rb')
    except OSError:
        print("Could not open/read file:", f)
        sys.exit()

    ##^^ Data structures
    online_training_times = {}
    action_info = {}
    metrics_info = {}
    reward_info = {}
    action_change_info = {}

    ##^^ Reading files
    for line in f:
        line = str(line.rstrip())
        # print(line)

        """
        
        From here on. The log files contain different key strings depending on the strategy for
                      action steering, and we need to retrieve several information, namely:
                      - users' KPIs -> "Received data" from the E2
                      - action the agent takes -> "Sending to DU"
                      - all info related to action steering (former action, substitution, reward)
        
        """

        ###^^^ metrics
        substring_prb = "Received data:"
        if substring_prb in line:
            # print(line)
            string_split = line.split("data:")
            # print(f'{timestep} - {string_split[1][2:-2]} ', end = '')

            ##^^ retrieving only the actual values
            ueMetric_string = str(string_split[1][2:-2]).split("\\\\n")# [2:-2] to pick only the data of PRB allocation taking out string delimiters
            timing_metrics = string_split[0].split(',')[0][2:]
            timing_metrics_datetime = datetime.strptime(timing_metrics, '%Y-%m-%d %H:%M:%S')
            # print(f'{timing_metrics} {ueMetric_string} ')
            if timing_metrics_datetime not in action_info:
                metrics_info[timing_metrics_datetime]=ueMetric_string

        ###^^^ actions
        substring_sched = "Sending to DU"
        if substring_sched in line:
            #^ parsing the info: timing, scheduling policy and prbs
            string_split = line.split("DU:")
            # print(string_split[1])
            timing_action = string_split[0].split(',')[0][2:]
            timing_action_datetime = datetime.strptime(timing_action, '%Y-%m-%d %H:%M:%S')
            
            st = string_split[1].index("n")
            
            sched = []
            for el in string_split[1][2:7].split(","):
                sched.append(int(el)) 
            reg = string_split[1][st+1:-2]
            # print(f'{timing_action} {sched} {reg}')
            prb = []
            for el in reg.split(","):
                prb.append(int(el)*3)
            state = prb + sched
            # print(f'{timing_action} {sched} {prb} {tuple(state)}')
            #^ inclusion into the data structure
            if timing_action_datetime not in action_info:
                action_info[timing_action_datetime] = tuple(state)

        ###^^^ reward
        if "INFO" in line:
            reward_sched = "reward"
            if reward_sched in line and "WEIGHT vector" not in line:
                #^ timing
                timing_reward = line.split(',')[0][2:]
                timing_reward_datetime = datetime.strptime(timing_reward, '%Y-%m-%d %H:%M:%S')

                #^ parsing the reward, make sure to remove the line that has the entire observation and the discount
                string_split = line.split(reward_sched)
                if "discount" not in string_split[1]:
                    reward = float(string_split[1][:-1])
                    # print(timing_reward_datetime,reward)
                    #^ inclusion into the data structure
                if timing_reward_datetime not in reward_info:
                    reward_info[timing_reward_datetime] = reward

        ###^^^ Action change info: we need to retrieve 
        #       1) action the agent wants to take, 2) its reward, 3) replaced action, 4) its reward, 5) avg-std or reward
        ##^^ getting the reward of action the agent would take, avg and std of past rewards
        substring_ac = "Reward comparison"
        if substring_ac in line:
            # print(line)
            #^ timing
            timing_ac = line.split(',')[0][2:]
            timing_ac_datetime = datetime.strptime(timing_reward, '%Y-%m-%d %H:%M:%S')

            string_split = line.split("=")
            #^ expected reward of the action the agent wants to take
            exp_rew_action_agent = string_split[1].split(',')[0]
            #^ avg and std:
            string_avg_std = string_split[2][2:-2]
            avg_std_split = string_avg_std.split(",")

            # print(f'{timing_ac} - {exp_rew_action_agent} -- Avg: {avg_std_split[0]} Std: {avg_std_split[1]}')

            dict_ac = {'exp_rew_action_agent': float(exp_rew_action_agent),
                 'avg_rew_past': float(avg_std_split[0]),
                 'std_rew_past': float(avg_std_split[1])
                 }

            if timing_ac_datetime not in action_change_info:
                action_change_info[timing_ac_datetime] = dict_ac
            else:
                action_change_info[timing_ac_datetime].update(dict_ac)

        ##^^ getting the actions
        substring_ac_action_agent = ", but was replaced with new action"
        if substring_ac_action_agent in line:
            # print(line)
            #^ timing
            timing_ac = line.split(',')[0][2:]
            timing_ac_datetime = datetime.strptime(timing_reward, '%Y-%m-%d %H:%M:%S')

            string_split = line.split(", but")
            #^ action suggested by the agent
            action_agent = string_split[0].split("was:")[1]

            sched = []
            prb = []
            for it, el in enumerate(action_agent[2:-1].split(",")):
                if it < 3:
                    prb.append(int(el))
                else:
                    sched.append(int(el))
            action_agent = tuple(prb + sched)

            #^ action replaced thanks to the graph
            action_graph = string_split[1].split("new action")[1]

            sched = []
            prb = []
            for it, el in enumerate(action_graph[2:-13].split(",")):
                if it < 3:
                    prb.append(int(el))
                else:
                    sched.append(int(el))
            action_graph = tuple(prb + sched)

            #^ insertion into the main dictionary
            dict_ac = {'action_agent': action_agent, 'action_graph': action_graph}

            if timing_ac_datetime not in action_change_info:
                action_change_info[timing_ac_datetime] = dict_ac
            else:
                action_change_info[timing_ac_datetime].update(dict_ac)

            # print(f'{timing_ac} - Agent: {action_agent} Graph: {action_graph[2:-13]}')
        ##^^ last, retrieving the reward of the action suggested by the graph
        substring_ac_rew_new_action = "Expected reward was"
        if substring_ac_rew_new_action in line:
            # print(line)
            #^ timing
            timing_ac = line.split(',')[0][2:]
            timing_ac_datetime = datetime.strptime(timing_reward, '%Y-%m-%d %H:%M:%S')

            string_split = ""
            if exp_strategy == "max_reward":
                string_split = line.split("give")
            elif exp_strategy == "max_reward_obs_20":
                string_split = line.split("give max_reward")
            elif exp_strategy == "min_reward" or exp_strategy== "min_reward_obs_20":
                string_split = line.split("give min_reward")

            # print(f'{timing_ac}: {string_split[1][1:-1]}')

            exp_rew_action_graph = float(string_split[1][1:-1])
            # print(f'{timing_ac}: {exp_rew_action_graph}')

            #^ insertion into the main dictionary
            dict_ac = {'exp_rew_action_graph': exp_rew_action_graph}
            if timing_ac_datetime not in action_change_info:
                action_change_info[timing_ac_datetime] = dict_ac
            else:
                action_change_info[timing_ac_datetime].update(dict_ac)

        ###^^^ Timing
        ##^^ online training starts
        substring_sched = "STARTED time-based online training"
        if substring_sched in line:
            timing_start = line.split(',')[0][2:]
            timing_start_datetime = datetime.strptime(timing_start, '%Y-%m-%d %H:%M:%S')
            # print("Start: ",timing_start, timing_start_datetime)
            #^ inclusion into the data structure
            if "start_times" not in online_training_times:
                online_training_times["start_times"]=[timing_start_datetime]
            else:
                online_training_times["start_times"].append(timing_start_datetime)

        ##^^ online training stops
        substring_sched = "STOPPED time-based online training"
        if substring_sched in line:
            timing_stop = line.split(',')[0][2:]            
            timing_stop_datetime = datetime.strptime(timing_stop, '%Y-%m-%d %H:%M:%S')
            # print("Stop: ",timing_stop,timing_stop_datetime)
            #^ inclusion into the data structure
            if "stop_times" not in online_training_times:
                online_training_times["stop_times"]=[timing_stop_datetime]
            else:
                online_training_times["stop_times"].append(timing_stop_datetime)

            
    return online_training_times, action_info, metrics_info, reward_info, action_change_info