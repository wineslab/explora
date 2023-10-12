"""
Action steering

10/13/2023
_______________

Summary: this script quantities the gains of action steering by processing the corresponding experiments
	 performed with the three strategies defined in Section 5.2 of the manuscript

DISCLAIMER: THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""

####^^^^ Imports and settings
##^^ misc
import os

##^^ importing project functions
from utils_db_process import *
from utils_attr_graphs import *
from utils_experiment_list_action_steering import *

##^^ functions

def analyzeActionComposition(online_training_times, action_change_info, action_info, type_experiment, printstats = False, plotsubstitution = False):
    """
    This function processes the actions that have been learned during OT, were known already before OT and those that are unknowns
    and links them to the reward that the reward

    Parameters
    _______________


    Output
    _______________

    N/A

    Notes
    _______________


    """

    online_training_stop = online_training_times["stop_times"][0]

    set_actions_used = set()
    set_actions_potentially_used_by_agent = set()
    set_actions_substituted = set()
    set_actions_not_substituted = set()

    num_changes = 0
    tot_times = 0

    #^ dictionary to count how many times the same action was substituted
    count_times_is_substituted = {}

    #^ dictionary to count how many times one action was used in susbstitution
    count_times_substitutes = {}

    for current_time,current_action_change in action_change_info.items():
        tot_times += 1
        current_action = tuple()
        if current_time > online_training_stop:
            if len(current_action_change) > 3:
                action_agent = current_action_change['action_agent']
                action_graph = current_action_change['action_graph']
                #^ adding into sets
                set_actions_used.add(action_graph)
                set_actions_substituted.add(action_graph)
                set_actions_potentially_used_by_agent.add(action_agent)
                num_changes+=1

                current_action = action_graph

                # print(f'{current_time} - {action_agent} substituted by {action_graph}')

                #^ adding in the respective dictionaries of substitution
                if action_agent not in count_times_is_substituted:
                    count_times_is_substituted[action_agent] = 1
                else:
                    count_times_is_substituted[action_agent] += 1

                if action_graph not in count_times_substitutes:
                    count_times_substitutes[action_graph] = 1
                else:
                    count_times_substitutes[action_graph] += 1

            else:
                current_action = action_info.get(current_time)
                #^ add to sets - check first that current_action does actually exists in "action_info"
                if current_action:
                    # print(f'{current_time} - {current_action}')

                    #^ adding into sets
                    set_actions_used.add(current_action)
                    set_actions_potentially_used_by_agent.add(current_action)
                    set_actions_not_substituted.add(current_action)

    #^ Printing the dictionaries
    # print(f'Stats on how often these actions were substituted: ')
    # for k,v in count_times_is_substituted.items():
    #     print(k,v)

    # print(f'\n--\nStats on how often these actions were used to substitute others: ')
    # for k,v in count_times_substitutes.items():
    #     print(k,v)

    # print(f'\n--\n')

    if plotsubstitution:
        outdir = "../results/action-steering/distribution-actions-steered/"
        times_is_substituted = np.array(list(count_times_is_substituted.values()))
        times_substitutes = np.array(list(count_times_substitutes.values()))

        print(f'Substitution distributions: \n{times_is_substituted}|{len(times_is_substituted)}\n{times_substitutes}|{len(times_substitutes)}')

        ax1, ax2 = plt.subplots(1, 2, figsize=(15, 8))

        max_y_lim = max(np.count_nonzero(times_is_substituted == 1),np.count_nonzero(times_substitutes == 1))

        bins_1 = np.linspace(np.min(times_is_substituted),np.max(times_is_substituted)+1)
        ax1.hist(times_is_substituted, bins=bins_1, color="darkorchid",align='mid')
        ax1.set_xticks(np.arange(1,np.max(times_is_substituted)+1,1))
        ax1.set_xlabel('Occurrences action from agent is substituted',fontsize=12)
        ax1.set_ylabel("Count",fontsize=12)
        ax1.set_ylim(bottom=0, top=max_y_lim)

        bins_2 = np.linspace(np.min(times_substitutes),np.max(times_substitutes)+1)
        ax2.hist(times_substitutes, bins=bins_2, color="forestgreen",align='mid')
        ax2.set_xticks(np.arange(1,np.max(times_substitutes)+1,1))
        ax2.set_xlabel('Occurrences action from graph substitutes action from agent',fontsize=12)
        ax2.set_ylabel("Count",fontsize=12)
        ax2.set_ylim(bottom=0, top=max_y_lim)
        # plt.plot()
        # plt.show()

        plotname = outdir+type_experiment+".png"
        plottikz = outdir+type_experiment+".tex"
        plt.savefig(plotname)
        tikzplotlib.save(plottikz)


    if printstats:
        reduction = round(100-(((len(list(set_actions_potentially_used_by_agent))-len(list(set_actions_used)))*100)/(len(list(set_actions_potentially_used_by_agent)))),2)
        print(f'Actions. Used: {len(list(set_actions_used))} - Potentially used: {len(list(set_actions_potentially_used_by_agent))} - {reduction} %')
        print(f'         Substituted: {len(list(set_actions_substituted))}. Not substituted: {len(list(set_actions_not_substituted))}')
        print(f'Changes: {num_changes} over {tot_times} - {round((num_changes/tot_times)*100,2)}%')

def getAvgStdKPIS(current_metrics,slice_id):

    # print(f'{current_time}: {current_metrics}')

    tx_pkts = []
    tx_brate = []
    dl_buffer = []
    current_prb = {}

    for el in current_metrics:
        split_el = el.split(",")
        # check length of the string as there may be spurious data at the end of the "Receive data" string
        if len(split_el) == 6:
            #^ computing the current PRB allocation across slices
            if int(split_el[metric_dict["slice_id"]]) == 0 and 0 not in current_prb:
                current_prb[0] = int(split_el[metric_dict["slice_prb"]])
            elif int(split_el[metric_dict["slice_id"]]) == 1 and 1 not in current_prb:
                current_prb[1] = int(split_el[metric_dict["slice_prb"]])
            elif int(split_el[metric_dict["slice_id"]]) == 2 and 2 not in current_prb:
                current_prb[2] = int(split_el[metric_dict["slice_prb"]])

            #^ only keeping KPIs for the specific slice_id selected
            if int(split_el[metric_dict["slice_id"]]) == slice_id:
                # print(f'{split_el[2]} {split_el[5]} {split_el[1]} - {split_el[4]}, ', end = '')
                tx_pkts.append(float(split_el[5]))
                tx_brate.append(float(split_el[2]))
                dl_buffer.append(float(split_el[1]))

    avg_tx_brate = round(np.average(np.array(tx_brate)),2)
    avg_tx_pkts = round(np.average(np.array(tx_pkts)),2)
    avg_dl_buffer = round(np.average(np.array(dl_buffer)),2)

    std_tx_brate = round(np.std(np.array(tx_brate)),2)
    std_tx_pkts = round(np.std(np.array(tx_pkts)),2)
    std_dl_buffer = round(np.std(np.array(dl_buffer)),2)

    dist_tx_brate = {"avg": avg_tx_brate, "std": std_tx_brate}
    dist_tx_pkts = {"avg": avg_tx_pkts, "std": std_tx_pkts}
    dist_dl_buffer = {"avg": avg_dl_buffer, "std": std_dl_buffer}

    return current_prb, dist_tx_brate, dist_tx_pkts, dist_dl_buffer

def linkActionsToKPIs(online_training_times, metrics_info, exportcdfdata = False):
    """
    This function links the actions to the KPIs: specifically it creates 3 subplots
    each one being a heatmap with scheduling, prb allocation and reward (on viridis - use bluered or anything with white for the zero)

    Parameters
    _______________

    online_training_times: dict with "start_times" and "stop_times" as keys, values are datetime objects
    learned_actions, known_actions, unknown_actions: lists
    action_info: dict with datetime objects as keys and list of actions as values
    metrics_info: dict with datetime objects as keys and KPI (tx_brate, tx_pkts, dl_buffer) as values

    Output
    _______________

    N/A

    Notes
    _______________


    """

    online_training_stop = online_training_times["stop_times"][0]

    ##^^ Executing all operations per slice

    for slice_id in range(0,3):

        #^ arrays to export the KPIs to file
        avg_tx_brate_toarray = []
        avg_tx_pkts_toarray = []
        avg_dl_buffer_toarray = []

        #^ looping over the metrics and filling in the matrices
        for keys, values in metrics_info.items():
            current_metrics = values
            current_time = keys

            #^ retrieving KPIs
            current_prb, dist_tx_brate, dist_tx_pkts, dist_dl_buffer = getAvgStdKPIS(current_metrics,slice_id)

            avg_tx_brate = dist_tx_brate["avg"]
            avg_tx_pkts = dist_tx_pkts["avg"]
            avg_dl_buffer = dist_dl_buffer["avg"]

            #^ The flow is as follows:
            # before_OT: agent runs with weights taken from offline checkpoints
            # after_OT: users drop from 6 to 5 -> perform online training to help the agent
            # after_final_exec: action steering has been enforced -> store the KPIs for later process
            if current_time > online_training_stop:
                # ##^^ storing the average KPIs
                avg_tx_brate_toarray.append(avg_tx_brate)
                avg_tx_pkts_toarray.append(avg_tx_pkts)
                avg_dl_buffer_toarray.append(avg_dl_buffer)

        if exportcdfdata:
            outdir = "../results/action-steering/data-for-cdfs/"+exp_strategy+"/"
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            #^ For the final resutls
            #  the "NEW" is just to make sure to not override anything (you can check that the file size is the same)
            filename_tx_brate = outdir + label+"-exp"+str(exp)+"_SL"+str(slice_id)+"_tx_brate.npy"
            filename_tx_pkts = outdir + label+"-exp"+str(exp)+"_SL"+str(slice_id)+"_tx_pkts.npy"
            filename_dl_buffer = outdir + label+"-exp"+str(exp)+"_SL"+str(slice_id)+"_dl_buffer.npy"

            with open(filename_tx_brate, 'wb') as f:
                np.save(f, avg_tx_brate_toarray)
            with open(filename_tx_pkts, 'wb') as f:
                np.save(f, avg_tx_pkts_toarray)
            with open(filename_dl_buffer, 'wb') as f:
                np.save(f, avg_dl_buffer_toarray)

if __name__ == "__main__":

    ##^^ Choose strategy in exp_strategy by selecting the index:
    ## 0: baseline
    ## 1: max_reward
    ## 2: max_reward_obs_20
    ## 3: min_reward
    ## 4: min_reward_obs_20
    ## 5: imp_bitrate
    ## 6: imp_bitrate_obs_20
    ## 7: baseline_obs_20

    #^ NOTE: run all at once via the for loop for process and export data for later plotting (linkActionsToKPIs)
    #        run one at a time to understand action composition and how many have been steered (analyzeActionComposition)

    for strategy in range(0,8):
        exp_strategy = list(exp_configuration.keys())[strategy]
        # exp_strategy = list(exp_configuration.keys())[0] # DE-COMMENT this to run one strategy at a time

        #^ next retrieves from exp_strategy the set of experiments [0] and the label [1] - DO NOT modify
        exp_list = exp_configuration[exp_strategy][0]
        exp_timing = exp_configuration[exp_strategy][1]

        print(f'Processing strategy "{exp_strategy}" with "{exp_list}" performed in "{exp_timing}" ')

        for label,exp in exp_list.items():

            print(f"= = = = = = = = = = = = = = = = =\nExp: {label}\n= = = = = = = = = = = = = = = = =")

            ##^^ PART I: read logs
            online_training_times, action_info, metrics_info, \
                reward_info, action_change_info = extractTimingOnlineTraining(exp,exp_timing,exp_strategy)

            ##^^ PART II: analysis

            linkActionsToKPIs(online_training_times, metrics_info, exportcdfdata = True)

            # #^ next function generates the stats shown in Appendix D
            # #^ NOTE: the script works for strategies that are not baseline, i.e., strategies with index 1-6
            # type_experiment = exp_strategy+"_"+label+"_exp"+str(exp)
            # analyzeActionComposition(online_training_times, action_change_info, action_info, 
            #                          type_experiment, printstats=True, plotsubstitution = False)

        print("-- Finished main --")
