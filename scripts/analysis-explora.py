"""
EXPLORA Core

10/12/2023
_______________
    
Summary: The script runs the main EXPLORA operations like generation and processing of attributed graphs,
         train of DT on top of the outcomes of the attributed graph processing and visualization of
         the results.
         These are Fig. 7, 8 of Section 6.2 and Fig. 13, 14 of Appendix C.
         
DISCLAIMER: THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING 
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""

####^^^^ Imports and settings
##^^ misc
from sklearn.preprocessing import MinMaxScaler
import graphviz
import tikzplotlib
from scipy.stats import norm

##^^ decision trees to evaluate the meaning of the transitions and their link to KPIs
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

#^ exporting decision trees
from sklearn.tree import export_graphviz
from six import StringIO
#from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

##^^ importing project functions
from utils_shap_plotting import *
from utils_db_process import *
from utils_generic_functions import computeMedoid
from utils_experiment_list import *
from utils_attr_graphs import *

##^^ functions

def processKPIStateTransitions(dict_kpi_ues, dict_trans,exp):
    """
    
    
    Parameters
    _______________
    
    dict_kpi_ues: the complex dictionary
    dict_trans: dictionary with transitions useful for later analysis
    exp: experiment ID

    Output
    _______________

    dict_kpi_distr_state: dictionary with STATE as key and avg/std per slice KPIs as value (as dictionary as well)
    dict_kpi_state_transition: dictionary with full STATE TRANSITION as key and CF-distance as value (as dictionary as well)

    """

    ##^^ list with transitions we want to study
    list_hot_trans = []

    ##^^ reading the dict_trans dictionary
    for keys,values in dict_trans.items():
        ##^^ picking up from and to states, then weight and then 
        keys = str(keys)
        #^ split in two to retrieve from and to state
        elements = keys[1:-1].split(',')
        from_state = [int(elements[el]) for el in range(6)]
        to_state = [int(elements[el]) for el in range(6,12)]

        ##^^ DEBUG: counting how many items for each slice ID we have in a given state

        # print(f'From state {from_state} with (',end=' ')#) to {to_state}: {values} ')
        from_kpis_count = dict_kpi_ues[tuple(from_state)]
        from_lens = []
        for ue_key, ue_item in from_kpis_count.items():
            if ue_key == 0:
                # print(f'SL{ue_key}: {len(ue_item)} ', end = '')
                from_lens.append(len(ue_item))
            if ue_key == 1:
                # print(f'SL{ue_key}: {len(ue_item)} ', end = '')
                from_lens.append(len(ue_item))
            if ue_key == 2:
                # print(f'SL{ue_key}: {len(ue_item)} ', end = '')
                from_lens.append(len(ue_item))
        # print(f') to state {to_state} with (',end=' ')
        to_kpis_count = dict_kpi_ues[tuple(from_state)]
        to_lens = []
        for ue_key, ue_item in to_kpis_count.items():
            if ue_key == 0:
                # print(f'SL{ue_key}: {len(ue_item)} ', end = '')
                to_lens.append(len(ue_item))
            if ue_key == 1:
                # print(f'SL{ue_key}: {len(ue_item)} ', end = '')
                to_lens.append(len(ue_item))
            if ue_key == 2:
                # print(f'SL{ue_key}: {len(ue_item)} ', end = '')
                to_lens.append(len(ue_item))
        # print(f') => {values}')

        ##^^ NOTE that the debugging highlights the following:
        ## there exists links between states with 1 occurrence that are fully negligible because also have little KPIs like 
        ## ___From state [30, 9, 11, 2, 2, 0] with ( SL0: 1 SL1: 1 SL2: 1 ) to state [12, 15, 23, 1, 2, 0] with ( SL0: 1 SL1: 1 SL2: 1 ) => 1
        ## but others do not:
        ## ___From state [12, 15, 23, 1, 2, 0] with ( SL0: 101 SL1: 101 SL2: 101 ) to state [12, 15, 23, 2, 1, 1] with 
        ## ( SL0: 101 SL1: 101 SL2: 101 ) => 1
        ## hence use a combined "values" + "num of KPIs reported" to set which transitions to use or not for the evaluation

        if values > 1 :
            list_hot_trans.append(keys)
        elif values == 1:
            reason_to_add = False
            for j in from_lens:
                if j > 50:
                    reason_to_add = True
            for j in to_lens:
                if j > 50:
                    reason_to_add = True
            if reason_to_add:
                list_hot_trans.append(keys)

    ##^^ cleaning unused structure
    dict_trans.clear()

    ##^^ dictionaries in output
    #^ dictionary with STATE as key and avg/std per slice KPIs as value (as dictionary as well)
    dict_kpi_distr_state = {}
    #^ dictionary with full STATE TRANSITION as key and 1 as value - FUTURE WORK: add  distribution distance as value (as dictionary as well)
    dict_kpi_state_transition = {}

    ##^^ Analysis of the transitions: linking KPIs and computing distances
    for el in list_hot_trans:
        keys = str(el)
        from_state, to_state = getFromToStates(keys)

        # print(f'{keys} - {from_state} to {to_state}')
        # print(f'{keys} - {from_state} to {to_state}: ', end='')

        ##^^ np array with the distribution of the from_state
        from_kpis = dict_kpi_ues[tuple(from_state)]

        dist_from_sl0 = []
        dist_from_sl1 = []
        dist_from_sl2 = []

        from_sl0_tx_brate = []
        from_sl0_tx_pkts = []
        from_sl0_dl_buffer = []
        from_sl1_tx_brate = []
        from_sl1_tx_pkts = []
        from_sl1_dl_buffer = []
        from_sl2_tx_brate = []
        from_sl2_tx_pkts = []
        from_sl2_dl_buffer = []

        for ue_key, ue_item in from_kpis.items():

            if ue_key==0:
                for el in ue_item:
                    for e in el:
                        dist_from_sl0.append((e.tx_brate, e.tx_pkts, e.dl_buffer))
                        from_sl0_tx_brate.append(e.tx_brate)
                        from_sl0_tx_pkts.append(e.tx_pkts)
                        from_sl0_dl_buffer.append(e.dl_buffer)
            if ue_key==1:
                for el in ue_item:
                    for e in el:
                        dist_from_sl1.append((e.tx_brate, e.tx_pkts, e.dl_buffer))
                        from_sl1_tx_brate.append(e.tx_brate)
                        from_sl1_tx_pkts.append(e.tx_pkts)
                        from_sl1_dl_buffer.append(e.dl_buffer)
            if ue_key==2:
                for el in ue_item:
                    for e in el:
                        dist_from_sl2.append((e.tx_brate, e.tx_pkts, e.dl_buffer))
                        from_sl2_tx_brate.append(e.tx_brate)
                        from_sl2_tx_pkts.append(e.tx_pkts)
                        from_sl2_dl_buffer.append(e.dl_buffer)
        # print(np.array(dist_from_sl0).shape,np.array(dist_from_sl1).shape,np.array(dist_from_sl2).shape)

        ##^^ np array with the distribution of the to_state
        to_kpis = dict_kpi_ues[tuple(to_state)]               
        dist_to_sl0 = []
        dist_to_sl1 = []
        dist_to_sl2 = []

        to_sl0_tx_brate = []
        to_sl0_tx_pkts = []
        to_sl0_dl_buffer = []
        to_sl1_tx_brate = []
        to_sl1_tx_pkts = []
        to_sl1_dl_buffer = []
        to_sl2_tx_brate = []
        to_sl2_tx_pkts = []
        to_sl2_dl_buffer = []

        for ue_key, ue_item in to_kpis.items():

            if ue_key==0:
                for el in ue_item:
                    for e in el:
                        dist_to_sl0.append((e.tx_brate, e.tx_pkts, e.dl_buffer))
                        to_sl0_tx_brate.append(e.tx_brate)
                        to_sl0_tx_pkts.append(e.tx_pkts)
                        to_sl0_dl_buffer.append(e.dl_buffer)
            if ue_key==1:
                for el in ue_item:
                    for e in el:
                        dist_to_sl1.append((e.tx_brate, e.tx_pkts, e.dl_buffer))
                        to_sl1_tx_brate.append(e.tx_brate)
                        to_sl1_tx_pkts.append(e.tx_pkts)
                        to_sl1_dl_buffer.append(e.dl_buffer)
            if ue_key==2:
                for el in ue_item:
                    for e in el:
                        dist_to_sl2.append((e.tx_brate, e.tx_pkts, e.dl_buffer))
                        to_sl2_tx_brate.append(e.tx_brate)
                        to_sl2_tx_pkts.append(e.tx_pkts)
                        to_sl2_dl_buffer.append(e.dl_buffer)

        # print(np.array(dist_to_sl0).shape,np.array(dist_to_sl1).shape,np.array(dist_to_sl2).shape)
        
        ##^^ At this point we have 6 np arrays with the distribution of "FROM/TO" states and each "slice"
        ##^^ Now create a dictionary with STATE as key and 1 as value // future: have distribution distance
        if keys not in dict_kpi_state_transition:
            dict_kpi_state_transition[keys]={"dist_SL0": 1, "dist_SL1": 1, "dist_SL2": 1}        

        ##^^ Now create a dictionary with STATE as key and (mean, std) KPIs per slice
        #^ helper printing
        # print("---")
        # print(f'FROM stats: ')
        # print(f'SL0 {round(np.array(from_sl0_tx_brate).mean(),2)}|{round(np.array(from_sl0_tx_brate).std(),2)}\
        #  {round(np.array(from_sl0_tx_pkts).mean(),2)}|{round(np.array(from_sl0_tx_pkts).std(),2)}\
        #  {round(np.array(from_sl0_dl_buffer).mean(),2)}|{round(np.array(from_sl0_dl_buffer).std(),2)} ')
        # print(f'SL1 {round(np.array(from_sl1_tx_brate).mean(),2)}|{round(np.array(from_sl1_tx_brate).std(),2)}\
        #  {round(np.array(from_sl1_tx_pkts).mean(),2)}|{round(np.array(from_sl1_tx_pkts).std(),2)}\
        #  {round(np.array(from_sl1_dl_buffer).mean(),2)}|{round(np.array(from_sl1_dl_buffer).std(),2)} ')
        # print(f'SL2 {round(np.array(from_sl2_tx_brate).mean(),2)}|{round(np.array(from_sl2_tx_brate).std(),2)}\
        #  {round(np.array(from_sl2_tx_pkts).mean(),2)}|{round(np.array(from_sl2_tx_pkts).std(),2)}\
        #  {round(np.array(from_sl2_dl_buffer).mean(),2)}|{round(np.array(from_sl2_dl_buffer).std(),2)} ')
        # print(f'Lengths. {len(from_sl0_tx_brate)}')
        # print("---")

        # print(f'TO stats: ')
        # print(f'SL0 {round(np.array(to_sl0_tx_brate).mean(),2)}|{round(np.array(to_sl0_tx_brate).std(),2)}\
        #  {round(np.array(to_sl0_tx_pkts).mean(),2)}|{round(np.array(to_sl0_tx_pkts).std(),2)}\
        #  {round(np.array(to_sl0_dl_buffer).mean(),2)}|{round(np.array(to_sl0_dl_buffer).std(),2)} ')
        # print(f'SL1 {round(np.array(to_sl1_tx_brate).mean(),2)}|{round(np.array(to_sl1_tx_brate).std(),2)}\
        #  {round(np.array(to_sl1_tx_pkts).mean(),2)}|{round(np.array(to_sl1_tx_pkts).std(),2)}\
        #  {round(np.array(to_sl1_dl_buffer).mean(),2)}|{round(np.array(to_sl1_dl_buffer).std(),2)} ')
        # print(f'SL2 {round(np.array(from_sl2_tx_brate).mean(),2)}|{round(np.array(to_sl2_tx_brate).std(),2)}\
        #  {round(np.array(to_sl2_tx_pkts).mean(),2)}|{round(np.array(to_sl2_tx_pkts).std(),2)}\
        #  {round(np.array(to_sl2_dl_buffer).mean(),2)}|{round(np.array(to_sl2_dl_buffer).std(),2)} ')
        # print(f'Lengths. {len(to_sl0_tx_brate)}')
        # print("---")

        from_state_sl0_kpi = stateSliceKPI(round(np.array(from_sl0_dl_buffer).mean(),2),round(np.array(from_sl0_dl_buffer).std(),2),\
            round(np.array(from_sl0_tx_brate).mean(),2),round(np.array(from_sl0_tx_brate).std(),2),\
            round(np.array(from_sl0_tx_pkts).mean(),2),round(np.array(from_sl0_tx_pkts).std(),2))
        from_state_sl1_kpi = stateSliceKPI(round(np.array(from_sl1_dl_buffer).mean(),2),round(np.array(from_sl1_dl_buffer).std(),2),\
            round(np.array(from_sl1_tx_brate).mean(),2),round(np.array(from_sl1_tx_brate).std(),2),\
            round(np.array(from_sl1_tx_pkts).mean(),2),round(np.array(from_sl1_tx_pkts).std(),2))
        from_state_sl2_kpi = stateSliceKPI(round(np.array(from_sl2_dl_buffer).mean(),2),round(np.array(from_sl2_dl_buffer).std(),2),\
            round(np.array(from_sl2_tx_brate).mean(),2),round(np.array(from_sl2_tx_brate).std(),2),\
            round(np.array(from_sl2_tx_pkts).mean(),2),round(np.array(from_sl2_tx_pkts).std(),2))

        to_state_sl0_kpi = stateSliceKPI(round(np.array(to_sl0_dl_buffer).mean(),2),round(np.array(to_sl0_dl_buffer).std(),2),\
            round(np.array(to_sl0_tx_brate).mean(),2),round(np.array(to_sl0_tx_brate).std(),2),\
            round(np.array(to_sl0_tx_pkts).mean(),2),round(np.array(to_sl0_tx_pkts).std(),2))
        to_state_sl1_kpi = stateSliceKPI(round(np.array(to_sl1_dl_buffer).mean(),2),round(np.array(to_sl1_dl_buffer).std(),2),\
            round(np.array(to_sl1_tx_brate).mean(),2),round(np.array(to_sl1_tx_brate).std(),2),\
            round(np.array(to_sl1_tx_pkts).mean(),2),round(np.array(to_sl1_tx_pkts).std(),2))
        to_state_sl2_kpi = stateSliceKPI(round(np.array(to_sl2_dl_buffer).mean(),2),round(np.array(to_sl2_dl_buffer).std(),2),\
            round(np.array(to_sl2_tx_brate).mean(),2),round(np.array(to_sl2_tx_brate).std(),2),\
            round(np.array(to_sl2_tx_pkts).mean(),2),round(np.array(to_sl2_tx_pkts).std(),2))


        ##^^ inclusion in dictionary if not present yet
        if tuple(from_state) not in dict_kpi_state_transition:
            dict_kpi_distr_state[tuple(from_state)]= {"SL0":from_state_sl0_kpi, 
                                                "SL1":from_state_sl1_kpi,
                                                "SL2":from_state_sl2_kpi,
                                                "Distr_elements":len(from_sl0_dl_buffer),
                                                }
        if tuple(to_state) not in dict_kpi_state_transition:
            dict_kpi_distr_state[tuple(to_state)]= {"SL0":to_state_sl0_kpi,
                                                "SL1":to_state_sl1_kpi,
                                                "SL2":to_state_sl2_kpi,
                                                "Distr_elements":len(to_sl0_dl_buffer),
                                                }


    return dict_kpi_state_transition, dict_kpi_distr_state

def printFullGraphwithKPIs(global_dict_kpi_state_transition,global_dict_kpi_distr_state,agent):
    """
    
    -
    Parameters
    _______________
    
    global_dict_kpi_distr_state: dictionary with STATE as key and avg/std per slice KPIs as value (as dictionary as well)
    global_dict_kpi_state_transition: dictionary with full STATE TRANSITION as key and CF-distance as value (as dictionary as well)

    Output
    _______________


    """
    ##^^ Printing results
    print(f'--- Results ---')
    print(f'--  Links between graph nodes --')
    #^ the "global_dict_kpi_state_transition", i.e., the EDGES
    for keys, values in global_dict_kpi_state_transition.items():
        # print(keys)
        from_state, to_state = getFromToStates(keys)
        print(f'From {from_state} to {to_state}')

    print(f'--  Graph node full attribute info --')    
    #^ the "global_dict_kpi_distr_state", i.e., the NODES
    for keys, values in global_dict_kpi_distr_state.items():
        print(keys)
        #^ looping over the list of dictionaries composed of {exp:dict_kpis_per_state}
        for el in values:
            for key, value in el.items():
                list_of_print = ["SL0","SL1","SL2"]
                for el in list_of_print:
                    val = value[el]
                    print(f'{key}, {value["Distr_elements"]}, {el}, {val.avg_tx_brate},  ',end = '')
                    print(f'{val.std_tx_brate}, {val.avg_tx_pkts}, {val.std_tx_pkts}, {val.avg_dl_buffer},{val.std_dl_buffer}')


def processExperiments(list_of_experiments,agent):
    """
    
    -
    Parameters
    _______________
    

    Output
    _______________


    """

    global_dict_kpi_state_transition = {}
    global_dict_kpi_distr_state = {}

    ##^^ ANALYSIS for each experiment
    for exp in list_of_experiments:
        # print(f'Current experiment under analysis: {exp} \n-------')

        ##^^ read the UE reported metrics
        dict_ue_metrics, dict_sched_pol = readUEMetrics(exp)

        ##^^ create dictionaries with distribution of user KPIs per slice per state [e.g. (24, 15, 11, 2, 1, 0)] and transitions between states
        dict_kpi_ues, dict_trans = createDictKPIStateTransitions(dict_ue_metrics,dict_sched_pol,exp)

        ##^^ cleaning 
        dict_ue_metrics.clear()
        dict_sched_pol.clear()
        
        dict_kpi_state_transition, dict_kpi_distr_state  = processKPIStateTransitions(dict_kpi_ues, dict_trans,exp)

        ##^^ analyze the distance metric for this experiment
        # analysisDistanceStateTransitions(dict_kpi_state_transition,dict_kpi_distr_state,exp)

        ##^^ cleaning
        dict_kpi_ues.clear()
        dict_trans.clear()

        ##^^ Including experiment-related info (from "dict_kpi_state_transition" and "dict_kpi_distr_state") into one big dictionary
        for keys, values in dict_kpi_state_transition.items():
            ##^^ Printing
            # from_state, to_state = getFromToStates(keys)
            # print(f'From {from_state} to {to_state}: ')
            # print(yaml.dump(values, default_flow_style=False))# very convenient way of printing dictionaries

            ##^^ Inclusion into the global dictionary
            if keys not in global_dict_kpi_state_transition:
                global_dict_kpi_state_transition[keys]=[values]
            else:
                global_dict_kpi_state_transition[keys].append(values)

        for keys, values in dict_kpi_distr_state.items():
            if agent == "embb-trf1":
                num_ues = exp_num_ues_embb_trf1[exp]
            if agent == "embb-trf2":
                num_ues = exp_num_ues_embb_trf2[exp]
            if agent == "urllc-trf1":
                num_ues = exp_num_ues_urllc_trf1[exp]
            if agent == "urllc-trf2":
                num_ues = exp_num_ues_urllc_trf2[exp]

            ##^^ Inclusion into the global dictionary
            if keys not in global_dict_kpi_distr_state:
                global_dict_kpi_distr_state[keys] = [{num_ues:values}]
            else:
                global_dict_kpi_distr_state[keys].append({num_ues:values})


        ##^^ cleaning structures
        dict_kpi_state_transition.clear()
        dict_kpi_distr_state.clear()

    #^ full print of the created structures
    # printFullGraphwithKPIs(global_dict_kpi_state_transition, global_dict_kpi_distr_state, agent)

    return global_dict_kpi_state_transition, global_dict_kpi_distr_state

    """ 
    NOTES
    
    - Regarding the "global_dict_kpi_state_transition," this structure is useful only to create the edges of the graph
    hence, for cases where for each edge there are multiple elements (i.e., two or more experiments have the same edge)

    """


def visualizeGraph(global_dict_kpi_state_transition,global_dict_kpi_distr_state,agent):
    """
    
    -
    Parameters
    _______________
    
    global_dict_kpi_state_transition: 
    global_dict_kpi_distr_state: 
    agent: self-explicative

    Output
    _______________


    """
    ##^^ Function variables
    outdir="../results/graphs-agents/"+str(agent)+"/"
    filetitle = outdir+"g-"+str(agent)
    filename = filetitle+".gv"

    ##^^ Creating the graph
    f = graphviz.Digraph(filetitle, filename=filename) #,engine='neato'

    for el in global_dict_kpi_distr_state.keys():
        # print(el)
        el = str(el)[1:-1]
        f.node(str(el))

    for keys, values in global_dict_kpi_state_transition.items():
        from_state, to_state = getFromToStates(keys)
        from_state = str(from_state)[1:-1]
        to_state = str(to_state)[1:-1]
        # print(f'From {from_state} to {to_state} - {keys}')

        f.edge(from_state,to_state)

    dotfile = filetitle + ".dot"
    f.render(filename=dotfile)

def plotTransitionEffectOnKPIs(sl0_df,sl1_df,sl2_df, agent, TikZExp=False):
    """
    The function makes the plots of Figure 7, those that show how two KPIs at a time
    vary according to the labels we have given to the current transition 
    (same-PRB, same-sched, distinct, self)
    -
    Parameters
    _______________
    
    three pandas dataframes, one per slice with columns:
    (trans  tx_brate  tx_pkts  dl_buffer  trans_cls)

    Output
    _______________


    """
    outdir="../results/plots-interpretation-agents/"+str(agent)+"/"

    #^ medoids computation

    classes = ['same-prbs', 'same-pol', 'distinct', 'self']

    #^ Plot of tx_brate vs dl_buffer

    fig,(ax0,ax1,ax2) = plt.subplots(nrows=1,ncols=3,squeeze=True,figsize=(10,4))

    #^ first pair: tx_brate and dl_buffer
    txbrate_dlbuff_medoids_sl0 = []
    txbrate_dlbuff_medoids_sl1 = []
    txbrate_dlbuff_medoids_sl2 = []

    for i, cl in enumerate(classes):
        medoid_i_sl0 = computeMedoid(sl0_df['tx_brate'][(sl0_df.trans_cls == cl)], sl0_df['dl_buffer'][(sl0_df.trans_cls == cl)])
        medoid_i_sl1 = computeMedoid(sl1_df['tx_brate'][(sl1_df.trans_cls == cl)], sl1_df['dl_buffer'][(sl1_df.trans_cls == cl)])
        medoid_i_sl2 = computeMedoid(sl2_df['tx_brate'][(sl2_df.trans_cls == cl)], sl2_df['dl_buffer'][(sl2_df.trans_cls == cl)])
        # print(f'TXBRATE vs DL_BUFFER: Medoid {cl}: {medoid_i_sl0[0]},{medoid_i_sl0[1]} ({i})')
        txbrate_dlbuff_medoids_sl0.append(medoid_i_sl0)
        txbrate_dlbuff_medoids_sl1.append(medoid_i_sl1)
        txbrate_dlbuff_medoids_sl2.append(medoid_i_sl2)

    #^ second pair: tx_packets and dl_buffer
    txpkts_dlbuff_medoids_sl0 = []
    txpkts_dlbuff_medoids_sl1 = []
    txpkts_dlbuff_medoids_sl2 = []

    for i, cl in enumerate(classes):
        medoid_i_sl0 = computeMedoid(sl0_df['tx_pkts'][(sl0_df.trans_cls == cl)], sl0_df['dl_buffer'][(sl0_df.trans_cls == cl)])
        medoid_i_sl1 = computeMedoid(sl1_df['tx_pkts'][(sl1_df.trans_cls == cl)], sl1_df['dl_buffer'][(sl1_df.trans_cls == cl)])
        medoid_i_sl2 = computeMedoid(sl2_df['tx_pkts'][(sl2_df.trans_cls == cl)], sl2_df['dl_buffer'][(sl2_df.trans_cls == cl)])
        # print(f'TX_PKTS vs DL_BUFFER: Medoid {cl}: {medoid_i_sl0[0]},{medoid_i_sl0[1]} ({i})')
        txpkts_dlbuff_medoids_sl0.append(medoid_i_sl0)
        txpkts_dlbuff_medoids_sl1.append(medoid_i_sl1)
        txpkts_dlbuff_medoids_sl2.append(medoid_i_sl2)

    #^ third pair: tx_brate and tx_packets
    txbrate_txpkts_medoids_sl0 = []
    txbrate_txpkts_medoids_sl1 = []
    txbrate_txpkts_medoids_sl2 = []

    for i, cl in enumerate(classes):
        medoid_i_sl0 = computeMedoid(sl0_df['tx_brate'][(sl0_df.trans_cls == cl)], sl0_df['tx_pkts'][(sl0_df.trans_cls == cl)])
        medoid_i_sl1 = computeMedoid(sl1_df['tx_brate'][(sl1_df.trans_cls == cl)], sl1_df['tx_pkts'][(sl1_df.trans_cls == cl)])
        medoid_i_sl2 = computeMedoid(sl2_df['tx_brate'][(sl2_df.trans_cls == cl)], sl2_df['tx_pkts'][(sl2_df.trans_cls == cl)])
        # print(f'tx_brate vs TX_PKTS: Medoid {cl}: {medoid_i_sl0[0]},{medoid_i_sl0[1]} ({i})')
        txbrate_txpkts_medoids_sl0.append(medoid_i_sl0)
        txbrate_txpkts_medoids_sl1.append(medoid_i_sl1)
        txbrate_txpkts_medoids_sl2.append(medoid_i_sl2)

    #^ SL0
    ax0.set(xlabel='SL0', ylabel='DL buffer size')
    ax0.scatter(txbrate_dlbuff_medoids_sl0[0][0], txbrate_dlbuff_medoids_sl0[0][1],
        marker='x',
        color='blue',
        label='same-prbs')
    ax0.scatter(txbrate_dlbuff_medoids_sl0[1][0], txbrate_dlbuff_medoids_sl0[1][1],
        marker='s',
        color='green',
        label='same-pol')
    ax0.scatter(txbrate_dlbuff_medoids_sl0[2][0], txbrate_dlbuff_medoids_sl0[2][1],
        marker='*',
        color='orange',
        label='distinct')
    ax0.scatter(sl0_df['tx_brate'][(sl0_df.trans_cls == 'self')], sl0_df['dl_buffer'][(sl0_df.trans_cls == 'self')],
        marker='D',
        color='red',
        label='self')
    #^ SL1
    ax1.set(xlabel='SL1\n tx_brate')
    ax1.scatter(txbrate_dlbuff_medoids_sl1[0][0], txbrate_dlbuff_medoids_sl1[0][1],
        marker='x',
        color='blue',
        label='same-prbs')
    ax1.scatter(txbrate_dlbuff_medoids_sl1[1][0], txbrate_dlbuff_medoids_sl1[1][1],
        marker='s',
        color='green',
        label='same-pol')
    ax1.scatter(txbrate_dlbuff_medoids_sl1[2][0], txbrate_dlbuff_medoids_sl1[2][1],
        marker='*',
        color='orange',
        label='distinct')
    ax1.scatter(sl1_df['tx_brate'][(sl1_df.trans_cls == 'self')], sl1_df['dl_buffer'][(sl1_df.trans_cls == 'self')],
        marker='D',
        color='red',
        label='self')
    #^ SL2 
    ax2.set(xlabel='SL2')
    ax2.scatter(txbrate_dlbuff_medoids_sl2[0][0], txbrate_dlbuff_medoids_sl2[0][1],
        marker='x',
        color='blue',
        label='same-prbs')
    ax2.scatter(txbrate_dlbuff_medoids_sl2[1][0], txbrate_dlbuff_medoids_sl2[1][1],
        marker='s',
        color='green',
        label='same-pol')
    ax2.scatter(txbrate_dlbuff_medoids_sl2[2][0], txbrate_dlbuff_medoids_sl2[2][1],
        marker='*',
        color='orange',
        label='distinct')
    ax2.scatter(sl2_df['tx_brate'][(sl2_df.trans_cls == 'self')], sl2_df['dl_buffer'][(sl2_df.trans_cls == 'self')],
        marker='D',
        color='red',
        label='self')    

    plt.legend(ncol=4, bbox_to_anchor=[0.3, 1.15])
    fig.subplots_adjust(top=0.9,bottom=0.15)
    
    filename = outdir + "tx_brate-vs-dl_buffer"

    if TikZExp:
        full_filename = filename + ".tex"
        tikzplotlib.save(full_filename)
    else:
        full_filename = filename + ".png"
        plt.show()
        plt.savefig(full_filename)
    plt.close()

    #^ Plot of tx_brate vs tx_pkts
    fig,(ax0,ax1,ax2) = plt.subplots(nrows=1,ncols=3,squeeze=True,figsize=(10,4))

    #^ SL0
    ax0.set(xlabel='SL0', ylabel='tx_pkts')
    ax0.scatter(txbrate_txpkts_medoids_sl0[0][0], txbrate_txpkts_medoids_sl0[0][1],
        marker='x',
        color='blue',
        label='same-prbs')
    ax0.scatter(txbrate_txpkts_medoids_sl0[1][0], txbrate_txpkts_medoids_sl0[1][1],
        marker='s',
        color='green',
        label='same-pol')
    ax0.scatter(txbrate_txpkts_medoids_sl0[2][0], txbrate_txpkts_medoids_sl0[2][1],
        marker='*',
        color='orange',
        label='distinct')
    ax0.scatter(sl0_df['tx_brate'][(sl0_df.trans_cls == 'self')], sl0_df['tx_pkts'][(sl0_df.trans_cls == 'self')],
        marker='D',
        color='red',
        label='self')
    #^ SL1
    ax1.set(xlabel='SL1\n tx_brate')
    ax1.scatter(txbrate_txpkts_medoids_sl1[0][0], txbrate_txpkts_medoids_sl1[0][1],
        marker='x',
        color='blue',
        label='same-prbs')
    ax1.scatter(txbrate_txpkts_medoids_sl1[1][0], txbrate_txpkts_medoids_sl1[1][1],
        marker='s',
        color='green',
        label='same-pol')    
    ax1.scatter(txbrate_txpkts_medoids_sl1[2][0], txbrate_txpkts_medoids_sl1[2][1],
        marker='*',
        color='orange',
        label='distinct')
    ax1.scatter(sl1_df['tx_brate'][(sl1_df.trans_cls == 'self')], sl1_df['tx_pkts'][(sl1_df.trans_cls == 'self')],
        marker='D',
        color='red',
        label='self')
    #^ SL2 
    ax2.set(xlabel='SL2')
    ax2.scatter(txbrate_txpkts_medoids_sl2[0][0], txbrate_txpkts_medoids_sl2[0][1],
        marker='x',
        color='blue',
        label='same-prbs')
    ax2.scatter(txbrate_txpkts_medoids_sl2[1][0], txbrate_txpkts_medoids_sl2[1][1],
        marker='s',
        color='green',
        label='same-pol')  
    ax2.scatter(txbrate_txpkts_medoids_sl2[2][0], txbrate_txpkts_medoids_sl2[2][1],
        marker='*',
        color='orange',
        label='distinct')
    ax2.scatter(sl2_df['tx_brate'][(sl2_df.trans_cls == 'self')], sl2_df['tx_pkts'][(sl2_df.trans_cls == 'self')],
        marker='D',
        color='red',
        label='self')

    plt.legend(ncol=4, bbox_to_anchor=[0.3, 1.15])
    fig.subplots_adjust(top=0.9,bottom=0.15)

    filename = outdir + "tx_brate-vs-tx_pkts"

    if TikZExp:
        full_filename = filename + ".tex"
        tikzplotlib.save(full_filename)
    else:
        full_filename = filename + ".png"
        plt.show()
        plt.savefig(full_filename)
    plt.close()

    #^ Plot of tx_pkts vs dl_buffer

    fig,(ax0,ax1,ax2) = plt.subplots(nrows=1,ncols=3,squeeze=True,figsize=(10,4))

    #^ SL0
    ax0.set(xlabel='SL0', ylabel='DL buffer size')
    ax0.scatter(sl0_df['tx_pkts'][(sl0_df.trans_cls == 'same-prbs')], sl0_df['dl_buffer'][(sl0_df.trans_cls == 'same-prbs')],
        marker='x',
        color='blue',
        label='same-prbs')
    ax0.scatter(sl0_df['tx_pkts'][(sl0_df.trans_cls == 'self')], sl0_df['dl_buffer'][(sl0_df.trans_cls == 'self')],
        marker='D',
        color='red',
        label='self')
    ax0.scatter(sl0_df['tx_pkts'][(sl0_df.trans_cls == 'same-pol')], sl0_df['dl_buffer'][(sl0_df.trans_cls == 'same-pol')],
        marker='s',
        color='green',
        label='same-pol')
    ax0.scatter(sl0_df['tx_pkts'][(sl0_df.trans_cls == 'distinct')], sl0_df['dl_buffer'][(sl0_df.trans_cls == 'distinct')],
        marker='*',
        color='orange',
        label='distinct')
    #^ SL1
    ax1.set(xlabel='SL1\n tx_pkts')
    ax1.set_xlim([min(sl1_df['tx_pkts']), max(sl1_df['tx_pkts'])])
    ax1.set_ylim([min(sl1_df['dl_buffer']), max(sl1_df['dl_buffer'])])
    ax1.scatter(sl1_df['tx_pkts'][(sl1_df.trans_cls == 'same-prbs')], sl1_df['dl_buffer'][(sl1_df.trans_cls == 'same-prbs')],
        marker='x',
        color='blue',
        label='same-prbs')
    ax1.scatter(sl1_df['tx_pkts'][(sl1_df.trans_cls == 'same-pol')], sl1_df['dl_buffer'][(sl1_df.trans_cls == 'same-pol')],
        marker='s',
        color='green',
        label='same-pol')
    ax1.scatter(sl1_df['tx_pkts'][(sl1_df.trans_cls == 'distinct')], sl1_df['dl_buffer'][(sl1_df.trans_cls == 'distinct')],
        marker='*',
        color='orange',
        label='distinct')
    ax1.scatter(sl1_df['tx_pkts'][(sl1_df.trans_cls == 'self')], sl1_df['dl_buffer'][(sl1_df.trans_cls == 'self')],
        marker='D',
        color='red',
        label='self')
    #^ SL2 
    ax2.set(xlabel='SL2')
    ax2.set_xlim([min(sl2_df['tx_pkts']), max(sl2_df['tx_pkts'])])
    ax2.set_ylim([min(sl2_df['dl_buffer']), max(sl2_df['dl_buffer'])])
    ax2.scatter(sl2_df['tx_pkts'][(sl2_df.trans_cls == 'same-prbs')], sl2_df['dl_buffer'][(sl2_df.trans_cls == 'same-prbs')],
        marker='x',
        color='blue',
        label='same-prbs')
    ax2.scatter(sl2_df['tx_pkts'][(sl2_df.trans_cls == 'same-pol')], sl2_df['dl_buffer'][(sl2_df.trans_cls == 'same-pol')],
        marker='s',
        color='green',
        label='same-pol')
    ax2.scatter(sl2_df['tx_pkts'][(sl2_df.trans_cls == 'distinct')], sl2_df['dl_buffer'][(sl2_df.trans_cls == 'distinct')],
        marker='*',
        color='orange',
        label='distinct')
    ax2.scatter(sl2_df['tx_pkts'][(sl2_df.trans_cls == 'self')], sl2_df['dl_buffer'][(sl2_df.trans_cls == 'self')],
        marker='D',
        color='red',
        label='self')

    plt.legend(ncol=4, bbox_to_anchor=[0.3, 1.15])
    fig.subplots_adjust(top=0.9,bottom=0.15)

    filename = outdir + "tx_pkts-vs-dl_buffer"

    if TikZExp:
        full_filename = filename + ".tex"
        tikzplotlib.save(full_filename)
    else:
        full_filename = filename + ".png"
        plt.show()
        plt.savefig(full_filename)
    plt.close()

def createDTonGraph(sl0_df,sl1_df,sl2_df, agent, ExportTree=False):
    """
    The function makes the plots of Figure 7, those that show how two KPIs at a time
    vary according to the labels we have given to the current transition 
    (same-PRB, same-sched, distinct, self)
    -
    Parameters
    _______________
    
    three pandas dataframes, one per slice with columns:
    (trans  tx_brate  tx_pkts  dl_buffer  trans_cls)
    agent: current agent under analysis
    ExportTree: to save the generated figure

    Output
    _______________


    """
    
    outdir="../results/dts-interpretations/"+str(agent)+"/"

    transition_classes = ['same-PRB', 'same-sched', 'distinct', 'self']
    list_of_dfs = [sl0_df,sl1_df,sl2_df]

    ##^^ Printing the panda dataframes
    # for slice in {0,1,2}:
    #     print(f'---\n Slice {slice}\n {list_of_dfs[slice]}')

    ##^^ buildling DTs per slice
    
    for sl_idx in range (0,3):
        df_considered = list_of_dfs[sl_idx]
        slice_tree_filename = outdir+"slice-"+str(sl_idx)+"_tree.png"
        df_considered["trans"] = df_considered["trans"].astype(str)

        # print(slice_tree_filename)

        feature_cols = ['tx_brate', 'dl_buffer'] #['tx_brate', 'tx_pkts'] #, 'dl_buffer']
        X = df_considered[feature_cols] # Features
        y = df_considered.trans_cls # Target variable

        #^ Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

        # print(f'Lens: Train {len(X_train)}, Test {len(X_test)}, - Y train {len(y_train)}, test {len(y_test)}')

        #^ Create Decision Tree classifer object
        clf = DecisionTreeClassifier()

        #^ Train Decision Tree Classifer
        clf = clf.fit(X_train,y_train)

        #^ Predict the response for test dataset
        y_pred = clf.predict(X_test)

        #^ Model Accuracy, how often is the classifier correct?
        # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

        if ExportTree:
            dot_data = StringIO()
            export_graphviz(clf, out_file=dot_data,  
                            filled=True, rounded=True,
                            special_characters=True,
                            feature_names = feature_cols,
                            class_names = transition_classes
                            )
            graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
            graph.write_png(slice_tree_filename)
            Image(graph.create_png())


def analysisAgentBehavior(global_dict_kpi_state_transition,global_dict_kpi_distr_state,agent):
    """
    Note: same traffic scenario means that we do this for experiments with different number of users

    -
    Parameters
    _______________
    

    Output
    _______________


    """

    print(f'Current set of experiments under analysis: {agent} \n-------')

    #^ dataframes for each slice
    cols = ['trans','tx_brate', 'tx_pkts', 'dl_buffer', 'trans_cls']
    sl0_df = pd.DataFrame([],columns=cols)
    sl1_df = pd.DataFrame([],columns=cols)
    sl2_df = pd.DataFrame([],columns=cols)

    #^ this is to count statistics on the type of transitions
    dict_classes_stats = {}

    ##^^ iteratate over all the possible transitions:
    for keys, values in global_dict_kpi_state_transition.items():
        #^ get FROM and TO states
        from_state, to_state = getFromToStates(keys)
        
        #^ get PRBs and Pol of both FROM and TO states
        from_state_prbs, from_state_pol = getFromPRBPol(from_state)
        to_state_prbs, to_state_pol = getFromPRBPol(to_state)

        #^ classes of transitions:
        transition_class = None
        if from_state_prbs == to_state_prbs and from_state_pol == to_state_pol:
            transition_class = "self"
        elif from_state_prbs == to_state_prbs and from_state_pol != to_state_pol:
            transition_class = "same-prbs"
        elif from_state_prbs != to_state_prbs and from_state_pol == to_state_pol:
            transition_class = "same-pol"
        elif from_state_prbs != to_state_prbs and from_state_pol != to_state_pol:
            transition_class = "distinct"

        # print(f'From {from_state} to {to_state} - Class: {transition_class}')

        #^ adding to the stats
        if transition_class not in dict_classes_stats:
            dict_classes_stats[transition_class] = 1
        else:
            dict_classes_stats[transition_class] += 1

        #^ get the KPIs from global_dict_kpi_distr_state
        from_state_kpi = global_dict_kpi_distr_state[tuple(from_state)]
        to_state_kpi = global_dict_kpi_distr_state[tuple(to_state)]       

        #^ iterate first over the "FROM" state
        for el_fs in from_state_kpi:
            for el_f, el_f_values in el_fs.items():
                #^ variables to compare the KPIs (tx brate, tx pkts, dl buffer) per slice
                from_kpis_values_sl0 = None
                from_kpis_values_sl1 = None
                from_kpis_values_sl2 = None
                #^ now I access the from values and store them
                list_of_sl = ["SL0","SL1","SL2"]
                for sl in list_of_sl:
                    value = el_f_values[sl]
                    if sl == "SL0":
                        from_kpis_values_sl0 = value
                    if sl == "SL1":
                        from_kpis_values_sl1 = value
                    if sl == "SL2":
                        from_kpis_values_sl2 = value
                #^ for each of the FROM elements, take a look if we can find the correspondent num. of users in the TO state
                #  if yes: pick the KPIs as well to compare
                for el_ts in to_state_kpi:
                    for el_t, el_t_values in el_ts.items():
                        #^ if the two are equal, then we can make the comparison
                        if el_t == el_f:
                            to_kpis_values_sl0 = None
                            to_kpis_values_sl1 = None
                            to_kpis_values_sl2 = None
                            for sl in list_of_sl:
                                value = el_t_values[sl]
                                if sl == "SL0":
                                    to_kpis_values_sl0 = value
                                    #^ adding to the df sl0_df
                                    sl0_df.loc[len(sl0_df)] = {'trans': keys,
                                     'tx_brate':round(from_kpis_values_sl0.avg_tx_brate-to_kpis_values_sl0.avg_tx_brate,2),
                                     'tx_pkts':round(from_kpis_values_sl0.avg_tx_pkts-to_kpis_values_sl0.avg_tx_pkts,2), 
                                     'dl_buffer':round(from_kpis_values_sl0.avg_dl_buffer-to_kpis_values_sl0.avg_dl_buffer,2), 
                                     'trans_cls':transition_class}
                                if sl == "SL1":
                                    to_kpis_values_sl1 = value
                                    #^ adding to the df sl0_df
                                    sl1_df.loc[len(sl1_df)] ={'trans': keys,
                                     'tx_brate':round(from_kpis_values_sl1.avg_tx_brate-to_kpis_values_sl1.avg_tx_brate,2),
                                     'tx_pkts':round(from_kpis_values_sl1.avg_tx_pkts-to_kpis_values_sl1.avg_tx_pkts,2), 
                                     'dl_buffer':round(from_kpis_values_sl1.avg_dl_buffer-to_kpis_values_sl1.avg_dl_buffer,2), 
                                     'trans_cls':transition_class}
                                if sl == "SL2":
                                    to_kpis_values_sl2 = value
                                    #^ adding to the df sl0_df
                                    sl2_df.loc[len(sl2_df)] = {'trans': keys,
                                     'tx_brate':round(from_kpis_values_sl2.avg_tx_brate-to_kpis_values_sl2.avg_tx_brate,2),
                                     'tx_pkts':round(from_kpis_values_sl2.avg_tx_pkts-to_kpis_values_sl2.avg_tx_pkts,2), 
                                     'dl_buffer':round(from_kpis_values_sl2.avg_dl_buffer-to_kpis_values_sl2.avg_dl_buffer,2), 
                                     'trans_cls':transition_class}

                            ##^ At this point we have all the info to compare the KPIs of one transition of one experiment
                            # print(f'SL0: tx_brate ({from_kpis_values_sl0.avg_tx_brate} {to_kpis_values_sl0.avg_tx_brate}) {round(from_kpis_values_sl0.avg_tx_brate-to_kpis_values_sl0.avg_tx_brate,2)}\
                            #     tx_pkts ({from_kpis_values_sl0.avg_tx_pkts} {to_kpis_values_sl0.avg_tx_pkts}) {round(from_kpis_values_sl0.avg_tx_pkts-to_kpis_values_sl0.avg_tx_pkts,2)}\
                            #     dl_buffer ({from_kpis_values_sl0.avg_dl_buffer} {to_kpis_values_sl0.avg_dl_buffer}) {round(from_kpis_values_sl0.avg_dl_buffer-to_kpis_values_sl0.avg_dl_buffer,2)}')
                            # print(f'SL1: tx_brate ({from_kpis_values_sl1.avg_tx_brate} {to_kpis_values_sl1.avg_tx_brate}) {round(from_kpis_values_sl1.avg_tx_brate-to_kpis_values_sl1.avg_tx_brate,2)}\
                            #     tx_pkts ({from_kpis_values_sl1.avg_tx_pkts} {to_kpis_values_sl1.avg_tx_pkts}) {round(from_kpis_values_sl1.avg_tx_pkts-to_kpis_values_sl1.avg_tx_pkts,2)}\
                            #     dl_buffer ({from_kpis_values_sl1.avg_dl_buffer} {to_kpis_values_sl1.avg_dl_buffer}) {round(from_kpis_values_sl1.avg_dl_buffer-to_kpis_values_sl1.avg_dl_buffer,2)}')
                            # print(f'SL2: tx_brate ({from_kpis_values_sl2.avg_tx_brate} {to_kpis_values_sl2.avg_tx_brate}) {round(from_kpis_values_sl2.avg_tx_brate-to_kpis_values_sl2.avg_tx_brate,2)}\
                            #     tx_pkts ({from_kpis_values_sl2.avg_tx_pkts} {to_kpis_values_sl2.avg_tx_pkts}) {round(from_kpis_values_sl2.avg_tx_pkts-to_kpis_values_sl2.avg_tx_pkts,2)}\
                            #     dl_buffer ({from_kpis_values_sl2.avg_dl_buffer} {to_kpis_values_sl2.avg_dl_buffer}) {round(from_kpis_values_sl2.avg_dl_buffer-to_kpis_values_sl2.avg_dl_buffer,2)}')
                            # print()

    ##^^ at this stage, all the information is collected: for each slice, we know for each transition the KPI variation
    plotTransitionEffectOnKPIs(sl0_df,sl1_df,sl2_df,agent,TikZExp=False)# use this one to plot

    createDTonGraph(sl0_df,sl1_df,sl2_df, agent, ExportTree=True)# use this one to create a DT that analyzes the explanations

    #^ exporting the stats on the transitions
    print("Stats. on classes of actions:")
    tot_items = sum(dict_classes_stats.values())
    for k,v in dict_classes_stats.items():
        print(f'{k}: {v} | {round((v/tot_items)*100,2)}%')

if __name__ == "__main__":
 
    ##^^ Analysis for to retrieve interpretations

    #^ defining the data structures: list_exp_<agent> makes it possible to run all experiments at once 
    g_d_kpi_state_transition_embb_trf1, g_d_kpi_distr_state_embb_trf1 = processExperiments(list_exp_embb_trf1,agent_embb_trf1)
    g_d_kpi_state_transition_embb_trf2, g_d_kpi_distr_state_embb_trf2 = processExperiments(list_exp_embb_trf2,agent_embb_trf2)
    g_d_kpi_state_transition_urllc_trf1, g_d_kpi_distr_state_urllc_trf1 = processExperiments(list_exp_urllc_trf1,agent_urllc_trf1)
    g_d_kpi_state_transition_urllc_trf2, g_d_kpi_distr_state_urllc_trf2 = processExperiments(list_exp_urllc_trf2,agent_urllc_trf2)

    exp_selection = {"embb-trf1":[g_d_kpi_state_transition_embb_trf1,g_d_kpi_distr_state_embb_trf1],
                     "embb-trf2":[g_d_kpi_state_transition_embb_trf2,g_d_kpi_distr_state_embb_trf2],
                     "urllc-trf1":[g_d_kpi_state_transition_urllc_trf1,g_d_kpi_distr_state_urllc_trf1],
                     "urllc-trf2":[g_d_kpi_state_transition_urllc_trf2,g_d_kpi_distr_state_urllc_trf2],
                     }

    #^ choose: 0 for  "embb-trf1", 1 for "embb-trf2", 2 for "urllc-trf1", and 3 for "urllc-trf2"
    agent = list(exp_selection.keys())[2]
    # configures the proper structures to be analyzed
    g_d_kpi_state_trans = exp_selection[agent][0]
    g_d_kpi_distr_state = exp_selection[agent][1]

    #^ visualizes and characterizes the graph without attributes (only NODES, i.e., actions)
    visualizeGraph(g_d_kpi_state_trans,g_d_kpi_distr_state,agent)
    
    #^ analysis to synthesize explanations in form of plots that highlight effect of transitions on two KPIs at a time 
    analysisAgentBehavior(g_d_kpi_state_trans,g_d_kpi_distr_state,agent)

    #^ clearing the data structures
    g_d_kpi_state_trans.clear()
    g_d_kpi_distr_state.clear()

    print("-- Finished main --")
