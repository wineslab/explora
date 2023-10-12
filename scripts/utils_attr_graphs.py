"""
Attributed graph

10/12/2023
_______________
    
Summary: functions for the handling attributed graphs

DISCLAIMER: THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING 
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""

##^^ imports
from utils_db_process import *

##^^ settings
state_zero = [0,0,0] # this sets an initial state for PRB allocation and scheduling policy

def populateSliceState(metrics):
    """
    
    Parameters
    _______________
    


    Output
    _______________

    
    """

    sl0 = []
    sl1 = []
    sl2 = []
    for value in metrics:
        if int(value.slice_id) == 0:
            sl0.append(int(value.slice_prb))
        if int(value.slice_id) == 1:
            sl1.append(int(value.slice_prb))
        if int(value.slice_id) == 2:
            sl2.append(int(value.slice_prb))
    return sl0, sl1, sl2

def populateUEKPIState(metrics):
    """
    
    Parameters
    _______________
    
    metrics:

    Output
    _______________

    sl0, sl1, sl2: metrics per individual slice
    
    """

    sl0 = []
    sl1 = []
    sl2 = []
    for value in metrics:
        if int(value.slice_id) == 0:
            uekpi = ueKPI(int(value.dl_buffer),float(value.tx_brate),int(value.tx_pkts))
            sl0.append(uekpi)
        if int(value.slice_id) == 1:
            uekpi = ueKPI(int(value.dl_buffer),float(value.tx_brate),int(value.tx_pkts))
            sl1.append(uekpi)
        if int(value.slice_id) == 2:
            uekpi = ueKPI(int(value.dl_buffer),float(value.tx_brate),int(value.tx_pkts))
            sl2.append(uekpi)
    return sl0, sl1, sl2    

def defineState(prev_sl0,prev_sl1,prev_sl2):
    """
    
    Parameters
    _______________
    


    Output
    _______________

    
    """
    prb_sl0 = 0
    prb_sl1 = 0
    prb_sl2 = 0 
    if len(prev_sl0) > 0 and len(prev_sl1) > 0 and len(prev_sl2) > 0:
        if prev_sl0.count(prev_sl0[0]) == len(prev_sl0):
            prb_sl0 = prev_sl0[0]
        else:
            prb_sl0 = mostFrequent(prev_sl0)
        if prev_sl1.count(prev_sl1[0]) == len(prev_sl1):
            prb_sl1 = prev_sl1[0]
        else:
            prb_sl1 = mostFrequent(prev_sl1)
        if prev_sl2.count(prev_sl2[0]) == len(prev_sl2):
            prb_sl2 = prev_sl2[0]
        else:
            prb_sl2 = mostFrequent(prev_sl2)

    prb_l = [prb_sl0,prb_sl1,prb_sl2]

    return prb_l   


def createDictKPIStateTransitions(dict_ue_metrics, dict_sched_pol, exp):
    """
    Function to create the a complex dictionary:
    - key: the state (tuple [[PRBs], [SCHED]])
    - value: dictionary with
             - key: slice id [0,1,2]
             - value: list of KPIs [tx_brate, tx_pkts, dl_buffer]
    
    Parameters
    _______________
    
    dict_ue_metrics: dictionary with metrics reported by the UEs at a given timestep
    dict_sched_pol: dictionary with the scheduling policy at a given timestep
    exp: current experiment

    Output
    _______________
    
    dict_kpi_ues: the complex dictionary
    dict_trans: dictionary with transitions useful for later analysis
    
    """

    ##^^ Populating a dictionary with state and corresponding explanation
    dict_kpi_ues = {}

    ##^^ Function variables
    dict_state = {}
    dict_trans = {}
    

    for j in range(2,len(dict_ue_metrics)):
        curr_metrics = dict_ue_metrics[j]
        curr_scheduling = dict_sched_pol[j]

        prev_metrics = dict_ue_metrics[j-1]
        prev_scheduling = dict_sched_pol[j-1]

        # print(f'{j} - {curr_scheduling} ',end='')

        prev_sl0, prev_sl1, prev_sl2 = populateSliceState(prev_metrics)
        curr_sl0, curr_sl1, curr_sl2 = populateSliceState(curr_metrics)

        prev_kpi_sl0, prev_kpi_sl1, prev_kpi_sl2 = populateUEKPIState(prev_metrics)
        curr_kpi_sl0, curr_kpi_sl1, curr_kpi_sl2 = populateUEKPIState(curr_metrics)


        ###^^^ 3 operations: state, transition (for the graph), populating the dictionary with distribution of the UE metrics to return

        ##^^ Definition of a state: populate dict_state

        prb_l = defineState(prev_sl0,prev_sl1,prev_sl2)
        if prb_l != state_zero:
            drl_sched_l=list(map(int, prev_scheduling.split(",")))
            state = prb_l + drl_sched_l
        
            if tuple(state) in dict_state:
                dict_state[tuple(state)]+=1
            else:
                dict_state[tuple(state)]=1

            ##^^ definition of a transition: populate dict_trans
            prb_l_to = defineState(curr_sl0,curr_sl1,curr_sl2)
            drl_sched_l_to = drl_sched_l=list(map(int, curr_scheduling.split(",")))
            state_to = prb_l_to + drl_sched_l_to

            from_to_transition = tuple(state + state_to)

            if from_to_transition in dict_trans:
                dict_trans[from_to_transition]+=1
            else:
                dict_trans[from_to_transition]=1


            ##^ creation final dictionary
            if tuple(state) in dict_kpi_ues:
                #^ update individual slices
                dict_kpi_ues[tuple(state)][0].append(prev_kpi_sl0)
                dict_kpi_ues[tuple(state)][1].append(prev_kpi_sl1)
                dict_kpi_ues[tuple(state)][2].append(prev_kpi_sl2)

            else:
                #^ create dictionary for individual slices
                dict_sl = {0: [], 1: [], 2: []} # new local dictionary
                dict_sl[0]=[prev_kpi_sl0]
                dict_sl[1]=[prev_kpi_sl1]
                dict_sl[2]=[prev_kpi_sl2]
                #^ storing into the complete dictionary
                dict_kpi_ues[tuple(state)]=dict_sl

    ##^^ debugging printing
    
    # for state_keys, state_items in dict_kpi_ues.items():
    #     print(f'Key: {state_keys} - UE KPIs: {state_items.keys()}')
    #     for ue_key, ue_item in state_items.items():
    #         print(f'SL{ue_key}: ')
    #         for el in ue_item:
    #             for e in el:
    #                 print(f'{e.tx_brate} {e.tx_pkts} {e.dl_buffer} ', end = ' ')
    #         print()
    #     print()

    return dict_kpi_ues, dict_trans

def getFromToStates(keys):
    
    #^ split in two to retrieve from and to states
    elements = keys[1:-1].split(',')
    from_state = [int(elements[el]) for el in range(6)]
    to_state = [int(elements[el]) for el in range(6,12)]

    return from_state, to_state

def getFromPRBPol(state):
    
    # print(state,end=' ')
    #^ split in two to retrieve from and to states
    elements = str(state)[1:-1].split(',')
    st_prb = [int(elements[el]) for el in range(3)]
    st_pol = [int(elements[el]) for el in range(3,6)]

    # print(f'{st_prb} {st_pol}')
    return st_prb, st_pol