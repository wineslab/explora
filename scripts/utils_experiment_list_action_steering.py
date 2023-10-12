"""
Information on Experiments on Action Steering

10/12/2023
_______________
    
Summary: utility to process experiments on action steering

DISCLAIMER: THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING 
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""

##^^ Choose strategy in exp_strategy by selecting the index:
## 0: baseline
## 1: max_reward
## 2: max_reward_obs_20
## 3: min_reward
## 4: min_reward_obs_20
## 5: imp_bitrate
## 6: imp_bitrate_obs_20
## 7: baseline_obs_20

#^ experiments - winter-2023
exp_baseline = {"embb-trf1": 27,"embb-trf2": 29,"urllc-trf1": 28,"urllc-trf2": 30}
exp_max_reward = {"embb-trf1": 31,"embb-trf2": 33,"urllc-trf1": 32,"urllc-trf2": 34}

#^ experiments - spring-2023
exp_max_reward_obs_20 = {"embb-trf1": 9,"embb-trf2": 11,"urllc-trf1": 10,"urllc-trf2": 12}
exp_min_reward = {"embb-trf1": 13,"embb-trf2": 17,"urllc-trf1": 14,"urllc-trf2": 18}
exp_min_reward_obs_20 = {"embb-trf1": 15,"embb-trf2": 19,"urllc-trf1": 16,"urllc-trf2": 20}
exp_imp_tx_brate= {"embb-trf1": 25,"embb-trf2": 27,"urllc-trf1": 26,"urllc-trf2": 28}
exp_imp_tx_brate_obs_20 = {"embb-trf1": 29,"embb-trf2": 31,"urllc-trf1": 30,"urllc-trf2": 32}
exp_baseline_obs_20 = {"embb-trf1": 21,"embb-trf2": 23,"urllc-trf1": 22,"urllc-trf2": 24}

#^ configuration
exp_configuration = {"baseline":[exp_baseline,"winter-2023"],
                     "max_reward":[exp_max_reward,"winter-2023"],
                     "max_reward_obs_20":[exp_max_reward_obs_20,"spring-2023"],
                     "min_reward":[exp_min_reward,"spring-2023"],
                     "min_reward_obs_20":[exp_min_reward_obs_20,"spring-2023"],
                     "imp_bitrate":[exp_imp_tx_brate,"spring-2023"],
                     "imp_bitrate_obs_20":[exp_imp_tx_brate_obs_20,"spring-2023"],
                     "baseline_obs_20":[exp_baseline_obs_20,"spring-2023"]
                    }

list_agents = ["embb-trf1", "embb-trf2", "urllc-trf1", "urllc-trf2"]