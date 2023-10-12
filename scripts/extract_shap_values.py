"""
Temporal Analysis with experiments run on SCOPE
    
10/12/2023
_______________
    
Summary: script to use SHAP to explain directly the DRL agent

USE: just set the experiment number (start_exp),
     the corresponding agent (agent), and run it

NOTE: the script returns on cmd line the execution time

DISCLAIMER: THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING 
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""

####^^^^ Imports and settings
##^^ misc
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import pickle as pkl
from datetime import datetime, timedelta
from scipy.stats import mode
from collections import Counter
import time

##^^ importing tensorflow
import tensorflow as tf
from tf_agents.trajectories import time_step as ts
from typing import NamedTuple
from tf_agents.typing import types

##^^ importing project functions
from utils_shap import *

def generate_timestep_for_policy(obs_tmp=None):
    step_type = tf.convert_to_tensor(
        [0], dtype=tf.int32, name='step_type')
    reward = tf.convert_to_tensor(
        [0], dtype=tf.float32, name='reward')
    discount = tf.convert_to_tensor(
        [1], dtype=tf.float32, name='discount')
    observations = tf.convert_to_tensor(
        [obs_tmp], dtype=tf.float32, name='observations')
    return ts.TimeStep(step_type, reward, discount, observations)

def funcDRLAgent(inputs):
    # print('Input shape: ', inputs.shape)
    inputs = tf.cast(inputs, dtype='float32')

    step_type = tf.convert_to_tensor(
        [0], dtype=tf.int32, name='step_type')
    reward = tf.convert_to_tensor(
        [0], dtype=tf.float32, name='reward')
    discount = tf.convert_to_tensor(
        [1], dtype=tf.float32, name='discount')

    if inputs.shape[0] > 1:
        observations = tf.convert_to_tensor(
            [inputs[l] for l in range(inputs.shape[0])], dtype=tf.float32, name='observations')
        out = tf.convert_to_tensor(
            [infer(discount=discount, observation=tf.expand_dims(observations[l],axis=0), reward=reward,
                   step_type=step_type)['action'] for l in range(inputs.shape[0])])
    else:
        out = tf.convert_to_tensor(
            infer(discount=discount, observation=inputs, reward=reward, step_type=step_type)[
                'action'])
        out = tf.expand_dims(out, axis=1)

    out = tf.cast(out, dtype=tf.float32)
    return out

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim=1):
        self.output_dim = output_dim
        self._trainable = False
        super(CustomLayer, self).__init__()
        self.step_type = tf.convert_to_tensor(
            [0], dtype=tf.int32, name='step_type')
        self.reward = tf.convert_to_tensor(
            [0], dtype=tf.float32, name='reward')
        self.discount = tf.convert_to_tensor(
            [1], dtype=tf.float32, name='discount')

    def build(self, input_shape):
        super(CustomLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=False):
        self._trainable = training

        if inputs.shape[0] is not None:
            if inputs.shape[0] > 1:
                observations = tf.convert_to_tensor(
                    [inputs[l] for l in range(inputs.shape[0])], dtype=tf.float32, name='observations')
                # tmp = [infer(discount=self.discount, observation=observations[l], reward=self.reward, step_type=self.step_type)['action'] for l in range(inputs.shape[0])]
                out = tf.convert_to_tensor(
                    [infer(discount=self.discount, observation=observations[l], reward=self.reward, step_type=self.step_type)['action'] for l in range(inputs.shape[0])])
            else:
                # observations = tf.convert_to_tensor(
                #     inputs, dtype=tf.float32, name='observations')
                out = tf.convert_to_tensor(
                    infer(discount=self.discount, observation=inputs[0], reward=self.reward, step_type=self.step_type)['action'])

        else:
            out = tf.compat.v1.placeholder(tf.float32, shape=(inputs.shape[0], 1))

        out = tf.cast(out, dtype=tf.float32)
        # print(out)
        return out

####^^ ANALYSIS

agent = "urllc" # choose between "embb" and "urllc"

#^ starting time
startTime = time.time()

start_exp = 32
tot_num_exp = start_exp+1 # this runs one experiment only
## NOTE: set start_exp to 1 and tot_num_exp to 9 to iterate over all the experiments
##       This is at your own risk, experiments with 4+ users take approx 26 h each
##       to complete

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Or 2, 3, etc. other than 0
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

print("Num GPUs Available: ", len(gpu_devices))

###^^^ Running the analysis
for exp in range(start_exp,tot_num_exp):
    print(f'Current experiment under analysis: {exp} \n-------')

    ##^^ Reading the pickled dataset, DRL agent checkpoints
    exp_name = "../data/motivation_main-results/exp"+str(exp)+ "/ai_log_dict.pkl"

    if agent == "embb":
        #^ path
        drl_checkpoint_folder = '../data/offline-training/pre-trained-agents/embb-agent/'    
    elif agent=="urllc":
        #^ path
        drl_checkpoint_folder = '../data/offline-training/pre-trained-agents/urllc-agent/'
    else:
        print("Please choose among [embb, urllc].")   

    ##^^ Loading the pre-trained DRL agent
    drl_checkpoint_name = drl_checkpoint_folder + '/saved_model.pb'
    drl_agents = [tf.saved_model.load(drl_checkpoint_folder)]
    infer = drl_agents[0].signatures["action"]
    infer.function_def.arg_attr[3].attr['_user_specified_name'].s = b'step_type'
    infer.function_def.arg_attr[2].attr['_user_specified_name'].s = b'reward'
    infer.function_def.arg_attr[1].attr['_user_specified_name'].s = b'discount'
    infer.function_def.arg_attr[0].attr['_user_specified_name'].s = b'observation'
    infer.structured_input_signature[1]['0/observation'] = tf.TensorSpec(shape=(None, 9), dtype=tf.float32, name='observation')
    infer.structured_input_signature[1]['0/reward'] = tf.TensorSpec(shape=(None, 9), dtype=tf.float32, name='reward')
    infer.structured_input_signature[1]['0/discount'] = tf.TensorSpec(shape=(None, 9), dtype=tf.float32, name='discount')
    infer.structured_input_signature[1]['0/step_type'] = tf.TensorSpec(shape=(None, 9), dtype=tf.int32, name='step_type')
    infer.structured_input_signature[1]['observation'] = infer.structured_input_signature[1]['0/observation']
    infer.structured_input_signature[1]['step_type'] = infer.structured_input_signature[1]['0/step_type']
    infer.structured_input_signature[1]['discount'] = infer.structured_input_signature[1]['0/discount']
    infer.structured_input_signature[1]['reward'] = infer.structured_input_signature[1]['0/reward']
    del(infer.structured_input_signature[1]['0/observation'])
    del (infer.structured_input_signature[1]['0/reward'])
    del (infer.structured_input_signature[1]['0/discount'])
    del (infer.structured_input_signature[1]['0/step_type'])
    infer._arg_keywords = ['discount', 'observation', 'reward', 'step_type']

    infer._function_spec.fullargspec.kwonlydefaults['reward'] = infer._function_spec.fullargspec.kwonlydefaults['0/reward']
    infer._function_spec.fullargspec.kwonlydefaults['observation'] = infer._function_spec.fullargspec.kwonlydefaults[
        '0/observation']
    infer._function_spec.fullargspec.kwonlydefaults['discount'] = infer._function_spec.fullargspec.kwonlydefaults[
        '0/discount']
    infer._function_spec.fullargspec.kwonlydefaults['step_type'] = infer._function_spec.fullargspec.kwonlydefaults[
        '0/step_type']
    del(infer._function_spec.fullargspec.kwonlydefaults['0/observation'])
    del (infer._function_spec.fullargspec.kwonlydefaults['0/reward'])
    del (infer._function_spec.fullargspec.kwonlydefaults['0/discount'])
    del (infer._function_spec.fullargspec.kwonlydefaults['0/step_type'])

    ##^^ Reading the dataset: picking timestamps and printing them into datetime format
    ##   ---
    ##   As UEs initially have to attach, at the beginning the first samples
    ##   have to be discarded. Use a threshold of 2 min

    f = open(exp_name, 'rb')
    dataset = pkl.load(f)
    # print(dataset.keys())

    datetime_arr = []
    for j in dataset.keys():
        datetime_arr.append(datetime.utcfromtimestamp(j/1000.0))#.strftime('%Y-%m-%d %H:%M:%S'))

    time_threshold = timedelta(minutes=2, seconds=0)
    cut_th=0 # this is telling at which point i
    for i in range(1,len(datetime_arr)):
        diff=datetime_arr[i]-datetime_arr[i-1]
        if diff > time_threshold:
            cut_th=i
    datetime_arr=datetime_arr[cut_th+1:]

    # print(dataset.values())

    ##^^ Picking up the inputs of the DRL agent, i.e., the outputs of the autoencoder
    output_ae = []

    for idx, item in enumerate(dataset.values()):
        if idx > cut_th:
            output_ae.append(item['AUTOENCODER_OUTPUT'])

    ##^^ formatting the data to be explained 
    data_exp = np.array(output_ae)
    data_exp = data_exp[100:200] # DE-COMMENT if you want to try the script on a sub-sample of the full array

    explainer = shap.KernelExplainer(funcDRLAgent, data=data_exp)
    shap_values = explainer.shap_values(data_exp)  # calculates shap values

    # print(shap_values)
    # print(np.array(shap_values).shape)

    ##^^ storing the shap values in binary files for later use
    outdir="../results/shap_explanations/shap-values/exp-"+str(exp)+"/"
    pathExist = os.path.exists(outdir)
    if not pathExist:
        os.makedirs(outdir)
    outfile=outdir+"explanation-new.npy" # adding "-new" to avoid overriding the original files (although are the same, see file size)
    # for the format see: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format

    #^ writing to file
    with open(outfile, 'wb') as f:
        np.save(f, np.array(shap_values))

###^^^ computing execution time
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
