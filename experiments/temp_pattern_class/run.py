#! /usr/bin/env python3

from model.network import Network

import numpy as np

from datasets.mnist1d import mnist1d_dataset_array

import os

from .settings import *

BASE_FOLD = os.path.dirname(__file__)

########################################
data_train_in, data_train_out, attention_train = mnist1d_dataset_array(N_PATTERNS_TRAIN, t=T_EPISODE,
                                                      t_pattern=T_PATTERN,
                                                      t0_attention=T0_ATTENTION,
                                                      t1_attention=T1_ATTENTION,
                                                      t_output_act=T_OUTPUT_ACT,
                                                      width_output_act=WIDTH_OUTPUT_ACT,
                                                      dt=DT_DATA,
                                                      sigm_uncorr_noise=0.0, sigm_corr_noise=0.0)

data_train_in = np.expand_dims(data_train_in, 1)
data_train_out = np.expand_dims(data_train_out, 1)

network = Network(N_I, N_H, N_O, DT, NT_DATA_IN_MAX, NT_DATA_OUT_MAX, EVENT_BASED, MOD_PARAMS)

weights_start = network.weights

train_time, loss = network.train_network(NT_TRAIN, data_train_in, data_train_out,
                                         DT_DATA, True, attention_train,
                                         align_fb_weight_init=True,
                                         nt_plast_step=NT_PLAST_STEP,
                                         nt_skip_rec=NT_SKIP_REC)
weights_end = network.weights
########################################

########################################
data_test_in, data_test_out, attention_test = mnist1d_dataset_array(N_PATTERNS_TEST, t=T_EPISODE,
                                                    t_pattern=T_PATTERN,
                                                    t0_attention=T0_ATTENTION,
                                                    t1_attention=T1_ATTENTION,
                                                    t_output_act=T_OUTPUT_ACT,
                                                    width_output_act=WIDTH_OUTPUT_ACT,
                                                    dt=DT_DATA,
                                                    sigm_uncorr_noise=0.0, sigm_corr_noise=0.0)

data_test_in = np.expand_dims(data_test_in, 1)
data_test_out = np.expand_dims(data_test_out, 1)


r_targ, r_o, err, r_in = network.test_network(NT_TEST, data_test_in, data_test_out,
                                              DT_DATA, True, attention_test,
                                              nt_skip_rec=NT_SKIP_REC,
                                              t_wash_in=2.*T_EPISODE)

loss_test = 0.5 * (err[:,:]**2.).mean(axis=1)

r_targ_ep = np.reshape(r_targ.T, (N_O, N_PATTERNS_TEST, NT_REC_EPISODE))
r_o_ep = np.reshape(r_o.T, (N_O, N_PATTERNS_TEST, NT_REC_EPISODE))

labels_targ = np.argmax(r_targ_ep.mean(axis=-1), axis=0)
labels_o = np.argmax(r_o_ep.mean(axis=-1), axis=0)

acc = (1.*(labels_o == labels_targ)).mean()

output_id = np.ndarray((NT_TEST//NT_SKIP_REC, N_O))
output_id[:,:] = np.arange(N_O)

np.savez(os.path.join(BASE_FOLD, "../../data_results/results_temp_pattern_class.npz"),
         train_time=train_time, loss_train=loss,
         r_targ_test=r_targ, r_out_test=r_o,
         r_in_test=r_in, accuracy=acc,
         labels_targ=labels_targ,
         labels_output=labels_o,
         t_ax_test=np.arange(NT_TEST//NT_SKIP_REC) * DT * NT_SKIP_REC,
         output_id=output_id,
         nt_episode=NT_REC_EPISODE)
###########################################

###########################################
data_example_in, data_example_out, attention_example = mnist1d_dataset_array(np.arange(10).tolist(), t=T_EPISODE_EXAMPLE,
                                                                             t_pattern=T_PATTERN,
                                                                             t0_attention=T0_ATTENTION,
                                                                             t1_attention=T1_ATTENTION,
                                                                             t_output_act=T_OUTPUT_ACT,
                                                                             width_output_act=WIDTH_OUTPUT_ACT,
                                                                             dt=DT_DATA,
                                                                             sigm_uncorr_noise=0.0, sigm_corr_noise=0.0)

data_example_in = np.expand_dims(data_example_in, 1)
data_example_out = np.expand_dims(data_example_out, 1)

r_targ, r_o, err, r_in = network.test_network(NT_EXAMPLE, data_example_in, data_example_out,
                                              DT_DATA, True, attention_example,
                                              nt_skip_rec=NT_SKIP_REC,
                                              t_wash_in=2.*T_EPISODE)

loss_test = 0.5 * (err[:,:]**2.).mean(axis=1)

r_targ_ep = np.reshape(r_targ.T, (N_O, 10, NT_REC_EPISODE_EXAMPLE))
r_o_ep = np.reshape(r_o.T, (N_O, 10, NT_REC_EPISODE_EXAMPLE))

labels_targ = np.argmax(r_targ_ep.mean(axis=-1), axis=0)
labels_o = np.argmax(r_o_ep.mean(axis=-1), axis=0)

acc = (1.*(labels_o == labels_targ)).mean()

output_id = np.ndarray((NT_EXAMPLE//NT_SKIP_REC, N_O))
output_id[:,:] = np.arange(N_O)

np.savez(os.path.join(BASE_FOLD, "../../data_results/example_temp_pattern_class.npz"),
         r_targ_test=r_targ, r_out_test=r_o,
         r_in_test=r_in, accuracy=acc,
         labels_targ=labels_targ,
         labels_output=labels_o,
         t_ax_test=np.arange(NT_EXAMPLE//NT_SKIP_REC) * DT * NT_SKIP_REC,
         output_id=output_id,
         nt_episode=NT_REC_EPISODE_EXAMPLE)

##########################################