from pygenn.genn_model import init_var
import numpy as np

N_PATTERNS_TRAIN = 10000
N_PATTERNS_TEST = 100

T_EPISODE = 35.
T_EPISODE_EXAMPLE = 35.
T_PATTERN = 5.
T0_ATTENTION = 0.
T1_ATTENTION = 35.
T_OUTPUT_ACT = 10.
WIDTH_OUTPUT_ACT = 2.0

T_TRAIN = N_PATTERNS_TRAIN * T_EPISODE
T_TEST = N_PATTERNS_TEST * T_EPISODE
T_EXAMPLE = 10 * T_EPISODE_EXAMPLE

N_I = 1
N_H = 500
N_O = 10

N_BATCH = 1

DT = .05
DT_DATA = .05

NT_SKIP_REC = 10

NT_TRAIN = int(T_TRAIN / DT)
NT_TEST = int(T_TEST / DT)
NT_EXAMPLE = int(T_EXAMPLE / DT)

NT_EPISODE = int(T_EPISODE / DT)
NT_EPISODE_EXAMPLE = int(T_EPISODE_EXAMPLE / DT)

assert NT_TRAIN == NT_EPISODE * N_PATTERNS_TRAIN
assert NT_TEST == NT_EPISODE * N_PATTERNS_TEST
assert NT_EXAMPLE == NT_EPISODE_EXAMPLE * 10

assert NT_EPISODE % NT_SKIP_REC == 0
assert NT_EPISODE_EXAMPLE % NT_SKIP_REC == 0

NT_PLAST_STEP = int(NT_EPISODE / 10)

NT_REC_EPISODE = int(NT_EPISODE/NT_SKIP_REC)
NT_REC_EPISODE_EXAMPLE = int(NT_EPISODE_EXAMPLE/NT_SKIP_REC)

NT_DATA_IN_MAX = NT_TRAIN
NT_DATA_OUT_MAX = NT_TRAIN

EVENT_BASED = False

##########################

P_H_PARAMS = {"th": 1e-1}
P_H_VAR_INIT = {
    "h": 0.0,
    "r_min_1": 0.0,
    "r": 0.0,
    "r_trace": 0.0,
    "r_trace_min_1": 0.0,
    "r_trace_min_2": 0.0,
    "dr_min_2": 1.0,
    "dr_min_1": 1.0,
    "dr": 1.0,
    "b": 0.01,
    "db": 0.0,
    "scale_fb": 0.0,
    "err_fb": 0.0,
    "r_event": 0.0,
    "r_prev_event": 0.0
}

P_O_PARAMS = {"th": 1e-3}
P_O_VAR_INIT = {
    "r": 0.0,
    "r_targ": 0.0,
    "err": 0.0,
    "err_event": 0.0,
    "err_prev_event": 0.0,
    "learning": 1,
    "attention": 0.0,
    "b": 0.0,
    "db": 0.0
}

P_I_PARAMS = {"th": 1e-3}
P_I_VAR_INIT = {
    "r_min_1": 0.0,
    "r": 0.0,
    "r_trace": 0.0,
    "r_trace_min_1": 0.0,
    "r_trace_min_2": 0.0,
    "r_event": 0.0,
    "r_prev_event": 0.0,
    "b": 0.0
}

W_HH_PARAMS = {"w_penalty": 0e-5}
W_HH_VAR_INIT = {
    "g": init_var("Normal", {"mean": 0.0, "sd": 0.98/np.sqrt(N_H)}),
    "inp_prev": 0.0,
    "dg": 0.0,
    "dg_new": 0.0
}

W_OH_PARAMS = {"w_penalty": 0e-6}
W_OH_VAR_INIT = {
    "g": init_var("Uniform", {"min": -1.0/np.sqrt(N_H), "max": 1.0/np.sqrt(N_H)}),
    "inp_prev": 0.0,
    "dg": 0.0,
    "dg_new": 0.0
}

W_HO_PARAMS = {"w_penalty": 0e-6}
W_HO_VAR_INIT = {
    "g": init_var("Normal", {"mean": 0.0, "sd": 0.5/np.sqrt(N_O)}),
    "inp_prev": 0.0,
    "dg": 0.0,
    "dg_new": 0.0
}

W_HI_PARAMS = {"w_penalty": 0e-5}
W_HI_VAR_INIT = {
    "g": init_var("Normal", {"mean": 0.0, "sd": 0.5/np.sqrt(N_I)}),
    "inp_prev": 0.0,
    "dg": 0.0,
    "dg_new": 0.0
}



#'''
LR_SCALE = 1.5
default_adam_params = {"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-5}

P_H_OPT_PARAMS = {"optimizer": "adam",
                  "optimizer_params": {"lr": LR_SCALE * 1e-4} | default_adam_params}

P_O_OPT_PARAMS = {"optimizer": "adam",
                  "optimizer_params": {"lr": LR_SCALE * 1e-4} | default_adam_params}

W_HH_OPT_PARAMS = {"optimizer": "adam",
                   "optimizer_params": {"lr": LR_SCALE * 1e-4} | default_adam_params}

W_OH_OPT_PARAMS = {"optimizer": "adam",
                   "optimizer_params": {"lr": LR_SCALE * 1e-4} | default_adam_params}

W_HO_OPT_PARAMS = {"optimizer": "adam",
                   "optimizer_params": {"lr": LR_SCALE * 1e-4} | default_adam_params}

W_HI_OPT_PARAMS = {"optimizer": "adam",
                   "optimizer_params": {"lr": LR_SCALE * 1e-4} | default_adam_params}
#'''
'''
P_H_OPT_PARAMS = {"optimizer": "sgd",
                  "optimizer_params": {"lr": LR_SCALE * 1e-2}}

P_O_OPT_PARAMS = {"optimizer": "sgd",
                  "optimizer_params": {"lr": LR_SCALE * 1e-3}}

W_HH_OPT_PARAMS = {"optimizer": "sgd",
                   "optimizer_params": {"lr": LR_SCALE * 1e-2}}

W_OH_OPT_PARAMS = {"optimizer": "sgd",
                   "optimizer_params": {"lr": LR_SCALE * 1e-3}}

W_HO_OPT_PARAMS = {"optimizer": "sgd",
                   "optimizer_params": {"lr": LR_SCALE * 1e-3}}

W_HI_OPT_PARAMS = {"optimizer": "sgd",
                   "optimizer_params": {"lr": LR_SCALE * 1e-2}}
'''

'''
LR_SCALE = 0.05

default_momentum_params = {"beta": 0.9}

P_H_OPT_PARAMS = {"optimizer": "momentum",
                  "optimizer_params": {"lr": LR_SCALE * 1e-2} | default_momentum_params}

P_O_OPT_PARAMS = {"optimizer": "momentum",
                  "optimizer_params": {"lr": LR_SCALE * 1e-3} | default_momentum_params}

W_HH_OPT_PARAMS = {"optimizer": "momentum",
                   "optimizer_params": {"lr": LR_SCALE * 1e-2} | default_momentum_params}

W_OH_OPT_PARAMS = {"optimizer": "momentum",
                   "optimizer_params": {"lr": LR_SCALE * 1e-3} | default_momentum_params}

W_HO_OPT_PARAMS = {"optimizer": "momentum",
                   "optimizer_params": {"lr": LR_SCALE * 1e-3} | default_momentum_params}

W_HI_OPT_PARAMS = {"optimizer": "momentum",
                   "optimizer_params": {"lr": LR_SCALE * 1e-2} | default_momentum_params}
'''


MOD_PARAMS = {
    "p_h_params": P_H_PARAMS,
    "p_h_var_init": P_H_VAR_INIT,
    "p_h_opt_params": P_H_OPT_PARAMS,
    "p_o_params": P_O_PARAMS,
    "p_o_var_init": P_O_VAR_INIT,
    "p_o_opt_params": P_O_OPT_PARAMS,
    "p_i_params": P_I_PARAMS,
    "p_i_var_init": P_I_VAR_INIT,
    "w_hh_params": W_HH_PARAMS,
    "w_hh_var_init": W_HH_VAR_INIT,
    "w_hh_opt_params": W_HH_OPT_PARAMS,
    "w_oh_params": W_OH_PARAMS,
    "w_oh_var_init": W_OH_VAR_INIT,
    "w_oh_opt_params": W_OH_OPT_PARAMS,
    "w_ho_params": W_HO_PARAMS,
    "w_ho_var_init": W_HO_VAR_INIT,
    "w_ho_opt_params": W_HO_OPT_PARAMS,
    "w_hi_params": W_HI_PARAMS,
    "w_hi_var_init": W_HI_VAR_INIT,
    "w_hi_opt_params": W_HI_OPT_PARAMS
}