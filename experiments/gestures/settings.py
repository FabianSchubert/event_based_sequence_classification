from pygenn.genn_model import init_var
import numpy as np

N_EPOCHS = 100

N_FOLDS = 50

N_DIM_DATA = 9
T_BUFFER = 1
N_H = 750
N_O = 11

N_BATCH = 1

#DT = 0.05
DT = 0.25

TAU_H = 2.5

EVIDENCE_THRESHOLD = 0.3
TAU_EVIDENCE = 1.5

EVENT_BASED = True

INPUT_FILES = ["na", "s", "j", "l", "ni"]

TRAIN_SPLIT = 0.8

SPIKE_RECORDING_POPS = ["p_i", "p_h", "p_o"]
MAX_SPIKE_RECORDING_STEPS = 1000

INPUT_MODE = "shift"
USE_KERNEL_EMBEDDING = True

RANDOMIZE_T_BUFFER_OFFSET = True

R_EMBED = 10.
N_EMBED = 500
R_STEP_FACTOR = 1.25

##########################

# TH_SCALE = 1e-2

P_I_PARAMS = {"th": 0.5, "tau_trace": TAU_H}
P_I_VAR_INIT = {
    "r": 0.0,
    "r_event": 0.0,
    "r_prev_event": 0.0,
    "r_trace": 0.0,
}

P_H_PARAMS = {"th": 0.25, "tau": TAU_H}
P_H_VAR_INIT = {
    "r": 0.0,
    # "r_trace": 0.0,
    "x": 0.0,
    "dr": 1.0,
    "b": 0.0,
    "db": 0.0,
    "err_fb": 0.0,
    "r_event": 0.0,
    "r_prev_event": 0.0,
    "dr_err_prod": 0.0,
    "weight_factor": 1.0,
}

P_O_PARAMS = {"th": 0.025}
P_O_VAR_INIT = {
    "x": 0.0,
    "targ": 0.0,
    "r": 0.0,
    "dr": 0.0,
    "err": 0.0,
    "err_event": 0.0,
    "err_prev_event": 0.0,
    "loss": 0.0,
    "b": 0.0,
    "db": 0.0,
}

W_HI_PARAMS = {"w_penalty": 0e-3}
W_HI_VAR_INIT = {
    "g": init_var("Normal", {"mean": 0.0, "sd": 0.5 / np.sqrt(N_DIM_DATA * T_BUFFER)}),
    "inp_prev": 0.0,
    "dg": 0.0,
    "dg_prev": 0.0,
}

W_OH_PARAMS = {"w_penalty": 0e-4}
W_OH_VAR_INIT = {
    "g": init_var("Normal", {"mean": 0.0, "sd": 0.5 / np.sqrt(N_H)}),
    "inp_prev": 0.0,
    "dg": 0.0,
    "dg_prev": 0.0,
}

W_HO_PARAMS = {"w_penalty": 0e-4}
W_HO_VAR_INIT = {
    "g": init_var("Normal", {"mean": 0.0, "sd": 0.5 / np.sqrt(N_O)}),
    "inp_prev": 0.0,
    "dg": 0.0,
    "dg_prev": 0.0,
}

LR_SCALE = 3e-4
default_adam_params = {"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-7}

P_H_OPT_PARAMS = {
    "optimizer": "adam",
    "optimizer_params": {"lr": LR_SCALE * 1.0} | default_adam_params,
}

P_O_OPT_PARAMS = {
    "optimizer": "adam",
    "optimizer_params": {"lr": LR_SCALE * 1.0} | default_adam_params,
}

W_HI_OPT_PARAMS = {
    "optimizer": "adam",
    "optimizer_params": {"lr": LR_SCALE * 1.0} | default_adam_params,
}

W_OH_OPT_PARAMS = {
    "optimizer": "adam",
    "optimizer_params": {"lr": LR_SCALE * 1.0} | default_adam_params,
}

W_OO_OPT_PARAMS = {
    "optimizer": "adam",
    "optimizer_params": {"lr": LR_SCALE * 1.0} | default_adam_params,
}

W_HO_OPT_PARAMS = {
    "optimizer": "adam",
    "optimizer_params": {"lr": LR_SCALE * 1.0} | default_adam_params,
}

MOD_PARAMS = {
    "p_i_params": P_I_PARAMS,
    "p_i_var_init": P_I_VAR_INIT,
    "p_h_params": P_H_PARAMS,
    "p_h_var_init": P_H_VAR_INIT,
    "p_h_opt_params": P_H_OPT_PARAMS,
    "p_o_params": P_O_PARAMS,
    "p_o_var_init": P_O_VAR_INIT,
    "p_o_opt_params": P_O_OPT_PARAMS,
    "w_hi_params": W_HI_PARAMS,
    "w_hi_var_init": W_HI_VAR_INIT,
    "w_hi_opt_params": W_HI_OPT_PARAMS,
    "w_oh_params": W_OH_PARAMS,
    "w_oh_var_init": W_OH_VAR_INIT,
    "w_oh_opt_params": W_OH_OPT_PARAMS,
    "w_ho_params": W_HO_PARAMS,
    "w_ho_var_init": W_HO_VAR_INIT,
    "w_ho_opt_params": W_HO_OPT_PARAMS,
}
