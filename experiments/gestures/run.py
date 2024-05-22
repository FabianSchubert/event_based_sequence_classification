#! /usr/bin/env python3

import matplotlib.pyplot as plt

from model.network import NetworkGestureBufferFF

from embedding import step_kernel_embedding

from torch.utils.data import DataLoader

from datetime import datetime

import numpy as np

from gesture_dataset.DataSet import UniHHIMUGestures, generate_train_test_split

import os

import json

import argparse

FILE_BASE = os.path.dirname(__file__)

with open(os.path.join(FILE_BASE, "settings.json"), "r") as f:
    SETTINGS = json.load(f)


INPUT_FILES = SETTINGS["INPUT_FILES"]
N_BATCH = SETTINGS["N_BATCH"]
N_DIM_DATA = SETTINGS["N_DIM_DATA"]
T_BUFFER = SETTINGS["T_BUFFER"]
N_H = SETTINGS["N_H"]
N_O = SETTINGS["N_O"]
DT = SETTINGS["DT"]
TAU_EVIDENCE = SETTINGS["TAU_EVIDENCE"]
N_EPOCHS = SETTINGS["N_EPOCHS"]
EVENT_BASED = SETTINGS["EVENT_BASED"]
EVIDENCE_THRESHOLD = SETTINGS["EVIDENCE_THRESHOLD"]
MOD_PARAMS = SETTINGS["MOD_PARAMS"]
N_FOLDS = SETTINGS["N_FOLDS"]
TRAIN_SPLIT = SETTINGS["TRAIN_SPLIT"]
SPIKE_RECORDING_POPS = SETTINGS["SPIKE_RECORDING_POPS"]
MAX_SPIKE_RECORDING_STEPS = SETTINGS["MAX_SPIKE_RECORDING_STEPS"]
INPUT_MODE = SETTINGS["INPUT_MODE"]
RANDOMIZE_T_BUFFER_OFFSET = SETTINGS["RANDOMIZE_T_BUFFER_OFFSET"]
USE_KERNEL_EMBEDDING = SETTINGS["USE_KERNEL_EMBEDDING"]
R_EMBED = SETTINGS["R_EMBED"]
N_EMBED = SETTINGS["N_EMBED"]
R_STEP_FACTOR = SETTINGS["R_STEP_FACTOR"]


parser = argparse.ArgumentParser(description="Gesture classification")
parser.add_argument(
    "--threshold_scale",
    type=float,
    default=1.0,
    help="scale the threshold for the hidden and the output layer",
)

args = parser.parse_args()

TH_SCALE = args.threshold_scale

SETTINGS["TH_SCALE"] = TH_SCALE

MOD_PARAMS["p_h_params"]["th"] *= TH_SCALE
MOD_PARAMS["p_o_params"]["th"] *= TH_SCALE

BASE_FOLD = os.path.dirname(__file__)


train_data, test_data = generate_train_test_split(
    dataDir="/home/fabian/Work/Repos/UHH-IMU-gestures-comparison/gesture_dataset/dataSets/",
    inputFiles=INPUT_FILES,
    train_split=TRAIN_SPLIT,
    shuffle=True,
    nFolds=N_FOLDS,
    nRepeat=1,
)

if USE_KERNEL_EMBEDDING:
    _seed = np.random.randint(int(1e6))

    for i, (inputs, targets) in enumerate(train_data):
        train_data[i] = (
            step_kernel_embedding(inputs, N_EMBED, R_EMBED, R_STEP_FACTOR, seed=_seed)[
                0
            ],
            targets,
        )
    for i, (inputs, targets) in enumerate(test_data):
        test_data[i] = (
            step_kernel_embedding(inputs, N_EMBED, R_EMBED, R_STEP_FACTOR, seed=_seed)[
                0
            ],
            targets,
        )


trainset = UniHHIMUGestures(data=(train_data, test_data), train=True)

testset = UniHHIMUGestures(data=(train_data, test_data), train=False)

trainloader = DataLoader(trainset, batch_size=N_BATCH, shuffle=True, num_workers=1)
testloader = DataLoader(testset, batch_size=N_BATCH, shuffle=True, num_workers=1)


NT_DATA_MAX = 0
for _i, _t in trainloader:
    NT_DATA_MAX = np.maximum(NT_DATA_MAX, _i.shape[1])
for _i, _t in testloader:
    NT_DATA_MAX = np.maximum(NT_DATA_MAX, _i.shape[1])

#####################################################

# import pdb; pdb.set_trace()

# Network
network = NetworkGestureBufferFF(
    N_EMBED if USE_KERNEL_EMBEDDING else N_DIM_DATA,
    T_BUFFER,
    N_H,
    N_O,
    DT,
    TAU_EVIDENCE,
    NT_DATA_MAX,
    NT_DATA_MAX,
    EVENT_BASED,
    MOD_PARAMS,
    n_batch=N_BATCH,
    spike_recoring_pops=SPIKE_RECORDING_POPS,
    max_spike_recording_steps=MAX_SPIKE_RECORDING_STEPS,
    input_mode=INPUT_MODE,
)


loss, T = network.train_network(
    N_EPOCHS,
    trainloader,
    align_fb_weight_init=True,
    print_train_loss=True,
    randomize_t_buffer_offset=RANDOMIZE_T_BUFFER_OFFSET,
)


(
    label_targ,
    label_pred,
    false_pos,
    false_neg,
    i_rec,
    h_rec,
    o_rec,
    o_x_rec,
    targ_rec,
    evidence_rec,
    evidence_rec_bin,
    spike_rec,
) = network.test_network(
    testloader,
    ev_th=EVIDENCE_THRESHOLD,
    return_spike_rec=True,
    randomize_t_buffer_offset=RANDOMIZE_T_BUFFER_OFFSET,
)


print("test loss: ", 0.5 * ((targ_rec - o_rec) ** 2.0).mean())

plt.style.use(
    "https://raw.githubusercontent.com/FabianSchubert/mpl_style/main/custom_style.mplstyle"
)


if EVENT_BASED:
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))

    ax[0].plot(spike_rec["p_i"][0], spike_rec["p_i"][1], ".", markersize=1)
    ax[1].plot(spike_rec["p_h"][0], spike_rec["p_h"][1], ".", markersize=1)
    ax[2].plot(spike_rec["p_o"][0], spike_rec["p_o"][1], ".", markersize=1)

    ax[0].set_title("input")
    ax[1].set_title("hidden")
    ax[2].set_title("output")

    ax[0].set_xlabel("time steps")
    ax[1].set_xlabel("time steps")
    ax[2].set_xlabel("time steps")

    ax[0].set_ylabel("neuron index")
    ax[1].set_ylabel("neuron index")
    ax[2].set_ylabel("neuron index")

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            BASE_FOLD,
            "figures/spikes_test.png",
        )
    )


fig, ax = plt.subplots(2, 1)

ax[0].pcolormesh(targ_rec[0, :1000].T)
# ax[1].pcolormesh(o_rec[0, :1000].T)
ax[1].pcolormesh(evidence_rec[0, :1000].T)
# ax[1].pcolormesh(evidence_rec_bin[0, :1000].T)

ax[0].set_title("target")
ax[1].set_title("prediction")

ax[0].set_xlabel("time steps")
ax[1].set_xlabel("time steps")

ax[0].set_ylabel("class index")
ax[1].set_ylabel("class index")

fig.tight_layout()

fig.savefig(
    os.path.join(
        BASE_FOLD,
        f"figures/output_test_{'event_based' if EVENT_BASED else 'continuous'}.png",
    )
)

plt.close()

fig, ax = plt.subplots()
ax.plot(loss)
# ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("MSE loss")

fig.savefig(
    os.path.join(
        BASE_FOLD,
        f"figures/train_loss_{'event_based' if EVENT_BASED else 'continuous'}.png",
    )
)

"""
if EVENT_BASED:
    fig, ax = plt.subplots()

    network.network.pull_recording_buffers_from_device()
    ax.plot(
        network.p_r.spike_recording_data[0],
        network.p_r.spike_recording_data[1],
        ".",
        markersize=2,
    )
    ax.set_xlim(
        [
            network.p_r.spike_recording_data[0].min(),
            network.p_r.spike_recording_data[0].min() + 250 * DT,
        ]
    )

    fig.savefig(os.path.join(BASE_FOLD, "figures/spikes.png"))
"""

np.savez(
    os.path.join(
        BASE_FOLD,
        f"results_data/train_results_{datetime.now().strftime('%H-%M_%d-%m-%y')}.npz",
    ),
    event_based=EVENT_BASED,
    t_buffer=T_BUFFER,
    n_h=N_H,
    n_epochs=N_EPOCHS,
    label_targ=label_targ,
    label_pred=label_pred,
    err=1.0 - (label_targ == label_pred).mean(),
    false_pos=false_pos.mean(),
    false_neg=false_neg.mean(),
    h_rec=h_rec,
    o_x_rec=o_x_rec,
    o_rec=o_rec,
    targ_rec=targ_rec,
    train_time=T,
    train_loss=loss,
    settings=SETTINGS,
)
