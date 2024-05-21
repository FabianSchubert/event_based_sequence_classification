#! /usr/bin/env python3

import numpy as np

from pygenn.genn_model import GeNNModel, create_wu_var_ref, create_var_ref

from .neurons import neur_h, neur_i, neur_o#, neur_r
from .synapses import (
    w_u_rec,
    w_u_rec_cont,
    w_u_out,
    w_u_out_cont,
    w_u_in,
    w_u_in_cont,
    w_u_err_fb,
    w_u_err_fb_cont,
    w_u_hr,
    w_u_hr_cont,
    w_u_hi,
    w_u_hi_cont,
)

from .ps_models import ps_model, ps_model_cont

from .current_sources import StreamDataCS

from .custom_updates import (
    param_change_batch_reduce,
    optimizers,
    softmax_1_model,
    softmax_2_model,
    softmax_3_model,
)

import time

from tqdm import tqdm


class NetworkBase:
    def __init__(self):
        pass

    def add_weight_plast_update(self, syn_name, optimizer, optimizer_params):
        _update_reduce_batch_weight_change = self.network.add_custom_update(
            f"reduce_batch_weight_change_{syn_name}",
            "WeightChangeBatchReduce",
            param_change_batch_reduce,
            {},
            {"reducedChange": 0.0},
            {
                "change": create_wu_var_ref(
                    self.network.synapse_populations[syn_name], "dg"
                )
            },
        )

        self.network.add_custom_update(
            f"plast_step_reduced_{syn_name}",
            "Plast",
            optimizers[optimizer]["model"],
            {"batch_size": self.network.batch_size} | optimizer_params,
            optimizers[optimizer]["var_init"],
            {
                "change": create_wu_var_ref(
                    _update_reduce_batch_weight_change, "reducedChange"
                ),
                "variable": create_wu_var_ref(
                    self.network.synapse_populations[syn_name], "g"
                ),
            },
        )

    def add_bias_plast_update(self, neur_name, optimizer, optimizer_params):
        _update_reduce_batch_bias_change = self.network.add_custom_update(
            f"reduce_batch_bias_change_{neur_name}",
            "BiasChangeBatchReduce",
            param_change_batch_reduce,
            {},
            {"reducedChange": 0.0},
            {
                "change": create_var_ref(
                    self.network.neuron_populations[neur_name], "db"
                )
            },
        )

        self.network.add_custom_update(
            f"plast_step_reduced_{neur_name}",
            "Plast",
            optimizers[optimizer]["model"],
            {"batch_size": self.network.batch_size} | optimizer_params,
            optimizers[optimizer]["var_init"],
            {
                "change": create_var_ref(
                    _update_reduce_batch_bias_change, "reducedChange"
                ),
                "variable": create_var_ref(
                    self.network.neuron_populations[neur_name], "b"
                ),
            },
        )

    def reset_weight_plast_vars(self):
        for _synpop in self.network.synapse_populations.values():
            if "dg" in _synpop.vars.keys():
                _synpop.vars["dg"].view[:] = 0.0
                _synpop.push_var_to_device("dg")
            if self.event_based and ("dg_prev" in _synpop.vars.keys()):
                _synpop.vars["dg_prev"].view[:] = 0.0
                _synpop.push_var_to_device("dg_prev")

                _synpop.vars["t_prev"].view[:] = 0.0
                _synpop.push_var_to_device("t_prev")

    def reset_bias_plast_vars(self):
        for _neurpop in self.network.neuron_populations.values():
            if "db" in _neurpop.vars.keys():
                _neurpop.vars["db"].view[:] = 0.0
                _neurpop.push_var_to_device("db")

    @property
    def biases(self):
        _b_dict = {}

        for key, nrp in self.network.neuron_populations.items():
            _b = nrp.vars["b"].view
            nrp.pull_var_from_device("b")
            _b_dict[key] = np.array(_b)

        return _b_dict

    @biases.setter
    def biases(self, b_dict):
        for key, _b in b_dict.items():
            _nrp = self.network.neuron_populations[key]
            _nrp.vars["b"].view[:] = _b
            _nrp.push_var_to_device("b")

    @property
    def weights(self):
        _w_dict = {}

        for key, sp in self.network.synapse_populations.items():
            _w = sp.vars["g"].view
            sp.pull_var_from_device("g")
            _w_dict[key] = np.reshape(_w, (sp.src.size, sp.trg.size)).T

        return _w_dict

    @weights.setter
    def weights(self, w_dict):
        for key, _w in w_dict.items():
            _sp = self.network.synapse_populations[key]
            _sp.vars["g"].view[:] = _w.T.flatten()
            _sp.push_var_to_device("g")

    def reset_state_keep_weights_biases(self):
        _weights = self.weights
        _biases = self.biases

        self.network.reinitialise()
        self.network.timestep = 0

        self.weights = _weights
        self.biases = _biases


class NetworkGestureBufferFF(NetworkBase):
    def __init__(
        self,
        N_DIM_DATA_IN,
        T_BUFFER,
        N_H,
        N_O,
        DT,
        tau_evidence,
        nt_data_in_max,
        nt_data_out_max,
        event_based,
        mod_params,
        n_batch=1,
        spike_recoring_pops=[],
        max_spike_recording_steps=0,
        input_mode="shift",
    ):
        self.N_DIM_DATA = N_DIM_DATA_IN
        self.T_BUFFER = T_BUFFER
        self.input_mode = input_mode

        if input_mode == "shift":
            N_I = N_DIM_DATA_IN * T_BUFFER
        elif input_mode == "roll":
            N_I = (
                N_DIM_DATA_IN + 1
            ) * T_BUFFER  # one extra input dimension for the periodic time embedding

        self.N_I = N_I
        self.N_H = N_H
        self.N_O = N_O
        self.DT = DT
        self.tau_evidence = tau_evidence
        self.nt_data_in_max = nt_data_in_max
        self.nt_data_out_max = nt_data_out_max
        self.event_based = event_based
        self.mod_params = mod_params
        self.n_batch = n_batch
        self.spike_recoring_pops = spike_recoring_pops
        self.max_spike_recording_steps = max_spike_recording_steps

        self.P_I_PARAMS = dict(mod_params["p_i_params"])
        self.P_I_VAR_INIT = mod_params["p_i_var_init"]
        if not event_based:
            self.P_I_PARAMS["th"] = 1000.0

        self.P_H_PARAMS = dict(mod_params["p_h_params"])
        self.P_H_VAR_INIT = mod_params["p_h_var_init"]
        if not event_based:
            self.P_H_PARAMS["th"] = 1000.0

        self.P_O_PARAMS = dict(mod_params["p_o_params"])
        self.P_O_VAR_INIT = mod_params["p_o_var_init"]
        if not event_based:
            self.P_O_PARAMS["th"] = 1000.0

        self.W_HI_PARAMS = mod_params["w_hi_params"]
        self.W_HI_VAR_INIT = dict(mod_params["w_hi_var_init"])
        if not event_based:
            self.W_HI_VAR_INIT.pop("inp_prev")
            self.W_HI_VAR_INIT.pop("dg_prev")
        else:
            self.W_HI_VAR_INIT["t_prev"] = 0.0

        self.W_OH_PARAMS = mod_params["w_oh_params"]
        self.W_OH_VAR_INIT = dict(mod_params["w_oh_var_init"])
        if not event_based:
            self.W_OH_VAR_INIT.pop("inp_prev")
            self.W_OH_VAR_INIT.pop("dg_prev")
        else:
            self.W_OH_VAR_INIT["t_prev"] = 0.0

        self.W_HO_PARAMS = mod_params["w_ho_params"]
        self.W_HO_VAR_INIT = dict(mod_params["w_ho_var_init"])
        if not event_based:
            self.W_HO_VAR_INIT.pop("inp_prev")
            self.W_HO_VAR_INIT.pop("dg_prev")
        else:
            self.W_HO_VAR_INIT["t_prev"] = 0.0

        self.network = GeNNModel("float", "BufferFF")
        self.network.dT = DT
        self.network.batch_size = n_batch

        self.p_i = self.network.add_neuron_population(
            "p_i", N_I, neur_i, self.P_I_PARAMS, self.P_I_VAR_INIT
        )

        self.p_h = self.network.add_neuron_population(
            "p_h", N_H, neur_h, self.P_H_PARAMS, self.P_H_VAR_INIT
        )

        self.p_o = self.network.add_neuron_population(
            "p_o", N_O, neur_o, self.P_O_PARAMS, self.P_O_VAR_INIT
        )

        for _pop in self.spike_recoring_pops:
            self.network.neuron_populations[_pop].spike_recording_enabled = True

        self.cs_in = StreamDataCS(
            "cs_in",
            self.network,
            self.p_i,
            nt_data_in_max,
            T_BUFFER,
            stream_type=self.input_mode,
        )
        self.cs_out = StreamDataCS(
            "cs_out", self.network, self.p_o, nt_data_out_max, 1, stream_type="shift"
        )

        self.w_hi = self.network.add_synapse_population(
            "w_hi",
            "DENSE_INDIVIDUALG",
            0,
            self.p_i,
            self.p_h,
            w_u_hi if event_based else w_u_hi_cont,
            self.W_HI_PARAMS,
            self.W_HI_VAR_INIT,
            {},
            {},
            ps_model if event_based else ps_model_cont,
            {"gamma": 0.0} if event_based else {},
            {},
        )

        self.w_oh = self.network.add_synapse_population(
            "w_oh",
            "DENSE_INDIVIDUALG",
            0,
            self.p_h,
            self.p_o,
            w_u_out if event_based else w_u_out_cont,
            self.W_OH_PARAMS,
            self.W_OH_VAR_INIT,
            {},
            {},
            ps_model if event_based else ps_model_cont,
            {"gamma": 0.0} if event_based else {},
            {},
        )

        self.w_oh.ps_target_var = "Isyn_net"

        self.w_ho = self.network.add_synapse_population(
            "w_ho",
            "DENSE_INDIVIDUALG",
            0,
            self.p_o,
            self.p_h,
            w_u_err_fb if event_based else w_u_err_fb_cont,
            self.W_HO_PARAMS,
            self.W_HO_VAR_INIT,
            {},
            {},
            ps_model if event_based else ps_model_cont,
            {"gamma": 0.0} if event_based else {},
            {},
        )

        self.w_ho.ps_target_var = "Isyn_err_fb"

        self.add_weight_plast_update(
            "w_hi",
            mod_params["w_hi_opt_params"]["optimizer"],
            mod_params["w_hi_opt_params"]["optimizer_params"],
        )

        self.add_weight_plast_update(
            "w_oh",
            mod_params["w_oh_opt_params"]["optimizer"],
            mod_params["w_oh_opt_params"]["optimizer_params"],
        )

        self.add_weight_plast_update(
            "w_ho",
            mod_params["w_ho_opt_params"]["optimizer"],
            mod_params["w_ho_opt_params"]["optimizer_params"],
        )

        self.add_bias_plast_update(
            "p_h",
            mod_params["p_h_opt_params"]["optimizer"],
            mod_params["p_h_opt_params"]["optimizer_params"],
        )

        self.add_bias_plast_update(
            "p_o",
            mod_params["p_o_opt_params"]["optimizer"],
            mod_params["p_o_opt_params"]["optimizer_params"],
        )

        _sm_1 = self.network.add_custom_update(
            "softmax_1",
            "softmax1",
            softmax_1_model,
            {},
            {"MaxVal": 0.0},
            {"Val": create_var_ref(self.p_o, "x")},
        )

        _sm_2 = self.network.add_custom_update(
            "softmax_2",
            "softmax2",
            softmax_2_model,
            {},
            {"SumExpVal": 0.0},
            {
                "Val": create_var_ref(self.p_o, "x"),
                "MaxVal": create_var_ref(_sm_1, "MaxVal"),
            },
        )

        self.network.add_custom_update(
            "softmax_3",
            "softmax3",
            softmax_3_model,
            {},
            {},
            {
                "Val": create_var_ref(self.p_o, "x"),
                "MaxVal": create_var_ref(_sm_1, "MaxVal"),
                "SumExpVal": create_var_ref(_sm_2, "SumExpVal"),
                "SoftmaxVal": create_var_ref(self.p_o, "r"),
            },
        )

        self.network.build()
        self.network.load(num_recording_timesteps=self.max_spike_recording_steps)

    @property
    def weight_hi(self):
        self.w_hi.pull_var_from_device("g")
        _w = np.reshape(self.w_hi.vars["g"].view, (self.N_I, self.N_H)).T
        return _w

    @weight_hi.setter
    def weight_hi(self, w):
        _w = self.w_hi.vars["g"].view
        _w[:] = w.T.flatten()
        self.w_hi.push_var_to_device("g")

    @property
    def weight_oh(self):
        self.w_oh.pull_var_from_device("g")
        _w = np.reshape(self.w_oh.vars["g"].view, (self.N_H, self.N_O)).T
        return _w

    @weight_oh.setter
    def weight_oh(self, w):
        _w = self.w_oh.vars["g"].view
        _w[:] = w.T.flatten()
        self.w_oh.push_var_to_device("g")

    @property
    def weight_ho(self):
        self.w_ho.pull_var_from_device("g")
        _w = np.reshape(self.w_ho.vars["g"].view, (self.N_O, self.N_H)).T
        return _w

    @weight_ho.setter
    def weight_ho(self, w):
        _w = self.w_ho.vars["g"].view
        _w[:] = w.T.flatten()
        self.w_ho.push_var_to_device("g")

    def train_network(
        self,
        n_epochs,
        data_loader,
        align_fb_weight_init=False,
        print_train_loss=False,
        randomize_t_buffer_offset=False,
    ):
        if align_fb_weight_init:
            self.weight_ho = self.weight_oh.T

        self.reset_weight_plast_vars()
        self.reset_bias_plast_vars()

        loss = np.zeros((n_epochs))

        nt_epoch = 0
        for inputs, targets in data_loader:
            nt_epoch += inputs.shape[1]

        t0 = time.time()

        for k in range(n_epochs):
            for inputs, targets in data_loader:
                inputs = inputs.numpy()
                targets = targets.numpy()

                if inputs.ndim == 2:
                    inputs = np.expand_dims(inputs, axis=0)
                    targets = np.expand_dims(targets, axis=0)

                assert inputs.shape[0] == self.n_batch, "error, batch size mismatch"

                self.cs_in.set_data(
                    inputs.astype("float32"),
                    self.network.dT,
                    periodic=False,
                    randomize_t_buffer_offset=randomize_t_buffer_offset,
                )
                self.cs_out.set_data(
                    targets.astype("float32"), self.network.dT, periodic=False
                )

                _nt = inputs.shape[1]

                for t in range(_nt):
                    self.network.step_time()

                    #self.network.custom_update("softmax1")
                    #self.network.custom_update("softmax2")
                    #self.network.custom_update("softmax3")

                self.network.custom_update("WeightChangeBatchReduce")
                self.network.custom_update("BiasChangeBatchReduce")
                self.network.custom_update("Plast")

                self.weight_ho = self.weight_oh.T

                # set the weight factors to the l1 norm of outgoing weights
                self.p_h.vars["weight_factor"].view[:] = np.abs(self.weight_oh).sum(axis=0)/np.abs(self.weight_oh).sum(axis=0).mean()
                self.p_h.push_var_to_device("weight_factor")

            self.p_o.pull_var_from_device("loss")
            loss[k] = self.p_o.vars["loss"].view.mean() / nt_epoch

            self.p_o.vars["loss"].view[:] = 0.0
            self.p_o.push_var_to_device("loss")

            if print_train_loss:
                print(f"Epoch {k+1}/{n_epochs}, Loss {loss[k]}", end="\r")

        t1 = time.time()

        return loss, (t1 - t0)

    def test_network(self, data_loader, ev_th=0.5, return_spike_rec=False, randomize_t_buffer_offset=False):
        nt_epoch = 0
        for inputs, targets in data_loader:
            inputs = inputs.numpy()
            if inputs.ndim == 2:
                inputs = np.expand_dims(inputs, axis=0)
            nt_epoch += inputs.shape[1]

        evidence = np.zeros((self.n_batch, nt_epoch, self.N_O))
        evidence_bin = np.zeros((self.n_batch, nt_epoch, self.N_O))

        label_pred = [[] * self.n_batch]
        label_targ = [[] * self.n_batch]

        false_pos = [[False] * self.n_batch]

        gesture_detected = [False] * self.n_batch
        gesture_detected_corr = [False] * self.n_batch

        t_start_detect = [0] * self.n_batch

        t = 0

        for inputs, targets in tqdm(data_loader):
            inputs = inputs.numpy()
            targets = targets.numpy()

            if inputs.ndim == 2:
                inputs = np.expand_dims(inputs, axis=0)
                targets = np.expand_dims(targets, axis=0)

            assert inputs.shape[0] == self.n_batch, "error, batch size mismatch"

            _nt = inputs.shape[1]

            self.cs_in.set_data(
                inputs.astype("float32"),
                self.network.dT,
                periodic=False,
                randomize_t_buffer_offset=randomize_t_buffer_offset,
            )
            self.cs_out.set_data(
                targets.astype("float32"), self.network.dT, periodic=False
            )

            i_rec = np.ndarray((self.n_batch, _nt, self.N_I))
            i_view = self.p_i.vars["r"].view

            h_rec = np.ndarray((self.n_batch, _nt, self.N_H))
            h_view = self.p_h.vars["r"].view

            o_rec = np.ndarray((self.n_batch, _nt, self.N_O))
            o_view = self.p_o.vars["r"].view

            o_x_rec = np.ndarray((self.n_batch, _nt, self.N_O))
            o_x_view = self.p_o.vars["x"].view

            targ_rec = np.ndarray((self.n_batch, _nt, self.N_O))
            targ_view = self.p_o.vars["targ"].view

            evidence_rec = np.zeros((self.n_batch, _nt, self.N_O))
            evidence_rec_bin = np.zeros((self.n_batch, _nt, self.N_O))

            for _t in range(_nt):
                self.network.step_time()

                #self.network.custom_update("softmax1")
                #self.network.custom_update("softmax2")
                #self.network.custom_update("softmax3")

                self.p_i.pull_var_from_device("r")
                i_rec[:, _t] = i_view

                if np.any(np.isnan(i_view)):
                    import pdb

                    pdb.set_trace()

                self.p_h.pull_var_from_device("r")
                h_rec[:, _t] = h_view

                self.p_o.pull_var_from_device("r")
                o_rec[:, _t] = o_view

                self.p_o.pull_var_from_device("x")
                o_x_rec[:, _t] = o_x_view

                self.p_o.pull_var_from_device("targ")
                targ_rec[:, _t] = targ_view

                if t > 0:
                    _o = self.p_o.vars["r"].view.reshape((self.n_batch, -1))

                    #_o = (self.p_o.vars["r"].view + 1e-3).reshape((self.n_batch, -1))
                    #_o /= _o.sum(axis=1, keepdims=True)

                    #_o = np.zeros((self.n_batch, self.N_O))
                    #_o[np.arange(self.n_batch), np.argmax(o_view.reshape((self.n_batch, -1)), axis=1)] = 1.0

                    # _o = self.p_o.vars["r"].view.reshape((self.n_batch, -1))
                    # _o = np.exp(_o) / np.exp(_o).sum(axis=1, keepdims=True)

                    evidence[:, t] = evidence[:, t - 1] + self.network.dT * (
                        _o - evidence[:, t - 1]
                    ) / self.tau_evidence

                    evidence_rec[:, _t] = evidence[:, t]

                    evidence_bin[:, t] = np.zeros((self.n_batch, self.N_O))
                    evidence_bin[np.arange(self.n_batch), t, np.argmax(evidence[:, t], axis=1)] = 1.0

                    evidence_rec_bin[:, _t] = evidence_bin[:, t]

                    _ev_sm_prev = evidence[:, t - 1, :-1].sum(axis=1)
                    _ev_sm = evidence[:, t, :-1].sum(axis=1)
                    #_ev_sm_prev = 1.-evidence[:, t - 1, -1]
                    #_ev_sm = 1.-evidence[:, t, -1]
                    #_ev_sm_prev = 1.*(np.argmax(evidence[:, t - 1, :], axis=1) != 10)
                    #_ev_sm = 1.*(np.argmax(evidence[:, t, :], axis=1) != 10)

                    _label = np.argmax(targets[:, _t], axis=1)
                    _label_prev = np.argmax(targets[:, _t - 1], axis=1)

                    for k in range(self.n_batch):
                        if (_label[k] != 10) and (_label_prev[k] == 10):
                            label_targ[k].append(_label)
                            label_pred[k].append(10)

                        if (_label[k] != 10) and gesture_detected[k]:
                            gesture_detected_corr[k] = True

                        if (_label[k] == 10) and (_label_prev[k] != 10):
                            false_pos[k].append(False)

                        if (_ev_sm[k] >= ev_th) and (_ev_sm_prev[k] < ev_th):
                            gesture_detected[k] = True
                            t_start_detect[k] = t

                        if (_ev_sm[k] < ev_th) and (_ev_sm_prev[k] >= ev_th):
                            if gesture_detected_corr[k]:
                                label_pred[k][-1] = np.argmax(evidence[k, t_start_detect[k]:t, :-1].sum(axis=0))
                            else:
                                false_pos[k][-1] = True

                            gesture_detected[k] = False
                            gesture_detected_corr[k] = False

                t += 1

        label_targ = np.array(label_targ).flatten()
        label_pred = np.array(label_pred).flatten()

        false_pos = 1.0 * np.array(false_pos).flatten()
        false_neg = 1.0 * (label_pred == 10).flatten()

        label_targ = label_targ[label_pred != 10]
        label_pred = label_pred[label_pred != 10]

        if return_spike_rec:
            self.network.pull_recording_buffers_from_device()
            spike_data = {
                pop: self.network.neuron_populations[pop].spike_recording_data
                for pop in self.spike_recoring_pops
            }
            return (
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
                spike_data,
            )

        return (
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
        )


class NetworkGestureResFF(NetworkBase):
    def __init__(
        self,
        N_I,
        N_R,
        N_H,
        N_O,
        DT,
        nt_data_in_max,
        nt_data_out_max,
        event_based,
        mod_params,
        specrad_recurrent=0.9,
        p_sparse_rec=0.1,
        p_sparse_input=0.1,
    ):
        self.N_I = N_I
        self.N_R = N_R
        self.N_H = N_H
        self.N_O = N_O

        self.event_based = event_based

        self.P_I_PARAMS = dict(mod_params["p_i_params"])
        self.P_I_VAR_INIT = mod_params["p_i_var_init"]
        if not event_based:
            self.P_I_PARAMS["th"] = 1000.0

        self.P_R_PARAMS = dict(mod_params["p_r_params"])
        self.P_R_VAR_INIT = mod_params["p_r_var_init"]
        if not event_based:
            self.P_R_PARAMS["th"] = 1000.0

        self.P_H_PARAMS = dict(mod_params["p_h_params"])
        self.P_H_VAR_INIT = mod_params["p_h_var_init"]
        if not event_based:
            self.P_H_PARAMS["th"] = 1000.0

        self.P_O_PARAMS = dict(mod_params["p_o_params"])
        self.P_O_VAR_INIT = mod_params["p_o_var_init"]
        if not event_based:
            self.P_O_PARAMS["th"] = 1000.0

        self.W_RI_PARAMS = mod_params["w_ri_params"]
        self.W_RI_VAR_INIT = dict(mod_params["w_ri_var_init"])
        if not event_based:
            self.W_RI_VAR_INIT.pop("inp_prev")

        self.W_RR_PARAMS = mod_params["w_rr_params"]
        self.W_RR_VAR_INIT = dict(mod_params["w_rr_var_init"])
        if not event_based:
            self.W_RR_VAR_INIT.pop("inp_prev")

        self.W_HR_PARAMS = mod_params["w_hr_params"]
        self.W_HR_VAR_INIT = dict(mod_params["w_hr_var_init"])
        if not event_based:
            self.W_HR_VAR_INIT.pop("inp_prev")
            self.W_HR_VAR_INIT.pop("dg_prev")
        else:
            self.W_HR_VAR_INIT["t_prev"] = 0.0

        self.W_OH_PARAMS = mod_params["w_oh_params"]
        self.W_OH_VAR_INIT = dict(mod_params["w_oh_var_init"])
        if not event_based:
            self.W_OH_VAR_INIT.pop("inp_prev")
            self.W_OH_VAR_INIT.pop("dg_prev")
        else:
            self.W_OH_VAR_INIT["t_prev"] = 0.0

        self.W_HO_PARAMS = mod_params["w_ho_params"]
        self.W_HO_VAR_INIT = dict(mod_params["w_ho_var_init"])
        if not event_based:
            self.W_HO_VAR_INIT.pop("inp_prev")
            self.W_HO_VAR_INIT.pop("dg_prev")
        else:
            self.W_HO_VAR_INIT["t_prev"] = 0.0

        self.network = GeNNModel("float", "rflo")
        self.network.dT = DT

        self.p_i = self.network.add_neuron_population(
            "p_i", N_I, neur_i, self.P_I_PARAMS, self.P_I_VAR_INIT
        )

        self.p_r = self.network.add_neuron_population(
            "p_r", N_R, neur_r, self.P_R_PARAMS, self.P_R_VAR_INIT
        )

        self.p_r.spike_recording_enabled = True

        self.p_h = self.network.add_neuron_population(
            "p_h", N_H, neur_h, self.P_H_PARAMS, self.P_H_VAR_INIT
        )

        self.p_o = self.network.add_neuron_population(
            "p_o", N_O, neur_o, self.P_O_PARAMS, self.P_O_VAR_INIT
        )

        self.cs_in = StreamDataCS("cs_in", self.network, self.p_i, nt_data_in_max)
        self.cs_out = StreamDataCS("cs_out", self.network, self.p_o, nt_data_out_max)

        self.w_ri = self.network.add_synapse_population(
            "w_ri",
            "DENSE_INDIVIDUALG",
            0,
            self.p_i,
            self.p_r,
            w_u_in if event_based else w_u_in_cont,
            self.W_RI_PARAMS,
            self.W_RI_VAR_INIT,
            {},
            {},
            ps_model if event_based else ps_model_cont,
            {"gamma": 0.0} if event_based else {},
            {},
        )

        self.w_ri.ps_target_var = "Isyn_in"

        self.w_rr = self.network.add_synapse_population(
            "w_rr",
            "DENSE_INDIVIDUALG",
            0,
            self.p_r,
            self.p_r,
            w_u_rec if event_based else w_u_rec_cont,
            self.W_RR_PARAMS,
            self.W_RR_VAR_INIT,
            {},
            {},
            ps_model if event_based else ps_model_cont,
            {"gamma": 0.0} if event_based else {},
            {},
        )

        self.w_hr = self.network.add_synapse_population(
            "w_hr",
            "DENSE_INDIVIDUALG",
            0,
            self.p_r,
            self.p_h,
            w_u_hr if event_based else w_u_hr_cont,
            self.W_HR_PARAMS,
            self.W_HR_VAR_INIT,
            {},
            {},
            ps_model if event_based else ps_model_cont,
            {"gamma": 0.0} if event_based else {},
            {},
        )

        self.w_oh = self.network.add_synapse_population(
            "w_oh",
            "DENSE_INDIVIDUALG",
            0,
            self.p_h,
            self.p_o,
            w_u_out if event_based else w_u_out_cont,
            self.W_OH_PARAMS,
            self.W_OH_VAR_INIT,
            {},
            {},
            ps_model if event_based else ps_model_cont,
            {"gamma": 0.0} if event_based else {},
            {},
        )

        self.w_oh.ps_target_var = "Isyn_net"

        self.w_ho = self.network.add_synapse_population(
            "w_ho",
            "DENSE_INDIVIDUALG",
            0,
            self.p_o,
            self.p_h,
            w_u_err_fb if event_based else w_u_err_fb_cont,
            self.W_HO_PARAMS,
            self.W_HO_VAR_INIT,
            {},
            {},
            ps_model if event_based else ps_model_cont,
            {"gamma": 0.0} if event_based else {},
            {},
        )

        self.w_ho.ps_target_var = "Isyn_err_fb"

        self.add_weight_plast_update(
            "w_hr",
            mod_params["w_hr_opt_params"]["optimizer"],
            mod_params["w_hr_opt_params"]["optimizer_params"],
        )

        self.add_weight_plast_update(
            "w_oh",
            mod_params["w_oh_opt_params"]["optimizer"],
            mod_params["w_oh_opt_params"]["optimizer_params"],
        )

        self.add_weight_plast_update(
            "w_ho",
            mod_params["w_ho_opt_params"]["optimizer"],
            mod_params["w_ho_opt_params"]["optimizer_params"],
        )

        self.add_bias_plast_update(
            "p_h",
            mod_params["p_h_opt_params"]["optimizer"],
            mod_params["p_h_opt_params"]["optimizer_params"],
        )

        self.add_bias_plast_update(
            "p_o",
            mod_params["p_o_opt_params"]["optimizer"],
            mod_params["p_o_opt_params"]["optimizer_params"],
        )

        _sm_1 = self.network.add_custom_update(
            "softmax_1",
            "softmax1",
            softmax_1_model,
            {},
            {"MaxVal": 0.0},
            {"Val": create_var_ref(self.p_o, "x")},
        )

        _sm_2 = self.network.add_custom_update(
            "softmax_2",
            "softmax2",
            softmax_2_model,
            {},
            {"SumExpVal": 0.0},
            {
                "Val": create_var_ref(self.p_o, "x"),
                "MaxVal": create_var_ref(_sm_1, "MaxVal"),
            },
        )

        self.network.add_custom_update(
            "softmax_3",
            "softmax3",
            softmax_3_model,
            {},
            {},
            {
                "Val": create_var_ref(self.p_o, "x"),
                "MaxVal": create_var_ref(_sm_1, "MaxVal"),
                "SumExpVal": create_var_ref(_sm_2, "SumExpVal"),
                "SoftmaxVal": create_var_ref(self.p_o, "r"),
            },
        )

        self.network.build()
        self.network.load(num_recording_timesteps=1000)

        # set diagonal recurrent weights to zero,
        # set random entries to 0 and normalise spectral radius
        _w_rr = self.weight_rr
        _w_rr[range(self.N_R), range(self.N_R)] = 0.0
        _w_rr *= np.random.rand(self.N_R, self.N_R) <= p_sparse_rec
        _w_rr *= specrad_recurrent / np.abs(np.linalg.eigvals(_w_rr)).max()
        self.weight_rr = _w_rr

        # sparse input
        _w_ri = self.weight_ri
        _w_ri *= np.random.rand(self.N_R, self.N_I) <= p_sparse_input
        self.weight_ri = _w_ri

    @property
    def weight_ri(self):
        self.w_ri.pull_var_from_device("g")
        _w = np.reshape(self.w_ri.vars["g"].view, (self.N_I, self.N_R)).T
        return _w

    @weight_ri.setter
    def weight_ri(self, w):
        _w = self.w_ri.vars["g"].view
        _w[:] = w.T.flatten()
        self.w_ri.push_var_to_device("g")

    @property
    def weight_rr(self):
        self.w_rr.pull_var_from_device("g")
        _w = np.reshape(self.w_rr.vars["g"].view, (self.N_R, self.N_R)).T
        return _w

    @weight_rr.setter
    def weight_rr(self, w):
        _w = self.w_rr.vars["g"].view
        _w[:] = w.T.flatten()
        self.w_rr.push_var_to_device("g")

    @property
    def weight_hr(self):
        self.w_hr.pull_var_from_device("g")
        _w = np.reshape(self.w_hr.vars["g"].view, (self.N_R, self.N_H)).T
        return _w

    @weight_hr.setter
    def weight_hr(self, w):
        _w = self.w_hr.vars["g"].view
        _w[:] = w.T.flatten()
        self.w_hr.push_var_to_device("g")

    @property
    def weight_oh(self):
        self.w_oh.pull_var_from_device("g")
        _w = np.reshape(self.w_oh.vars["g"].view, (self.N_H, self.N_O)).T
        return _w

    @weight_oh.setter
    def weight_oh(self, w):
        _w = self.w_oh.vars["g"].view
        _w[:] = w.T.flatten()
        self.w_oh.push_var_to_device("g")

    @property
    def weight_ho(self):
        self.w_ho.pull_var_from_device("g")
        _w = np.reshape(self.w_ho.vars["g"].view, (self.N_O, self.N_H)).T
        return _w

    @weight_ho.setter
    def weight_ho(self, w):
        _w = self.w_ho.vars["g"].view
        _w[:] = w.T.flatten()
        self.w_ho.push_var_to_device("g")

    def train_network(
        self, n_epochs, data_loader, align_fb_weight_init=False, print_train_loss=False
    ):
        if align_fb_weight_init:
            self.weight_ho = self.weight_oh.T

        self.reset_weight_plast_vars()
        self.reset_bias_plast_vars()

        loss = np.zeros((n_epochs))

        nt_epoch = 0
        for inputs, targets in data_loader:
            nt_epoch += inputs.shape[1]

        t0 = time.time()

        for k in range(n_epochs):
            for inputs, targets in data_loader:
                X = np.array(inputs[0])
                Y = np.array(targets[0])

                self.cs_in.set_data(
                    np.expand_dims(X, axis=1), self.network.dT, periodic=False
                )
                self.cs_out.set_data(
                    np.expand_dims(Y, axis=1), self.network.dT, periodic=False
                )

                _nt = inputs.shape[1]

                for t in range(_nt):
                    self.network.step_time()

                    # self.network.custom_update("softmax1")
                    # self.network.custom_update("softmax2")
                    # self.network.custom_update("softmax3")

                self.network.custom_update("WeightChangeBatchReduce")
                self.network.custom_update("BiasChangeBatchReduce")
                self.network.custom_update("Plast")

                self.weight_ho = self.weight_oh.T

            self.p_o.pull_var_from_device("loss")
            loss[k] = self.p_o.vars["loss"].view.mean() / nt_epoch

            self.p_o.vars["loss"].view[:] = 0.0
            self.p_o.push_var_to_device("loss")

            if print_train_loss:
                print(f"Epoch {k+1}/{n_epochs}, Loss {loss[k]}", end="\r")

        t1 = time.time()

        return loss, (t1 - t0)

    def test_network(self, data_loader, ev_th=0.5):
        nt_epoch = 0
        for inputs, targets in data_loader:
            nt_epoch += inputs.shape[1]

        evidence = np.zeros((nt_epoch, self.N_O))

        label_pred = []
        label_targ = []

        false_pos = [False]

        gesture_detected = False
        gesture_detected_corr = False

        t = 0

        for inputs, targets in tqdm(data_loader):
            X = np.array(inputs[0])
            Y = np.array(targets[0])

            _nt = inputs.shape[1]

            self.cs_in.set_data(
                np.expand_dims(X, axis=1), self.network.dT, periodic=False
            )
            self.cs_out.set_data(
                np.expand_dims(Y, axis=1), self.network.dT, periodic=False
            )

            r_rec = np.ndarray((_nt, self.N_R))
            r_view = self.p_r.vars["r"].view

            h_rec = np.ndarray((_nt, self.N_H))
            h_view = self.p_h.vars["r"].view

            o_rec = np.ndarray((_nt, self.N_O))
            o_view = self.p_o.vars["r"].view

            o_x_rec = np.ndarray((_nt, self.N_O))
            o_x_view = self.p_o.vars["x"].view

            targ_rec = np.ndarray((_nt, self.N_O))
            targ_view = self.p_o.vars["targ"].view

            for _t in range(_nt):
                self.network.step_time()

                # self.network.custom_update("softmax1")
                # self.network.custom_update("softmax2")
                # self.network.custom_update("softmax3")

                self.p_r.pull_var_from_device("r")
                r_rec[_t] = r_view

                self.p_h.pull_var_from_device("r")
                h_rec[_t] = h_view

                self.p_o.pull_var_from_device("r")
                o_rec[_t] = o_view

                self.p_o.pull_var_from_device("x")
                o_x_rec[_t] = o_x_view

                self.p_o.pull_var_from_device("targ")
                targ_rec[_t] = targ_view

                self.p_o.pull_var_from_device("r")

                if t > 0:
                    _o = self.p_o.vars["r"].view + 1e-3
                    _o /= _o.sum()

                    evidence[t] = evidence[t - 1] + self.network.dT * (
                        _o - evidence[t - 1]
                    )

                    _ev_sm_prev = evidence[t - 1, :-1].sum()
                    _ev_sm = evidence[t, :-1].sum()

                    _label = np.argmax(Y[_t])
                    _label_prev = np.argmax(Y[_t - 1])

                    if (_label != 10) and (_label_prev == 10):
                        label_targ.append(np.argmax(Y[_t]))
                        label_pred.append(10)

                    if (_label != 10) and gesture_detected:
                        gesture_detected_corr = True

                    if (_label == 10) and (_label_prev != 10):
                        false_pos.append(False)

                    if (_ev_sm >= ev_th) and (_ev_sm_prev < ev_th):
                        gesture_detected = True

                    if (_ev_sm < ev_th) and (_ev_sm_prev >= ev_th):
                        if gesture_detected_corr:
                            label_pred[-1] = np.argmax(evidence[t, :-1])
                        else:
                            false_pos[-1] = True

                        gesture_detected = False
                        gesture_detected_corr = False

                t += 1

        label_targ = np.array(label_targ)
        label_pred = np.array(label_pred)

        label_targ = label_targ[label_pred != 10]
        label_pred = label_pred[label_pred != 10]

        false_pos = 1.0 * np.array(false_pos)
        false_neg = 1.0 * (label_pred == 10)

        return (
            label_targ,
            label_pred,
            false_pos,
            false_neg,
            r_rec,
            h_rec,
            o_rec,
            o_x_rec,
            targ_rec,
        )
