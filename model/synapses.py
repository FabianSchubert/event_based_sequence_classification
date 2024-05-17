#! /usr/bin/env python3

import numpy as np

from pygenn.genn_model import create_custom_weight_update_class
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY

w_u_rec = create_custom_weight_update_class(
    "recurrent_weights",
    sim_code="""
        $(addToInSyn, $(g) * ($(r_pre) - $(r_prev_event_pre)));
        //$(addToInSyn, $(g) * $(r_pre) - $(inp_prev));
        //$(inp_prev) = $(g) * $(r_pre);
    """,
    var_name_types=[("g", "scalar", VarAccess_READ_ONLY),
                    ("inp_prev", "scalar")],
    param_names=[],
    is_prev_pre_spike_time_required=True
)

w_u_rec_cont = create_custom_weight_update_class(
    "recurrent_weights_continuous",
    synapse_dynamics_code="""
        $(addToInSyn, $(g) * $(r_pre));
    """,
    var_name_types=[("g", "scalar", VarAccess_READ_ONLY)],
    param_names=[]
)

w_u_hr = create_custom_weight_update_class(
    "recurrent_to_hidden_weights",
    sim_code="""
        $(addToInSyn, $(g) * $(r_pre) - $(inp_prev));
        $(inp_prev) = $(g) * $(r_pre);

        $(dg) += (t - $(t_prev)) * $(dg_prev) / DT;
        $(dg_prev) = $(err_fb_post) * $(dr_post) * $(r_trace_pre) - $(g) * $(w_penalty);
        $(t_prev) = t;
    """,
    var_name_types=[("g", "scalar", VarAccess_READ_ONLY),
                    ("inp_prev", "scalar"),
                    ("dg", "scalar"),
                    ("dg_prev", "scalar"),
                    ("t_prev", "scalar")],
    param_names=["w_penalty"],
    is_prev_pre_spike_time_required=True
    )

w_u_hr_cont = create_custom_weight_update_class(
    "recurrent_to_hidden_weights_continuous",
    synapse_dynamics_code="""
        $(addToInSyn, $(g) * $(r_pre));

        $(dg) += ($(err_fb_post) * $(dr_post) * $(r_trace_pre) - $(g) * $(w_penalty));
    """,
    var_name_types=[("g", "scalar", VarAccess_READ_ONLY),
                    ("dg", "scalar")],
    param_names=["w_penalty"]
)

w_u_hi = create_custom_weight_update_class(
    "input_to_hidden_weights",
    sim_code="""
        //const scalar input_new = $(g) * $(r_pre);
        $(addToInSyn, $(g) * $(r_pre) - $(inp_prev));
        $(inp_prev) = $(g) * $(r_pre);

        $(dg) += (t - max(0.0, $(prev_sT_pre))) * $(dg_prev);
        $(dg_prev) = $(dr_err_prod_post) * $(r_trace_pre) - $(g) * $(w_penalty);
    """,
    var_name_types=[("g", "scalar", VarAccess_READ_ONLY),
                    ("inp_prev", "scalar"),
                    ("dg", "scalar"),
                    ("dg_prev", "scalar"),
                    ("t_prev", "scalar")],
    param_names=["w_penalty"],
    is_prev_pre_spike_time_required=True
    )

w_u_hi_cont = create_custom_weight_update_class(
    "input_to_hidden_weights_continuous",
    synapse_dynamics_code="""
        $(addToInSyn, $(g) * $(r_pre));

        $(dg) += $(dr_err_prod_post) * $(r_trace_pre) - $(g) * $(w_penalty);
    """,
    var_name_types=[("g", "scalar", VarAccess_READ_ONLY),
                    ("dg", "scalar")],
    param_names=["w_penalty"]
)

w_u_out = create_custom_weight_update_class(
    "output_weights",
    sim_code="""
        $(addToInSyn, $(g) * $(r_pre) - $(inp_prev));
        $(inp_prev) = $(g) * $(r_pre);

        $(dg) += (t - max(0.0, $(prev_sT_pre))) * $(dg_prev);
        $(dg_prev) = $(err_post) * $(dr_post) * $(r_prev_event_pre) - $(w_penalty) * $(g);
    """,
    var_name_types=[("g", "scalar", VarAccess_READ_ONLY),
                    ("inp_prev", "scalar"),
                    ("dg", "scalar"),
                    ("dg_prev", "scalar"),
                    ("t_prev", "scalar")],
    param_names=["w_penalty"],
    is_prev_pre_spike_time_required=True
)

w_u_out_cont = create_custom_weight_update_class(
    "output_weights_continuous",
    synapse_dynamics_code="""
        $(addToInSyn, $(g) * $(r_pre));
        $(dg) += ($(err_post) * $(dr_post) * $(r_pre) - $(w_penalty) * $(g));
    """,
    var_name_types=[("g", "scalar", VarAccess_READ_ONLY),
                    ("dg", "scalar")],
    param_names=["w_penalty"]
)

w_u_in = create_custom_weight_update_class(
    "input_weights",
    sim_code="""
        $(addToInSyn, $(g) * $(r_pre) - $(inp_prev));
        $(inp_prev) = $(g) * $(r_pre);
    """,
    var_name_types=[("g", "scalar", VarAccess_READ_ONLY),
                    ("inp_prev", "scalar")],
    param_names=[],
    is_prev_pre_spike_time_required=True
)

w_u_in_cont = create_custom_weight_update_class(
    "input_weights_continuous",
    synapse_dynamics_code="""
        $(addToInSyn, $(g) * $(r_pre));
    """,
    var_name_types=[("g", "scalar", VarAccess_READ_ONLY)],
    param_names=[]
)

w_u_err_fb = create_custom_weight_update_class(
    "err_feedback",
    sim_code="""
        $(addToInSyn, $(g) * $(err_pre) - $(inp_prev));
        $(inp_prev) = $(g) * $(err_pre);

        //$(dg) += (t - $(t_prev)) * $(dg_prev) / DT;
        //$(dg_prev) = $(r_post) * $(err_pre) * $(dr_pre) - $(w_penalty) * $(g);
        //$(t_prev) = t;
    """,
    var_name_types=[("g", "scalar", VarAccess_READ_ONLY),
                    ("inp_prev", "scalar"),
                    ("dg", "scalar"),
                    ("dg_prev", "scalar"),
                    ("t_prev", "scalar")],
    param_names=["w_penalty"],
    is_prev_pre_spike_time_required=False
)

w_u_err_fb_cont = create_custom_weight_update_class(
    "err_feedback_continuous",
    synapse_dynamics_code="""
        $(addToInSyn, $(g) * $(err_pre));
        $(dg) += ($(r_post) * $(err_pre) * $(dr_pre) - $(w_penalty) * $(g));
    """,
    var_name_types=[("g", "scalar", VarAccess_READ_ONLY),
                    ("dg", "scalar")],
    param_names=["w_penalty"]
)
