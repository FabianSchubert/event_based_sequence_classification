#! /usr/bin/env python3

import numpy as np

from pygenn.genn_model import create_custom_neuron_class
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY


def act_func(x):
    #return f'({x})'
    return f"(({x}) < 0.0 ? 0.0 : ({x}))"
    # return f'({x})'
    #return f'tanh({x})'
    #return f'((tanh(2.0*({x}))+1.0)/2.0)'
    #return f'max(0.0, tanh({x}))'


def d_act_func(x):
    #return "1.0"
    return f"(({x}) < 0.0 ? 0.0 : 1.0)"
    # return "1.0"
    #return f'(1.0 - tanh({x})*tanh({x}))'
    #return f'(1.0 - tanh(2.0*({x}))*tanh(2.0*({x})))'
    #return f'(({x}) < 0.0 ? 0.0 : (1.0 - tanh({x})*tanh({x})))'


def act_func_out(x):
    return f"exp({x})"

neur_h = create_custom_neuron_class(
    "hidden_nonlinear",
    sim_code=f"""
        $(err_fb) = $(Isyn_err_fb);

        $(x) += DT * ($(Isyn) + $(b) - $(x)) / $(tau);
        //$(x) = $(Isyn) + $(b);
        const scalar r_prev = $(r);
        $(r) = {act_func("$(x)")};
        $(dr) = {d_act_func("$(x)")};
        //$(err_fb) += 1.1 * (r_prev - $(r));
        $(err_fb) -= 0.01 * ($(r) > 0.0);
        //$(r_event) *= (1. - DT);

        $(db) += $(err_fb) * $(dr);

        $(dr_err_prod) = $(dr) * $(err_fb);
    """,
    threshold_condition_code="abs($(r) - $(r_event)) >= ($(th) / $(weight_factor))",
    reset_code="""
        $(r_prev_event) = $(r_event);
        $(r_event) = $(r);
    """,
    param_names=["th", "tau"],
    var_name_types=[
        ("r", "scalar"),
        ("x", "scalar"),
        ("dr", "scalar"),
        ("b", "scalar", VarAccess_READ_ONLY),
        ("db", "scalar"),
        ("err_fb", "scalar"),
        ("r_event", "scalar"),
        ("r_prev_event", "scalar"),
        ("dr_err_prod", "scalar"),
        ("weight_factor", "scalar"),
    ],
    additional_input_vars=[("Isyn_err_fb", "scalar", 0.0)],
    is_auto_refractory_required=False,
)

neur_o = create_custom_neuron_class(
    "output",
    sim_code="""
        //$(x) += DT * ($(Isyn_net) + $(b) - $(x));
        $(x) = $(Isyn_net) + $(b);
        //$(targ) += DT * ($(Isyn) - $(targ));
        $(targ) = $(Isyn);

        //$(r) = $(x);
        //$(r) = max(0.0, $(x));
        //$(r) = min(1.0, max(0.0, $(x)));
        //$(r) = 1./(1. + exp(-$(x)));
        $(r) = max(0.0, tanh($(x)));
        //$(dr) = (($(x) > 0.0) && ($(x) < 1.0) ? 1.0 : 0.0);
        //$(dr) = ($(x) > 0.0 ? 1.0 : 0.0);
        //$(dr) = 1.0;
        //$(dr) = $(r) * (1. - $(r));
        $(dr) = $(x) < 0.0 ? 0.0 : (1.0 - tanh($(x))*tanh($(x)));
        $(err) = $(targ) - $(r) - 0.005 * ($(r) > 0.0);

        $(loss) += 0.5 * $(err) * $(err);

        $(db) += $(err) * $(dr);
    """,
    threshold_condition_code="abs($(err) - $(err_event)) >= $(th)",
    reset_code="""
        $(err_prev_event) = $(err_event);
        $(err_event) = $(err);
    """,
    param_names=["th"],
    var_name_types=[
        ("x", "scalar"),
        ("r", "scalar"),
        ("dr", "scalar"),
        ("targ", "scalar"),
        ("err", "scalar"),
        ("err_event", "scalar"),
        ("err_prev_event", "scalar"),
        ("b", "scalar", VarAccess_READ_ONLY),
        ("db", "scalar"),
        ("loss", "scalar"),
    ],
    additional_input_vars=[("Isyn_net", "scalar", 0.0)],
    is_auto_refractory_required=False,
)

neur_i = create_custom_neuron_class(
    "input",
    sim_code="""
        $(r) = $(Isyn);
        $(r_trace) += DT * ($(r) - $(r_trace)) / $(tau_trace);
    """,
    threshold_condition_code="abs($(r) - $(r_event)) >= $(th)",
    reset_code="""
        $(r_prev_event) = $(r_event);
        $(r_event) = $(r);
    """,
    param_names=["th", "tau_trace"],
    var_name_types=[
        ("r", "scalar"),
        ("r_event", "scalar"),
        ("r_prev_event", "scalar"),
        ("r_trace", "scalar"),
    ],
    is_auto_refractory_required=False,
)
