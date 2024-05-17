#! /usr/bin/env python3

from pygenn.genn_wrapper.Models import (VarAccess_REDUCE_BATCH_SUM,
                                        VarAccess_REDUCE_NEURON_SUM,
                                        VarAccess_REDUCE_NEURON_MAX,
                                        VarAccessMode_READ_ONLY)

from pygenn.genn_model import create_custom_custom_update_class

softmax_1_model = create_custom_custom_update_class(
    "softmax_1_model",
    var_name_types=[("MaxVal", "scalar", VarAccess_REDUCE_NEURON_MAX)],
    var_refs=[("Val", "scalar", VarAccessMode_READ_ONLY)],
    update_code="""
    $(MaxVal) = $(Val);
    """)

# Second pass of softmax - calculate scaled sum of exp(value)
softmax_2_model = create_custom_custom_update_class(
    "softmax_2_model",
    var_name_types=[("SumExpVal", "scalar", VarAccess_REDUCE_NEURON_SUM)],
    var_refs=[("Val", "scalar", VarAccessMode_READ_ONLY),
              ("MaxVal", "scalar", VarAccessMode_READ_ONLY)],
    update_code="""
    $(SumExpVal) = exp($(Val) - $(MaxVal));
    """)

# Third pass of softmax - calculate softmax value
softmax_3_model = create_custom_custom_update_class(
    "softmax_3_model",
    var_refs=[("Val", "scalar", VarAccessMode_READ_ONLY),
              ("MaxVal", "scalar", VarAccessMode_READ_ONLY),
              ("SumExpVal", "scalar", VarAccessMode_READ_ONLY),
              ("SoftmaxVal", "scalar")],
    update_code="""
    $(SoftmaxVal) = exp($(Val) - $(MaxVal)) / $(SumExpVal);
    """)


param_change_batch_reduce = create_custom_custom_update_class(
    "param_change_batch_reduce",
    var_name_types=[("reducedChange", "scalar", VarAccess_REDUCE_BATCH_SUM)],
    var_refs=[("change", "scalar")],
    update_code="""
        $(reducedChange) = $(change);
        $(change) = 0.0;
    """)

sgd_param = create_custom_custom_update_class(
    "sgd",
    var_refs=[("change", "scalar"), ("variable", "scalar")],
    param_names=["batch_size", "lr"],
    update_code="""
        $(variable) += $(lr) * $(change) / $(batch_size);
    """)

momentum_param = create_custom_custom_update_class(
    "momentum",
    var_refs=[("change", "scalar"), ("variable", "scalar")],
    var_name_types=[("m", "scalar")],
    param_names=["batch_size", "lr", "beta"],
    update_code="""
        const scalar change_norm = $(change) / $(batch_size);
        $(m) = $(beta) * $(m) + (1.0 - $(beta)) * change_norm;
        $(variable) += $(lr) * $(m);
    """)

adam_param = create_custom_custom_update_class(
    "adam",
    var_refs=[("change", "scalar"), ("variable", "scalar")],
    var_name_types=[("m", "scalar"), ("v", "scalar"), ("time", "scalar")],
    param_names=["batch_size", "lr", "beta1", "beta2", "epsilon"],
    update_code="""
        const scalar change_norm = $(change) / $(batch_size);
        $(m) = $(beta1) * $(m) + (1.0 - $(beta1)) * change_norm;
        $(v) = $(beta2) * $(v) + (1.0 - $(beta2)) * change_norm * change_norm;
        const scalar m_hat = $(m)/(1.0 - pow($(beta1), $(time)));
        const scalar v_hat = $(v)/(1.0 - pow($(beta2), $(time)));
        //$(variable) += $(lr) * $(m) / (sqrt($(v)) +  $(epsilon));
        $(variable) += $(lr) * m_hat / (sqrt(v_hat) +  $(epsilon));
        $(time) += 1.0;
    """)

optimizers = {
    "sgd": {"model": sgd_param,
            "var_init": {}},
    "momentum": {"model": momentum_param,
                 "var_init": {"m": 0.0}},
    "adam": {"model": adam_param,
             "var_init": {"m": 0.0, "v": 0.0, "time": 1.0}}
}
