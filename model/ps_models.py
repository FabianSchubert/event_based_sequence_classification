#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from pygenn.genn_model import create_custom_postsynaptic_class

ps_model_cont = create_custom_postsynaptic_class(
    "cont_ps",
    apply_input_code="$(Isyn) += $(inSyn); $(inSyn) = 0.0;"
)

#ps_model = create_custom_postsynaptic_class(
#    "delta_ps",
#    apply_input_code="$(Isyn) += $(inSyn);"
#)

ps_model = create_custom_postsynaptic_class(
    "delta_ps",
    param_names=["gamma"],
    apply_input_code="$(Isyn) += $(inSyn); $(inSyn) *= (1.0 - DT * $(gamma));"
)
