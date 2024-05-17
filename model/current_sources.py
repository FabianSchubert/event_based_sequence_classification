#! /usr/bin/env python3

import numpy as np

from pygenn.genn_model import create_custom_current_source_class


cs_data_stream_shift = create_custom_current_source_class(
    "cs_data_stream_shift",
    param_names=["t_buffer", "nt_data_max", "dims_data", "size_data_array"],
    var_name_types=[
        ("t_idx", "int"),
        ("periodic", "int"),
        ("nt_data", "int"),
        ("dt_data", "scalar"),
        ("t_data", "scalar"),
    ],
    extra_global_params=[("data", "scalar*")],
    injection_code="""
        const int nt_data_max_int = int($(nt_data_max));
        const int dims_data_int = int($(dims_data));
        const int t_buffer_int = int($(t_buffer));

        if($(periodic)){
            $(t_idx) = int($(t_data) / $(dt_data)) % $(nt_data);
        } else {
            $(t_idx) = min(int($(t_data) / $(dt_data)), $(nt_data)-1);
        }

        $(t_data) += DT;
                
        const int t_access = max(0, $(t_idx) - (int)($(id))%t_buffer_int);

        const int dim_id = $(id) / t_buffer_int;
        const int index_access = $(batch)*nt_data_max_int*dims_data_int + t_access * dims_data_int + dim_id;

        const scalar datapoint = $(data)[index_access];

        //const scalar datapoint = 0.0;

        $(injectCurrent, datapoint);
    """,
)

cs_data_stream_roll = create_custom_current_source_class(
    "cs_data_stream_roll",
    param_names=["t_buffer", "nt_data_max", "dims_data", "size_data_array"],
    var_name_types=[
        ("t_idx", "int"),
        ("periodic", "int"),
        ("nt_data", "int"),
        ("dt_data", "scalar"),
        ("t_data", "scalar"),
        ("current_data", "scalar"),
        ("t_ind_buffer_offset", "int"),
    ],
    extra_global_params=[("data", "scalar*")],
    injection_code="""
        const int nt_data_max_int = int($(nt_data_max));
        const int dims_data_int = int($(dims_data));
        const int t_buffer_int = int($(t_buffer));

        if($(periodic)){
            $(t_idx) = int($(t_data) / $(dt_data)) % $(nt_data);
        } else {
            $(t_idx) = min(int($(t_data) / $(dt_data)), $(nt_data)-1);
        }

        $(t_data) += DT;

        const int t_ind_buffer_update = ($(t_idx) + $(t_ind_buffer_offset)) % t_buffer_int;
        const int t_ind_buffer = (int)($(id)) % t_buffer_int;
        const int dim_id = (int)($(id)) / t_buffer_int;

        if(dim_id == 0){
            // zeroth dimension is the time embedding
            
            $(injectCurrent, 1.0 * (float)(t_ind_buffer_update == t_ind_buffer));
        } else {
            if(t_ind_buffer_update == t_ind_buffer){
                // printf("t_ind_buffer_update: %d \\n", t_ind_buffer_update);
                $(current_data) = $(data)[$(batch)*nt_data_max_int*dims_data_int + $(t_idx) * dims_data_int + (dim_id - 1)];
            }
            else{
                $(current_data) = 0.0;
            }
            $(injectCurrent, $(current_data));
        }
    """,
)

# //$(injectCurrent, 0.0 * (float)((t_ind_buffer + t_buffer_int - t_ind_buffer_update) % t_buffer_int) / t_buffer_int);


class StreamDataCS:
    def __init__(
        self, name, genn_model, target_pop, nt_max, t_buffer, stream_type="shift"
    ):
        self.name = name

        self.genn_model = genn_model
        self.target_pop = target_pop
        self.nt_max = nt_max
        self.t_buffer = t_buffer
        self.stream_type = stream_type

        assert (
            self.target_pop.size % self.t_buffer == 0
        ), "error, population size must be divisible by t_buffer"

        _cs_data_stream = (
            cs_data_stream_shift
            if stream_type == "shift"
            else cs_data_stream_roll
        )
        _var_init = (
            {"t_idx": 0, "periodic": 0, "nt_data": 0, "dt_data": 0.0, "t_data": 0.0}
            if stream_type == "shift"
            else {
                "t_idx": 0,
                "periodic": 0,
                "nt_data": 0,
                "dt_data": 0.0,
                "t_data": 0.0,
                "current_data": 0.0,
                "t_ind_buffer_offset": 0,
            }
        )

        assert (
            self.target_pop.size % self.t_buffer == 0
        ), "error, population size must be divisible by t_buffer"
        if self.stream_type == "shift":
            self.data_dim = int(self.target_pop.size / self.t_buffer)
        else:
            # for the roll stream, the first dimension is the time embedding, so we subtract 1
            # to get the number of dimensions for the actual sensory data
            self.data_dim = int(self.target_pop.size / self.t_buffer) - 1

        self.flat_data_size = self.genn_model.batch_size * self.nt_max * self.data_dim

        self.cs_data_stream = self.genn_model.add_current_source(
            self.name,
            _cs_data_stream,
            self.target_pop,
            {
                "t_buffer": float(self.t_buffer),
                "nt_data_max": float(self.nt_max),
                "dims_data": float(int(self.target_pop.size / self.t_buffer)),
                "size_data_array": float(self.flat_data_size),
            },
            _var_init,
        )

        self.cs_data_stream.set_extra_global_param(
            "data", np.zeros(self.flat_data_size).astype("float32")
        )

    def set_data(
        self,
        data,
        dt_data,
        periodic,
        randomize_t_buffer_offset=False,
    ):
        assert data.ndim == 3, "error, data must be 3d array"

        assert (
            data.shape[0] == self.genn_model.batch_size
        ), "error, first dimension of data array must match batch size"
        assert (
            data.shape[2] == self.data_dim
        ), "error, third dimension of data array must match data_dim := target_pop.size / t_buffer"

        _nt = data.shape[1]
        assert (
            _nt <= self.nt_max
        ), "error, number of time steps in data exceeds given maximum"

        self.cs_data_stream.vars["t_idx"].view[:] = 0
        self.cs_data_stream.push_var_to_device("t_idx")

        self.cs_data_stream.vars["periodic"].view[:] = periodic
        self.cs_data_stream.push_var_to_device("periodic")

        self.cs_data_stream.vars["nt_data"].view[:] = _nt
        self.cs_data_stream.push_var_to_device("nt_data")

        self.cs_data_stream.vars["dt_data"].view[:] = dt_data
        self.cs_data_stream.push_var_to_device("dt_data")

        self.cs_data_stream.vars["t_data"].view[:] = 0.0
        self.cs_data_stream.push_var_to_device("t_data")

        if self.stream_type == "roll":
            self.cs_data_stream.vars["current_data"].view[:] = 0.0
            self.cs_data_stream.push_var_to_device("current_data")

            if randomize_t_buffer_offset:
                self.cs_data_stream.vars["t_ind_buffer_offset"].view[:] = (
                    np.random.randint(0, self.t_buffer)
                )
            else:
                self.cs_data_stream.vars["t_ind_buffer_offset"].view[:] = 0
            self.cs_data_stream.push_var_to_device("t_ind_buffer_offset")

        # _data_flat = data.flatten()

        # zero out the data array
        self.cs_data_stream.extra_global_params["data"].view[:] = 0.0
        # write the new data into the data array for each batch. Since _nt <= self.nt_max,
        # the written data is segmented, and the rest of the array is left as zeros.
        for k in range(self.genn_model.batch_size):
            self.cs_data_stream.extra_global_params["data"].view[
                k * self.nt_max * self.target_pop.size : k
                * self.nt_max
                * self.target_pop.size
                + _nt * self.data_dim
            ] = data[k].flatten()

        # push the data array to the device
        self.cs_data_stream.push_extra_global_param_to_device(
            "data", self.flat_data_size
        )
