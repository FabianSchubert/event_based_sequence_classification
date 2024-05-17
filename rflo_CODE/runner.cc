#include "definitionsInternal.h"

extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
unsigned long long iT;
float t;
unsigned long long numRecordingTimesteps = 0;
__device__ curandStatePhilox4_32_10_t d_rng;

// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
double initTime = 0.0;
double initSparseTime = 0.0;
double neuronUpdateTime = 0.0;
double presynapticUpdateTime = 0.0;
double postsynapticUpdateTime = 0.0;
double synapseDynamicsTime = 0.0;
double customUpdateBiasChangeBatchReduceTime = 0.0;
double customUpdateBiasChangeBatchReduceTransposeTime = 0.0;
double customUpdatePlastTime = 0.0;
double customUpdatePlastTransposeTime = 0.0;
double customUpdateWeightChangeBatchReduceTime = 0.0;
double customUpdateWeightChangeBatchReduceTransposeTime = 0.0;
double customUpdatesoftmax1Time = 0.0;
double customUpdatesoftmax1TransposeTime = 0.0;
double customUpdatesoftmax2Time = 0.0;
double customUpdatesoftmax2TransposeTime = 0.0;
double customUpdatesoftmax3Time = 0.0;
double customUpdatesoftmax3TransposeTime = 0.0;
// ------------------------------------------------------------------------
// merged group arrays
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// local neuron groups
// ------------------------------------------------------------------------
unsigned int* glbSpkCntp_h;
unsigned int* d_glbSpkCntp_h;
unsigned int* glbSpkp_h;
unsigned int* d_glbSpkp_h;
float* prevSTp_h;
float* d_prevSTp_h;
scalar* rp_h;
scalar* d_rp_h;
scalar* xp_h;
scalar* d_xp_h;
scalar* drp_h;
scalar* d_drp_h;
scalar* bp_h;
scalar* d_bp_h;
scalar* dbp_h;
scalar* d_dbp_h;
scalar* err_fbp_h;
scalar* d_err_fbp_h;
scalar* r_eventp_h;
scalar* d_r_eventp_h;
scalar* r_prev_eventp_h;
scalar* d_r_prev_eventp_h;
unsigned int* glbSpkCntp_i;
unsigned int* d_glbSpkCntp_i;
unsigned int* glbSpkp_i;
unsigned int* d_glbSpkp_i;
float* prevSTp_i;
float* d_prevSTp_i;
scalar* rp_i;
scalar* d_rp_i;
scalar* r_eventp_i;
scalar* d_r_eventp_i;
scalar* r_prev_eventp_i;
scalar* d_r_prev_eventp_i;
// current source variables
int* t_idxcs_in;
int* d_t_idxcs_in;
int* periodiccs_in;
int* d_periodiccs_in;
scalar* t_datacs_in;
scalar* d_t_datacs_in;
scalar* dt_datacs_in;
scalar* d_dt_datacs_in;
int* ntcs_in;
int* d_ntcs_in;
scalar* datacs_in;
scalar* d_datacs_in;
unsigned int* glbSpkCntp_o;
unsigned int* d_glbSpkCntp_o;
unsigned int* glbSpkp_o;
unsigned int* d_glbSpkp_o;
float* prevSTp_o;
float* d_prevSTp_o;
scalar* xp_o;
scalar* d_xp_o;
scalar* rp_o;
scalar* d_rp_o;
scalar* drp_o;
scalar* d_drp_o;
scalar* targp_o;
scalar* d_targp_o;
scalar* errp_o;
scalar* d_errp_o;
scalar* err_eventp_o;
scalar* d_err_eventp_o;
scalar* err_prev_eventp_o;
scalar* d_err_prev_eventp_o;
scalar* bp_o;
scalar* d_bp_o;
scalar* dbp_o;
scalar* d_dbp_o;
scalar* lossp_o;
scalar* d_lossp_o;
// current source variables
int* t_idxcs_out;
int* d_t_idxcs_out;
int* periodiccs_out;
int* d_periodiccs_out;
scalar* t_datacs_out;
scalar* d_t_datacs_out;
scalar* dt_datacs_out;
scalar* d_dt_datacs_out;
int* ntcs_out;
int* d_ntcs_out;
scalar* datacs_out;
scalar* d_datacs_out;
unsigned int* glbSpkCntp_r;
unsigned int* d_glbSpkCntp_r;
unsigned int* glbSpkp_r;
unsigned int* d_glbSpkp_r;
uint32_t* recordSpkp_r;
uint32_t* d_recordSpkp_r;
float* prevSTp_r;
float* d_prevSTp_r;
scalar* r_tracep_r;
scalar* d_r_tracep_r;
scalar* rp_r;
scalar* d_rp_r;
scalar* r_eventp_r;
scalar* d_r_eventp_r;
scalar* r_prev_eventp_r;
scalar* d_r_prev_eventp_r;

// ------------------------------------------------------------------------
// custom update variables
// ------------------------------------------------------------------------
scalar* mplast_step_reduced_p_h;
scalar* d_mplast_step_reduced_p_h;
scalar* vplast_step_reduced_p_h;
scalar* d_vplast_step_reduced_p_h;
scalar* timeplast_step_reduced_p_h;
scalar* d_timeplast_step_reduced_p_h;
scalar* mplast_step_reduced_p_o;
scalar* d_mplast_step_reduced_p_o;
scalar* vplast_step_reduced_p_o;
scalar* d_vplast_step_reduced_p_o;
scalar* timeplast_step_reduced_p_o;
scalar* d_timeplast_step_reduced_p_o;
scalar* reducedChangereduce_batch_bias_change_p_h;
scalar* d_reducedChangereduce_batch_bias_change_p_h;
scalar* reducedChangereduce_batch_bias_change_p_o;
scalar* d_reducedChangereduce_batch_bias_change_p_o;
scalar* MaxValsoftmax_1;
scalar* d_MaxValsoftmax_1;
scalar* SumExpValsoftmax_2;
scalar* d_SumExpValsoftmax_2;
scalar* mplast_step_reduced_w_ho;
scalar* d_mplast_step_reduced_w_ho;
scalar* vplast_step_reduced_w_ho;
scalar* d_vplast_step_reduced_w_ho;
scalar* timeplast_step_reduced_w_ho;
scalar* d_timeplast_step_reduced_w_ho;
scalar* mplast_step_reduced_w_hr;
scalar* d_mplast_step_reduced_w_hr;
scalar* vplast_step_reduced_w_hr;
scalar* d_vplast_step_reduced_w_hr;
scalar* timeplast_step_reduced_w_hr;
scalar* d_timeplast_step_reduced_w_hr;
scalar* mplast_step_reduced_w_oh;
scalar* d_mplast_step_reduced_w_oh;
scalar* vplast_step_reduced_w_oh;
scalar* d_vplast_step_reduced_w_oh;
scalar* timeplast_step_reduced_w_oh;
scalar* d_timeplast_step_reduced_w_oh;
scalar* reducedChangereduce_batch_weight_change_w_ho;
scalar* d_reducedChangereduce_batch_weight_change_w_ho;
scalar* reducedChangereduce_batch_weight_change_w_hr;
scalar* d_reducedChangereduce_batch_weight_change_w_hr;
scalar* reducedChangereduce_batch_weight_change_w_oh;
scalar* d_reducedChangereduce_batch_weight_change_w_oh;

// ------------------------------------------------------------------------
// pre and postsynaptic variables
// ------------------------------------------------------------------------
float* inSynw_ho;
float* d_inSynw_ho;
float* inSynw_hr;
float* d_inSynw_hr;
float* inSynw_oh;
float* d_inSynw_oh;
float* inSynw_rr;
float* d_inSynw_rr;
float* inSynw_ri;
float* d_inSynw_ri;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------
scalar* gw_ho;
scalar* d_gw_ho;
scalar* inp_prevw_ho;
scalar* d_inp_prevw_ho;
scalar* dgw_ho;
scalar* d_dgw_ho;
scalar* dg_prevw_ho;
scalar* d_dg_prevw_ho;
scalar* t_prevw_ho;
scalar* d_t_prevw_ho;
scalar* gw_hr;
scalar* d_gw_hr;
scalar* inp_prevw_hr;
scalar* d_inp_prevw_hr;
scalar* dgw_hr;
scalar* d_dgw_hr;
scalar* dg_prevw_hr;
scalar* d_dg_prevw_hr;
scalar* t_prevw_hr;
scalar* d_t_prevw_hr;
scalar* gw_oh;
scalar* d_gw_oh;
scalar* inp_prevw_oh;
scalar* d_inp_prevw_oh;
scalar* dgw_oh;
scalar* d_dgw_oh;
scalar* dg_prevw_oh;
scalar* d_dg_prevw_oh;
scalar* t_prevw_oh;
scalar* d_t_prevw_oh;
scalar* gw_ri;
scalar* d_gw_ri;
scalar* inp_prevw_ri;
scalar* d_inp_prevw_ri;
scalar* gw_rr;
scalar* d_gw_rr;
scalar* inp_prevw_rr;
scalar* d_inp_prevw_rr;

}  // extern "C"
// ------------------------------------------------------------------------
// extra global params
// ------------------------------------------------------------------------
void allocatedatacs_in(unsigned int count) {
    CHECK_CUDA_ERRORS(cudaHostAlloc(&datacs_in, count * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_datacs_in, count * sizeof(scalar)));
    pushMergedNeuronUpdate2dataCS0ToDevice(0, d_datacs_in);
}
void freedatacs_in() {
    CHECK_CUDA_ERRORS(cudaFreeHost(datacs_in));
    CHECK_CUDA_ERRORS(cudaFree(d_datacs_in));
}
void pushdatacs_inToDevice(unsigned int count) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_datacs_in, datacs_in, count * sizeof(scalar), cudaMemcpyHostToDevice));
}
void pulldatacs_inFromDevice(unsigned int count) {
    CHECK_CUDA_ERRORS(cudaMemcpy(datacs_in, d_datacs_in, count * sizeof(scalar), cudaMemcpyDeviceToHost));
}
void allocatedatacs_out(unsigned int count) {
    CHECK_CUDA_ERRORS(cudaHostAlloc(&datacs_out, count * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_datacs_out, count * sizeof(scalar)));
    pushMergedNeuronUpdate1dataCS0ToDevice(0, d_datacs_out);
}
void freedatacs_out() {
    CHECK_CUDA_ERRORS(cudaFreeHost(datacs_out));
    CHECK_CUDA_ERRORS(cudaFree(d_datacs_out));
}
void pushdatacs_outToDevice(unsigned int count) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_datacs_out, datacs_out, count * sizeof(scalar), cudaMemcpyHostToDevice));
}
void pulldatacs_outFromDevice(unsigned int count) {
    CHECK_CUDA_ERRORS(cudaMemcpy(datacs_out, d_datacs_out, count * sizeof(scalar), cudaMemcpyDeviceToHost));
}

// ------------------------------------------------------------------------
// copying things to device
// ------------------------------------------------------------------------
void pushp_hSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntp_h, glbSpkCntp_h, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkp_h, glbSpkp_h, 500 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushp_hCurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntp_h, glbSpkCntp_h, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkp_h, glbSpkp_h, glbSpkCntp_h[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushp_hPreviousSpikeTimesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_prevSTp_h, prevSTp_h, 500 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushrp_hToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_rp_h, rp_h, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentrp_hToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_rp_h, rp_h, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushxp_hToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_xp_h, xp_h, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentxp_hToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_xp_h, xp_h, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushdrp_hToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_drp_h, drp_h, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentdrp_hToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_drp_h, drp_h, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushbp_hToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_bp_h, bp_h, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentbp_hToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_bp_h, bp_h, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushdbp_hToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_dbp_h, dbp_h, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentdbp_hToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_dbp_h, dbp_h, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pusherr_fbp_hToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_err_fbp_h, err_fbp_h, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrenterr_fbp_hToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_err_fbp_h, err_fbp_h, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushr_eventp_hToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_r_eventp_h, r_eventp_h, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentr_eventp_hToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_r_eventp_h, r_eventp_h, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushr_prev_eventp_hToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_r_prev_eventp_h, r_prev_eventp_h, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentr_prev_eventp_hToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_r_prev_eventp_h, r_prev_eventp_h, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushp_hStateToDevice(bool uninitialisedOnly) {
    pushrp_hToDevice(uninitialisedOnly);
    pushxp_hToDevice(uninitialisedOnly);
    pushdrp_hToDevice(uninitialisedOnly);
    pushbp_hToDevice(uninitialisedOnly);
    pushdbp_hToDevice(uninitialisedOnly);
    pusherr_fbp_hToDevice(uninitialisedOnly);
    pushr_eventp_hToDevice(uninitialisedOnly);
    pushr_prev_eventp_hToDevice(uninitialisedOnly);
}

void pushp_iSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntp_i, glbSpkCntp_i, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkp_i, glbSpkp_i, 9 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushp_iCurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntp_i, glbSpkCntp_i, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkp_i, glbSpkp_i, glbSpkCntp_i[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushp_iPreviousSpikeTimesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_prevSTp_i, prevSTp_i, 9 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushrp_iToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_rp_i, rp_i, 9 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentrp_iToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_rp_i, rp_i, 9 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushr_eventp_iToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_r_eventp_i, r_eventp_i, 9 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentr_eventp_iToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_r_eventp_i, r_eventp_i, 9 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushr_prev_eventp_iToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_r_prev_eventp_i, r_prev_eventp_i, 9 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentr_prev_eventp_iToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_r_prev_eventp_i, r_prev_eventp_i, 9 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushp_iStateToDevice(bool uninitialisedOnly) {
    pushrp_iToDevice(uninitialisedOnly);
    pushr_eventp_iToDevice(uninitialisedOnly);
    pushr_prev_eventp_iToDevice(uninitialisedOnly);
}

void pusht_idxcs_inToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_t_idxcs_in, t_idxcs_in, 9 * sizeof(int), cudaMemcpyHostToDevice));
    }
}

void pushperiodiccs_inToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_periodiccs_in, periodiccs_in, 9 * sizeof(int), cudaMemcpyHostToDevice));
    }
}

void pusht_datacs_inToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_t_datacs_in, t_datacs_in, 9 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushdt_datacs_inToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_dt_datacs_in, dt_datacs_in, 9 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushntcs_inToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_ntcs_in, ntcs_in, 9 * sizeof(int), cudaMemcpyHostToDevice));
    }
}

void pushcs_inStateToDevice(bool uninitialisedOnly) {
    pusht_idxcs_inToDevice(uninitialisedOnly);
    pushperiodiccs_inToDevice(uninitialisedOnly);
    pusht_datacs_inToDevice(uninitialisedOnly);
    pushdt_datacs_inToDevice(uninitialisedOnly);
    pushntcs_inToDevice(uninitialisedOnly);
}

void pushp_oSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntp_o, glbSpkCntp_o, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkp_o, glbSpkp_o, 11 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushp_oCurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntp_o, glbSpkCntp_o, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkp_o, glbSpkp_o, glbSpkCntp_o[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushp_oPreviousSpikeTimesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_prevSTp_o, prevSTp_o, 11 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushxp_oToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_xp_o, xp_o, 11 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentxp_oToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_xp_o, xp_o, 11 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushrp_oToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_rp_o, rp_o, 11 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentrp_oToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_rp_o, rp_o, 11 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushdrp_oToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_drp_o, drp_o, 11 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentdrp_oToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_drp_o, drp_o, 11 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushtargp_oToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_targp_o, targp_o, 11 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrenttargp_oToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_targp_o, targp_o, 11 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pusherrp_oToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_errp_o, errp_o, 11 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrenterrp_oToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_errp_o, errp_o, 11 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pusherr_eventp_oToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_err_eventp_o, err_eventp_o, 11 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrenterr_eventp_oToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_err_eventp_o, err_eventp_o, 11 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pusherr_prev_eventp_oToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_err_prev_eventp_o, err_prev_eventp_o, 11 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrenterr_prev_eventp_oToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_err_prev_eventp_o, err_prev_eventp_o, 11 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushbp_oToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_bp_o, bp_o, 11 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentbp_oToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_bp_o, bp_o, 11 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushdbp_oToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_dbp_o, dbp_o, 11 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentdbp_oToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_dbp_o, dbp_o, 11 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushlossp_oToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_lossp_o, lossp_o, 11 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentlossp_oToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_lossp_o, lossp_o, 11 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushp_oStateToDevice(bool uninitialisedOnly) {
    pushxp_oToDevice(uninitialisedOnly);
    pushrp_oToDevice(uninitialisedOnly);
    pushdrp_oToDevice(uninitialisedOnly);
    pushtargp_oToDevice(uninitialisedOnly);
    pusherrp_oToDevice(uninitialisedOnly);
    pusherr_eventp_oToDevice(uninitialisedOnly);
    pusherr_prev_eventp_oToDevice(uninitialisedOnly);
    pushbp_oToDevice(uninitialisedOnly);
    pushdbp_oToDevice(uninitialisedOnly);
    pushlossp_oToDevice(uninitialisedOnly);
}

void pusht_idxcs_outToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_t_idxcs_out, t_idxcs_out, 11 * sizeof(int), cudaMemcpyHostToDevice));
    }
}

void pushperiodiccs_outToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_periodiccs_out, periodiccs_out, 11 * sizeof(int), cudaMemcpyHostToDevice));
    }
}

void pusht_datacs_outToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_t_datacs_out, t_datacs_out, 11 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushdt_datacs_outToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_dt_datacs_out, dt_datacs_out, 11 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushntcs_outToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_ntcs_out, ntcs_out, 11 * sizeof(int), cudaMemcpyHostToDevice));
    }
}

void pushcs_outStateToDevice(bool uninitialisedOnly) {
    pusht_idxcs_outToDevice(uninitialisedOnly);
    pushperiodiccs_outToDevice(uninitialisedOnly);
    pusht_datacs_outToDevice(uninitialisedOnly);
    pushdt_datacs_outToDevice(uninitialisedOnly);
    pushntcs_outToDevice(uninitialisedOnly);
}

void pushp_rSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntp_r, glbSpkCntp_r, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkp_r, glbSpkp_r, 500 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushp_rCurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntp_r, glbSpkCntp_r, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkp_r, glbSpkp_r, glbSpkCntp_r[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushp_rPreviousSpikeTimesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_prevSTp_r, prevSTp_r, 500 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushr_tracep_rToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_r_tracep_r, r_tracep_r, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentr_tracep_rToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_r_tracep_r, r_tracep_r, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushrp_rToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_rp_r, rp_r, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentrp_rToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_rp_r, rp_r, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushr_eventp_rToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_r_eventp_r, r_eventp_r, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentr_eventp_rToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_r_eventp_r, r_eventp_r, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushr_prev_eventp_rToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_r_prev_eventp_r, r_prev_eventp_r, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentr_prev_eventp_rToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_r_prev_eventp_r, r_prev_eventp_r, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushp_rStateToDevice(bool uninitialisedOnly) {
    pushr_tracep_rToDevice(uninitialisedOnly);
    pushrp_rToDevice(uninitialisedOnly);
    pushr_eventp_rToDevice(uninitialisedOnly);
    pushr_prev_eventp_rToDevice(uninitialisedOnly);
}

void pushmplast_step_reduced_p_hToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_mplast_step_reduced_p_h, mplast_step_reduced_p_h, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushvplast_step_reduced_p_hToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_vplast_step_reduced_p_h, vplast_step_reduced_p_h, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushtimeplast_step_reduced_p_hToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_timeplast_step_reduced_p_h, timeplast_step_reduced_p_h, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushplast_step_reduced_p_hStateToDevice(bool uninitialisedOnly) {
    pushmplast_step_reduced_p_hToDevice(uninitialisedOnly);
    pushvplast_step_reduced_p_hToDevice(uninitialisedOnly);
    pushtimeplast_step_reduced_p_hToDevice(uninitialisedOnly);
}

void pushmplast_step_reduced_p_oToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_mplast_step_reduced_p_o, mplast_step_reduced_p_o, 11 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushvplast_step_reduced_p_oToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_vplast_step_reduced_p_o, vplast_step_reduced_p_o, 11 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushtimeplast_step_reduced_p_oToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_timeplast_step_reduced_p_o, timeplast_step_reduced_p_o, 11 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushplast_step_reduced_p_oStateToDevice(bool uninitialisedOnly) {
    pushmplast_step_reduced_p_oToDevice(uninitialisedOnly);
    pushvplast_step_reduced_p_oToDevice(uninitialisedOnly);
    pushtimeplast_step_reduced_p_oToDevice(uninitialisedOnly);
}

void pushreducedChangereduce_batch_bias_change_p_hToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_reducedChangereduce_batch_bias_change_p_h, reducedChangereduce_batch_bias_change_p_h, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushreduce_batch_bias_change_p_hStateToDevice(bool uninitialisedOnly) {
    pushreducedChangereduce_batch_bias_change_p_hToDevice(uninitialisedOnly);
}

void pushreducedChangereduce_batch_bias_change_p_oToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_reducedChangereduce_batch_bias_change_p_o, reducedChangereduce_batch_bias_change_p_o, 11 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushreduce_batch_bias_change_p_oStateToDevice(bool uninitialisedOnly) {
    pushreducedChangereduce_batch_bias_change_p_oToDevice(uninitialisedOnly);
}

void pushMaxValsoftmax_1ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_MaxValsoftmax_1, MaxValsoftmax_1, 1 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushsoftmax_1StateToDevice(bool uninitialisedOnly) {
    pushMaxValsoftmax_1ToDevice(uninitialisedOnly);
}

void pushSumExpValsoftmax_2ToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_SumExpValsoftmax_2, SumExpValsoftmax_2, 1 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushsoftmax_2StateToDevice(bool uninitialisedOnly) {
    pushSumExpValsoftmax_2ToDevice(uninitialisedOnly);
}

void pushsoftmax_3StateToDevice(bool uninitialisedOnly) {
}

void pushmplast_step_reduced_w_hoToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_mplast_step_reduced_w_ho, mplast_step_reduced_w_ho, 5500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushvplast_step_reduced_w_hoToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_vplast_step_reduced_w_ho, vplast_step_reduced_w_ho, 5500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushtimeplast_step_reduced_w_hoToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_timeplast_step_reduced_w_ho, timeplast_step_reduced_w_ho, 5500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushplast_step_reduced_w_hoStateToDevice(bool uninitialisedOnly) {
    pushmplast_step_reduced_w_hoToDevice(uninitialisedOnly);
    pushvplast_step_reduced_w_hoToDevice(uninitialisedOnly);
    pushtimeplast_step_reduced_w_hoToDevice(uninitialisedOnly);
}

void pushmplast_step_reduced_w_hrToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_mplast_step_reduced_w_hr, mplast_step_reduced_w_hr, 250000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushvplast_step_reduced_w_hrToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_vplast_step_reduced_w_hr, vplast_step_reduced_w_hr, 250000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushtimeplast_step_reduced_w_hrToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_timeplast_step_reduced_w_hr, timeplast_step_reduced_w_hr, 250000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushplast_step_reduced_w_hrStateToDevice(bool uninitialisedOnly) {
    pushmplast_step_reduced_w_hrToDevice(uninitialisedOnly);
    pushvplast_step_reduced_w_hrToDevice(uninitialisedOnly);
    pushtimeplast_step_reduced_w_hrToDevice(uninitialisedOnly);
}

void pushmplast_step_reduced_w_ohToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_mplast_step_reduced_w_oh, mplast_step_reduced_w_oh, 5500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushvplast_step_reduced_w_ohToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_vplast_step_reduced_w_oh, vplast_step_reduced_w_oh, 5500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushtimeplast_step_reduced_w_ohToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_timeplast_step_reduced_w_oh, timeplast_step_reduced_w_oh, 5500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushplast_step_reduced_w_ohStateToDevice(bool uninitialisedOnly) {
    pushmplast_step_reduced_w_ohToDevice(uninitialisedOnly);
    pushvplast_step_reduced_w_ohToDevice(uninitialisedOnly);
    pushtimeplast_step_reduced_w_ohToDevice(uninitialisedOnly);
}

void pushreducedChangereduce_batch_weight_change_w_hoToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_reducedChangereduce_batch_weight_change_w_ho, reducedChangereduce_batch_weight_change_w_ho, 5500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushreduce_batch_weight_change_w_hoStateToDevice(bool uninitialisedOnly) {
    pushreducedChangereduce_batch_weight_change_w_hoToDevice(uninitialisedOnly);
}

void pushreducedChangereduce_batch_weight_change_w_hrToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_reducedChangereduce_batch_weight_change_w_hr, reducedChangereduce_batch_weight_change_w_hr, 250000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushreduce_batch_weight_change_w_hrStateToDevice(bool uninitialisedOnly) {
    pushreducedChangereduce_batch_weight_change_w_hrToDevice(uninitialisedOnly);
}

void pushreducedChangereduce_batch_weight_change_w_ohToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_reducedChangereduce_batch_weight_change_w_oh, reducedChangereduce_batch_weight_change_w_oh, 5500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushreduce_batch_weight_change_w_ohStateToDevice(bool uninitialisedOnly) {
    pushreducedChangereduce_batch_weight_change_w_ohToDevice(uninitialisedOnly);
}

void pushgw_hoToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_gw_ho, gw_ho, 5500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushinp_prevw_hoToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inp_prevw_ho, inp_prevw_ho, 5500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushdgw_hoToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_dgw_ho, dgw_ho, 5500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushdg_prevw_hoToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_dg_prevw_ho, dg_prevw_ho, 5500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pusht_prevw_hoToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_t_prevw_ho, t_prevw_ho, 5500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushinSynw_hoToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynw_ho, inSynw_ho, 500 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushw_hoStateToDevice(bool uninitialisedOnly) {
    pushgw_hoToDevice(uninitialisedOnly);
    pushinp_prevw_hoToDevice(uninitialisedOnly);
    pushdgw_hoToDevice(uninitialisedOnly);
    pushdg_prevw_hoToDevice(uninitialisedOnly);
    pusht_prevw_hoToDevice(uninitialisedOnly);
    pushinSynw_hoToDevice(uninitialisedOnly);
}

void pushgw_hrToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_gw_hr, gw_hr, 250000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushinp_prevw_hrToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inp_prevw_hr, inp_prevw_hr, 250000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushdgw_hrToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_dgw_hr, dgw_hr, 250000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushdg_prevw_hrToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_dg_prevw_hr, dg_prevw_hr, 250000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pusht_prevw_hrToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_t_prevw_hr, t_prevw_hr, 250000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushinSynw_hrToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynw_hr, inSynw_hr, 500 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushw_hrStateToDevice(bool uninitialisedOnly) {
    pushgw_hrToDevice(uninitialisedOnly);
    pushinp_prevw_hrToDevice(uninitialisedOnly);
    pushdgw_hrToDevice(uninitialisedOnly);
    pushdg_prevw_hrToDevice(uninitialisedOnly);
    pusht_prevw_hrToDevice(uninitialisedOnly);
    pushinSynw_hrToDevice(uninitialisedOnly);
}

void pushgw_ohToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_gw_oh, gw_oh, 5500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushinp_prevw_ohToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inp_prevw_oh, inp_prevw_oh, 5500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushdgw_ohToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_dgw_oh, dgw_oh, 5500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushdg_prevw_ohToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_dg_prevw_oh, dg_prevw_oh, 5500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pusht_prevw_ohToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_t_prevw_oh, t_prevw_oh, 5500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushinSynw_ohToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynw_oh, inSynw_oh, 11 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushw_ohStateToDevice(bool uninitialisedOnly) {
    pushgw_ohToDevice(uninitialisedOnly);
    pushinp_prevw_ohToDevice(uninitialisedOnly);
    pushdgw_ohToDevice(uninitialisedOnly);
    pushdg_prevw_ohToDevice(uninitialisedOnly);
    pusht_prevw_ohToDevice(uninitialisedOnly);
    pushinSynw_ohToDevice(uninitialisedOnly);
}

void pushgw_riToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_gw_ri, gw_ri, 4500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushinp_prevw_riToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inp_prevw_ri, inp_prevw_ri, 4500 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushinSynw_riToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynw_ri, inSynw_ri, 500 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushw_riStateToDevice(bool uninitialisedOnly) {
    pushgw_riToDevice(uninitialisedOnly);
    pushinp_prevw_riToDevice(uninitialisedOnly);
    pushinSynw_riToDevice(uninitialisedOnly);
}

void pushgw_rrToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_gw_rr, gw_rr, 250000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushinp_prevw_rrToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inp_prevw_rr, inp_prevw_rr, 250000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushinSynw_rrToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynw_rr, inSynw_rr, 500 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushw_rrStateToDevice(bool uninitialisedOnly) {
    pushgw_rrToDevice(uninitialisedOnly);
    pushinp_prevw_rrToDevice(uninitialisedOnly);
    pushinSynw_rrToDevice(uninitialisedOnly);
}


// ------------------------------------------------------------------------
// copying things from device
// ------------------------------------------------------------------------
void pullp_hSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntp_h, d_glbSpkCntp_h, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkp_h, d_glbSpkp_h, 500 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullp_hCurrentSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntp_h, d_glbSpkCntp_h, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkp_h, d_glbSpkp_h, glbSpkCntp_h[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullp_hPreviousSpikeTimesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(prevSTp_h, d_prevSTp_h, 500 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullrp_hFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rp_h, d_rp_h, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentrp_hFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rp_h, d_rp_h, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullxp_hFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(xp_h, d_xp_h, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentxp_hFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(xp_h, d_xp_h, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pulldrp_hFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(drp_h, d_drp_h, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentdrp_hFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(drp_h, d_drp_h, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullbp_hFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(bp_h, d_bp_h, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentbp_hFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(bp_h, d_bp_h, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pulldbp_hFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(dbp_h, d_dbp_h, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentdbp_hFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(dbp_h, d_dbp_h, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullerr_fbp_hFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(err_fbp_h, d_err_fbp_h, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrenterr_fbp_hFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(err_fbp_h, d_err_fbp_h, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullr_eventp_hFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(r_eventp_h, d_r_eventp_h, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentr_eventp_hFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(r_eventp_h, d_r_eventp_h, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullr_prev_eventp_hFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(r_prev_eventp_h, d_r_prev_eventp_h, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentr_prev_eventp_hFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(r_prev_eventp_h, d_r_prev_eventp_h, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullp_hStateFromDevice() {
    pullrp_hFromDevice();
    pullxp_hFromDevice();
    pulldrp_hFromDevice();
    pullbp_hFromDevice();
    pulldbp_hFromDevice();
    pullerr_fbp_hFromDevice();
    pullr_eventp_hFromDevice();
    pullr_prev_eventp_hFromDevice();
}

void pullp_iSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntp_i, d_glbSpkCntp_i, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkp_i, d_glbSpkp_i, 9 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullp_iCurrentSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntp_i, d_glbSpkCntp_i, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkp_i, d_glbSpkp_i, glbSpkCntp_i[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullp_iPreviousSpikeTimesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(prevSTp_i, d_prevSTp_i, 9 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullrp_iFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rp_i, d_rp_i, 9 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentrp_iFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rp_i, d_rp_i, 9 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullr_eventp_iFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(r_eventp_i, d_r_eventp_i, 9 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentr_eventp_iFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(r_eventp_i, d_r_eventp_i, 9 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullr_prev_eventp_iFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(r_prev_eventp_i, d_r_prev_eventp_i, 9 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentr_prev_eventp_iFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(r_prev_eventp_i, d_r_prev_eventp_i, 9 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullp_iStateFromDevice() {
    pullrp_iFromDevice();
    pullr_eventp_iFromDevice();
    pullr_prev_eventp_iFromDevice();
}

void pullt_idxcs_inFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(t_idxcs_in, d_t_idxcs_in, 9 * sizeof(int), cudaMemcpyDeviceToHost));
}

void pullperiodiccs_inFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(periodiccs_in, d_periodiccs_in, 9 * sizeof(int), cudaMemcpyDeviceToHost));
}

void pullt_datacs_inFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(t_datacs_in, d_t_datacs_in, 9 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pulldt_datacs_inFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(dt_datacs_in, d_dt_datacs_in, 9 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullntcs_inFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(ntcs_in, d_ntcs_in, 9 * sizeof(int), cudaMemcpyDeviceToHost));
}

void pullcs_inStateFromDevice() {
    pullt_idxcs_inFromDevice();
    pullperiodiccs_inFromDevice();
    pullt_datacs_inFromDevice();
    pulldt_datacs_inFromDevice();
    pullntcs_inFromDevice();
}

void pullp_oSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntp_o, d_glbSpkCntp_o, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkp_o, d_glbSpkp_o, 11 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullp_oCurrentSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntp_o, d_glbSpkCntp_o, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkp_o, d_glbSpkp_o, glbSpkCntp_o[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullp_oPreviousSpikeTimesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(prevSTp_o, d_prevSTp_o, 11 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullxp_oFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(xp_o, d_xp_o, 11 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentxp_oFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(xp_o, d_xp_o, 11 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullrp_oFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rp_o, d_rp_o, 11 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentrp_oFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rp_o, d_rp_o, 11 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pulldrp_oFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(drp_o, d_drp_o, 11 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentdrp_oFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(drp_o, d_drp_o, 11 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pulltargp_oFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(targp_o, d_targp_o, 11 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrenttargp_oFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(targp_o, d_targp_o, 11 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullerrp_oFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(errp_o, d_errp_o, 11 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrenterrp_oFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(errp_o, d_errp_o, 11 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullerr_eventp_oFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(err_eventp_o, d_err_eventp_o, 11 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrenterr_eventp_oFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(err_eventp_o, d_err_eventp_o, 11 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullerr_prev_eventp_oFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(err_prev_eventp_o, d_err_prev_eventp_o, 11 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrenterr_prev_eventp_oFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(err_prev_eventp_o, d_err_prev_eventp_o, 11 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullbp_oFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(bp_o, d_bp_o, 11 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentbp_oFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(bp_o, d_bp_o, 11 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pulldbp_oFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(dbp_o, d_dbp_o, 11 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentdbp_oFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(dbp_o, d_dbp_o, 11 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pulllossp_oFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(lossp_o, d_lossp_o, 11 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentlossp_oFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(lossp_o, d_lossp_o, 11 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullp_oStateFromDevice() {
    pullxp_oFromDevice();
    pullrp_oFromDevice();
    pulldrp_oFromDevice();
    pulltargp_oFromDevice();
    pullerrp_oFromDevice();
    pullerr_eventp_oFromDevice();
    pullerr_prev_eventp_oFromDevice();
    pullbp_oFromDevice();
    pulldbp_oFromDevice();
    pulllossp_oFromDevice();
}

void pullt_idxcs_outFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(t_idxcs_out, d_t_idxcs_out, 11 * sizeof(int), cudaMemcpyDeviceToHost));
}

void pullperiodiccs_outFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(periodiccs_out, d_periodiccs_out, 11 * sizeof(int), cudaMemcpyDeviceToHost));
}

void pullt_datacs_outFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(t_datacs_out, d_t_datacs_out, 11 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pulldt_datacs_outFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(dt_datacs_out, d_dt_datacs_out, 11 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullntcs_outFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(ntcs_out, d_ntcs_out, 11 * sizeof(int), cudaMemcpyDeviceToHost));
}

void pullcs_outStateFromDevice() {
    pullt_idxcs_outFromDevice();
    pullperiodiccs_outFromDevice();
    pullt_datacs_outFromDevice();
    pulldt_datacs_outFromDevice();
    pullntcs_outFromDevice();
}

void pullp_rSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntp_r, d_glbSpkCntp_r, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkp_r, d_glbSpkp_r, 500 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullp_rCurrentSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntp_r, d_glbSpkCntp_r, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkp_r, d_glbSpkp_r, glbSpkCntp_r[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullp_rPreviousSpikeTimesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(prevSTp_r, d_prevSTp_r, 500 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullr_tracep_rFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(r_tracep_r, d_r_tracep_r, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentr_tracep_rFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(r_tracep_r, d_r_tracep_r, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullrp_rFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rp_r, d_rp_r, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentrp_rFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rp_r, d_rp_r, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullr_eventp_rFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(r_eventp_r, d_r_eventp_r, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentr_eventp_rFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(r_eventp_r, d_r_eventp_r, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullr_prev_eventp_rFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(r_prev_eventp_r, d_r_prev_eventp_r, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentr_prev_eventp_rFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(r_prev_eventp_r, d_r_prev_eventp_r, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullp_rStateFromDevice() {
    pullr_tracep_rFromDevice();
    pullrp_rFromDevice();
    pullr_eventp_rFromDevice();
    pullr_prev_eventp_rFromDevice();
}

void pullmplast_step_reduced_p_hFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(mplast_step_reduced_p_h, d_mplast_step_reduced_p_h, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullvplast_step_reduced_p_hFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(vplast_step_reduced_p_h, d_vplast_step_reduced_p_h, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pulltimeplast_step_reduced_p_hFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(timeplast_step_reduced_p_h, d_timeplast_step_reduced_p_h, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullplast_step_reduced_p_hStateFromDevice() {
    pullmplast_step_reduced_p_hFromDevice();
    pullvplast_step_reduced_p_hFromDevice();
    pulltimeplast_step_reduced_p_hFromDevice();
}

void pullmplast_step_reduced_p_oFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(mplast_step_reduced_p_o, d_mplast_step_reduced_p_o, 11 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullvplast_step_reduced_p_oFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(vplast_step_reduced_p_o, d_vplast_step_reduced_p_o, 11 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pulltimeplast_step_reduced_p_oFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(timeplast_step_reduced_p_o, d_timeplast_step_reduced_p_o, 11 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullplast_step_reduced_p_oStateFromDevice() {
    pullmplast_step_reduced_p_oFromDevice();
    pullvplast_step_reduced_p_oFromDevice();
    pulltimeplast_step_reduced_p_oFromDevice();
}

void pullreducedChangereduce_batch_bias_change_p_hFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(reducedChangereduce_batch_bias_change_p_h, d_reducedChangereduce_batch_bias_change_p_h, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullreduce_batch_bias_change_p_hStateFromDevice() {
    pullreducedChangereduce_batch_bias_change_p_hFromDevice();
}

void pullreducedChangereduce_batch_bias_change_p_oFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(reducedChangereduce_batch_bias_change_p_o, d_reducedChangereduce_batch_bias_change_p_o, 11 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullreduce_batch_bias_change_p_oStateFromDevice() {
    pullreducedChangereduce_batch_bias_change_p_oFromDevice();
}

void pullMaxValsoftmax_1FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(MaxValsoftmax_1, d_MaxValsoftmax_1, 1 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullsoftmax_1StateFromDevice() {
    pullMaxValsoftmax_1FromDevice();
}

void pullSumExpValsoftmax_2FromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(SumExpValsoftmax_2, d_SumExpValsoftmax_2, 1 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullsoftmax_2StateFromDevice() {
    pullSumExpValsoftmax_2FromDevice();
}

void pullsoftmax_3StateFromDevice() {
}

void pullmplast_step_reduced_w_hoFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(mplast_step_reduced_w_ho, d_mplast_step_reduced_w_ho, 5500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullvplast_step_reduced_w_hoFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(vplast_step_reduced_w_ho, d_vplast_step_reduced_w_ho, 5500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pulltimeplast_step_reduced_w_hoFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(timeplast_step_reduced_w_ho, d_timeplast_step_reduced_w_ho, 5500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullplast_step_reduced_w_hoStateFromDevice() {
    pullmplast_step_reduced_w_hoFromDevice();
    pullvplast_step_reduced_w_hoFromDevice();
    pulltimeplast_step_reduced_w_hoFromDevice();
}

void pullmplast_step_reduced_w_hrFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(mplast_step_reduced_w_hr, d_mplast_step_reduced_w_hr, 250000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullvplast_step_reduced_w_hrFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(vplast_step_reduced_w_hr, d_vplast_step_reduced_w_hr, 250000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pulltimeplast_step_reduced_w_hrFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(timeplast_step_reduced_w_hr, d_timeplast_step_reduced_w_hr, 250000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullplast_step_reduced_w_hrStateFromDevice() {
    pullmplast_step_reduced_w_hrFromDevice();
    pullvplast_step_reduced_w_hrFromDevice();
    pulltimeplast_step_reduced_w_hrFromDevice();
}

void pullmplast_step_reduced_w_ohFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(mplast_step_reduced_w_oh, d_mplast_step_reduced_w_oh, 5500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullvplast_step_reduced_w_ohFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(vplast_step_reduced_w_oh, d_vplast_step_reduced_w_oh, 5500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pulltimeplast_step_reduced_w_ohFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(timeplast_step_reduced_w_oh, d_timeplast_step_reduced_w_oh, 5500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullplast_step_reduced_w_ohStateFromDevice() {
    pullmplast_step_reduced_w_ohFromDevice();
    pullvplast_step_reduced_w_ohFromDevice();
    pulltimeplast_step_reduced_w_ohFromDevice();
}

void pullreducedChangereduce_batch_weight_change_w_hoFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(reducedChangereduce_batch_weight_change_w_ho, d_reducedChangereduce_batch_weight_change_w_ho, 5500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullreduce_batch_weight_change_w_hoStateFromDevice() {
    pullreducedChangereduce_batch_weight_change_w_hoFromDevice();
}

void pullreducedChangereduce_batch_weight_change_w_hrFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(reducedChangereduce_batch_weight_change_w_hr, d_reducedChangereduce_batch_weight_change_w_hr, 250000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullreduce_batch_weight_change_w_hrStateFromDevice() {
    pullreducedChangereduce_batch_weight_change_w_hrFromDevice();
}

void pullreducedChangereduce_batch_weight_change_w_ohFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(reducedChangereduce_batch_weight_change_w_oh, d_reducedChangereduce_batch_weight_change_w_oh, 5500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullreduce_batch_weight_change_w_ohStateFromDevice() {
    pullreducedChangereduce_batch_weight_change_w_ohFromDevice();
}

void pullgw_hoFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(gw_ho, d_gw_ho, 5500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullinp_prevw_hoFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inp_prevw_ho, d_inp_prevw_ho, 5500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pulldgw_hoFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(dgw_ho, d_dgw_ho, 5500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pulldg_prevw_hoFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(dg_prevw_ho, d_dg_prevw_ho, 5500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullt_prevw_hoFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(t_prevw_ho, d_t_prevw_ho, 5500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullinSynw_hoFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynw_ho, d_inSynw_ho, 500 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullw_hoStateFromDevice() {
    pullgw_hoFromDevice();
    pullinp_prevw_hoFromDevice();
    pulldgw_hoFromDevice();
    pulldg_prevw_hoFromDevice();
    pullt_prevw_hoFromDevice();
    pullinSynw_hoFromDevice();
}

void pullgw_hrFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(gw_hr, d_gw_hr, 250000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullinp_prevw_hrFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inp_prevw_hr, d_inp_prevw_hr, 250000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pulldgw_hrFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(dgw_hr, d_dgw_hr, 250000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pulldg_prevw_hrFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(dg_prevw_hr, d_dg_prevw_hr, 250000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullt_prevw_hrFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(t_prevw_hr, d_t_prevw_hr, 250000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullinSynw_hrFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynw_hr, d_inSynw_hr, 500 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullw_hrStateFromDevice() {
    pullgw_hrFromDevice();
    pullinp_prevw_hrFromDevice();
    pulldgw_hrFromDevice();
    pulldg_prevw_hrFromDevice();
    pullt_prevw_hrFromDevice();
    pullinSynw_hrFromDevice();
}

void pullgw_ohFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(gw_oh, d_gw_oh, 5500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullinp_prevw_ohFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inp_prevw_oh, d_inp_prevw_oh, 5500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pulldgw_ohFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(dgw_oh, d_dgw_oh, 5500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pulldg_prevw_ohFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(dg_prevw_oh, d_dg_prevw_oh, 5500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullt_prevw_ohFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(t_prevw_oh, d_t_prevw_oh, 5500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullinSynw_ohFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynw_oh, d_inSynw_oh, 11 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullw_ohStateFromDevice() {
    pullgw_ohFromDevice();
    pullinp_prevw_ohFromDevice();
    pulldgw_ohFromDevice();
    pulldg_prevw_ohFromDevice();
    pullt_prevw_ohFromDevice();
    pullinSynw_ohFromDevice();
}

void pullgw_riFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(gw_ri, d_gw_ri, 4500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullinp_prevw_riFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inp_prevw_ri, d_inp_prevw_ri, 4500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullinSynw_riFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynw_ri, d_inSynw_ri, 500 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullw_riStateFromDevice() {
    pullgw_riFromDevice();
    pullinp_prevw_riFromDevice();
    pullinSynw_riFromDevice();
}

void pullgw_rrFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(gw_rr, d_gw_rr, 250000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullinp_prevw_rrFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inp_prevw_rr, d_inp_prevw_rr, 250000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullinSynw_rrFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynw_rr, d_inSynw_rr, 500 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullw_rrStateFromDevice() {
    pullgw_rrFromDevice();
    pullinp_prevw_rrFromDevice();
    pullinSynw_rrFromDevice();
}


// ------------------------------------------------------------------------
// helper getter functions
// ------------------------------------------------------------------------
unsigned int* getp_hCurrentSpikes(unsigned int batch) {
    return (glbSpkp_h);
}

unsigned int& getp_hCurrentSpikeCount(unsigned int batch) {
    return glbSpkCntp_h[0];
}

scalar* getCurrentrp_h(unsigned int batch) {
    return rp_h;
}

scalar* getCurrentxp_h(unsigned int batch) {
    return xp_h;
}

scalar* getCurrentdrp_h(unsigned int batch) {
    return drp_h;
}

scalar* getCurrentbp_h(unsigned int batch) {
    return bp_h;
}

scalar* getCurrentdbp_h(unsigned int batch) {
    return dbp_h;
}

scalar* getCurrenterr_fbp_h(unsigned int batch) {
    return err_fbp_h;
}

scalar* getCurrentr_eventp_h(unsigned int batch) {
    return r_eventp_h;
}

scalar* getCurrentr_prev_eventp_h(unsigned int batch) {
    return r_prev_eventp_h;
}

unsigned int* getp_iCurrentSpikes(unsigned int batch) {
    return (glbSpkp_i);
}

unsigned int& getp_iCurrentSpikeCount(unsigned int batch) {
    return glbSpkCntp_i[0];
}

scalar* getCurrentrp_i(unsigned int batch) {
    return rp_i;
}

scalar* getCurrentr_eventp_i(unsigned int batch) {
    return r_eventp_i;
}

scalar* getCurrentr_prev_eventp_i(unsigned int batch) {
    return r_prev_eventp_i;
}

unsigned int* getp_oCurrentSpikes(unsigned int batch) {
    return (glbSpkp_o);
}

unsigned int& getp_oCurrentSpikeCount(unsigned int batch) {
    return glbSpkCntp_o[0];
}

scalar* getCurrentxp_o(unsigned int batch) {
    return xp_o;
}

scalar* getCurrentrp_o(unsigned int batch) {
    return rp_o;
}

scalar* getCurrentdrp_o(unsigned int batch) {
    return drp_o;
}

scalar* getCurrenttargp_o(unsigned int batch) {
    return targp_o;
}

scalar* getCurrenterrp_o(unsigned int batch) {
    return errp_o;
}

scalar* getCurrenterr_eventp_o(unsigned int batch) {
    return err_eventp_o;
}

scalar* getCurrenterr_prev_eventp_o(unsigned int batch) {
    return err_prev_eventp_o;
}

scalar* getCurrentbp_o(unsigned int batch) {
    return bp_o;
}

scalar* getCurrentdbp_o(unsigned int batch) {
    return dbp_o;
}

scalar* getCurrentlossp_o(unsigned int batch) {
    return lossp_o;
}

unsigned int* getp_rCurrentSpikes(unsigned int batch) {
    return (glbSpkp_r);
}

unsigned int& getp_rCurrentSpikeCount(unsigned int batch) {
    return glbSpkCntp_r[0];
}

scalar* getCurrentr_tracep_r(unsigned int batch) {
    return r_tracep_r;
}

scalar* getCurrentrp_r(unsigned int batch) {
    return rp_r;
}

scalar* getCurrentr_eventp_r(unsigned int batch) {
    return r_eventp_r;
}

scalar* getCurrentr_prev_eventp_r(unsigned int batch) {
    return r_prev_eventp_r;
}


void copyStateToDevice(bool uninitialisedOnly) {
    pushp_hStateToDevice(uninitialisedOnly);
    pushp_iStateToDevice(uninitialisedOnly);
    pushcs_inStateToDevice(uninitialisedOnly);
    pushp_oStateToDevice(uninitialisedOnly);
    pushcs_outStateToDevice(uninitialisedOnly);
    pushp_rStateToDevice(uninitialisedOnly);
    pushplast_step_reduced_p_hStateToDevice(uninitialisedOnly);
    pushplast_step_reduced_p_oStateToDevice(uninitialisedOnly);
    pushreduce_batch_bias_change_p_hStateToDevice(uninitialisedOnly);
    pushreduce_batch_bias_change_p_oStateToDevice(uninitialisedOnly);
    pushsoftmax_1StateToDevice(uninitialisedOnly);
    pushsoftmax_2StateToDevice(uninitialisedOnly);
    pushsoftmax_3StateToDevice(uninitialisedOnly);
    pushplast_step_reduced_w_hoStateToDevice(uninitialisedOnly);
    pushplast_step_reduced_w_hrStateToDevice(uninitialisedOnly);
    pushplast_step_reduced_w_ohStateToDevice(uninitialisedOnly);
    pushreduce_batch_weight_change_w_hoStateToDevice(uninitialisedOnly);
    pushreduce_batch_weight_change_w_hrStateToDevice(uninitialisedOnly);
    pushreduce_batch_weight_change_w_ohStateToDevice(uninitialisedOnly);
    pushw_hoStateToDevice(uninitialisedOnly);
    pushw_hrStateToDevice(uninitialisedOnly);
    pushw_ohStateToDevice(uninitialisedOnly);
    pushw_riStateToDevice(uninitialisedOnly);
    pushw_rrStateToDevice(uninitialisedOnly);
}

void copyConnectivityToDevice(bool uninitialisedOnly) {
}

void copyStateFromDevice() {
    pullp_hStateFromDevice();
    pullp_iStateFromDevice();
    pullcs_inStateFromDevice();
    pullp_oStateFromDevice();
    pullcs_outStateFromDevice();
    pullp_rStateFromDevice();
    pullplast_step_reduced_p_hStateFromDevice();
    pullplast_step_reduced_p_oStateFromDevice();
    pullreduce_batch_bias_change_p_hStateFromDevice();
    pullreduce_batch_bias_change_p_oStateFromDevice();
    pullsoftmax_1StateFromDevice();
    pullsoftmax_2StateFromDevice();
    pullsoftmax_3StateFromDevice();
    pullplast_step_reduced_w_hoStateFromDevice();
    pullplast_step_reduced_w_hrStateFromDevice();
    pullplast_step_reduced_w_ohStateFromDevice();
    pullreduce_batch_weight_change_w_hoStateFromDevice();
    pullreduce_batch_weight_change_w_hrStateFromDevice();
    pullreduce_batch_weight_change_w_ohStateFromDevice();
    pullw_hoStateFromDevice();
    pullw_hrStateFromDevice();
    pullw_ohStateFromDevice();
    pullw_riStateFromDevice();
    pullw_rrStateFromDevice();
}

void copyCurrentSpikesFromDevice() {
    pullp_hCurrentSpikesFromDevice();
    pullp_iCurrentSpikesFromDevice();
    pullp_oCurrentSpikesFromDevice();
    pullp_rCurrentSpikesFromDevice();
}

void copyCurrentSpikeEventsFromDevice() {
}

void allocateRecordingBuffers(unsigned int timesteps) {
    numRecordingTimesteps = timesteps;
     {
    }
     {
    }
     {
    }
     {
        const unsigned int numWords = 16 * timesteps;
         {
            CHECK_CUDA_ERRORS(cudaHostAlloc(&recordSpkp_r, numWords * sizeof(uint32_t), cudaHostAllocPortable));
            CHECK_CUDA_ERRORS(cudaMalloc(&d_recordSpkp_r, numWords * sizeof(uint32_t)));
            pushMergedNeuronUpdate0recordSpkToDevice(0, d_recordSpkp_r);
        }
    }
}

void pullRecordingBuffersFromDevice() {
    if(numRecordingTimesteps == 0) {
        throw std::runtime_error("Recording buffer not allocated - cannot pull from device");
    }
     {
    }
     {
    }
     {
    }
     {
        const unsigned int numWords = 16 * numRecordingTimesteps;
         {
            CHECK_CUDA_ERRORS(cudaMemcpy(recordSpkp_r, d_recordSpkp_r, numWords * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        }
    }
}

void allocateMem() {
    int deviceID;
    CHECK_CUDA_ERRORS(cudaDeviceGetByPCIBusId(&deviceID, "0000:81:00.0"));
    CHECK_CUDA_ERRORS(cudaSetDevice(deviceID));
    
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // timers
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // local neuron groups
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkCntp_h, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkCntp_h, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkp_h, 500 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkp_h, 500 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&prevSTp_h, 500 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_prevSTp_h, 500 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rp_h, 500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rp_h, 500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&xp_h, 500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_xp_h, 500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&drp_h, 500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_drp_h, 500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&bp_h, 500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_bp_h, 500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&dbp_h, 500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_dbp_h, 500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&err_fbp_h, 500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_err_fbp_h, 500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&r_eventp_h, 500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_r_eventp_h, 500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&r_prev_eventp_h, 500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_r_prev_eventp_h, 500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkCntp_i, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkCntp_i, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkp_i, 9 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkp_i, 9 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&prevSTp_i, 9 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_prevSTp_i, 9 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rp_i, 9 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rp_i, 9 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&r_eventp_i, 9 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_r_eventp_i, 9 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&r_prev_eventp_i, 9 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_r_prev_eventp_i, 9 * sizeof(scalar)));
    // current source variables
    CHECK_CUDA_ERRORS(cudaHostAlloc(&t_idxcs_in, 9 * sizeof(int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_t_idxcs_in, 9 * sizeof(int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&periodiccs_in, 9 * sizeof(int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_periodiccs_in, 9 * sizeof(int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&t_datacs_in, 9 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_t_datacs_in, 9 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&dt_datacs_in, 9 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_dt_datacs_in, 9 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&ntcs_in, 9 * sizeof(int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_ntcs_in, 9 * sizeof(int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkCntp_o, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkCntp_o, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkp_o, 11 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkp_o, 11 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&prevSTp_o, 11 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_prevSTp_o, 11 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&xp_o, 11 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_xp_o, 11 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rp_o, 11 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rp_o, 11 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&drp_o, 11 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_drp_o, 11 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&targp_o, 11 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_targp_o, 11 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&errp_o, 11 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_errp_o, 11 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&err_eventp_o, 11 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_err_eventp_o, 11 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&err_prev_eventp_o, 11 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_err_prev_eventp_o, 11 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&bp_o, 11 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_bp_o, 11 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&dbp_o, 11 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_dbp_o, 11 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&lossp_o, 11 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_lossp_o, 11 * sizeof(scalar)));
    // current source variables
    CHECK_CUDA_ERRORS(cudaHostAlloc(&t_idxcs_out, 11 * sizeof(int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_t_idxcs_out, 11 * sizeof(int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&periodiccs_out, 11 * sizeof(int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_periodiccs_out, 11 * sizeof(int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&t_datacs_out, 11 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_t_datacs_out, 11 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&dt_datacs_out, 11 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_dt_datacs_out, 11 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&ntcs_out, 11 * sizeof(int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_ntcs_out, 11 * sizeof(int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkCntp_r, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkCntp_r, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkp_r, 500 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkp_r, 500 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&prevSTp_r, 500 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_prevSTp_r, 500 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&r_tracep_r, 500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_r_tracep_r, 500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rp_r, 500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rp_r, 500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&r_eventp_r, 500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_r_eventp_r, 500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&r_prev_eventp_r, 500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_r_prev_eventp_r, 500 * sizeof(scalar)));
    
    // ------------------------------------------------------------------------
    // custom update variables
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaHostAlloc(&mplast_step_reduced_p_h, 500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_mplast_step_reduced_p_h, 500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&vplast_step_reduced_p_h, 500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_vplast_step_reduced_p_h, 500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&timeplast_step_reduced_p_h, 500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_timeplast_step_reduced_p_h, 500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&mplast_step_reduced_p_o, 11 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_mplast_step_reduced_p_o, 11 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&vplast_step_reduced_p_o, 11 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_vplast_step_reduced_p_o, 11 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&timeplast_step_reduced_p_o, 11 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_timeplast_step_reduced_p_o, 11 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&reducedChangereduce_batch_bias_change_p_h, 500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_reducedChangereduce_batch_bias_change_p_h, 500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&reducedChangereduce_batch_bias_change_p_o, 11 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_reducedChangereduce_batch_bias_change_p_o, 11 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&MaxValsoftmax_1, 1 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_MaxValsoftmax_1, 1 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&SumExpValsoftmax_2, 1 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_SumExpValsoftmax_2, 1 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&mplast_step_reduced_w_ho, 5500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_mplast_step_reduced_w_ho, 5500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&vplast_step_reduced_w_ho, 5500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_vplast_step_reduced_w_ho, 5500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&timeplast_step_reduced_w_ho, 5500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_timeplast_step_reduced_w_ho, 5500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&mplast_step_reduced_w_hr, 250000 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_mplast_step_reduced_w_hr, 250000 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&vplast_step_reduced_w_hr, 250000 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_vplast_step_reduced_w_hr, 250000 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&timeplast_step_reduced_w_hr, 250000 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_timeplast_step_reduced_w_hr, 250000 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&mplast_step_reduced_w_oh, 5500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_mplast_step_reduced_w_oh, 5500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&vplast_step_reduced_w_oh, 5500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_vplast_step_reduced_w_oh, 5500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&timeplast_step_reduced_w_oh, 5500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_timeplast_step_reduced_w_oh, 5500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&reducedChangereduce_batch_weight_change_w_ho, 5500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_reducedChangereduce_batch_weight_change_w_ho, 5500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&reducedChangereduce_batch_weight_change_w_hr, 250000 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_reducedChangereduce_batch_weight_change_w_hr, 250000 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&reducedChangereduce_batch_weight_change_w_oh, 5500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_reducedChangereduce_batch_weight_change_w_oh, 5500 * sizeof(scalar)));
    
    // ------------------------------------------------------------------------
    // pre and postsynaptic variables
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynw_ho, 500 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynw_ho, 500 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynw_hr, 500 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynw_hr, 500 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynw_oh, 11 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynw_oh, 11 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynw_rr, 500 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynw_rr, 500 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynw_ri, 500 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynw_ri, 500 * sizeof(float)));
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaHostAlloc(&gw_ho, 5500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_gw_ho, 5500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inp_prevw_ho, 5500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inp_prevw_ho, 5500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&dgw_ho, 5500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_dgw_ho, 5500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&dg_prevw_ho, 5500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_dg_prevw_ho, 5500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&t_prevw_ho, 5500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_t_prevw_ho, 5500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&gw_hr, 250000 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_gw_hr, 250000 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inp_prevw_hr, 250000 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inp_prevw_hr, 250000 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&dgw_hr, 250000 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_dgw_hr, 250000 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&dg_prevw_hr, 250000 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_dg_prevw_hr, 250000 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&t_prevw_hr, 250000 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_t_prevw_hr, 250000 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&gw_oh, 5500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_gw_oh, 5500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inp_prevw_oh, 5500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inp_prevw_oh, 5500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&dgw_oh, 5500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_dgw_oh, 5500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&dg_prevw_oh, 5500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_dg_prevw_oh, 5500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&t_prevw_oh, 5500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_t_prevw_oh, 5500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&gw_ri, 4500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_gw_ri, 4500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inp_prevw_ri, 4500 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inp_prevw_ri, 4500 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&gw_rr, 250000 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_gw_rr, 250000 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inp_prevw_rr, 250000 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inp_prevw_rr, 250000 * sizeof(scalar)));
    
    pushMergedNeuronInitGroup0ToDevice(0, d_glbSpkCntp_r, d_glbSpkp_r, d_prevSTp_r, d_r_tracep_r, d_rp_r, d_r_eventp_r, d_r_prev_eventp_r, d_inSynw_rr, d_inSynw_ri, 500);
    pushMergedNeuronInitGroup1ToDevice(0, d_err_prev_eventp_o, d_ntcs_out, d_dt_datacs_out, d_t_datacs_out, d_periodiccs_out, d_t_idxcs_out, d_inSynw_oh, d_lossp_o, d_dbp_o, d_bp_o, d_err_eventp_o, d_errp_o, d_targp_o, d_drp_o, d_rp_o, d_xp_o, d_prevSTp_o, d_glbSpkp_o, d_glbSpkCntp_o, 11);
    pushMergedNeuronInitGroup2ToDevice(0, d_glbSpkCntp_i, d_glbSpkp_i, d_prevSTp_i, d_rp_i, d_r_eventp_i, d_r_prev_eventp_i, d_t_idxcs_in, d_periodiccs_in, d_t_datacs_in, d_dt_datacs_in, d_ntcs_in, 9);
    pushMergedNeuronInitGroup3ToDevice(0, d_glbSpkCntp_h, d_glbSpkp_h, d_prevSTp_h, d_rp_h, d_xp_h, d_drp_h, d_bp_h, d_dbp_h, d_err_fbp_h, d_r_eventp_h, d_r_prev_eventp_h, d_inSynw_ho, d_inSynw_hr, 500);
    pushMergedSynapseInitGroup0ToDevice(0, d_gw_ri, d_inp_prevw_ri, 9, 500, 500, 1.66666666666666657e-01f);
    pushMergedSynapseInitGroup0ToDevice(1, d_gw_rr, d_inp_prevw_rr, 500, 500, 500, 1.00000000000000000e+00f);
    pushMergedSynapseInitGroup1ToDevice(0, d_gw_ho, d_inp_prevw_ho, d_dgw_ho, d_dg_prevw_ho, d_t_prevw_ho, 11, 500, 500, 1.50755672288881815e-01f);
    pushMergedSynapseInitGroup1ToDevice(1, d_gw_hr, d_inp_prevw_hr, d_dgw_hr, d_dg_prevw_hr, d_t_prevw_hr, 500, 500, 500, 2.23606797749978971e-02f);
    pushMergedSynapseInitGroup1ToDevice(2, d_gw_oh, d_inp_prevw_oh, d_dgw_oh, d_dg_prevw_oh, d_t_prevw_oh, 500, 11, 11, 2.23606797749978971e-02f);
    pushMergedCustomUpdateInitGroup0ToDevice(0, d_SumExpValsoftmax_2, 11);
    pushMergedCustomUpdateInitGroup1ToDevice(0, d_MaxValsoftmax_1, 11);
    pushMergedCustomUpdateInitGroup2ToDevice(0, d_reducedChangereduce_batch_bias_change_p_h, 500);
    pushMergedCustomUpdateInitGroup2ToDevice(1, d_reducedChangereduce_batch_bias_change_p_o, 11);
    pushMergedCustomUpdateInitGroup3ToDevice(0, d_mplast_step_reduced_p_h, d_vplast_step_reduced_p_h, d_timeplast_step_reduced_p_h, 500);
    pushMergedCustomUpdateInitGroup3ToDevice(1, d_mplast_step_reduced_p_o, d_vplast_step_reduced_p_o, d_timeplast_step_reduced_p_o, 11);
    pushMergedCustomWUUpdateInitGroup0ToDevice(0, d_reducedChangereduce_batch_weight_change_w_ho, 11, 500, 500);
    pushMergedCustomWUUpdateInitGroup0ToDevice(1, d_reducedChangereduce_batch_weight_change_w_hr, 500, 500, 500);
    pushMergedCustomWUUpdateInitGroup0ToDevice(2, d_reducedChangereduce_batch_weight_change_w_oh, 500, 11, 11);
    pushMergedCustomWUUpdateInitGroup1ToDevice(0, d_mplast_step_reduced_w_ho, d_vplast_step_reduced_w_ho, d_timeplast_step_reduced_w_ho, 11, 500, 500);
    pushMergedCustomWUUpdateInitGroup1ToDevice(1, d_mplast_step_reduced_w_hr, d_vplast_step_reduced_w_hr, d_timeplast_step_reduced_w_hr, 500, 500, 500);
    pushMergedCustomWUUpdateInitGroup1ToDevice(2, d_mplast_step_reduced_w_oh, d_vplast_step_reduced_w_oh, d_timeplast_step_reduced_w_oh, 500, 11, 11);
    pushMergedNeuronUpdateGroup0ToDevice(0, d_glbSpkCntp_r, d_glbSpkp_r, d_prevSTp_r, d_r_tracep_r, d_rp_r, d_r_eventp_r, d_r_prev_eventp_r, d_inSynw_rr, d_inSynw_ri, d_recordSpkp_r, 500);
    pushMergedNeuronUpdateGroup1ToDevice(0, d_err_prev_eventp_o, d_datacs_out, d_ntcs_out, d_dt_datacs_out, d_t_datacs_out, d_periodiccs_out, d_t_idxcs_out, d_inSynw_oh, d_lossp_o, d_dbp_o, d_bp_o, d_err_eventp_o, d_errp_o, d_targp_o, d_drp_o, d_rp_o, d_xp_o, d_prevSTp_o, d_glbSpkp_o, d_glbSpkCntp_o, 11);
    pushMergedNeuronUpdateGroup2ToDevice(0, d_glbSpkCntp_i, d_glbSpkp_i, d_prevSTp_i, d_rp_i, d_r_eventp_i, d_r_prev_eventp_i, d_t_idxcs_in, d_periodiccs_in, d_t_datacs_in, d_dt_datacs_in, d_ntcs_in, d_datacs_in, 9);
    pushMergedNeuronUpdateGroup3ToDevice(0, d_glbSpkCntp_h, d_glbSpkp_h, d_prevSTp_h, d_rp_h, d_xp_h, d_drp_h, d_bp_h, d_dbp_h, d_err_fbp_h, d_r_eventp_h, d_r_prev_eventp_h, d_inSynw_hr, d_inSynw_ho, 500);
    pushMergedPresynapticUpdateGroup0ToDevice(0, d_inSynw_rr, d_glbSpkCntp_r, d_glbSpkp_r, d_rp_r, d_r_prev_eventp_r, d_prevSTp_r, d_gw_rr, d_inp_prevw_rr, 500, 500, 500);
    pushMergedPresynapticUpdateGroup1ToDevice(0, d_inSynw_ri, d_glbSpkCntp_i, d_glbSpkp_i, d_rp_i, d_prevSTp_i, d_gw_ri, d_inp_prevw_ri, 9, 500, 500);
    pushMergedPresynapticUpdateGroup2ToDevice(0, d_inSynw_oh, d_glbSpkCntp_h, d_glbSpkp_h, d_rp_h, d_drp_o, d_errp_o, d_prevSTp_h, d_gw_oh, d_inp_prevw_oh, d_dgw_oh, d_dg_prevw_oh, d_t_prevw_oh, 500, 11, 11);
    pushMergedPresynapticUpdateGroup3ToDevice(0, d_inSynw_hr, d_glbSpkCntp_r, d_glbSpkp_r, d_r_tracep_r, d_rp_r, d_drp_h, d_err_fbp_h, d_prevSTp_r, d_gw_hr, d_inp_prevw_hr, d_dgw_hr, d_dg_prevw_hr, d_t_prevw_hr, 500, 500, 500);
    pushMergedPresynapticUpdateGroup4ToDevice(0, d_inSynw_ho, d_glbSpkCntp_o, d_glbSpkp_o, d_drp_o, d_errp_o, d_rp_h, d_prevSTp_o, d_gw_ho, d_inp_prevw_ho, d_dgw_ho, d_dg_prevw_ho, d_t_prevw_ho, 11, 500, 500);
    pushMergedNeuronPrevSpikeTimeUpdateGroup0ToDevice(0, d_glbSpkCntp_h, d_glbSpkp_h, d_prevSTp_h, 500);
    pushMergedNeuronPrevSpikeTimeUpdateGroup0ToDevice(1, d_glbSpkCntp_i, d_glbSpkp_i, d_prevSTp_i, 9);
    pushMergedNeuronPrevSpikeTimeUpdateGroup0ToDevice(2, d_glbSpkCntp_o, d_glbSpkp_o, d_prevSTp_o, 11);
    pushMergedNeuronPrevSpikeTimeUpdateGroup0ToDevice(3, d_glbSpkCntp_r, d_glbSpkp_r, d_prevSTp_r, 500);
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(0, d_glbSpkCntp_h);
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(1, d_glbSpkCntp_i);
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(2, d_glbSpkCntp_o);
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(3, d_glbSpkCntp_r);
    pushMergedCustomUpdateGroup0ToDevice(0, d_xp_o, d_MaxValsoftmax_1, d_SumExpValsoftmax_2, d_rp_o, 11);
    pushMergedCustomUpdateGroup1ToDevice(0, d_SumExpValsoftmax_2, d_xp_o, d_MaxValsoftmax_1, 11);
    pushMergedCustomUpdateGroup2ToDevice(0, d_MaxValsoftmax_1, d_xp_o, 11);
    pushMergedCustomUpdateGroup3ToDevice(0, d_reducedChangereduce_batch_bias_change_p_h, d_dbp_h, 500);
    pushMergedCustomUpdateGroup3ToDevice(1, d_reducedChangereduce_batch_bias_change_p_o, d_dbp_o, 11);
    pushMergedCustomUpdateGroup4ToDevice(0, d_mplast_step_reduced_p_h, d_vplast_step_reduced_p_h, d_timeplast_step_reduced_p_h, d_reducedChangereduce_batch_bias_change_p_h, d_bp_h, 500);
    pushMergedCustomUpdateGroup4ToDevice(1, d_mplast_step_reduced_p_o, d_vplast_step_reduced_p_o, d_timeplast_step_reduced_p_o, d_reducedChangereduce_batch_bias_change_p_o, d_bp_o, 11);
    pushMergedCustomUpdateWUGroup0ToDevice(0, d_reducedChangereduce_batch_weight_change_w_ho, d_dgw_ho, 11, 500, 500);
    pushMergedCustomUpdateWUGroup0ToDevice(1, d_reducedChangereduce_batch_weight_change_w_hr, d_dgw_hr, 500, 500, 500);
    pushMergedCustomUpdateWUGroup0ToDevice(2, d_reducedChangereduce_batch_weight_change_w_oh, d_dgw_oh, 500, 11, 11);
    pushMergedCustomUpdateWUGroup1ToDevice(0, d_mplast_step_reduced_w_ho, d_vplast_step_reduced_w_ho, d_timeplast_step_reduced_w_ho, d_reducedChangereduce_batch_weight_change_w_ho, d_gw_ho, 11, 500, 500);
    pushMergedCustomUpdateWUGroup1ToDevice(1, d_mplast_step_reduced_w_hr, d_vplast_step_reduced_w_hr, d_timeplast_step_reduced_w_hr, d_reducedChangereduce_batch_weight_change_w_hr, d_gw_hr, 500, 500, 500);
    pushMergedCustomUpdateWUGroup1ToDevice(2, d_mplast_step_reduced_w_oh, d_vplast_step_reduced_w_oh, d_timeplast_step_reduced_w_oh, d_reducedChangereduce_batch_weight_change_w_oh, d_gw_oh, 500, 11, 11);
}

void freeMem() {
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // timers
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // local neuron groups
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntp_h));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntp_h));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkp_h));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkp_h));
    CHECK_CUDA_ERRORS(cudaFreeHost(prevSTp_h));
    CHECK_CUDA_ERRORS(cudaFree(d_prevSTp_h));
    CHECK_CUDA_ERRORS(cudaFreeHost(rp_h));
    CHECK_CUDA_ERRORS(cudaFree(d_rp_h));
    CHECK_CUDA_ERRORS(cudaFreeHost(xp_h));
    CHECK_CUDA_ERRORS(cudaFree(d_xp_h));
    CHECK_CUDA_ERRORS(cudaFreeHost(drp_h));
    CHECK_CUDA_ERRORS(cudaFree(d_drp_h));
    CHECK_CUDA_ERRORS(cudaFreeHost(bp_h));
    CHECK_CUDA_ERRORS(cudaFree(d_bp_h));
    CHECK_CUDA_ERRORS(cudaFreeHost(dbp_h));
    CHECK_CUDA_ERRORS(cudaFree(d_dbp_h));
    CHECK_CUDA_ERRORS(cudaFreeHost(err_fbp_h));
    CHECK_CUDA_ERRORS(cudaFree(d_err_fbp_h));
    CHECK_CUDA_ERRORS(cudaFreeHost(r_eventp_h));
    CHECK_CUDA_ERRORS(cudaFree(d_r_eventp_h));
    CHECK_CUDA_ERRORS(cudaFreeHost(r_prev_eventp_h));
    CHECK_CUDA_ERRORS(cudaFree(d_r_prev_eventp_h));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntp_i));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntp_i));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkp_i));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkp_i));
    CHECK_CUDA_ERRORS(cudaFreeHost(prevSTp_i));
    CHECK_CUDA_ERRORS(cudaFree(d_prevSTp_i));
    CHECK_CUDA_ERRORS(cudaFreeHost(rp_i));
    CHECK_CUDA_ERRORS(cudaFree(d_rp_i));
    CHECK_CUDA_ERRORS(cudaFreeHost(r_eventp_i));
    CHECK_CUDA_ERRORS(cudaFree(d_r_eventp_i));
    CHECK_CUDA_ERRORS(cudaFreeHost(r_prev_eventp_i));
    CHECK_CUDA_ERRORS(cudaFree(d_r_prev_eventp_i));
    // current source variables
    CHECK_CUDA_ERRORS(cudaFreeHost(t_idxcs_in));
    CHECK_CUDA_ERRORS(cudaFree(d_t_idxcs_in));
    CHECK_CUDA_ERRORS(cudaFreeHost(periodiccs_in));
    CHECK_CUDA_ERRORS(cudaFree(d_periodiccs_in));
    CHECK_CUDA_ERRORS(cudaFreeHost(t_datacs_in));
    CHECK_CUDA_ERRORS(cudaFree(d_t_datacs_in));
    CHECK_CUDA_ERRORS(cudaFreeHost(dt_datacs_in));
    CHECK_CUDA_ERRORS(cudaFree(d_dt_datacs_in));
    CHECK_CUDA_ERRORS(cudaFreeHost(ntcs_in));
    CHECK_CUDA_ERRORS(cudaFree(d_ntcs_in));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntp_o));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntp_o));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkp_o));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkp_o));
    CHECK_CUDA_ERRORS(cudaFreeHost(prevSTp_o));
    CHECK_CUDA_ERRORS(cudaFree(d_prevSTp_o));
    CHECK_CUDA_ERRORS(cudaFreeHost(xp_o));
    CHECK_CUDA_ERRORS(cudaFree(d_xp_o));
    CHECK_CUDA_ERRORS(cudaFreeHost(rp_o));
    CHECK_CUDA_ERRORS(cudaFree(d_rp_o));
    CHECK_CUDA_ERRORS(cudaFreeHost(drp_o));
    CHECK_CUDA_ERRORS(cudaFree(d_drp_o));
    CHECK_CUDA_ERRORS(cudaFreeHost(targp_o));
    CHECK_CUDA_ERRORS(cudaFree(d_targp_o));
    CHECK_CUDA_ERRORS(cudaFreeHost(errp_o));
    CHECK_CUDA_ERRORS(cudaFree(d_errp_o));
    CHECK_CUDA_ERRORS(cudaFreeHost(err_eventp_o));
    CHECK_CUDA_ERRORS(cudaFree(d_err_eventp_o));
    CHECK_CUDA_ERRORS(cudaFreeHost(err_prev_eventp_o));
    CHECK_CUDA_ERRORS(cudaFree(d_err_prev_eventp_o));
    CHECK_CUDA_ERRORS(cudaFreeHost(bp_o));
    CHECK_CUDA_ERRORS(cudaFree(d_bp_o));
    CHECK_CUDA_ERRORS(cudaFreeHost(dbp_o));
    CHECK_CUDA_ERRORS(cudaFree(d_dbp_o));
    CHECK_CUDA_ERRORS(cudaFreeHost(lossp_o));
    CHECK_CUDA_ERRORS(cudaFree(d_lossp_o));
    // current source variables
    CHECK_CUDA_ERRORS(cudaFreeHost(t_idxcs_out));
    CHECK_CUDA_ERRORS(cudaFree(d_t_idxcs_out));
    CHECK_CUDA_ERRORS(cudaFreeHost(periodiccs_out));
    CHECK_CUDA_ERRORS(cudaFree(d_periodiccs_out));
    CHECK_CUDA_ERRORS(cudaFreeHost(t_datacs_out));
    CHECK_CUDA_ERRORS(cudaFree(d_t_datacs_out));
    CHECK_CUDA_ERRORS(cudaFreeHost(dt_datacs_out));
    CHECK_CUDA_ERRORS(cudaFree(d_dt_datacs_out));
    CHECK_CUDA_ERRORS(cudaFreeHost(ntcs_out));
    CHECK_CUDA_ERRORS(cudaFree(d_ntcs_out));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntp_r));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntp_r));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkp_r));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkp_r));
    CHECK_CUDA_ERRORS(cudaFreeHost(recordSpkp_r));
    CHECK_CUDA_ERRORS(cudaFree(d_recordSpkp_r));
    CHECK_CUDA_ERRORS(cudaFreeHost(prevSTp_r));
    CHECK_CUDA_ERRORS(cudaFree(d_prevSTp_r));
    CHECK_CUDA_ERRORS(cudaFreeHost(r_tracep_r));
    CHECK_CUDA_ERRORS(cudaFree(d_r_tracep_r));
    CHECK_CUDA_ERRORS(cudaFreeHost(rp_r));
    CHECK_CUDA_ERRORS(cudaFree(d_rp_r));
    CHECK_CUDA_ERRORS(cudaFreeHost(r_eventp_r));
    CHECK_CUDA_ERRORS(cudaFree(d_r_eventp_r));
    CHECK_CUDA_ERRORS(cudaFreeHost(r_prev_eventp_r));
    CHECK_CUDA_ERRORS(cudaFree(d_r_prev_eventp_r));
    
    // ------------------------------------------------------------------------
    // custom update variables
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaFreeHost(mplast_step_reduced_p_h));
    CHECK_CUDA_ERRORS(cudaFree(d_mplast_step_reduced_p_h));
    CHECK_CUDA_ERRORS(cudaFreeHost(vplast_step_reduced_p_h));
    CHECK_CUDA_ERRORS(cudaFree(d_vplast_step_reduced_p_h));
    CHECK_CUDA_ERRORS(cudaFreeHost(timeplast_step_reduced_p_h));
    CHECK_CUDA_ERRORS(cudaFree(d_timeplast_step_reduced_p_h));
    CHECK_CUDA_ERRORS(cudaFreeHost(mplast_step_reduced_p_o));
    CHECK_CUDA_ERRORS(cudaFree(d_mplast_step_reduced_p_o));
    CHECK_CUDA_ERRORS(cudaFreeHost(vplast_step_reduced_p_o));
    CHECK_CUDA_ERRORS(cudaFree(d_vplast_step_reduced_p_o));
    CHECK_CUDA_ERRORS(cudaFreeHost(timeplast_step_reduced_p_o));
    CHECK_CUDA_ERRORS(cudaFree(d_timeplast_step_reduced_p_o));
    CHECK_CUDA_ERRORS(cudaFreeHost(reducedChangereduce_batch_bias_change_p_h));
    CHECK_CUDA_ERRORS(cudaFree(d_reducedChangereduce_batch_bias_change_p_h));
    CHECK_CUDA_ERRORS(cudaFreeHost(reducedChangereduce_batch_bias_change_p_o));
    CHECK_CUDA_ERRORS(cudaFree(d_reducedChangereduce_batch_bias_change_p_o));
    CHECK_CUDA_ERRORS(cudaFreeHost(MaxValsoftmax_1));
    CHECK_CUDA_ERRORS(cudaFree(d_MaxValsoftmax_1));
    CHECK_CUDA_ERRORS(cudaFreeHost(SumExpValsoftmax_2));
    CHECK_CUDA_ERRORS(cudaFree(d_SumExpValsoftmax_2));
    CHECK_CUDA_ERRORS(cudaFreeHost(mplast_step_reduced_w_ho));
    CHECK_CUDA_ERRORS(cudaFree(d_mplast_step_reduced_w_ho));
    CHECK_CUDA_ERRORS(cudaFreeHost(vplast_step_reduced_w_ho));
    CHECK_CUDA_ERRORS(cudaFree(d_vplast_step_reduced_w_ho));
    CHECK_CUDA_ERRORS(cudaFreeHost(timeplast_step_reduced_w_ho));
    CHECK_CUDA_ERRORS(cudaFree(d_timeplast_step_reduced_w_ho));
    CHECK_CUDA_ERRORS(cudaFreeHost(mplast_step_reduced_w_hr));
    CHECK_CUDA_ERRORS(cudaFree(d_mplast_step_reduced_w_hr));
    CHECK_CUDA_ERRORS(cudaFreeHost(vplast_step_reduced_w_hr));
    CHECK_CUDA_ERRORS(cudaFree(d_vplast_step_reduced_w_hr));
    CHECK_CUDA_ERRORS(cudaFreeHost(timeplast_step_reduced_w_hr));
    CHECK_CUDA_ERRORS(cudaFree(d_timeplast_step_reduced_w_hr));
    CHECK_CUDA_ERRORS(cudaFreeHost(mplast_step_reduced_w_oh));
    CHECK_CUDA_ERRORS(cudaFree(d_mplast_step_reduced_w_oh));
    CHECK_CUDA_ERRORS(cudaFreeHost(vplast_step_reduced_w_oh));
    CHECK_CUDA_ERRORS(cudaFree(d_vplast_step_reduced_w_oh));
    CHECK_CUDA_ERRORS(cudaFreeHost(timeplast_step_reduced_w_oh));
    CHECK_CUDA_ERRORS(cudaFree(d_timeplast_step_reduced_w_oh));
    CHECK_CUDA_ERRORS(cudaFreeHost(reducedChangereduce_batch_weight_change_w_ho));
    CHECK_CUDA_ERRORS(cudaFree(d_reducedChangereduce_batch_weight_change_w_ho));
    CHECK_CUDA_ERRORS(cudaFreeHost(reducedChangereduce_batch_weight_change_w_hr));
    CHECK_CUDA_ERRORS(cudaFree(d_reducedChangereduce_batch_weight_change_w_hr));
    CHECK_CUDA_ERRORS(cudaFreeHost(reducedChangereduce_batch_weight_change_w_oh));
    CHECK_CUDA_ERRORS(cudaFree(d_reducedChangereduce_batch_weight_change_w_oh));
    
    // ------------------------------------------------------------------------
    // pre and postsynaptic variables
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynw_ho));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynw_ho));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynw_hr));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynw_hr));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynw_oh));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynw_oh));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynw_rr));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynw_rr));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynw_ri));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynw_ri));
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaFreeHost(gw_ho));
    CHECK_CUDA_ERRORS(cudaFree(d_gw_ho));
    CHECK_CUDA_ERRORS(cudaFreeHost(inp_prevw_ho));
    CHECK_CUDA_ERRORS(cudaFree(d_inp_prevw_ho));
    CHECK_CUDA_ERRORS(cudaFreeHost(dgw_ho));
    CHECK_CUDA_ERRORS(cudaFree(d_dgw_ho));
    CHECK_CUDA_ERRORS(cudaFreeHost(dg_prevw_ho));
    CHECK_CUDA_ERRORS(cudaFree(d_dg_prevw_ho));
    CHECK_CUDA_ERRORS(cudaFreeHost(t_prevw_ho));
    CHECK_CUDA_ERRORS(cudaFree(d_t_prevw_ho));
    CHECK_CUDA_ERRORS(cudaFreeHost(gw_hr));
    CHECK_CUDA_ERRORS(cudaFree(d_gw_hr));
    CHECK_CUDA_ERRORS(cudaFreeHost(inp_prevw_hr));
    CHECK_CUDA_ERRORS(cudaFree(d_inp_prevw_hr));
    CHECK_CUDA_ERRORS(cudaFreeHost(dgw_hr));
    CHECK_CUDA_ERRORS(cudaFree(d_dgw_hr));
    CHECK_CUDA_ERRORS(cudaFreeHost(dg_prevw_hr));
    CHECK_CUDA_ERRORS(cudaFree(d_dg_prevw_hr));
    CHECK_CUDA_ERRORS(cudaFreeHost(t_prevw_hr));
    CHECK_CUDA_ERRORS(cudaFree(d_t_prevw_hr));
    CHECK_CUDA_ERRORS(cudaFreeHost(gw_oh));
    CHECK_CUDA_ERRORS(cudaFree(d_gw_oh));
    CHECK_CUDA_ERRORS(cudaFreeHost(inp_prevw_oh));
    CHECK_CUDA_ERRORS(cudaFree(d_inp_prevw_oh));
    CHECK_CUDA_ERRORS(cudaFreeHost(dgw_oh));
    CHECK_CUDA_ERRORS(cudaFree(d_dgw_oh));
    CHECK_CUDA_ERRORS(cudaFreeHost(dg_prevw_oh));
    CHECK_CUDA_ERRORS(cudaFree(d_dg_prevw_oh));
    CHECK_CUDA_ERRORS(cudaFreeHost(t_prevw_oh));
    CHECK_CUDA_ERRORS(cudaFree(d_t_prevw_oh));
    CHECK_CUDA_ERRORS(cudaFreeHost(gw_ri));
    CHECK_CUDA_ERRORS(cudaFree(d_gw_ri));
    CHECK_CUDA_ERRORS(cudaFreeHost(inp_prevw_ri));
    CHECK_CUDA_ERRORS(cudaFree(d_inp_prevw_ri));
    CHECK_CUDA_ERRORS(cudaFreeHost(gw_rr));
    CHECK_CUDA_ERRORS(cudaFree(d_gw_rr));
    CHECK_CUDA_ERRORS(cudaFreeHost(inp_prevw_rr));
    CHECK_CUDA_ERRORS(cudaFree(d_inp_prevw_rr));
    
}

size_t getFreeDeviceMemBytes() {
    size_t free;
    size_t total;
    CHECK_CUDA_ERRORS(cudaMemGetInfo(&free, &total));
    return free;
}

void stepTime() {
    updateSynapses(t);
    updateNeurons(t, (unsigned int)(iT % numRecordingTimesteps)); 
    iT++;
    t = iT*DT;
}

