#pragma once
#define EXPORT_VAR extern
#define EXPORT_FUNC
// Standard C++ includes
#include <random>
#include <string>
#include <stdexcept>

// Standard C includes
#include <cassert>
#include <cstdint>
#define DT 2.50000000000000000e-01f
typedef float scalar;
#define SCALAR_MIN 1.175494351e-38f
#define SCALAR_MAX 3.402823466e+38f

#define TIME_MIN 1.175494351e-38f
#define TIME_MAX 3.402823466e+38f

// ------------------------------------------------------------------------
// bit tool macros
#define B(x,i) ((x) & (0x80000000 >> (i))) //!< Extract the bit at the specified position i from x
#define setB(x,i) x= ((x) | (0x80000000 >> (i))) //!< Set the bit at the specified position i in x to 1
#define delB(x,i) x= ((x) & (~(0x80000000 >> (i)))) //!< Set the bit at the specified position i in x to 0

extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
EXPORT_VAR unsigned long long iT;
EXPORT_VAR float t;

// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
EXPORT_VAR double initTime;
EXPORT_VAR double initSparseTime;
EXPORT_VAR double neuronUpdateTime;
EXPORT_VAR double presynapticUpdateTime;
EXPORT_VAR double postsynapticUpdateTime;
EXPORT_VAR double synapseDynamicsTime;
EXPORT_VAR double customUpdateBiasChangeBatchReduceTime;
EXPORT_VAR double customUpdateBiasChangeBatchReduceTransposeTime;
EXPORT_VAR double customUpdatePlastTime;
EXPORT_VAR double customUpdatePlastTransposeTime;
EXPORT_VAR double customUpdateWeightChangeBatchReduceTime;
EXPORT_VAR double customUpdateWeightChangeBatchReduceTransposeTime;
EXPORT_VAR double customUpdatesoftmax1Time;
EXPORT_VAR double customUpdatesoftmax1TransposeTime;
EXPORT_VAR double customUpdatesoftmax2Time;
EXPORT_VAR double customUpdatesoftmax2TransposeTime;
EXPORT_VAR double customUpdatesoftmax3Time;
EXPORT_VAR double customUpdatesoftmax3TransposeTime;
// ------------------------------------------------------------------------
// local neuron groups
// ------------------------------------------------------------------------
#define spikeCount_p_h glbSpkCntp_h[0]
#define spike_p_h glbSpkp_h
#define glbSpkShiftp_h 0

EXPORT_VAR unsigned int* glbSpkCntp_h;
EXPORT_VAR unsigned int* d_glbSpkCntp_h;
EXPORT_VAR unsigned int* glbSpkp_h;
EXPORT_VAR unsigned int* d_glbSpkp_h;
EXPORT_VAR float* prevSTp_h;
EXPORT_VAR float* d_prevSTp_h;
EXPORT_VAR scalar* rp_h;
EXPORT_VAR scalar* d_rp_h;
EXPORT_VAR scalar* xp_h;
EXPORT_VAR scalar* d_xp_h;
EXPORT_VAR scalar* drp_h;
EXPORT_VAR scalar* d_drp_h;
EXPORT_VAR scalar* bp_h;
EXPORT_VAR scalar* d_bp_h;
EXPORT_VAR scalar* dbp_h;
EXPORT_VAR scalar* d_dbp_h;
EXPORT_VAR scalar* err_fbp_h;
EXPORT_VAR scalar* d_err_fbp_h;
EXPORT_VAR scalar* r_eventp_h;
EXPORT_VAR scalar* d_r_eventp_h;
EXPORT_VAR scalar* r_prev_eventp_h;
EXPORT_VAR scalar* d_r_prev_eventp_h;
#define spikeCount_p_i glbSpkCntp_i[0]
#define spike_p_i glbSpkp_i
#define glbSpkShiftp_i 0

EXPORT_VAR unsigned int* glbSpkCntp_i;
EXPORT_VAR unsigned int* d_glbSpkCntp_i;
EXPORT_VAR unsigned int* glbSpkp_i;
EXPORT_VAR unsigned int* d_glbSpkp_i;
EXPORT_VAR float* prevSTp_i;
EXPORT_VAR float* d_prevSTp_i;
EXPORT_VAR scalar* rp_i;
EXPORT_VAR scalar* d_rp_i;
EXPORT_VAR scalar* r_eventp_i;
EXPORT_VAR scalar* d_r_eventp_i;
EXPORT_VAR scalar* r_prev_eventp_i;
EXPORT_VAR scalar* d_r_prev_eventp_i;
// current source variables
EXPORT_VAR int* t_idxcs_in;
EXPORT_VAR int* d_t_idxcs_in;
EXPORT_VAR int* periodiccs_in;
EXPORT_VAR int* d_periodiccs_in;
EXPORT_VAR scalar* t_datacs_in;
EXPORT_VAR scalar* d_t_datacs_in;
EXPORT_VAR scalar* dt_datacs_in;
EXPORT_VAR scalar* d_dt_datacs_in;
EXPORT_VAR int* ntcs_in;
EXPORT_VAR int* d_ntcs_in;
EXPORT_VAR scalar* datacs_in;
EXPORT_VAR scalar* d_datacs_in;
#define spikeCount_p_o glbSpkCntp_o[0]
#define spike_p_o glbSpkp_o
#define glbSpkShiftp_o 0

EXPORT_VAR unsigned int* glbSpkCntp_o;
EXPORT_VAR unsigned int* d_glbSpkCntp_o;
EXPORT_VAR unsigned int* glbSpkp_o;
EXPORT_VAR unsigned int* d_glbSpkp_o;
EXPORT_VAR float* prevSTp_o;
EXPORT_VAR float* d_prevSTp_o;
EXPORT_VAR scalar* xp_o;
EXPORT_VAR scalar* d_xp_o;
EXPORT_VAR scalar* rp_o;
EXPORT_VAR scalar* d_rp_o;
EXPORT_VAR scalar* drp_o;
EXPORT_VAR scalar* d_drp_o;
EXPORT_VAR scalar* targp_o;
EXPORT_VAR scalar* d_targp_o;
EXPORT_VAR scalar* errp_o;
EXPORT_VAR scalar* d_errp_o;
EXPORT_VAR scalar* err_eventp_o;
EXPORT_VAR scalar* d_err_eventp_o;
EXPORT_VAR scalar* err_prev_eventp_o;
EXPORT_VAR scalar* d_err_prev_eventp_o;
EXPORT_VAR scalar* bp_o;
EXPORT_VAR scalar* d_bp_o;
EXPORT_VAR scalar* dbp_o;
EXPORT_VAR scalar* d_dbp_o;
EXPORT_VAR scalar* lossp_o;
EXPORT_VAR scalar* d_lossp_o;
// current source variables
EXPORT_VAR int* t_idxcs_out;
EXPORT_VAR int* d_t_idxcs_out;
EXPORT_VAR int* periodiccs_out;
EXPORT_VAR int* d_periodiccs_out;
EXPORT_VAR scalar* t_datacs_out;
EXPORT_VAR scalar* d_t_datacs_out;
EXPORT_VAR scalar* dt_datacs_out;
EXPORT_VAR scalar* d_dt_datacs_out;
EXPORT_VAR int* ntcs_out;
EXPORT_VAR int* d_ntcs_out;
EXPORT_VAR scalar* datacs_out;
EXPORT_VAR scalar* d_datacs_out;
#define spikeCount_p_r glbSpkCntp_r[0]
#define spike_p_r glbSpkp_r
#define glbSpkShiftp_r 0

EXPORT_VAR unsigned int* glbSpkCntp_r;
EXPORT_VAR unsigned int* d_glbSpkCntp_r;
EXPORT_VAR unsigned int* glbSpkp_r;
EXPORT_VAR unsigned int* d_glbSpkp_r;
EXPORT_VAR uint32_t* recordSpkp_r;
EXPORT_VAR uint32_t* d_recordSpkp_r;
EXPORT_VAR float* prevSTp_r;
EXPORT_VAR float* d_prevSTp_r;
EXPORT_VAR scalar* r_tracep_r;
EXPORT_VAR scalar* d_r_tracep_r;
EXPORT_VAR scalar* rp_r;
EXPORT_VAR scalar* d_rp_r;
EXPORT_VAR scalar* r_eventp_r;
EXPORT_VAR scalar* d_r_eventp_r;
EXPORT_VAR scalar* r_prev_eventp_r;
EXPORT_VAR scalar* d_r_prev_eventp_r;

// ------------------------------------------------------------------------
// custom update variables
// ------------------------------------------------------------------------
EXPORT_VAR scalar* mplast_step_reduced_p_h;
EXPORT_VAR scalar* d_mplast_step_reduced_p_h;
EXPORT_VAR scalar* vplast_step_reduced_p_h;
EXPORT_VAR scalar* d_vplast_step_reduced_p_h;
EXPORT_VAR scalar* timeplast_step_reduced_p_h;
EXPORT_VAR scalar* d_timeplast_step_reduced_p_h;
EXPORT_VAR scalar* mplast_step_reduced_p_o;
EXPORT_VAR scalar* d_mplast_step_reduced_p_o;
EXPORT_VAR scalar* vplast_step_reduced_p_o;
EXPORT_VAR scalar* d_vplast_step_reduced_p_o;
EXPORT_VAR scalar* timeplast_step_reduced_p_o;
EXPORT_VAR scalar* d_timeplast_step_reduced_p_o;
EXPORT_VAR scalar* reducedChangereduce_batch_bias_change_p_h;
EXPORT_VAR scalar* d_reducedChangereduce_batch_bias_change_p_h;
EXPORT_VAR scalar* reducedChangereduce_batch_bias_change_p_o;
EXPORT_VAR scalar* d_reducedChangereduce_batch_bias_change_p_o;
EXPORT_VAR scalar* MaxValsoftmax_1;
EXPORT_VAR scalar* d_MaxValsoftmax_1;
EXPORT_VAR scalar* SumExpValsoftmax_2;
EXPORT_VAR scalar* d_SumExpValsoftmax_2;
EXPORT_VAR scalar* mplast_step_reduced_w_ho;
EXPORT_VAR scalar* d_mplast_step_reduced_w_ho;
EXPORT_VAR scalar* vplast_step_reduced_w_ho;
EXPORT_VAR scalar* d_vplast_step_reduced_w_ho;
EXPORT_VAR scalar* timeplast_step_reduced_w_ho;
EXPORT_VAR scalar* d_timeplast_step_reduced_w_ho;
EXPORT_VAR scalar* mplast_step_reduced_w_hr;
EXPORT_VAR scalar* d_mplast_step_reduced_w_hr;
EXPORT_VAR scalar* vplast_step_reduced_w_hr;
EXPORT_VAR scalar* d_vplast_step_reduced_w_hr;
EXPORT_VAR scalar* timeplast_step_reduced_w_hr;
EXPORT_VAR scalar* d_timeplast_step_reduced_w_hr;
EXPORT_VAR scalar* mplast_step_reduced_w_oh;
EXPORT_VAR scalar* d_mplast_step_reduced_w_oh;
EXPORT_VAR scalar* vplast_step_reduced_w_oh;
EXPORT_VAR scalar* d_vplast_step_reduced_w_oh;
EXPORT_VAR scalar* timeplast_step_reduced_w_oh;
EXPORT_VAR scalar* d_timeplast_step_reduced_w_oh;
EXPORT_VAR scalar* reducedChangereduce_batch_weight_change_w_ho;
EXPORT_VAR scalar* d_reducedChangereduce_batch_weight_change_w_ho;
EXPORT_VAR scalar* reducedChangereduce_batch_weight_change_w_hr;
EXPORT_VAR scalar* d_reducedChangereduce_batch_weight_change_w_hr;
EXPORT_VAR scalar* reducedChangereduce_batch_weight_change_w_oh;
EXPORT_VAR scalar* d_reducedChangereduce_batch_weight_change_w_oh;

// ------------------------------------------------------------------------
// pre and postsynaptic variables
// ------------------------------------------------------------------------
EXPORT_VAR float* inSynw_ho;
EXPORT_VAR float* d_inSynw_ho;
EXPORT_VAR float* inSynw_hr;
EXPORT_VAR float* d_inSynw_hr;
EXPORT_VAR float* inSynw_oh;
EXPORT_VAR float* d_inSynw_oh;
EXPORT_VAR float* inSynw_rr;
EXPORT_VAR float* d_inSynw_rr;
EXPORT_VAR float* inSynw_ri;
EXPORT_VAR float* d_inSynw_ri;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------
EXPORT_VAR scalar* gw_ho;
EXPORT_VAR scalar* d_gw_ho;
EXPORT_VAR scalar* inp_prevw_ho;
EXPORT_VAR scalar* d_inp_prevw_ho;
EXPORT_VAR scalar* dgw_ho;
EXPORT_VAR scalar* d_dgw_ho;
EXPORT_VAR scalar* dg_prevw_ho;
EXPORT_VAR scalar* d_dg_prevw_ho;
EXPORT_VAR scalar* t_prevw_ho;
EXPORT_VAR scalar* d_t_prevw_ho;
EXPORT_VAR scalar* gw_hr;
EXPORT_VAR scalar* d_gw_hr;
EXPORT_VAR scalar* inp_prevw_hr;
EXPORT_VAR scalar* d_inp_prevw_hr;
EXPORT_VAR scalar* dgw_hr;
EXPORT_VAR scalar* d_dgw_hr;
EXPORT_VAR scalar* dg_prevw_hr;
EXPORT_VAR scalar* d_dg_prevw_hr;
EXPORT_VAR scalar* t_prevw_hr;
EXPORT_VAR scalar* d_t_prevw_hr;
EXPORT_VAR scalar* gw_oh;
EXPORT_VAR scalar* d_gw_oh;
EXPORT_VAR scalar* inp_prevw_oh;
EXPORT_VAR scalar* d_inp_prevw_oh;
EXPORT_VAR scalar* dgw_oh;
EXPORT_VAR scalar* d_dgw_oh;
EXPORT_VAR scalar* dg_prevw_oh;
EXPORT_VAR scalar* d_dg_prevw_oh;
EXPORT_VAR scalar* t_prevw_oh;
EXPORT_VAR scalar* d_t_prevw_oh;
EXPORT_VAR scalar* gw_ri;
EXPORT_VAR scalar* d_gw_ri;
EXPORT_VAR scalar* inp_prevw_ri;
EXPORT_VAR scalar* d_inp_prevw_ri;
EXPORT_VAR scalar* gw_rr;
EXPORT_VAR scalar* d_gw_rr;
EXPORT_VAR scalar* inp_prevw_rr;
EXPORT_VAR scalar* d_inp_prevw_rr;

EXPORT_FUNC void pushp_hSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullp_hSpikesFromDevice();
EXPORT_FUNC void pushp_hCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullp_hCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getp_hCurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getp_hCurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushp_hPreviousSpikeTimesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullp_hPreviousSpikeTimesFromDevice();
EXPORT_FUNC void pushrp_hToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullrp_hFromDevice();
EXPORT_FUNC void pushCurrentrp_hToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentrp_hFromDevice();
EXPORT_FUNC scalar* getCurrentrp_h(unsigned int batch = 0); 
EXPORT_FUNC void pushxp_hToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullxp_hFromDevice();
EXPORT_FUNC void pushCurrentxp_hToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentxp_hFromDevice();
EXPORT_FUNC scalar* getCurrentxp_h(unsigned int batch = 0); 
EXPORT_FUNC void pushdrp_hToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulldrp_hFromDevice();
EXPORT_FUNC void pushCurrentdrp_hToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentdrp_hFromDevice();
EXPORT_FUNC scalar* getCurrentdrp_h(unsigned int batch = 0); 
EXPORT_FUNC void pushbp_hToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullbp_hFromDevice();
EXPORT_FUNC void pushCurrentbp_hToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentbp_hFromDevice();
EXPORT_FUNC scalar* getCurrentbp_h(unsigned int batch = 0); 
EXPORT_FUNC void pushdbp_hToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulldbp_hFromDevice();
EXPORT_FUNC void pushCurrentdbp_hToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentdbp_hFromDevice();
EXPORT_FUNC scalar* getCurrentdbp_h(unsigned int batch = 0); 
EXPORT_FUNC void pusherr_fbp_hToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullerr_fbp_hFromDevice();
EXPORT_FUNC void pushCurrenterr_fbp_hToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrenterr_fbp_hFromDevice();
EXPORT_FUNC scalar* getCurrenterr_fbp_h(unsigned int batch = 0); 
EXPORT_FUNC void pushr_eventp_hToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullr_eventp_hFromDevice();
EXPORT_FUNC void pushCurrentr_eventp_hToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentr_eventp_hFromDevice();
EXPORT_FUNC scalar* getCurrentr_eventp_h(unsigned int batch = 0); 
EXPORT_FUNC void pushr_prev_eventp_hToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullr_prev_eventp_hFromDevice();
EXPORT_FUNC void pushCurrentr_prev_eventp_hToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentr_prev_eventp_hFromDevice();
EXPORT_FUNC scalar* getCurrentr_prev_eventp_h(unsigned int batch = 0); 
EXPORT_FUNC void pushp_hStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullp_hStateFromDevice();
EXPORT_FUNC void pushp_iSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullp_iSpikesFromDevice();
EXPORT_FUNC void pushp_iCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullp_iCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getp_iCurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getp_iCurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushp_iPreviousSpikeTimesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullp_iPreviousSpikeTimesFromDevice();
EXPORT_FUNC void pushrp_iToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullrp_iFromDevice();
EXPORT_FUNC void pushCurrentrp_iToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentrp_iFromDevice();
EXPORT_FUNC scalar* getCurrentrp_i(unsigned int batch = 0); 
EXPORT_FUNC void pushr_eventp_iToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullr_eventp_iFromDevice();
EXPORT_FUNC void pushCurrentr_eventp_iToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentr_eventp_iFromDevice();
EXPORT_FUNC scalar* getCurrentr_eventp_i(unsigned int batch = 0); 
EXPORT_FUNC void pushr_prev_eventp_iToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullr_prev_eventp_iFromDevice();
EXPORT_FUNC void pushCurrentr_prev_eventp_iToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentr_prev_eventp_iFromDevice();
EXPORT_FUNC scalar* getCurrentr_prev_eventp_i(unsigned int batch = 0); 
EXPORT_FUNC void pushp_iStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullp_iStateFromDevice();
EXPORT_FUNC void pusht_idxcs_inToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullt_idxcs_inFromDevice();
EXPORT_FUNC void pushperiodiccs_inToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullperiodiccs_inFromDevice();
EXPORT_FUNC void pusht_datacs_inToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullt_datacs_inFromDevice();
EXPORT_FUNC void pushdt_datacs_inToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulldt_datacs_inFromDevice();
EXPORT_FUNC void pushntcs_inToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullntcs_inFromDevice();
EXPORT_FUNC void pushcs_inStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullcs_inStateFromDevice();
EXPORT_FUNC void allocatedatacs_in(unsigned int count);
EXPORT_FUNC void freedatacs_in();
EXPORT_FUNC void pushdatacs_inToDevice(unsigned int count);
EXPORT_FUNC void pulldatacs_inFromDevice(unsigned int count);
EXPORT_FUNC void pushp_oSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullp_oSpikesFromDevice();
EXPORT_FUNC void pushp_oCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullp_oCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getp_oCurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getp_oCurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushp_oPreviousSpikeTimesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullp_oPreviousSpikeTimesFromDevice();
EXPORT_FUNC void pushxp_oToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullxp_oFromDevice();
EXPORT_FUNC void pushCurrentxp_oToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentxp_oFromDevice();
EXPORT_FUNC scalar* getCurrentxp_o(unsigned int batch = 0); 
EXPORT_FUNC void pushrp_oToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullrp_oFromDevice();
EXPORT_FUNC void pushCurrentrp_oToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentrp_oFromDevice();
EXPORT_FUNC scalar* getCurrentrp_o(unsigned int batch = 0); 
EXPORT_FUNC void pushdrp_oToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulldrp_oFromDevice();
EXPORT_FUNC void pushCurrentdrp_oToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentdrp_oFromDevice();
EXPORT_FUNC scalar* getCurrentdrp_o(unsigned int batch = 0); 
EXPORT_FUNC void pushtargp_oToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulltargp_oFromDevice();
EXPORT_FUNC void pushCurrenttargp_oToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrenttargp_oFromDevice();
EXPORT_FUNC scalar* getCurrenttargp_o(unsigned int batch = 0); 
EXPORT_FUNC void pusherrp_oToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullerrp_oFromDevice();
EXPORT_FUNC void pushCurrenterrp_oToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrenterrp_oFromDevice();
EXPORT_FUNC scalar* getCurrenterrp_o(unsigned int batch = 0); 
EXPORT_FUNC void pusherr_eventp_oToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullerr_eventp_oFromDevice();
EXPORT_FUNC void pushCurrenterr_eventp_oToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrenterr_eventp_oFromDevice();
EXPORT_FUNC scalar* getCurrenterr_eventp_o(unsigned int batch = 0); 
EXPORT_FUNC void pusherr_prev_eventp_oToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullerr_prev_eventp_oFromDevice();
EXPORT_FUNC void pushCurrenterr_prev_eventp_oToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrenterr_prev_eventp_oFromDevice();
EXPORT_FUNC scalar* getCurrenterr_prev_eventp_o(unsigned int batch = 0); 
EXPORT_FUNC void pushbp_oToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullbp_oFromDevice();
EXPORT_FUNC void pushCurrentbp_oToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentbp_oFromDevice();
EXPORT_FUNC scalar* getCurrentbp_o(unsigned int batch = 0); 
EXPORT_FUNC void pushdbp_oToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulldbp_oFromDevice();
EXPORT_FUNC void pushCurrentdbp_oToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentdbp_oFromDevice();
EXPORT_FUNC scalar* getCurrentdbp_o(unsigned int batch = 0); 
EXPORT_FUNC void pushlossp_oToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulllossp_oFromDevice();
EXPORT_FUNC void pushCurrentlossp_oToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentlossp_oFromDevice();
EXPORT_FUNC scalar* getCurrentlossp_o(unsigned int batch = 0); 
EXPORT_FUNC void pushp_oStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullp_oStateFromDevice();
EXPORT_FUNC void pusht_idxcs_outToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullt_idxcs_outFromDevice();
EXPORT_FUNC void pushperiodiccs_outToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullperiodiccs_outFromDevice();
EXPORT_FUNC void pusht_datacs_outToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullt_datacs_outFromDevice();
EXPORT_FUNC void pushdt_datacs_outToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulldt_datacs_outFromDevice();
EXPORT_FUNC void pushntcs_outToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullntcs_outFromDevice();
EXPORT_FUNC void pushcs_outStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullcs_outStateFromDevice();
EXPORT_FUNC void allocatedatacs_out(unsigned int count);
EXPORT_FUNC void freedatacs_out();
EXPORT_FUNC void pushdatacs_outToDevice(unsigned int count);
EXPORT_FUNC void pulldatacs_outFromDevice(unsigned int count);
EXPORT_FUNC void pushp_rSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullp_rSpikesFromDevice();
EXPORT_FUNC void pushp_rCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullp_rCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getp_rCurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getp_rCurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushp_rPreviousSpikeTimesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullp_rPreviousSpikeTimesFromDevice();
EXPORT_FUNC void pushr_tracep_rToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullr_tracep_rFromDevice();
EXPORT_FUNC void pushCurrentr_tracep_rToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentr_tracep_rFromDevice();
EXPORT_FUNC scalar* getCurrentr_tracep_r(unsigned int batch = 0); 
EXPORT_FUNC void pushrp_rToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullrp_rFromDevice();
EXPORT_FUNC void pushCurrentrp_rToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentrp_rFromDevice();
EXPORT_FUNC scalar* getCurrentrp_r(unsigned int batch = 0); 
EXPORT_FUNC void pushr_eventp_rToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullr_eventp_rFromDevice();
EXPORT_FUNC void pushCurrentr_eventp_rToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentr_eventp_rFromDevice();
EXPORT_FUNC scalar* getCurrentr_eventp_r(unsigned int batch = 0); 
EXPORT_FUNC void pushr_prev_eventp_rToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullr_prev_eventp_rFromDevice();
EXPORT_FUNC void pushCurrentr_prev_eventp_rToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentr_prev_eventp_rFromDevice();
EXPORT_FUNC scalar* getCurrentr_prev_eventp_r(unsigned int batch = 0); 
EXPORT_FUNC void pushp_rStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullp_rStateFromDevice();
EXPORT_FUNC void pushmplast_step_reduced_p_hToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullmplast_step_reduced_p_hFromDevice();
EXPORT_FUNC void pushvplast_step_reduced_p_hToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullvplast_step_reduced_p_hFromDevice();
EXPORT_FUNC void pushtimeplast_step_reduced_p_hToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulltimeplast_step_reduced_p_hFromDevice();
EXPORT_FUNC void pushplast_step_reduced_p_hStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullplast_step_reduced_p_hStateFromDevice();
EXPORT_FUNC void pushmplast_step_reduced_p_oToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullmplast_step_reduced_p_oFromDevice();
EXPORT_FUNC void pushvplast_step_reduced_p_oToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullvplast_step_reduced_p_oFromDevice();
EXPORT_FUNC void pushtimeplast_step_reduced_p_oToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulltimeplast_step_reduced_p_oFromDevice();
EXPORT_FUNC void pushplast_step_reduced_p_oStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullplast_step_reduced_p_oStateFromDevice();
EXPORT_FUNC void pushreducedChangereduce_batch_bias_change_p_hToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullreducedChangereduce_batch_bias_change_p_hFromDevice();
EXPORT_FUNC void pushreduce_batch_bias_change_p_hStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullreduce_batch_bias_change_p_hStateFromDevice();
EXPORT_FUNC void pushreducedChangereduce_batch_bias_change_p_oToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullreducedChangereduce_batch_bias_change_p_oFromDevice();
EXPORT_FUNC void pushreduce_batch_bias_change_p_oStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullreduce_batch_bias_change_p_oStateFromDevice();
EXPORT_FUNC void pushMaxValsoftmax_1ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullMaxValsoftmax_1FromDevice();
EXPORT_FUNC void pushsoftmax_1StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullsoftmax_1StateFromDevice();
EXPORT_FUNC void pushSumExpValsoftmax_2ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullSumExpValsoftmax_2FromDevice();
EXPORT_FUNC void pushsoftmax_2StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullsoftmax_2StateFromDevice();
EXPORT_FUNC void pushsoftmax_3StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullsoftmax_3StateFromDevice();
EXPORT_FUNC void pushmplast_step_reduced_w_hoToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullmplast_step_reduced_w_hoFromDevice();
EXPORT_FUNC void pushvplast_step_reduced_w_hoToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullvplast_step_reduced_w_hoFromDevice();
EXPORT_FUNC void pushtimeplast_step_reduced_w_hoToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulltimeplast_step_reduced_w_hoFromDevice();
EXPORT_FUNC void pushplast_step_reduced_w_hoStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullplast_step_reduced_w_hoStateFromDevice();
EXPORT_FUNC void pushmplast_step_reduced_w_hrToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullmplast_step_reduced_w_hrFromDevice();
EXPORT_FUNC void pushvplast_step_reduced_w_hrToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullvplast_step_reduced_w_hrFromDevice();
EXPORT_FUNC void pushtimeplast_step_reduced_w_hrToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulltimeplast_step_reduced_w_hrFromDevice();
EXPORT_FUNC void pushplast_step_reduced_w_hrStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullplast_step_reduced_w_hrStateFromDevice();
EXPORT_FUNC void pushmplast_step_reduced_w_ohToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullmplast_step_reduced_w_ohFromDevice();
EXPORT_FUNC void pushvplast_step_reduced_w_ohToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullvplast_step_reduced_w_ohFromDevice();
EXPORT_FUNC void pushtimeplast_step_reduced_w_ohToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulltimeplast_step_reduced_w_ohFromDevice();
EXPORT_FUNC void pushplast_step_reduced_w_ohStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullplast_step_reduced_w_ohStateFromDevice();
EXPORT_FUNC void pushreducedChangereduce_batch_weight_change_w_hoToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullreducedChangereduce_batch_weight_change_w_hoFromDevice();
EXPORT_FUNC void pushreduce_batch_weight_change_w_hoStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullreduce_batch_weight_change_w_hoStateFromDevice();
EXPORT_FUNC void pushreducedChangereduce_batch_weight_change_w_hrToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullreducedChangereduce_batch_weight_change_w_hrFromDevice();
EXPORT_FUNC void pushreduce_batch_weight_change_w_hrStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullreduce_batch_weight_change_w_hrStateFromDevice();
EXPORT_FUNC void pushreducedChangereduce_batch_weight_change_w_ohToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullreducedChangereduce_batch_weight_change_w_ohFromDevice();
EXPORT_FUNC void pushreduce_batch_weight_change_w_ohStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullreduce_batch_weight_change_w_ohStateFromDevice();
EXPORT_FUNC void pushgw_hoToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgw_hoFromDevice();
EXPORT_FUNC void pushinp_prevw_hoToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinp_prevw_hoFromDevice();
EXPORT_FUNC void pushdgw_hoToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulldgw_hoFromDevice();
EXPORT_FUNC void pushdg_prevw_hoToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulldg_prevw_hoFromDevice();
EXPORT_FUNC void pusht_prevw_hoToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullt_prevw_hoFromDevice();
EXPORT_FUNC void pushinSynw_hoToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynw_hoFromDevice();
EXPORT_FUNC void pushw_hoStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullw_hoStateFromDevice();
EXPORT_FUNC void pushgw_hrToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgw_hrFromDevice();
EXPORT_FUNC void pushinp_prevw_hrToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinp_prevw_hrFromDevice();
EXPORT_FUNC void pushdgw_hrToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulldgw_hrFromDevice();
EXPORT_FUNC void pushdg_prevw_hrToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulldg_prevw_hrFromDevice();
EXPORT_FUNC void pusht_prevw_hrToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullt_prevw_hrFromDevice();
EXPORT_FUNC void pushinSynw_hrToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynw_hrFromDevice();
EXPORT_FUNC void pushw_hrStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullw_hrStateFromDevice();
EXPORT_FUNC void pushgw_ohToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgw_ohFromDevice();
EXPORT_FUNC void pushinp_prevw_ohToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinp_prevw_ohFromDevice();
EXPORT_FUNC void pushdgw_ohToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulldgw_ohFromDevice();
EXPORT_FUNC void pushdg_prevw_ohToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulldg_prevw_ohFromDevice();
EXPORT_FUNC void pusht_prevw_ohToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullt_prevw_ohFromDevice();
EXPORT_FUNC void pushinSynw_ohToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynw_ohFromDevice();
EXPORT_FUNC void pushw_ohStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullw_ohStateFromDevice();
EXPORT_FUNC void pushgw_riToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgw_riFromDevice();
EXPORT_FUNC void pushinp_prevw_riToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinp_prevw_riFromDevice();
EXPORT_FUNC void pushinSynw_riToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynw_riFromDevice();
EXPORT_FUNC void pushw_riStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullw_riStateFromDevice();
EXPORT_FUNC void pushgw_rrToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgw_rrFromDevice();
EXPORT_FUNC void pushinp_prevw_rrToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinp_prevw_rrFromDevice();
EXPORT_FUNC void pushinSynw_rrToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynw_rrFromDevice();
EXPORT_FUNC void pushw_rrStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullw_rrStateFromDevice();
// Runner functions
EXPORT_FUNC void copyStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void copyConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void copyStateFromDevice();
EXPORT_FUNC void copyCurrentSpikesFromDevice();
EXPORT_FUNC void copyCurrentSpikeEventsFromDevice();
EXPORT_FUNC void allocateRecordingBuffers(unsigned int timesteps);
EXPORT_FUNC void pullRecordingBuffersFromDevice();
EXPORT_FUNC void allocateMem();
EXPORT_FUNC void freeMem();
EXPORT_FUNC size_t getFreeDeviceMemBytes();
EXPORT_FUNC void stepTime();

// Functions generated by backend
EXPORT_FUNC void updateNeurons(float t, unsigned int recordingTimestep); 
EXPORT_FUNC void updateSynapses(float t);
EXPORT_FUNC void initialize();
EXPORT_FUNC void initializeSparse();
EXPORT_FUNC void updateBiasChangeBatchReduce();
EXPORT_FUNC void updatePlast();
EXPORT_FUNC void updateWeightChangeBatchReduce();
EXPORT_FUNC void updatesoftmax1();
EXPORT_FUNC void updatesoftmax2();
EXPORT_FUNC void updatesoftmax3();
}  // extern "C"
