#pragma once
#include "definitions.h"

// CUDA includes
#include <curand_kernel.h>
#include <cuda_fp16.h>

// ------------------------------------------------------------------------
// Helper macro for error-checking CUDA calls
#define CHECK_CUDA_ERRORS(call) {\
    cudaError_t error = call;\
    if (error != cudaSuccess) {\
        throw std::runtime_error(__FILE__": " + std::to_string(__LINE__) + ": cuda error " + std::to_string(error) + ": " + cudaGetErrorString(error));\
    }\
}

#define SUPPORT_CODE_FUNC __device__ __host__ inline


template<typename RNG>
__device__ inline float exponentialDistFloat(RNG *rng) {
    while (true) {
        const float u = curand_uniform(rng);
        if (u != 0.0f) {
            return -logf(u);
        }
    }
}

template<typename RNG>
__device__ inline double exponentialDistDouble(RNG *rng) {
    while (true) {
        const double u = curand_uniform_double(rng);
        if (u != 0.0) {
            return -log(u);
        }
    }
}

template<typename RNG>
__device__ inline float gammaDistFloatInternal(RNG *rng, float c, float d)
 {
    float x, v, u;
    while (true) {
        do {
            x = curand_normal(rng);
            v = 1.0f + c*x;
        }
        while (v <= 0.0f);
        
        v = v*v*v;
        do {
            u = curand_uniform(rng);
        }
        while (u == 1.0f);
        
        if (u < 1.0f - 0.0331f*x*x*x*x) break;
        if (logf(u) < 0.5f*x*x + d*(1.0f - v + logf(v))) break;
    }
    
    return d*v;
}

template<typename RNG>
__device__ inline float gammaDistFloat(RNG *rng, float a)
 {
    if (a > 1)
     {
        const float u = curand_uniform (rng);
        const float d = (1.0f + a) - 1.0f / 3.0f;
        const float c = (1.0f / 3.0f) / sqrtf(d);
        return gammaDistFloatInternal (rng, c, d) * powf(u, 1.0f / a);
    }
    else
     {
        const float d = a - 1.0f / 3.0f;
        const float c = (1.0f / 3.0f) / sqrtf(d);
        return gammaDistFloatInternal(rng, c, d);
    }
}

template<typename RNG>
__device__ inline float gammaDistDoubleInternal(RNG *rng, double c, double d)
 {
    double x, v, u;
    while (true) {
        do {
            x = curand_normal_double(rng);
            v = 1.0 + c*x;
        }
        while (v <= 0.0);
        
        v = v*v*v;
        do {
            u = curand_uniform_double(rng);
        }
        while (u == 1.0);
        
        if (u < 1.0 - 0.0331*x*x*x*x) break;
        if (log(u) < 0.5*x*x + d*(1.0 - v + log(v))) break;
    }
    
    return d*v;
}

template<typename RNG>
__device__ inline float gammaDistDouble(RNG *rng, double a)
 {
    if (a > 1.0)
     {
        const double u = curand_uniform (rng);
        const double d = (1.0 + a) - 1.0 / 3.0;
        const double c = (1.0 / 3.0) / sqrt(d);
        return gammaDistDoubleInternal (rng, c, d) * pow(u, 1.0 / a);
    }
    else
     {
        const float d = a - 1.0 / 3.0;
        const float c = (1.0 / 3.0) / sqrt(d);
        return gammaDistDoubleInternal(rng, c, d);
    }
}

template<typename RNG>
__device__ inline unsigned int binomialDistFloatInternal(RNG *rng, unsigned int n, float p)
 {
    const float q = 1.0f - p;
    const float qn = expf(n * logf(q));
    const float np = n * p;
    const unsigned int bound = min(n, (unsigned int)(np + (10.0f * sqrtf((np * q) + 1.0f))));
    unsigned int x = 0;
    float px = qn;
    float u = curand_uniform(rng);
    while(u > px)
     {
        x++;
        if(x > bound) {
            x = 0;
            px = qn;
            u = curand_uniform(rng);
        }
        else {
            u -= px;
            px = ((n - x + 1) * p * px) / (x * q);
        }
    }
    return x;
}

template<typename RNG>
__device__ inline unsigned int binomialDistFloat(RNG *rng, unsigned int n, float p)
 {
    if(p <= 0.5f) {
        return binomialDistFloatInternal(rng, n, p);
    }
    else {
        return (n - binomialDistFloatInternal(rng, n, 1.0f - p));
    }
}
template<typename RNG>
__device__ inline unsigned int binomialDistDoubleInternal(RNG *rng, unsigned int n, double p)
 {
    const double q = 1.0 - p;
    const double qn = exp(n * log(q));
    const double np = n * p;
    const unsigned int bound = min(n, (unsigned int)(np + (10.0 * sqrt((np * q) + 1.0))));
    unsigned int x = 0;
    double px = qn;
    double u = curand_uniform_double(rng);
    while(u > px)
     {
        x++;
        if(x > bound) {
            x = 0;
            px = qn;
            u = curand_uniform_double(rng);
        }
        else {
            u -= px;
            px = ((n - x + 1) * p * px) / (x * q);
        }
    }
    return x;
}

template<typename RNG>
__device__ inline unsigned int binomialDistDouble(RNG *rng, unsigned int n, double p)
 {
    if(p <= 0.5) {
        return binomialDistDoubleInternal(rng, n, p);
    }
    else {
        return (n - binomialDistDoubleInternal(rng, n, 1.0 - p));
    }
}
// ------------------------------------------------------------------------
// merged group structures
// ------------------------------------------------------------------------
extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
EXPORT_VAR __device__ curandStatePhilox4_32_10_t d_rng;

// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged group arrays for host initialisation
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// local neuron groups
// ------------------------------------------------------------------------
// current source variables
// current source variables

// ------------------------------------------------------------------------
// custom update variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// pre and postsynaptic variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// copying merged group structures to device
// ------------------------------------------------------------------------
EXPORT_FUNC void pushMergedNeuronInitGroup0ToDevice(unsigned int idx, scalar* err_prev_event, scalar* t_dataCS0, scalar* dt_dataCS0, int* nt_dataCS0, int* periodicCS0, int* t_idxCS0, float* inSynInSyn0, scalar* loss, scalar* db, scalar* b, scalar* err_event, scalar* err, scalar* targ, scalar* dr, scalar* r, scalar* x, unsigned int* spk, unsigned int* spkCnt, unsigned int numNeurons);
EXPORT_FUNC void pushMergedNeuronInitGroup1ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, float* prevST, scalar* r, scalar* r_event, scalar* r_prev_event, scalar* r_trace, int* t_idxCS0, int* periodicCS0, int* nt_dataCS0, scalar* dt_dataCS0, scalar* t_dataCS0, unsigned int numNeurons);
EXPORT_FUNC void pushMergedNeuronInitGroup2ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, float* prevST, scalar* r, scalar* x, scalar* dr, scalar* b, scalar* db, scalar* err_fb, scalar* r_event, scalar* r_prev_event, scalar* dr_err_prod, scalar* weight_factor, float* inSynInSyn0, float* inSynInSyn1, unsigned int numNeurons);
EXPORT_FUNC void pushMergedSynapseInitGroup0ToDevice(unsigned int idx, scalar* g, scalar* inp_prev, scalar* dg, scalar* dg_prev, scalar* t_prev, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int rowStride, scalar sdg);
EXPORT_FUNC void pushMergedCustomUpdateInitGroup0ToDevice(unsigned int idx, scalar* SumExpVal, unsigned int size);
EXPORT_FUNC void pushMergedCustomUpdateInitGroup1ToDevice(unsigned int idx, scalar* MaxVal, unsigned int size);
EXPORT_FUNC void pushMergedCustomUpdateInitGroup2ToDevice(unsigned int idx, scalar* reducedChange, unsigned int size);
EXPORT_FUNC void pushMergedCustomUpdateInitGroup3ToDevice(unsigned int idx, scalar* m, scalar* v, scalar* time, unsigned int size);
EXPORT_FUNC void pushMergedCustomWUUpdateInitGroup0ToDevice(unsigned int idx, scalar* reducedChange, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int rowStride);
EXPORT_FUNC void pushMergedCustomWUUpdateInitGroup1ToDevice(unsigned int idx, scalar* m, scalar* v, scalar* time, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int rowStride);
EXPORT_FUNC void pushMergedNeuronUpdateGroup0ToDevice(unsigned int idx, scalar* b, uint32_t* recordSpk, scalar* dataCS0, scalar* t_dataCS0, scalar* dt_dataCS0, int* nt_dataCS0, int* periodicCS0, int* t_idxCS0, float* inSynInSyn0, scalar* loss, scalar* db, scalar* err_prev_event, scalar* err_event, scalar* err, scalar* targ, scalar* dr, scalar* r, scalar* x, unsigned int* spk, unsigned int* spkCnt, unsigned int numNeurons);
EXPORT_FUNC void pushMergedNeuronUpdate0recordSpkToDevice(unsigned int idx, uint32_t* value);
EXPORT_FUNC void pushMergedNeuronUpdate0dataCS0ToDevice(unsigned int idx, scalar* value);
EXPORT_FUNC void pushMergedNeuronUpdateGroup1ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, float* prevST, scalar* r, scalar* r_event, scalar* r_prev_event, scalar* r_trace, int* t_idxCS0, int* periodicCS0, int* nt_dataCS0, scalar* dt_dataCS0, scalar* t_dataCS0, scalar* dataCS0, uint32_t* recordSpk, unsigned int numNeurons);
EXPORT_FUNC void pushMergedNeuronUpdate1dataCS0ToDevice(unsigned int idx, scalar* value);
EXPORT_FUNC void pushMergedNeuronUpdate1recordSpkToDevice(unsigned int idx, uint32_t* value);
EXPORT_FUNC void pushMergedNeuronUpdateGroup2ToDevice(unsigned int idx, scalar* db, uint32_t* recordSpk, float* inSynInSyn1, float* inSynInSyn0, scalar* weight_factor, scalar* dr_err_prod, scalar* r_prev_event, scalar* r_event, scalar* err_fb, scalar* b, scalar* dr, scalar* x, scalar* r, float* prevST, unsigned int* spk, unsigned int* spkCnt, unsigned int numNeurons);
EXPORT_FUNC void pushMergedNeuronUpdate2recordSpkToDevice(unsigned int idx, uint32_t* value);
EXPORT_FUNC void pushMergedPresynapticUpdateGroup0ToDevice(unsigned int idx, float* inSyn, unsigned int* srcSpkCnt, unsigned int* srcSpk, scalar* rPre, scalar* r_prev_eventPre, scalar* drPost, scalar* errPost, float* prevSTPre, scalar* g, scalar* inp_prev, scalar* dg, scalar* dg_prev, scalar* t_prev, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int rowStride);
EXPORT_FUNC void pushMergedPresynapticUpdateGroup1ToDevice(unsigned int idx, float* inSyn, unsigned int* srcSpkCnt, unsigned int* srcSpk, scalar* drPre, scalar* errPre, scalar* rPost, scalar* g, scalar* inp_prev, scalar* dg, scalar* dg_prev, scalar* t_prev, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int rowStride);
EXPORT_FUNC void pushMergedPresynapticUpdateGroup2ToDevice(unsigned int idx, float* inSyn, unsigned int* srcSpkCnt, unsigned int* srcSpk, scalar* rPre, scalar* r_tracePre, scalar* dr_err_prodPost, float* prevSTPre, scalar* g, scalar* inp_prev, scalar* dg, scalar* dg_prev, scalar* t_prev, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int rowStride);
EXPORT_FUNC void pushMergedNeuronPrevSpikeTimeUpdateGroup0ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, float* prevST, unsigned int numNeurons);
EXPORT_FUNC void pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(unsigned int idx, unsigned int* spkCnt);
EXPORT_FUNC void pushMergedCustomUpdateGroup0ToDevice(unsigned int idx, scalar* Val, scalar* MaxVal, scalar* SumExpVal, scalar* SoftmaxVal, unsigned int size);
EXPORT_FUNC void pushMergedCustomUpdateGroup1ToDevice(unsigned int idx, scalar* SumExpVal, scalar* Val, scalar* MaxVal, unsigned int size);
EXPORT_FUNC void pushMergedCustomUpdateGroup2ToDevice(unsigned int idx, scalar* MaxVal, scalar* Val, unsigned int size);
EXPORT_FUNC void pushMergedCustomUpdateGroup3ToDevice(unsigned int idx, scalar* reducedChange, scalar* change, unsigned int size);
EXPORT_FUNC void pushMergedCustomUpdateGroup4ToDevice(unsigned int idx, scalar* m, scalar* v, scalar* time, scalar* change, scalar* variable, unsigned int size);
EXPORT_FUNC void pushMergedCustomUpdateWUGroup0ToDevice(unsigned int idx, scalar* reducedChange, scalar* change, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int rowStride);
EXPORT_FUNC void pushMergedCustomUpdateWUGroup1ToDevice(unsigned int idx, scalar* m, scalar* v, scalar* time, scalar* change, scalar* variable, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int rowStride);
}  // extern "C"
