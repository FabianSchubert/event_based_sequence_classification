#include "definitionsInternal.h"
#include <iostream>
#include <random>
#include <cstdint>

struct MergedNeuronInitGroup0
 {
    scalar* err_prev_event;
    scalar* t_dataCS0;
    scalar* dt_dataCS0;
    int* nt_dataCS0;
    int* periodicCS0;
    int* t_idxCS0;
    float* inSynInSyn0;
    scalar* loss;
    scalar* db;
    scalar* b;
    scalar* err_event;
    scalar* err;
    scalar* targ;
    scalar* dr;
    scalar* r;
    scalar* x;
    unsigned int* spk;
    unsigned int* spkCnt;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronInitGroup1
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    float* prevST;
    scalar* r;
    scalar* r_event;
    scalar* r_prev_event;
    scalar* r_trace;
    int* t_idxCS0;
    int* periodicCS0;
    int* nt_dataCS0;
    scalar* dt_dataCS0;
    scalar* t_dataCS0;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronInitGroup2
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    float* prevST;
    scalar* r;
    scalar* x;
    scalar* dr;
    scalar* b;
    scalar* db;
    scalar* err_fb;
    scalar* r_event;
    scalar* r_prev_event;
    scalar* dr_err_prod;
    scalar* weight_factor;
    float* inSynInSyn0;
    float* inSynInSyn1;
    unsigned int numNeurons;
    
}
;
struct MergedSynapseInitGroup0
 {
    scalar* g;
    scalar* inp_prev;
    scalar* dg;
    scalar* dg_prev;
    scalar* t_prev;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    unsigned int rowStride;
    scalar sdg;
    
}
;
struct MergedCustomUpdateInitGroup0
 {
    scalar* SumExpVal;
    unsigned int size;
    
}
;
struct MergedCustomUpdateInitGroup1
 {
    scalar* MaxVal;
    unsigned int size;
    
}
;
struct MergedCustomUpdateInitGroup2
 {
    scalar* reducedChange;
    unsigned int size;
    
}
;
struct MergedCustomUpdateInitGroup3
 {
    scalar* m;
    scalar* v;
    scalar* time;
    unsigned int size;
    
}
;
struct MergedCustomWUUpdateInitGroup0
 {
    scalar* reducedChange;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    unsigned int rowStride;
    
}
;
struct MergedCustomWUUpdateInitGroup1
 {
    scalar* m;
    scalar* v;
    scalar* time;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    unsigned int rowStride;
    
}
;
__device__ __constant__ MergedNeuronInitGroup0 d_mergedNeuronInitGroup0[1];
void pushMergedNeuronInitGroup0ToDevice(unsigned int idx, scalar* err_prev_event, scalar* t_dataCS0, scalar* dt_dataCS0, int* nt_dataCS0, int* periodicCS0, int* t_idxCS0, float* inSynInSyn0, scalar* loss, scalar* db, scalar* b, scalar* err_event, scalar* err, scalar* targ, scalar* dr, scalar* r, scalar* x, unsigned int* spk, unsigned int* spkCnt, unsigned int numNeurons) {
    MergedNeuronInitGroup0 group = {err_prev_event, t_dataCS0, dt_dataCS0, nt_dataCS0, periodicCS0, t_idxCS0, inSynInSyn0, loss, db, b, err_event, err, targ, dr, r, x, spk, spkCnt, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronInitGroup0, &group, sizeof(MergedNeuronInitGroup0), idx * sizeof(MergedNeuronInitGroup0)));
}
__device__ __constant__ MergedNeuronInitGroup1 d_mergedNeuronInitGroup1[1];
void pushMergedNeuronInitGroup1ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, float* prevST, scalar* r, scalar* r_event, scalar* r_prev_event, scalar* r_trace, int* t_idxCS0, int* periodicCS0, int* nt_dataCS0, scalar* dt_dataCS0, scalar* t_dataCS0, unsigned int numNeurons) {
    MergedNeuronInitGroup1 group = {spkCnt, spk, prevST, r, r_event, r_prev_event, r_trace, t_idxCS0, periodicCS0, nt_dataCS0, dt_dataCS0, t_dataCS0, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronInitGroup1, &group, sizeof(MergedNeuronInitGroup1), idx * sizeof(MergedNeuronInitGroup1)));
}
__device__ __constant__ MergedNeuronInitGroup2 d_mergedNeuronInitGroup2[1];
void pushMergedNeuronInitGroup2ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, float* prevST, scalar* r, scalar* x, scalar* dr, scalar* b, scalar* db, scalar* err_fb, scalar* r_event, scalar* r_prev_event, scalar* dr_err_prod, scalar* weight_factor, float* inSynInSyn0, float* inSynInSyn1, unsigned int numNeurons) {
    MergedNeuronInitGroup2 group = {spkCnt, spk, prevST, r, x, dr, b, db, err_fb, r_event, r_prev_event, dr_err_prod, weight_factor, inSynInSyn0, inSynInSyn1, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronInitGroup2, &group, sizeof(MergedNeuronInitGroup2), idx * sizeof(MergedNeuronInitGroup2)));
}
__device__ __constant__ MergedSynapseInitGroup0 d_mergedSynapseInitGroup0[3];
void pushMergedSynapseInitGroup0ToDevice(unsigned int idx, scalar* g, scalar* inp_prev, scalar* dg, scalar* dg_prev, scalar* t_prev, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int rowStride, scalar sdg) {
    MergedSynapseInitGroup0 group = {g, inp_prev, dg, dg_prev, t_prev, numSrcNeurons, numTrgNeurons, rowStride, sdg, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedSynapseInitGroup0, &group, sizeof(MergedSynapseInitGroup0), idx * sizeof(MergedSynapseInitGroup0)));
}
__device__ __constant__ MergedCustomUpdateInitGroup0 d_mergedCustomUpdateInitGroup0[1];
void pushMergedCustomUpdateInitGroup0ToDevice(unsigned int idx, scalar* SumExpVal, unsigned int size) {
    MergedCustomUpdateInitGroup0 group = {SumExpVal, size, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomUpdateInitGroup0, &group, sizeof(MergedCustomUpdateInitGroup0), idx * sizeof(MergedCustomUpdateInitGroup0)));
}
__device__ __constant__ MergedCustomUpdateInitGroup1 d_mergedCustomUpdateInitGroup1[1];
void pushMergedCustomUpdateInitGroup1ToDevice(unsigned int idx, scalar* MaxVal, unsigned int size) {
    MergedCustomUpdateInitGroup1 group = {MaxVal, size, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomUpdateInitGroup1, &group, sizeof(MergedCustomUpdateInitGroup1), idx * sizeof(MergedCustomUpdateInitGroup1)));
}
__device__ __constant__ MergedCustomUpdateInitGroup2 d_mergedCustomUpdateInitGroup2[2];
void pushMergedCustomUpdateInitGroup2ToDevice(unsigned int idx, scalar* reducedChange, unsigned int size) {
    MergedCustomUpdateInitGroup2 group = {reducedChange, size, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomUpdateInitGroup2, &group, sizeof(MergedCustomUpdateInitGroup2), idx * sizeof(MergedCustomUpdateInitGroup2)));
}
__device__ __constant__ MergedCustomUpdateInitGroup3 d_mergedCustomUpdateInitGroup3[2];
void pushMergedCustomUpdateInitGroup3ToDevice(unsigned int idx, scalar* m, scalar* v, scalar* time, unsigned int size) {
    MergedCustomUpdateInitGroup3 group = {m, v, time, size, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomUpdateInitGroup3, &group, sizeof(MergedCustomUpdateInitGroup3), idx * sizeof(MergedCustomUpdateInitGroup3)));
}
__device__ __constant__ MergedCustomWUUpdateInitGroup0 d_mergedCustomWUUpdateInitGroup0[3];
void pushMergedCustomWUUpdateInitGroup0ToDevice(unsigned int idx, scalar* reducedChange, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int rowStride) {
    MergedCustomWUUpdateInitGroup0 group = {reducedChange, numSrcNeurons, numTrgNeurons, rowStride, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomWUUpdateInitGroup0, &group, sizeof(MergedCustomWUUpdateInitGroup0), idx * sizeof(MergedCustomWUUpdateInitGroup0)));
}
__device__ __constant__ MergedCustomWUUpdateInitGroup1 d_mergedCustomWUUpdateInitGroup1[3];
void pushMergedCustomWUUpdateInitGroup1ToDevice(unsigned int idx, scalar* m, scalar* v, scalar* time, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int rowStride) {
    MergedCustomWUUpdateInitGroup1 group = {m, v, time, numSrcNeurons, numTrgNeurons, rowStride, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomWUUpdateInitGroup1, &group, sizeof(MergedCustomWUUpdateInitGroup1), idx * sizeof(MergedCustomWUUpdateInitGroup1)));
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
__device__ unsigned int d_mergedNeuronInitGroupStartID0[] = {0, };
__device__ unsigned int d_mergedNeuronInitGroupStartID1[] = {64, };
__device__ unsigned int d_mergedNeuronInitGroupStartID2[] = {576, };
__device__ unsigned int d_mergedSynapseInitGroupStartID0[] = {1344, 2112, 2880, };
__device__ unsigned int d_mergedCustomUpdateInitGroupStartID0[] = {2944, };
__device__ unsigned int d_mergedCustomUpdateInitGroupStartID1[] = {3008, };
__device__ unsigned int d_mergedCustomUpdateInitGroupStartID2[] = {3072, 3840, };
__device__ unsigned int d_mergedCustomUpdateInitGroupStartID3[] = {3904, 4672, };
__device__ unsigned int d_mergedCustomWUUpdateInitGroupStartID0[] = {4736, 5504, 6272, };
__device__ unsigned int d_mergedCustomWUUpdateInitGroupStartID1[] = {6336, 7104, 7872, };

extern "C" __global__ void initializeRNGKernel(unsigned long long deviceRNGSeed) {
    if(threadIdx.x == 0) {
        curand_init(deviceRNGSeed, 0, 0, &d_rng);
    }
}

extern "C" __global__ void initializeKernel(unsigned long long deviceRNGSeed) {
    const unsigned int id = 64 * blockIdx.x + threadIdx.x;
    // ------------------------------------------------------------------------
    // Local neuron groups
    // merged0
    if(id < 64) {
        struct MergedNeuronInitGroup0 *group = &d_mergedNeuronInitGroup0[0]; 
        const unsigned int lid = id - 0;
        // only do this for existing neurons
        if(lid < group->numNeurons) {
            if(lid == 0) {
                group->spkCnt[0] = 0;
            }
            group->spk[lid] = 0;
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->x[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->r[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->dr[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->targ[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->err[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->err_event[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->err_prev_event[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->b[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->db[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->loss[lid] = initVal;
            }
             {
                group->inSynInSyn0[lid] = 0.000000000e+00f;
            }
            // current source variables
             {
                int initVal;
                initVal = (0.00000000000000000e+00f);
                group->t_idxCS0[lid] = initVal;
            }
             {
                int initVal;
                initVal = (0.00000000000000000e+00f);
                group->periodicCS0[lid] = initVal;
            }
             {
                int initVal;
                initVal = (0.00000000000000000e+00f);
                group->nt_dataCS0[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->dt_dataCS0[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->t_dataCS0[lid] = initVal;
            }
        }
    }
    // merged1
    if(id >= 64 && id < 576) {
        struct MergedNeuronInitGroup1 *group = &d_mergedNeuronInitGroup1[0]; 
        const unsigned int lid = id - 64;
        // only do this for existing neurons
        if(lid < group->numNeurons) {
            if(lid == 0) {
                group->spkCnt[0] = 0;
            }
            group->spk[lid] = 0;
            group->prevST[lid] = -TIME_MAX;
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->r[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->r_event[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->r_prev_event[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->r_trace[lid] = initVal;
            }
            // current source variables
             {
                int initVal;
                initVal = (0.00000000000000000e+00f);
                group->t_idxCS0[lid] = initVal;
            }
             {
                int initVal;
                initVal = (0.00000000000000000e+00f);
                group->periodicCS0[lid] = initVal;
            }
             {
                int initVal;
                initVal = (0.00000000000000000e+00f);
                group->nt_dataCS0[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->dt_dataCS0[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->t_dataCS0[lid] = initVal;
            }
        }
    }
    // merged2
    if(id >= 576 && id < 1344) {
        struct MergedNeuronInitGroup2 *group = &d_mergedNeuronInitGroup2[0]; 
        const unsigned int lid = id - 576;
        // only do this for existing neurons
        if(lid < group->numNeurons) {
            if(lid == 0) {
                group->spkCnt[0] = 0;
            }
            group->spk[lid] = 0;
            group->prevST[lid] = -TIME_MAX;
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->r[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->x[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (1.00000000000000000e+00f);
                group->dr[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->b[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->db[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->err_fb[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->r_event[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->r_prev_event[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->dr_err_prod[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (1.00000000000000000e+00f);
                group->weight_factor[lid] = initVal;
            }
             {
                group->inSynInSyn0[lid] = 0.000000000e+00f;
            }
             {
                group->inSynInSyn1[lid] = 0.000000000e+00f;
            }
            // current source variables
        }
    }
    
    // ------------------------------------------------------------------------
    // Synapse groups
    // merged0
    if(id >= 1344 && id < 2944) {
        unsigned int lo = 0;
        unsigned int hi = 3;
        while(lo < hi)
         {
            const unsigned int mid = (lo + hi) / 2;
            if(id < d_mergedSynapseInitGroupStartID0[mid]) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        struct MergedSynapseInitGroup0 *group = &d_mergedSynapseInitGroup0[lo - 1]; 
        const unsigned int groupStartID = d_mergedSynapseInitGroupStartID0[lo - 1];
        const unsigned int lid = id - groupStartID;
        if(lid < group->numTrgNeurons) {
            curandStatePhilox4_32_10_t localRNG = d_rng;
            skipahead_sequence((unsigned long long)id, &localRNG);
            for(unsigned int i = 0; i < group->numSrcNeurons; i++) {
                 {
                    scalar initVal;
                    initVal = (0.00000000000000000e+00f) + (curand_normal(&localRNG) * group->sdg);
                    group->g[(i * group->rowStride) + lid] = initVal;
                }
                 {
                    scalar initVal;
                    initVal = (0.00000000000000000e+00f);
                    group->inp_prev[(i * group->rowStride) + lid] = initVal;
                }
                 {
                    scalar initVal;
                    initVal = (0.00000000000000000e+00f);
                    group->dg[(i * group->rowStride) + lid] = initVal;
                }
                 {
                    scalar initVal;
                    initVal = (0.00000000000000000e+00f);
                    group->dg_prev[(i * group->rowStride) + lid] = initVal;
                }
                 {
                    scalar initVal;
                    initVal = (0.00000000000000000e+00f);
                    group->t_prev[(i * group->rowStride) + lid] = initVal;
                }
            }
        }
    }
    
    // ------------------------------------------------------------------------
    // Custom update groups
    // merged0
    if(id >= 2944 && id < 3008) {
        struct MergedCustomUpdateInitGroup0 *group = &d_mergedCustomUpdateInitGroup0[0]; 
        const unsigned int lid = id - 2944;
        // only do this for existing variables
        if(lid < group->size) {
             {
                if(lid == 0) {
                    scalar initVal;
                    initVal = (0.00000000000000000e+00f);
                    group->SumExpVal[0] = initVal;
                }
            }
        }
    }
    // merged1
    if(id >= 3008 && id < 3072) {
        struct MergedCustomUpdateInitGroup1 *group = &d_mergedCustomUpdateInitGroup1[0]; 
        const unsigned int lid = id - 3008;
        // only do this for existing variables
        if(lid < group->size) {
             {
                if(lid == 0) {
                    scalar initVal;
                    initVal = (0.00000000000000000e+00f);
                    group->MaxVal[0] = initVal;
                }
            }
        }
    }
    // merged2
    if(id >= 3072 && id < 3904) {
        unsigned int lo = 0;
        unsigned int hi = 2;
        while(lo < hi)
         {
            const unsigned int mid = (lo + hi) / 2;
            if(id < d_mergedCustomUpdateInitGroupStartID2[mid]) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        struct MergedCustomUpdateInitGroup2 *group = &d_mergedCustomUpdateInitGroup2[lo - 1]; 
        const unsigned int groupStartID = d_mergedCustomUpdateInitGroupStartID2[lo - 1];
        const unsigned int lid = id - groupStartID;
        // only do this for existing variables
        if(lid < group->size) {
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->reducedChange[lid] = initVal;
            }
        }
    }
    // merged3
    if(id >= 3904 && id < 4736) {
        unsigned int lo = 0;
        unsigned int hi = 2;
        while(lo < hi)
         {
            const unsigned int mid = (lo + hi) / 2;
            if(id < d_mergedCustomUpdateInitGroupStartID3[mid]) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        struct MergedCustomUpdateInitGroup3 *group = &d_mergedCustomUpdateInitGroup3[lo - 1]; 
        const unsigned int groupStartID = d_mergedCustomUpdateInitGroupStartID3[lo - 1];
        const unsigned int lid = id - groupStartID;
        // only do this for existing variables
        if(lid < group->size) {
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->m[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->v[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (1.00000000000000000e+00f);
                group->time[lid] = initVal;
            }
        }
    }
    
    // ------------------------------------------------------------------------
    // Custom WU update groups
    // merged0
    if(id >= 4736 && id < 6336) {
        unsigned int lo = 0;
        unsigned int hi = 3;
        while(lo < hi)
         {
            const unsigned int mid = (lo + hi) / 2;
            if(id < d_mergedCustomWUUpdateInitGroupStartID0[mid]) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        struct MergedCustomWUUpdateInitGroup0 *group = &d_mergedCustomWUUpdateInitGroup0[lo - 1]; 
        const unsigned int groupStartID = d_mergedCustomWUUpdateInitGroupStartID0[lo - 1];
        const unsigned int lid = id - groupStartID;
        if(lid < group->numTrgNeurons) {
            for(unsigned int i = 0; i < group->numSrcNeurons; i++) {
                 {
                    scalar initVal;
                    initVal = (0.00000000000000000e+00f);
                    group->reducedChange[(i * group->rowStride) + lid] = initVal;
                }
            }
        }
    }
    // merged1
    if(id >= 6336 && id < 7936) {
        unsigned int lo = 0;
        unsigned int hi = 3;
        while(lo < hi)
         {
            const unsigned int mid = (lo + hi) / 2;
            if(id < d_mergedCustomWUUpdateInitGroupStartID1[mid]) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        struct MergedCustomWUUpdateInitGroup1 *group = &d_mergedCustomWUUpdateInitGroup1[lo - 1]; 
        const unsigned int groupStartID = d_mergedCustomWUUpdateInitGroupStartID1[lo - 1];
        const unsigned int lid = id - groupStartID;
        if(lid < group->numTrgNeurons) {
            for(unsigned int i = 0; i < group->numSrcNeurons; i++) {
                 {
                    scalar initVal;
                    initVal = (0.00000000000000000e+00f);
                    group->m[(i * group->rowStride) + lid] = initVal;
                }
                 {
                    scalar initVal;
                    initVal = (0.00000000000000000e+00f);
                    group->v[(i * group->rowStride) + lid] = initVal;
                }
                 {
                    scalar initVal;
                    initVal = (1.00000000000000000e+00f);
                    group->time[(i * group->rowStride) + lid] = initVal;
                }
            }
        }
    }
    
    // ------------------------------------------------------------------------
    // Synapse groups with sparse connectivity
    
}
void initialize() {
    unsigned long long deviceRNGSeed = 0;
     {
        std::random_device seedSource;
        uint32_t *deviceRNGSeedWord = reinterpret_cast<uint32_t*>(&deviceRNGSeed);
        for(int i = 0; i < 2; i++) {
            deviceRNGSeedWord[i] = seedSource();
        }
    }
    initializeRNGKernel<<<1, 1>>>(deviceRNGSeed);
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
     {
        const dim3 threads(64, 1);
        const dim3 grid(124, 1);
        initializeKernel<<<grid, threads>>>(deviceRNGSeed);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}

void initializeSparse() {
    copyStateToDevice(true);
    copyConnectivityToDevice(true);
    
}
