#include "definitionsInternal.h"
#include "supportCode.h"

struct MergedNeuronUpdateGroup0
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    float* prevST;
    scalar* r_trace;
    scalar* r;
    scalar* r_event;
    scalar* r_prev_event;
    float* inSynInSyn0;
    float* inSynInSyn1;
    uint32_t* recordSpk;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronUpdateGroup1
 {
    scalar* err_prev_event;
    scalar* dataCS0;
    int* ntCS0;
    scalar* dt_dataCS0;
    scalar* t_dataCS0;
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
    float* prevST;
    unsigned int* spk;
    unsigned int* spkCnt;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronUpdateGroup2
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    float* prevST;
    scalar* r;
    scalar* r_event;
    scalar* r_prev_event;
    int* t_idxCS0;
    int* periodicCS0;
    scalar* t_dataCS0;
    scalar* dt_dataCS0;
    int* ntCS0;
    scalar* dataCS0;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronUpdateGroup3
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
    float* inSynInSyn0;
    float* inSynInSyn1;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronSpikeQueueUpdateGroup0
 {
    unsigned int* spkCnt;
    
}
;
struct MergedNeuronPrevSpikeTimeUpdateGroup0
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    float* prevST;
    unsigned int numNeurons;
    
}
;
__device__ __constant__ MergedNeuronSpikeQueueUpdateGroup0 d_mergedNeuronSpikeQueueUpdateGroup0[4];
void pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(unsigned int idx, unsigned int* spkCnt) {
    MergedNeuronSpikeQueueUpdateGroup0 group = {spkCnt, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronSpikeQueueUpdateGroup0, &group, sizeof(MergedNeuronSpikeQueueUpdateGroup0), idx * sizeof(MergedNeuronSpikeQueueUpdateGroup0)));
}
__device__ __constant__ MergedNeuronPrevSpikeTimeUpdateGroup0 d_mergedNeuronPrevSpikeTimeUpdateGroup0[4];
void pushMergedNeuronPrevSpikeTimeUpdateGroup0ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, float* prevST, unsigned int numNeurons) {
    MergedNeuronPrevSpikeTimeUpdateGroup0 group = {spkCnt, spk, prevST, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronPrevSpikeTimeUpdateGroup0, &group, sizeof(MergedNeuronPrevSpikeTimeUpdateGroup0), idx * sizeof(MergedNeuronPrevSpikeTimeUpdateGroup0)));
}
__device__ __constant__ MergedNeuronUpdateGroup0 d_mergedNeuronUpdateGroup0[1];
void pushMergedNeuronUpdateGroup0ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, float* prevST, scalar* r_trace, scalar* r, scalar* r_event, scalar* r_prev_event, float* inSynInSyn0, float* inSynInSyn1, uint32_t* recordSpk, unsigned int numNeurons) {
    MergedNeuronUpdateGroup0 group = {spkCnt, spk, prevST, r_trace, r, r_event, r_prev_event, inSynInSyn0, inSynInSyn1, recordSpk, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup0, &group, sizeof(MergedNeuronUpdateGroup0), idx * sizeof(MergedNeuronUpdateGroup0)));
}
__device__ __constant__ MergedNeuronUpdateGroup1 d_mergedNeuronUpdateGroup1[1];
void pushMergedNeuronUpdateGroup1ToDevice(unsigned int idx, scalar* err_prev_event, scalar* dataCS0, int* ntCS0, scalar* dt_dataCS0, scalar* t_dataCS0, int* periodicCS0, int* t_idxCS0, float* inSynInSyn0, scalar* loss, scalar* db, scalar* b, scalar* err_event, scalar* err, scalar* targ, scalar* dr, scalar* r, scalar* x, float* prevST, unsigned int* spk, unsigned int* spkCnt, unsigned int numNeurons) {
    MergedNeuronUpdateGroup1 group = {err_prev_event, dataCS0, ntCS0, dt_dataCS0, t_dataCS0, periodicCS0, t_idxCS0, inSynInSyn0, loss, db, b, err_event, err, targ, dr, r, x, prevST, spk, spkCnt, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup1, &group, sizeof(MergedNeuronUpdateGroup1), idx * sizeof(MergedNeuronUpdateGroup1)));
}
__device__ __constant__ MergedNeuronUpdateGroup2 d_mergedNeuronUpdateGroup2[1];
void pushMergedNeuronUpdateGroup2ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, float* prevST, scalar* r, scalar* r_event, scalar* r_prev_event, int* t_idxCS0, int* periodicCS0, scalar* t_dataCS0, scalar* dt_dataCS0, int* ntCS0, scalar* dataCS0, unsigned int numNeurons) {
    MergedNeuronUpdateGroup2 group = {spkCnt, spk, prevST, r, r_event, r_prev_event, t_idxCS0, periodicCS0, t_dataCS0, dt_dataCS0, ntCS0, dataCS0, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup2, &group, sizeof(MergedNeuronUpdateGroup2), idx * sizeof(MergedNeuronUpdateGroup2)));
}
__device__ __constant__ MergedNeuronUpdateGroup3 d_mergedNeuronUpdateGroup3[1];
void pushMergedNeuronUpdateGroup3ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, float* prevST, scalar* r, scalar* x, scalar* dr, scalar* b, scalar* db, scalar* err_fb, scalar* r_event, scalar* r_prev_event, float* inSynInSyn0, float* inSynInSyn1, unsigned int numNeurons) {
    MergedNeuronUpdateGroup3 group = {spkCnt, spk, prevST, r, x, dr, b, db, err_fb, r_event, r_prev_event, inSynInSyn0, inSynInSyn1, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup3, &group, sizeof(MergedNeuronUpdateGroup3), idx * sizeof(MergedNeuronUpdateGroup3)));
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
void pushMergedNeuronUpdate0recordSpkToDevice(unsigned int idx, uint32_t* value) {
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup0, &value, sizeof(value), (sizeof(MergedNeuronUpdateGroup0) * (idx)) + offsetof(MergedNeuronUpdateGroup0, recordSpk)));
}

void pushMergedNeuronUpdate1dataCS0ToDevice(unsigned int idx, scalar* value) {
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup1, &value, sizeof(value), (sizeof(MergedNeuronUpdateGroup1) * (idx)) + offsetof(MergedNeuronUpdateGroup1, dataCS0)));
}

void pushMergedNeuronUpdate2dataCS0ToDevice(unsigned int idx, scalar* value) {
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup2, &value, sizeof(value), (sizeof(MergedNeuronUpdateGroup2) * (idx)) + offsetof(MergedNeuronUpdateGroup2, dataCS0)));
}

__device__ __constant__ unsigned int d_mergedNeuronUpdateGroupStartID0[] = {0, };
__device__ __constant__ unsigned int d_mergedNeuronUpdateGroupStartID1[] = {512, };
__device__ __constant__ unsigned int d_mergedNeuronUpdateGroupStartID2[] = {544, };
__device__ __constant__ unsigned int d_mergedNeuronUpdateGroupStartID3[] = {576, };
__device__ __constant__ unsigned int d_mergedNeuronPrevSpikeTimeUpdateGroupStartID0[] = {0, 512, 544, 576, };

extern "C" __global__ void neuronPrevSpikeTimeUpdateKernel(float t) {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x;
    // merged0
    if(id < 1088) {
        unsigned int lo = 0;
        unsigned int hi = 4;
        while(lo < hi)
         {
            const unsigned int mid = (lo + hi) / 2;
            if(id < d_mergedNeuronPrevSpikeTimeUpdateGroupStartID0[mid]) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        struct MergedNeuronPrevSpikeTimeUpdateGroup0 *group = &d_mergedNeuronPrevSpikeTimeUpdateGroup0[lo - 1]; 
        const unsigned int groupStartID = d_mergedNeuronPrevSpikeTimeUpdateGroupStartID0[lo - 1];
        const unsigned int lid = id - groupStartID;
         {
            if(lid < group->spkCnt[0]) {
                group->prevST[group->spk[lid]] = t - DT;
            }
            
        }
    }
}

extern "C" __global__ void neuronSpikeQueueUpdateKernel() {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x;
    if(id < 4) {
        struct MergedNeuronSpikeQueueUpdateGroup0 *group = &d_mergedNeuronSpikeQueueUpdateGroup0[id - 0]; 
        group->spkCnt[0] = 0;
    }
}

extern "C" __global__ void updateNeuronsKernel(float t, unsigned int recordingTimestep)
 {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x; 
    __shared__ unsigned int shSpk[32];
    __shared__ unsigned int shPosSpk;
    __shared__ unsigned int shSpkCount;
    if (threadIdx.x == 0) {
        shSpkCount = 0;
    }
    
    __shared__ uint32_t shSpkRecord;
    if (threadIdx.x == 0) {
        shSpkRecord = 0;
    }
    __syncthreads();
    // merged0
    if(id < 512) {
        struct MergedNeuronUpdateGroup0 *group = &d_mergedNeuronUpdateGroup0[0]; 
        const unsigned int lid = id - 0;
        
        if(lid < group->numNeurons) {
            scalar lr_trace = group->r_trace[lid];
            scalar lr = group->r[lid];
            scalar lr_event = group->r_event[lid];
            scalar lr_prev_event = group->r_prev_event[lid];
            const float lprevST = group->prevST[lid];
            
            float Isyn = 0;
            scalar Isyn_in = 0.00000000000000000e+00f;
             {
                // pull inSyn values in a coalesced access
                float linSyn = group->inSynInSyn0[lid];
                Isyn += linSyn; linSyn *= (1.0f - DT * (0.00000000000000000e+00f));
                
                group->inSynInSyn0[lid] = linSyn;
            }
             {
                // pull inSyn values in a coalesced access
                float linSyn = group->inSynInSyn1[lid];
                Isyn_in += linSyn; linSyn *= (1.0f - DT * (0.00000000000000000e+00f));
                
                group->inSynInSyn1[lid] = linSyn;
            }
            // test whether spike condition was fulfilled previously
            // calculate membrane potential
            
            lr += DT * (tanh(Isyn + Isyn_in) - lr);
            //lr += DT * (Isyn + Isyn_in - lr);
            lr_trace += DT * (lr - lr_trace);
            
            // test for and register a true spike
            if (abs(lr - lr_event) >= (1.00000000000000006e-01f)) {
                const unsigned int spkIdx = atomicAdd(&shSpkCount, 1);
                shSpk[spkIdx] = lid;
                atomicOr(&shSpkRecord, 1 << threadIdx.x);
                // spike reset code
                
                lr_prev_event = lr_event;
                lr_event = lr;
                
            }
            group->r_trace[lid] = lr_trace;
            group->r[lid] = lr;
            group->r_event[lid] = lr_event;
            group->r_prev_event[lid] = lr_prev_event;
        }
        __syncthreads();
        if(threadIdx.x == 0) {
            if (shSpkCount > 0) {
                shPosSpk = atomicAdd(&group->spkCnt[0], shSpkCount);
            }
        }
        __syncthreads();
        if(threadIdx.x < shSpkCount) {
            const unsigned int n = shSpk[threadIdx.x];
            group->spk[shPosSpk + threadIdx.x] = n;
        }
        if(threadIdx.x < 1) {
            const unsigned int numRecordingWords = (group->numNeurons + 31) / 32;
            const unsigned int popWordIdx = (lid / 32) + threadIdx.x;
            if(popWordIdx < numRecordingWords) {
                group->recordSpk[(recordingTimestep * numRecordingWords * 1) + popWordIdx] = shSpkRecord;
            }
        }
    }
    // merged1
    if(id >= 512 && id < 544) {
        struct MergedNeuronUpdateGroup1 *group = &d_mergedNeuronUpdateGroup1[0]; 
        const unsigned int lid = id - 512;
        
        if(lid < group->numNeurons) {
            scalar lx = group->x[lid];
            scalar lr = group->r[lid];
            scalar ldr = group->dr[lid];
            scalar ltarg = group->targ[lid];
            scalar lerr = group->err[lid];
            scalar lerr_event = group->err_event[lid];
            scalar lerr_prev_event = group->err_prev_event[lid];
            const scalar lb = group->b[lid];
            scalar ldb = group->db[lid];
            scalar lloss = group->loss[lid];
            const float lprevST = group->prevST[lid];
            
            float Isyn = 0;
            scalar Isyn_net = 0.00000000000000000e+00f;
             {
                // pull inSyn values in a coalesced access
                float linSyn = group->inSynInSyn0[lid];
                Isyn_net += linSyn; linSyn *= (1.0f - DT * (0.00000000000000000e+00f));
                
                group->inSynInSyn0[lid] = linSyn;
            }
            // current source 0
             {
                int lcst_idx = group->t_idxCS0[lid];
                int lcsperiodic = group->periodicCS0[lid];
                scalar lcst_data = group->t_dataCS0[lid];
                scalar lcsdt_data = group->dt_dataCS0[lid];
                int lcsnt = group->ntCS0[lid];
                
                if(lcsperiodic){
                	lcst_idx = int(lcst_data / lcsdt_data) % lcsnt;
                } else {
                	lcst_idx = min(int(lcst_data / lcsdt_data), lcsnt-1);
                }
                
                lcst_data += DT;
                
                const int n_pop_int = int((1.10000000000000000e+01f));
                const int n_batch_int = int((1.00000000000000000e+00f));
                const int total_neurons = n_pop_int * n_batch_int;
                
                const scalar datapoint = group->dataCS0[lcst_idx*total_neurons + 0 * n_pop_int + lid];
                
                Isyn += datapoint;
                
                group->t_idxCS0[lid] = lcst_idx;
                group->periodicCS0[lid] = lcsperiodic;
                group->t_dataCS0[lid] = lcst_data;
                group->dt_dataCS0[lid] = lcsdt_data;
                group->ntCS0[lid] = lcsnt;
            }
            // test whether spike condition was fulfilled previously
            // calculate membrane potential
            
            //lx += DT * (Isyn_net + lb - lx);
            lx = Isyn_net + lb;
            ltarg += DT * (Isyn - ltarg);
            //ltarg = Isyn;
            //lr = 1.0f/(1.0f+exp(-lx));
            lr = max(0.0f, lx);
            //lr = lx;
            //ldr = lr * (1.f-lr);
            ldr = (lx > 0.0f ? 1.0f : 0.0f);
            lerr = ltarg - lr;
            lloss += 0.5f * lerr * lerr;
            ldb += lerr * ldr;
            
            // test for and register a true spike
            if (abs(lerr - lerr_event) >= (1.00000000000000002e-02f)) {
                const unsigned int spkIdx = atomicAdd(&shSpkCount, 1);
                shSpk[spkIdx] = lid;
                // spike reset code
                
                lerr_prev_event = lerr_event;
                lerr_event = lerr;
                
            }
            group->x[lid] = lx;
            group->r[lid] = lr;
            group->dr[lid] = ldr;
            group->targ[lid] = ltarg;
            group->err[lid] = lerr;
            group->err_event[lid] = lerr_event;
            group->err_prev_event[lid] = lerr_prev_event;
            group->db[lid] = ldb;
            group->loss[lid] = lloss;
        }
        __syncthreads();
        if(threadIdx.x == 0) {
            if (shSpkCount > 0) {
                shPosSpk = atomicAdd(&group->spkCnt[0], shSpkCount);
            }
        }
        __syncthreads();
        if(threadIdx.x < shSpkCount) {
            const unsigned int n = shSpk[threadIdx.x];
            group->spk[shPosSpk + threadIdx.x] = n;
        }
    }
    // merged2
    if(id >= 544 && id < 576) {
        struct MergedNeuronUpdateGroup2 *group = &d_mergedNeuronUpdateGroup2[0]; 
        const unsigned int lid = id - 544;
        
        if(lid < group->numNeurons) {
            scalar lr = group->r[lid];
            scalar lr_event = group->r_event[lid];
            scalar lr_prev_event = group->r_prev_event[lid];
            const float lprevST = group->prevST[lid];
            
            float Isyn = 0;
            // current source 0
             {
                int lcst_idx = group->t_idxCS0[lid];
                int lcsperiodic = group->periodicCS0[lid];
                scalar lcst_data = group->t_dataCS0[lid];
                scalar lcsdt_data = group->dt_dataCS0[lid];
                int lcsnt = group->ntCS0[lid];
                
                if(lcsperiodic){
                	lcst_idx = int(lcst_data / lcsdt_data) % lcsnt;
                } else {
                	lcst_idx = min(int(lcst_data / lcsdt_data), lcsnt-1);
                }
                
                lcst_data += DT;
                
                const int n_pop_int = int((9.00000000000000000e+00f));
                const int n_batch_int = int((1.00000000000000000e+00f));
                const int total_neurons = n_pop_int * n_batch_int;
                
                const scalar datapoint = group->dataCS0[lcst_idx*total_neurons + 0 * n_pop_int + lid];
                
                Isyn += datapoint;
                
                group->t_idxCS0[lid] = lcst_idx;
                group->periodicCS0[lid] = lcsperiodic;
                group->t_dataCS0[lid] = lcst_data;
                group->dt_dataCS0[lid] = lcsdt_data;
                group->ntCS0[lid] = lcsnt;
            }
            // test whether spike condition was fulfilled previously
            // calculate membrane potential
            
            lr = Isyn;
            
            // test for and register a true spike
            if (abs(lr - lr_event) >= (2.50000000000000000e-01f)) {
                const unsigned int spkIdx = atomicAdd(&shSpkCount, 1);
                shSpk[spkIdx] = lid;
                // spike reset code
                
                lr_prev_event = lr_event;
                lr_event = lr;
                
            }
            group->r[lid] = lr;
            group->r_event[lid] = lr_event;
            group->r_prev_event[lid] = lr_prev_event;
        }
        __syncthreads();
        if(threadIdx.x == 0) {
            if (shSpkCount > 0) {
                shPosSpk = atomicAdd(&group->spkCnt[0], shSpkCount);
            }
        }
        __syncthreads();
        if(threadIdx.x < shSpkCount) {
            const unsigned int n = shSpk[threadIdx.x];
            group->spk[shPosSpk + threadIdx.x] = n;
        }
    }
    // merged3
    if(id >= 576 && id < 1088) {
        struct MergedNeuronUpdateGroup3 *group = &d_mergedNeuronUpdateGroup3[0]; 
        const unsigned int lid = id - 576;
        
        if(lid < group->numNeurons) {
            scalar lr = group->r[lid];
            scalar lx = group->x[lid];
            scalar ldr = group->dr[lid];
            const scalar lb = group->b[lid];
            scalar ldb = group->db[lid];
            scalar lerr_fb = group->err_fb[lid];
            scalar lr_event = group->r_event[lid];
            scalar lr_prev_event = group->r_prev_event[lid];
            const float lprevST = group->prevST[lid];
            
            float Isyn = 0;
            scalar Isyn_err_fb = 0.00000000000000000e+00f;
             {
                // pull inSyn values in a coalesced access
                float linSyn = group->inSynInSyn0[lid];
                Isyn += linSyn; linSyn *= (1.0f - DT * (0.00000000000000000e+00f));
                
                group->inSynInSyn0[lid] = linSyn;
            }
             {
                // pull inSyn values in a coalesced access
                float linSyn = group->inSynInSyn1[lid];
                Isyn_err_fb += linSyn; linSyn *= (1.0f - DT * (0.00000000000000000e+00f));
                
                group->inSynInSyn1[lid] = linSyn;
            }
            // test whether spike condition was fulfilled previously
            // calculate membrane potential
            
            lerr_fb = Isyn_err_fb;
            
            lx += DT * (Isyn + lb - lx);
            //lx = Isyn + lb;
            
            lr = ((lx) < 0.0f ? 0.0f : (lx));
            ldr = ((lx) < 0.0f ? 0.0f : 1.0f);
            
            //lr_event *= (1.f - DT);
            
            ldb += lerr_fb * ldr;
            
            // test for and register a true spike
            if (abs(lr - lr_event) >= (1.00000000000000002e-02f)) {
                const unsigned int spkIdx = atomicAdd(&shSpkCount, 1);
                shSpk[spkIdx] = lid;
                // spike reset code
                
                lr_prev_event = lr_event;
                lr_event = lr;
                
            }
            group->r[lid] = lr;
            group->x[lid] = lx;
            group->dr[lid] = ldr;
            group->db[lid] = ldb;
            group->err_fb[lid] = lerr_fb;
            group->r_event[lid] = lr_event;
            group->r_prev_event[lid] = lr_prev_event;
        }
        __syncthreads();
        if(threadIdx.x == 0) {
            if (shSpkCount > 0) {
                shPosSpk = atomicAdd(&group->spkCnt[0], shSpkCount);
            }
        }
        __syncthreads();
        if(threadIdx.x < shSpkCount) {
            const unsigned int n = shSpk[threadIdx.x];
            group->spk[shPosSpk + threadIdx.x] = n;
        }
    }
}
void updateNeurons(float t, unsigned int recordingTimestep) {
     {
        const dim3 threads(32, 1);
        const dim3 grid(34, 1);
        neuronPrevSpikeTimeUpdateKernel<<<grid, threads>>>(t);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
     {
        const dim3 threads(32, 1);
        const dim3 grid(1, 1);
        neuronSpikeQueueUpdateKernel<<<grid, threads>>>();
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
     {
        const dim3 threads(32, 1);
        const dim3 grid(34, 1);
        updateNeuronsKernel<<<grid, threads>>>(t, recordingTimestep);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}
