#include "definitionsInternal.h"
#include "supportCode.h"

struct MergedNeuronUpdateGroup0
 {
    scalar* b;
    uint32_t* recordSpk;
    scalar* dataCS0;
    scalar* t_dataCS0;
    scalar* dt_dataCS0;
    int* nt_dataCS0;
    int* periodicCS0;
    int* t_idxCS0;
    float* inSynInSyn0;
    scalar* loss;
    scalar* db;
    scalar* err_prev_event;
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
struct MergedNeuronUpdateGroup1
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
    scalar* dataCS0;
    uint32_t* recordSpk;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronUpdateGroup2
 {
    scalar* db;
    uint32_t* recordSpk;
    float* inSynInSyn1;
    float* inSynInSyn0;
    scalar* weight_factor;
    scalar* dr_err_prod;
    scalar* r_prev_event;
    scalar* r_event;
    scalar* err_fb;
    scalar* b;
    scalar* dr;
    scalar* x;
    scalar* r;
    float* prevST;
    unsigned int* spk;
    unsigned int* spkCnt;
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
__device__ __constant__ MergedNeuronSpikeQueueUpdateGroup0 d_mergedNeuronSpikeQueueUpdateGroup0[3];
void pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(unsigned int idx, unsigned int* spkCnt) {
    MergedNeuronSpikeQueueUpdateGroup0 group = {spkCnt, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronSpikeQueueUpdateGroup0, &group, sizeof(MergedNeuronSpikeQueueUpdateGroup0), idx * sizeof(MergedNeuronSpikeQueueUpdateGroup0)));
}
__device__ __constant__ MergedNeuronPrevSpikeTimeUpdateGroup0 d_mergedNeuronPrevSpikeTimeUpdateGroup0[2];
void pushMergedNeuronPrevSpikeTimeUpdateGroup0ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, float* prevST, unsigned int numNeurons) {
    MergedNeuronPrevSpikeTimeUpdateGroup0 group = {spkCnt, spk, prevST, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronPrevSpikeTimeUpdateGroup0, &group, sizeof(MergedNeuronPrevSpikeTimeUpdateGroup0), idx * sizeof(MergedNeuronPrevSpikeTimeUpdateGroup0)));
}
__device__ __constant__ MergedNeuronUpdateGroup0 d_mergedNeuronUpdateGroup0[1];
void pushMergedNeuronUpdateGroup0ToDevice(unsigned int idx, scalar* b, uint32_t* recordSpk, scalar* dataCS0, scalar* t_dataCS0, scalar* dt_dataCS0, int* nt_dataCS0, int* periodicCS0, int* t_idxCS0, float* inSynInSyn0, scalar* loss, scalar* db, scalar* err_prev_event, scalar* err_event, scalar* err, scalar* targ, scalar* dr, scalar* r, scalar* x, unsigned int* spk, unsigned int* spkCnt, unsigned int numNeurons) {
    MergedNeuronUpdateGroup0 group = {b, recordSpk, dataCS0, t_dataCS0, dt_dataCS0, nt_dataCS0, periodicCS0, t_idxCS0, inSynInSyn0, loss, db, err_prev_event, err_event, err, targ, dr, r, x, spk, spkCnt, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup0, &group, sizeof(MergedNeuronUpdateGroup0), idx * sizeof(MergedNeuronUpdateGroup0)));
}
__device__ __constant__ MergedNeuronUpdateGroup1 d_mergedNeuronUpdateGroup1[1];
void pushMergedNeuronUpdateGroup1ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, float* prevST, scalar* r, scalar* r_event, scalar* r_prev_event, scalar* r_trace, int* t_idxCS0, int* periodicCS0, int* nt_dataCS0, scalar* dt_dataCS0, scalar* t_dataCS0, scalar* dataCS0, uint32_t* recordSpk, unsigned int numNeurons) {
    MergedNeuronUpdateGroup1 group = {spkCnt, spk, prevST, r, r_event, r_prev_event, r_trace, t_idxCS0, periodicCS0, nt_dataCS0, dt_dataCS0, t_dataCS0, dataCS0, recordSpk, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup1, &group, sizeof(MergedNeuronUpdateGroup1), idx * sizeof(MergedNeuronUpdateGroup1)));
}
__device__ __constant__ MergedNeuronUpdateGroup2 d_mergedNeuronUpdateGroup2[1];
void pushMergedNeuronUpdateGroup2ToDevice(unsigned int idx, scalar* db, uint32_t* recordSpk, float* inSynInSyn1, float* inSynInSyn0, scalar* weight_factor, scalar* dr_err_prod, scalar* r_prev_event, scalar* r_event, scalar* err_fb, scalar* b, scalar* dr, scalar* x, scalar* r, float* prevST, unsigned int* spk, unsigned int* spkCnt, unsigned int numNeurons) {
    MergedNeuronUpdateGroup2 group = {db, recordSpk, inSynInSyn1, inSynInSyn0, weight_factor, dr_err_prod, r_prev_event, r_event, err_fb, b, dr, x, r, prevST, spk, spkCnt, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup2, &group, sizeof(MergedNeuronUpdateGroup2), idx * sizeof(MergedNeuronUpdateGroup2)));
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
void pushMergedNeuronUpdate0dataCS0ToDevice(unsigned int idx, scalar* value) {
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup0, &value, sizeof(value), (sizeof(MergedNeuronUpdateGroup0) * (idx)) + offsetof(MergedNeuronUpdateGroup0, dataCS0)));
}

void pushMergedNeuronUpdate0recordSpkToDevice(unsigned int idx, uint32_t* value) {
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup0, &value, sizeof(value), (sizeof(MergedNeuronUpdateGroup0) * (idx)) + offsetof(MergedNeuronUpdateGroup0, recordSpk)));
}

void pushMergedNeuronUpdate1dataCS0ToDevice(unsigned int idx, scalar* value) {
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup1, &value, sizeof(value), (sizeof(MergedNeuronUpdateGroup1) * (idx)) + offsetof(MergedNeuronUpdateGroup1, dataCS0)));
}

void pushMergedNeuronUpdate1recordSpkToDevice(unsigned int idx, uint32_t* value) {
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup1, &value, sizeof(value), (sizeof(MergedNeuronUpdateGroup1) * (idx)) + offsetof(MergedNeuronUpdateGroup1, recordSpk)));
}

void pushMergedNeuronUpdate2recordSpkToDevice(unsigned int idx, uint32_t* value) {
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup2, &value, sizeof(value), (sizeof(MergedNeuronUpdateGroup2) * (idx)) + offsetof(MergedNeuronUpdateGroup2, recordSpk)));
}

__device__ __constant__ unsigned int d_mergedNeuronUpdateGroupStartID0[] = {0, };
__device__ __constant__ unsigned int d_mergedNeuronUpdateGroupStartID1[] = {32, };
__device__ __constant__ unsigned int d_mergedNeuronUpdateGroupStartID2[] = {544, };
__device__ __constant__ unsigned int d_mergedNeuronPrevSpikeTimeUpdateGroupStartID0[] = {0, 768, };

extern "C" __global__ void neuronPrevSpikeTimeUpdateKernel(float t) {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x;
    // merged0
    if(id < 1280) {
        unsigned int lo = 0;
        unsigned int hi = 2;
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
    if(id < 3) {
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
    if(id < 32) {
        struct MergedNeuronUpdateGroup0 *group = &d_mergedNeuronUpdateGroup0[0]; 
        const unsigned int lid = id - 0;
        
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
                int lcsnt_data = group->nt_dataCS0[lid];
                scalar lcsdt_data = group->dt_dataCS0[lid];
                scalar lcst_data = group->t_dataCS0[lid];
                
                const int nt_data_max_int = int((1.59500000000000000e+03f));
                const int dims_data_int = int((1.10000000000000000e+01f));
                const int t_buffer_int = int((1.00000000000000000e+00f));
                
                if(lcsperiodic){
                    lcst_idx = int(lcst_data / lcsdt_data) % lcsnt_data;
                } else {
                    lcst_idx = min(int(lcst_data / lcsdt_data), lcsnt_data-1);
                }
                
                lcst_data += DT;
                
                const int t_access = max(0, lcst_idx - (int)(lid)%t_buffer_int);
                
                const int dim_id = lid / t_buffer_int;
                const int index_access = 0*nt_data_max_int*dims_data_int + t_access * dims_data_int + dim_id;
                
                const scalar datapoint = group->dataCS0[index_access];
                
                //const scalar datapoint = 0.0f;
                
                Isyn += datapoint;
                
                group->t_idxCS0[lid] = lcst_idx;
                group->periodicCS0[lid] = lcsperiodic;
                group->nt_dataCS0[lid] = lcsnt_data;
                group->dt_dataCS0[lid] = lcsdt_data;
                group->t_dataCS0[lid] = lcst_data;
            }
            // test whether spike condition was fulfilled previously
            // calculate membrane potential
            
            //lx += DT * (Isyn_net + lb - lx);
            lx = Isyn_net + lb;
            //ltarg += DT * (Isyn - ltarg);
            ltarg = Isyn;
            
            //lr = max(0.0f, lx);
            //lr = min(1.0f, max(0.0f, lx));
            lr = 1.f/(1.f + exp(-lx));
            //lr = max(0.0f, tanh(lx));
            //ldr = ((lx > 0.0f) && (lx < 1.0f) ? 1.0f : 0.0f);
            //ldr = (lx > 0.0f ? 1.0f : 0.0f);
            ldr = lr * (1.f - lr);
            //ldr = lx < 0.0f ? 0.0f : (1.0f - tanh(lx)*tanh(lx));
            
            lerr = ltarg - lr;
            lloss += 0.5f * lerr * lerr;
            
            ldb += lerr * ldr;
            
            // test for and register a true spike
            if (abs(lerr - lerr_event) >= (2.50000000000000014e-02f)) {
                const unsigned int spkIdx = atomicAdd(&shSpkCount, 1);
                shSpk[spkIdx] = lid;
                atomicOr(&shSpkRecord, 1 << threadIdx.x);
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
        if(threadIdx.x < 1) {
            const unsigned int numRecordingWords = (group->numNeurons + 31) / 32;
            const unsigned int popWordIdx = (lid / 32) + threadIdx.x;
            if(popWordIdx < numRecordingWords) {
                group->recordSpk[(recordingTimestep * numRecordingWords * 1) + popWordIdx] = shSpkRecord;
            }
        }
    }
    // merged1
    if(id >= 32 && id < 544) {
        struct MergedNeuronUpdateGroup1 *group = &d_mergedNeuronUpdateGroup1[0]; 
        const unsigned int lid = id - 32;
        
        if(lid < group->numNeurons) {
            scalar lr = group->r[lid];
            scalar lr_event = group->r_event[lid];
            scalar lr_prev_event = group->r_prev_event[lid];
            scalar lr_trace = group->r_trace[lid];
            const float lprevST = group->prevST[lid];
            
            float Isyn = 0;
            // current source 0
             {
                int lcst_idx = group->t_idxCS0[lid];
                int lcsperiodic = group->periodicCS0[lid];
                int lcsnt_data = group->nt_dataCS0[lid];
                scalar lcsdt_data = group->dt_dataCS0[lid];
                scalar lcst_data = group->t_dataCS0[lid];
                
                const int nt_data_max_int = int((1.59500000000000000e+03f));
                const int dims_data_int = int((5.00000000000000000e+02f));
                const int t_buffer_int = int((1.00000000000000000e+00f));
                
                if(lcsperiodic){
                    lcst_idx = int(lcst_data / lcsdt_data) % lcsnt_data;
                } else {
                    lcst_idx = min(int(lcst_data / lcsdt_data), lcsnt_data-1);
                }
                
                lcst_data += DT;
                
                const int t_access = max(0, lcst_idx - (int)(lid)%t_buffer_int);
                
                const int dim_id = lid / t_buffer_int;
                const int index_access = 0*nt_data_max_int*dims_data_int + t_access * dims_data_int + dim_id;
                
                const scalar datapoint = group->dataCS0[index_access];
                
                //const scalar datapoint = 0.0f;
                
                Isyn += datapoint;
                
                group->t_idxCS0[lid] = lcst_idx;
                group->periodicCS0[lid] = lcsperiodic;
                group->nt_dataCS0[lid] = lcsnt_data;
                group->dt_dataCS0[lid] = lcsdt_data;
                group->t_dataCS0[lid] = lcst_data;
            }
            // test whether spike condition was fulfilled previously
            // calculate membrane potential
            
            lr = Isyn;
            lr_trace += DT * (lr - lr_trace) / (2.50000000000000000e+00f);
            
            // test for and register a true spike
            if (abs(lr - lr_event) >= (5.00000000000000000e-01f)) {
                const unsigned int spkIdx = atomicAdd(&shSpkCount, 1);
                shSpk[spkIdx] = lid;
                atomicOr(&shSpkRecord, 1 << threadIdx.x);
                // spike reset code
                
                lr_prev_event = lr_event;
                lr_event = lr;
                
            }
            group->r[lid] = lr;
            group->r_event[lid] = lr_event;
            group->r_prev_event[lid] = lr_prev_event;
            group->r_trace[lid] = lr_trace;
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
    // merged2
    if(id >= 544 && id < 1312) {
        struct MergedNeuronUpdateGroup2 *group = &d_mergedNeuronUpdateGroup2[0]; 
        const unsigned int lid = id - 544;
        
        if(lid < group->numNeurons) {
            scalar lr = group->r[lid];
            scalar lx = group->x[lid];
            scalar ldr = group->dr[lid];
            const scalar lb = group->b[lid];
            scalar ldb = group->db[lid];
            scalar lerr_fb = group->err_fb[lid];
            scalar lr_event = group->r_event[lid];
            scalar lr_prev_event = group->r_prev_event[lid];
            scalar ldr_err_prod = group->dr_err_prod[lid];
            scalar lweight_factor = group->weight_factor[lid];
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
            
            lx += DT * (Isyn + lb - lx) / (2.50000000000000000e+00f);
            //lx = Isyn + lb;
            const scalar r_prev = lr;
            lr = ((lx) < 0.0f ? 0.0f : (lx));
            ldr = ((lx) < 0.0f ? 0.0f : 1.0f);
            //lerr_fb += 1.1f * (r_prev - lr);
            lerr_fb -= 0.005f * (lr > 0.0f);
            //lr_event *= (1.f - DT);
            
            ldb += lerr_fb * ldr;
            
            ldr_err_prod = ldr * lerr_fb;
            
            // test for and register a true spike
            if (abs(lr - lr_event) >= ((2.50000000000000000e-01f) / lweight_factor)) {
                const unsigned int spkIdx = atomicAdd(&shSpkCount, 1);
                shSpk[spkIdx] = lid;
                atomicOr(&shSpkRecord, 1 << threadIdx.x);
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
            group->dr_err_prod[lid] = ldr_err_prod;
            group->weight_factor[lid] = lweight_factor;
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
}
void updateNeurons(float t, unsigned int recordingTimestep) {
     {
        const dim3 threads(32, 1);
        const dim3 grid(40, 1);
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
        const dim3 grid(41, 1);
        updateNeuronsKernel<<<grid, threads>>>(t, recordingTimestep);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}
