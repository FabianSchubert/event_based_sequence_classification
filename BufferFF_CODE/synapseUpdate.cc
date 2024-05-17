#include "definitionsInternal.h"
#include "supportCode.h"

struct MergedPresynapticUpdateGroup0
 {
    float* inSyn;
    unsigned int* srcSpkCnt;
    unsigned int* srcSpk;
    scalar* rPre;
    scalar* r_prev_eventPre;
    scalar* drPost;
    scalar* errPost;
    float* prevSTPre;
    scalar* g;
    scalar* inp_prev;
    scalar* dg;
    scalar* dg_prev;
    scalar* t_prev;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    unsigned int rowStride;
    
}
;
struct MergedPresynapticUpdateGroup1
 {
    float* inSyn;
    unsigned int* srcSpkCnt;
    unsigned int* srcSpk;
    scalar* drPre;
    scalar* errPre;
    scalar* rPost;
    scalar* g;
    scalar* inp_prev;
    scalar* dg;
    scalar* dg_prev;
    scalar* t_prev;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    unsigned int rowStride;
    
}
;
struct MergedPresynapticUpdateGroup2
 {
    float* inSyn;
    unsigned int* srcSpkCnt;
    unsigned int* srcSpk;
    scalar* rPre;
    scalar* r_tracePre;
    scalar* dr_err_prodPost;
    float* prevSTPre;
    scalar* g;
    scalar* inp_prev;
    scalar* dg;
    scalar* dg_prev;
    scalar* t_prev;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    unsigned int rowStride;
    
}
;
__device__ __constant__ MergedPresynapticUpdateGroup0 d_mergedPresynapticUpdateGroup0[1];
void pushMergedPresynapticUpdateGroup0ToDevice(unsigned int idx, float* inSyn, unsigned int* srcSpkCnt, unsigned int* srcSpk, scalar* rPre, scalar* r_prev_eventPre, scalar* drPost, scalar* errPost, float* prevSTPre, scalar* g, scalar* inp_prev, scalar* dg, scalar* dg_prev, scalar* t_prev, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int rowStride) {
    MergedPresynapticUpdateGroup0 group = {inSyn, srcSpkCnt, srcSpk, rPre, r_prev_eventPre, drPost, errPost, prevSTPre, g, inp_prev, dg, dg_prev, t_prev, numSrcNeurons, numTrgNeurons, rowStride, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedPresynapticUpdateGroup0, &group, sizeof(MergedPresynapticUpdateGroup0), idx * sizeof(MergedPresynapticUpdateGroup0)));
}
__device__ __constant__ MergedPresynapticUpdateGroup1 d_mergedPresynapticUpdateGroup1[1];
void pushMergedPresynapticUpdateGroup1ToDevice(unsigned int idx, float* inSyn, unsigned int* srcSpkCnt, unsigned int* srcSpk, scalar* drPre, scalar* errPre, scalar* rPost, scalar* g, scalar* inp_prev, scalar* dg, scalar* dg_prev, scalar* t_prev, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int rowStride) {
    MergedPresynapticUpdateGroup1 group = {inSyn, srcSpkCnt, srcSpk, drPre, errPre, rPost, g, inp_prev, dg, dg_prev, t_prev, numSrcNeurons, numTrgNeurons, rowStride, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedPresynapticUpdateGroup1, &group, sizeof(MergedPresynapticUpdateGroup1), idx * sizeof(MergedPresynapticUpdateGroup1)));
}
__device__ __constant__ MergedPresynapticUpdateGroup2 d_mergedPresynapticUpdateGroup2[1];
void pushMergedPresynapticUpdateGroup2ToDevice(unsigned int idx, float* inSyn, unsigned int* srcSpkCnt, unsigned int* srcSpk, scalar* rPre, scalar* r_tracePre, scalar* dr_err_prodPost, float* prevSTPre, scalar* g, scalar* inp_prev, scalar* dg, scalar* dg_prev, scalar* t_prev, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int rowStride) {
    MergedPresynapticUpdateGroup2 group = {inSyn, srcSpkCnt, srcSpk, rPre, r_tracePre, dr_err_prodPost, prevSTPre, g, inp_prev, dg, dg_prev, t_prev, numSrcNeurons, numTrgNeurons, rowStride, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedPresynapticUpdateGroup2, &group, sizeof(MergedPresynapticUpdateGroup2), idx * sizeof(MergedPresynapticUpdateGroup2)));
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
__device__ __constant__ unsigned int d_mergedPresynapticUpdateGroupStartID0[] = {0, };
__device__ __constant__ unsigned int d_mergedPresynapticUpdateGroupStartID1[] = {32, };
__device__ __constant__ unsigned int d_mergedPresynapticUpdateGroupStartID2[] = {800, };
extern "C" __global__ void updatePresynapticKernel(float t)
 {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x; 
    __shared__ float shLg[32];
    __shared__ unsigned int shSpk[32];
    // merged0
    if(id < 32) {
        struct MergedPresynapticUpdateGroup0 *group = &d_mergedPresynapticUpdateGroup0[0]; 
        const unsigned int lid = id - 0;
        float linSyn = 0;
         {
            const unsigned int numSpikes = group->srcSpkCnt[0];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                __syncthreads();
                if (threadIdx.x < numSpikesInBlock) {
                    const unsigned int spk = group->srcSpk[(r * 32) + threadIdx.x];
                    shSpk[threadIdx.x] = spk;
                }
                __syncthreads();
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < group->rowStride) {
                        const unsigned int synAddress = (shSpk[j] * group->rowStride) + lid;
                        
                        linSyn += group->g[synAddress]*group->rPre[shSpk[j]]-group->inp_prev[synAddress];
                        group->inp_prev[synAddress] = group->g[synAddress] * group->rPre[shSpk[j]];
                        
                        group->dg[synAddress] += (t - max(0.0f, (2.50000000000000000e-01f + group->prevSTPre[shSpk[j]]))) * group->dg_prev[synAddress];
                        group->dg_prev[synAddress] = group->errPost[lid] * group->drPost[lid] * group->r_prev_eventPre[shSpk[j]] - (0.00000000000000000e+00f) * group->g[synAddress];
                    }
                }
            }
        }
        
        // only do this for existing neurons
        if (lid < group->numTrgNeurons) {
            group->inSyn[lid] += linSyn;
        }
    }
    // merged1
    if(id >= 32 && id < 800) {
        struct MergedPresynapticUpdateGroup1 *group = &d_mergedPresynapticUpdateGroup1[0]; 
        const unsigned int lid = id - 32;
        float linSyn = 0;
         {
            const unsigned int numSpikes = group->srcSpkCnt[0];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                __syncthreads();
                if (threadIdx.x < numSpikesInBlock) {
                    const unsigned int spk = group->srcSpk[(r * 32) + threadIdx.x];
                    shSpk[threadIdx.x] = spk;
                }
                __syncthreads();
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < group->rowStride) {
                        const unsigned int synAddress = (shSpk[j] * group->rowStride) + lid;
                        
                        linSyn += group->g[synAddress]*group->errPre[shSpk[j]]-group->inp_prev[synAddress];
                        group->inp_prev[synAddress] = group->g[synAddress] * group->errPre[shSpk[j]];
                        
                        //group->dg[synAddress] += (t - group->t_prev[synAddress]) * group->dg_prev[synAddress] / DT;
                        //group->dg_prev[synAddress] = group->rPost[lid] * group->errPre[shSpk[j]] * group->drPre[shSpk[j]] - (0.00000000000000000e+00f) * group->g[synAddress];
                        //group->t_prev[synAddress] = t;
                    }
                }
            }
        }
        
        // only do this for existing neurons
        if (lid < group->numTrgNeurons) {
            group->inSyn[lid] += linSyn;
        }
    }
    // merged2
    if(id >= 800 && id < 1568) {
        struct MergedPresynapticUpdateGroup2 *group = &d_mergedPresynapticUpdateGroup2[0]; 
        const unsigned int lid = id - 800;
        float linSyn = 0;
         {
            const unsigned int numSpikes = group->srcSpkCnt[0];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                __syncthreads();
                if (threadIdx.x < numSpikesInBlock) {
                    const unsigned int spk = group->srcSpk[(r * 32) + threadIdx.x];
                    shSpk[threadIdx.x] = spk;
                }
                __syncthreads();
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < group->rowStride) {
                        const unsigned int synAddress = (shSpk[j] * group->rowStride) + lid;
                        
                        //const scalar input_new = group->g[synAddress] * group->rPre[shSpk[j]];
                        linSyn += group->g[synAddress]*group->rPre[shSpk[j]]-group->inp_prev[synAddress];
                        group->inp_prev[synAddress] = group->g[synAddress] * group->rPre[shSpk[j]];
                        
                        group->dg[synAddress] += (t - max(0.0f, (2.50000000000000000e-01f + group->prevSTPre[shSpk[j]]))) * group->dg_prev[synAddress];
                        group->dg_prev[synAddress] = group->dr_err_prodPost[lid] * group->r_tracePre[shSpk[j]] - group->g[synAddress] * (0.00000000000000000e+00f);
                    }
                }
            }
        }
        
        // only do this for existing neurons
        if (lid < group->numTrgNeurons) {
            group->inSyn[lid] += linSyn;
        }
    }
}
void updateSynapses(float t) {
     {
        const dim3 threads(32, 1);
        const dim3 grid(49, 1);
        updatePresynapticKernel<<<grid, threads>>>(t);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}
