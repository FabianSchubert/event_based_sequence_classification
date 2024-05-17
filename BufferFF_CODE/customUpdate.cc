#include "definitionsInternal.h"

struct MergedCustomUpdateGroup0
 {
    scalar* Val;
    scalar* MaxVal;
    scalar* SumExpVal;
    scalar* SoftmaxVal;
    unsigned int size;
    
}
;
struct MergedCustomUpdateGroup1
 {
    scalar* SumExpVal;
    scalar* Val;
    scalar* MaxVal;
    unsigned int size;
    
}
;
struct MergedCustomUpdateGroup2
 {
    scalar* MaxVal;
    scalar* Val;
    unsigned int size;
    
}
;
struct MergedCustomUpdateGroup3
 {
    scalar* reducedChange;
    scalar* change;
    unsigned int size;
    
}
;
struct MergedCustomUpdateGroup4
 {
    scalar* m;
    scalar* v;
    scalar* time;
    scalar* change;
    scalar* variable;
    unsigned int size;
    
}
;
struct MergedCustomUpdateWUGroup0
 {
    scalar* reducedChange;
    scalar* change;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    unsigned int rowStride;
    
}
;
struct MergedCustomUpdateWUGroup1
 {
    scalar* m;
    scalar* v;
    scalar* time;
    scalar* change;
    scalar* variable;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    unsigned int rowStride;
    
}
;
__device__ __constant__ MergedCustomUpdateGroup0 d_mergedCustomUpdateGroup0[1];
void pushMergedCustomUpdateGroup0ToDevice(unsigned int idx, scalar* Val, scalar* MaxVal, scalar* SumExpVal, scalar* SoftmaxVal, unsigned int size) {
    MergedCustomUpdateGroup0 group = {Val, MaxVal, SumExpVal, SoftmaxVal, size, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomUpdateGroup0, &group, sizeof(MergedCustomUpdateGroup0), idx * sizeof(MergedCustomUpdateGroup0)));
}
__device__ __constant__ MergedCustomUpdateGroup1 d_mergedCustomUpdateGroup1[1];
void pushMergedCustomUpdateGroup1ToDevice(unsigned int idx, scalar* SumExpVal, scalar* Val, scalar* MaxVal, unsigned int size) {
    MergedCustomUpdateGroup1 group = {SumExpVal, Val, MaxVal, size, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomUpdateGroup1, &group, sizeof(MergedCustomUpdateGroup1), idx * sizeof(MergedCustomUpdateGroup1)));
}
__device__ __constant__ MergedCustomUpdateGroup2 d_mergedCustomUpdateGroup2[1];
void pushMergedCustomUpdateGroup2ToDevice(unsigned int idx, scalar* MaxVal, scalar* Val, unsigned int size) {
    MergedCustomUpdateGroup2 group = {MaxVal, Val, size, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomUpdateGroup2, &group, sizeof(MergedCustomUpdateGroup2), idx * sizeof(MergedCustomUpdateGroup2)));
}
__device__ __constant__ MergedCustomUpdateGroup3 d_mergedCustomUpdateGroup3[2];
void pushMergedCustomUpdateGroup3ToDevice(unsigned int idx, scalar* reducedChange, scalar* change, unsigned int size) {
    MergedCustomUpdateGroup3 group = {reducedChange, change, size, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomUpdateGroup3, &group, sizeof(MergedCustomUpdateGroup3), idx * sizeof(MergedCustomUpdateGroup3)));
}
__device__ __constant__ MergedCustomUpdateGroup4 d_mergedCustomUpdateGroup4[2];
void pushMergedCustomUpdateGroup4ToDevice(unsigned int idx, scalar* m, scalar* v, scalar* time, scalar* change, scalar* variable, unsigned int size) {
    MergedCustomUpdateGroup4 group = {m, v, time, change, variable, size, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomUpdateGroup4, &group, sizeof(MergedCustomUpdateGroup4), idx * sizeof(MergedCustomUpdateGroup4)));
}
__device__ __constant__ MergedCustomUpdateWUGroup0 d_mergedCustomUpdateWUGroup0[3];
void pushMergedCustomUpdateWUGroup0ToDevice(unsigned int idx, scalar* reducedChange, scalar* change, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int rowStride) {
    MergedCustomUpdateWUGroup0 group = {reducedChange, change, numSrcNeurons, numTrgNeurons, rowStride, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomUpdateWUGroup0, &group, sizeof(MergedCustomUpdateWUGroup0), idx * sizeof(MergedCustomUpdateWUGroup0)));
}
__device__ __constant__ MergedCustomUpdateWUGroup1 d_mergedCustomUpdateWUGroup1[3];
void pushMergedCustomUpdateWUGroup1ToDevice(unsigned int idx, scalar* m, scalar* v, scalar* time, scalar* change, scalar* variable, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int rowStride) {
    MergedCustomUpdateWUGroup1 group = {m, v, time, change, variable, numSrcNeurons, numTrgNeurons, rowStride, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedCustomUpdateWUGroup1, &group, sizeof(MergedCustomUpdateWUGroup1), idx * sizeof(MergedCustomUpdateWUGroup1)));
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
__device__ __constant__ unsigned int d_mergedCustomUpdateGroupStartID3[] = {0, 768, };
extern "C" __global__ void customUpdateBiasChangeBatchReduce(float t)
 {
    const unsigned int id = 64 * blockIdx.x + threadIdx.x; 
    // ------------------------------------------------------------------------
    // Custom updates
    // merged3
    if(id < 832) {
        unsigned int lo = 0;
        unsigned int hi = 2;
        while(lo < hi)
         {
            const unsigned int mid = (lo + hi) / 2;
            if(id < d_mergedCustomUpdateGroupStartID3[mid]) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        struct MergedCustomUpdateGroup3 *group = &d_mergedCustomUpdateGroup3[lo - 1]; 
        const unsigned int groupStartID = d_mergedCustomUpdateGroupStartID3[lo - 1];
        const unsigned int lid = id - groupStartID;
        // only do this for existing neurons
        if(lid < group->size) {
            scalar lrreducedChange = 0;
            for(unsigned int batch = 0; batch < 1; batch++) {
                scalar lreducedChange;
                scalar lchange = group->change[lid];
                
                lreducedChange = lchange;
                lchange = 0.0f;
                group->change[lid] = lchange;
                lrreducedChange += lreducedChange;
            }
            group->reducedChange[lid] = lrreducedChange;
        }
    }
    // ------------------------------------------------------------------------
    // Custom WU updates
}
void updateBiasChangeBatchReduce() {
     {
        const dim3 threads(64, 1);
        const dim3 grid(13, 1);
        customUpdateBiasChangeBatchReduce<<<grid, threads>>>(t);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}
__device__ __constant__ unsigned int d_mergedCustomUpdateGroupStartID4[] = {0, 768, };
__device__ __constant__ unsigned int d_mergedCustomUpdateWUGroupStartID1[] = {832, 375872, 384128, };
extern "C" __global__ void customUpdatePlast(float t)
 {
    const unsigned int id = 64 * blockIdx.x + threadIdx.x; 
    // ------------------------------------------------------------------------
    // Custom updates
    // merged4
    if(id < 832) {
        unsigned int lo = 0;
        unsigned int hi = 2;
        while(lo < hi)
         {
            const unsigned int mid = (lo + hi) / 2;
            if(id < d_mergedCustomUpdateGroupStartID4[mid]) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        struct MergedCustomUpdateGroup4 *group = &d_mergedCustomUpdateGroup4[lo - 1]; 
        const unsigned int groupStartID = d_mergedCustomUpdateGroupStartID4[lo - 1];
        const unsigned int lid = id - groupStartID;
        // only do this for existing neurons
        if(lid < group->size) {
            scalar lm = group->m[lid];
            scalar lv = group->v[lid];
            scalar ltime = group->time[lid];
            scalar lchange = group->change[lid];
            scalar lvariable = group->variable[lid];
            
            const scalar change_norm = lchange / (1.00000000000000000e+00f);
            lm = (9.00000000000000022e-01f) * lm + (1.0f - (9.00000000000000022e-01f)) * change_norm;
            lv = (9.98999999999999999e-01f) * lv + (1.0f - (9.98999999999999999e-01f)) * change_norm * change_norm;
            const scalar m_hat = lm/(1.0f - pow((9.00000000000000022e-01f), ltime));
            const scalar v_hat = lv/(1.0f - pow((9.98999999999999999e-01f), ltime));
            //lvariable += (2.99999999999999974e-04f) * lm / (sqrt(lv) +  (9.99999999999999955e-08f));
            lvariable += (2.99999999999999974e-04f) * m_hat / (sqrt(v_hat) +  (9.99999999999999955e-08f));
            ltime += 1.0f;
            group->m[lid] = lm;
            group->v[lid] = lv;
            group->time[lid] = ltime;
            group->change[lid] = lchange;
            group->variable[lid] = lvariable;
        }
    }
    // ------------------------------------------------------------------------
    // Custom WU updates
    // merged1
    if(id >= 832 && id < 392384) {
        unsigned int lo = 0;
        unsigned int hi = 3;
        while(lo < hi)
         {
            const unsigned int mid = (lo + hi) / 2;
            if(id < d_mergedCustomUpdateWUGroupStartID1[mid]) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        struct MergedCustomUpdateWUGroup1 *group = &d_mergedCustomUpdateWUGroup1[lo - 1]; 
        const unsigned int groupStartID = d_mergedCustomUpdateWUGroupStartID1[lo - 1];
        const unsigned int lid = id - groupStartID;
        const unsigned int size = group->numSrcNeurons * group->rowStride;
        if (lid < size) {
            scalar lm = group->m[lid];
            scalar lv = group->v[lid];
            scalar ltime = group->time[lid];
            scalar lchange = group->change[lid];
            scalar lvariable = group->variable[lid];
            
            const scalar change_norm = lchange / (1.00000000000000000e+00f);
            lm = (9.00000000000000022e-01f) * lm + (1.0f - (9.00000000000000022e-01f)) * change_norm;
            lv = (9.98999999999999999e-01f) * lv + (1.0f - (9.98999999999999999e-01f)) * change_norm * change_norm;
            const scalar m_hat = lm/(1.0f - pow((9.00000000000000022e-01f), ltime));
            const scalar v_hat = lv/(1.0f - pow((9.98999999999999999e-01f), ltime));
            //lvariable += (2.99999999999999974e-04f) * lm / (sqrt(lv) +  (9.99999999999999955e-08f));
            lvariable += (2.99999999999999974e-04f) * m_hat / (sqrt(v_hat) +  (9.99999999999999955e-08f));
            ltime += 1.0f;
            group->m[lid] = lm;
            group->v[lid] = lv;
            group->time[lid] = ltime;
            group->change[lid] = lchange;
            group->variable[lid] = lvariable;
        }
    }
}
void updatePlast() {
     {
        const dim3 threads(64, 1);
        const dim3 grid(6131, 1);
        customUpdatePlast<<<grid, threads>>>(t);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}
__device__ __constant__ unsigned int d_mergedCustomUpdateWUGroupStartID0[] = {0, 375040, 383296, };
extern "C" __global__ void customUpdateWeightChangeBatchReduce(float t)
 {
    const unsigned int id = 64 * blockIdx.x + threadIdx.x; 
    // ------------------------------------------------------------------------
    // Custom updates
    // ------------------------------------------------------------------------
    // Custom WU updates
    // merged0
    if(id < 391552) {
        unsigned int lo = 0;
        unsigned int hi = 3;
        while(lo < hi)
         {
            const unsigned int mid = (lo + hi) / 2;
            if(id < d_mergedCustomUpdateWUGroupStartID0[mid]) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        struct MergedCustomUpdateWUGroup0 *group = &d_mergedCustomUpdateWUGroup0[lo - 1]; 
        const unsigned int groupStartID = d_mergedCustomUpdateWUGroupStartID0[lo - 1];
        const unsigned int lid = id - groupStartID;
        const unsigned int size = group->numSrcNeurons * group->rowStride;
        if (lid < size) {
            scalar lrreducedChange = 0;
            for(unsigned int batch = 0; batch < 1; batch++) {
                scalar lreducedChange;
                scalar lchange = group->change[lid];
                
                lreducedChange = lchange;
                lchange = 0.0f;
                group->change[lid] = lchange;
                lrreducedChange += lreducedChange;
            }
            group->reducedChange[lid] = lrreducedChange;
        }
    }
}
void updateWeightChangeBatchReduce() {
     {
        const dim3 threads(64, 1);
        const dim3 grid(6118, 1);
        customUpdateWeightChangeBatchReduce<<<grid, threads>>>(t);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}
__device__ __constant__ unsigned int d_mergedCustomUpdateGroupStartID2[] = {0, };
extern "C" __global__ void customUpdatesoftmax1(float t)
 {
    const unsigned int id = 64 * blockIdx.x + threadIdx.x; 
    // ------------------------------------------------------------------------
    // Custom updates
    // merged2
    if(id < 64) {
        struct MergedCustomUpdateGroup2 *group = &d_mergedCustomUpdateGroup2[0]; 
        const unsigned int lid = id - 0;
        // only do this for existing neurons
        if(lid < 32) {
            const unsigned int lane = lid % 32;
            const unsigned int batch = lid / 32;
            scalar lrMaxVal = -3.402823466e+38;
            for(unsigned int idx = lane; idx < group->size; idx += 32) {
                scalar lMaxVal;
                const scalar lVal = group->Val[idx];
                
                lMaxVal = lVal;
                lrMaxVal = fmax(lrMaxVal, lMaxVal);
            }
            lrMaxVal = fmax(lrMaxVal, __shfl_down_sync(0xFFFFFFFF, lrMaxVal, 16));
            lrMaxVal = fmax(lrMaxVal, __shfl_down_sync(0xFFFFFFFF, lrMaxVal, 8));
            lrMaxVal = fmax(lrMaxVal, __shfl_down_sync(0xFFFFFFFF, lrMaxVal, 4));
            lrMaxVal = fmax(lrMaxVal, __shfl_down_sync(0xFFFFFFFF, lrMaxVal, 2));
            lrMaxVal = fmax(lrMaxVal, __shfl_down_sync(0xFFFFFFFF, lrMaxVal, 1));
            if(lane == 0) {
                group->MaxVal[0] = lrMaxVal;
            }
        }
    }
    // ------------------------------------------------------------------------
    // Custom WU updates
}
void updatesoftmax1() {
     {
        const dim3 threads(64, 1);
        const dim3 grid(1, 1);
        customUpdatesoftmax1<<<grid, threads>>>(t);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}
__device__ __constant__ unsigned int d_mergedCustomUpdateGroupStartID1[] = {0, };
extern "C" __global__ void customUpdatesoftmax2(float t)
 {
    const unsigned int id = 64 * blockIdx.x + threadIdx.x; 
    // ------------------------------------------------------------------------
    // Custom updates
    // merged1
    if(id < 64) {
        struct MergedCustomUpdateGroup1 *group = &d_mergedCustomUpdateGroup1[0]; 
        const unsigned int lid = id - 0;
        // only do this for existing neurons
        if(lid < 32) {
            const unsigned int lane = lid % 32;
            const unsigned int batch = lid / 32;
            scalar lrSumExpVal = 0;
            for(unsigned int idx = lane; idx < group->size; idx += 32) {
                scalar lSumExpVal;
                const scalar lVal = group->Val[idx];
                const scalar lMaxVal = group->MaxVal[0];
                
                lSumExpVal = exp(lVal - lMaxVal);
                lrSumExpVal += lSumExpVal;
            }
            lrSumExpVal += __shfl_down_sync(0xFFFFFFFF, lrSumExpVal, 16);
            lrSumExpVal += __shfl_down_sync(0xFFFFFFFF, lrSumExpVal, 8);
            lrSumExpVal += __shfl_down_sync(0xFFFFFFFF, lrSumExpVal, 4);
            lrSumExpVal += __shfl_down_sync(0xFFFFFFFF, lrSumExpVal, 2);
            lrSumExpVal += __shfl_down_sync(0xFFFFFFFF, lrSumExpVal, 1);
            if(lane == 0) {
                group->SumExpVal[0] = lrSumExpVal;
            }
        }
    }
    // ------------------------------------------------------------------------
    // Custom WU updates
}
void updatesoftmax2() {
     {
        const dim3 threads(64, 1);
        const dim3 grid(1, 1);
        customUpdatesoftmax2<<<grid, threads>>>(t);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}
__device__ __constant__ unsigned int d_mergedCustomUpdateGroupStartID0[] = {0, };
extern "C" __global__ void customUpdatesoftmax3(float t)
 {
    const unsigned int id = 64 * blockIdx.x + threadIdx.x; 
    // ------------------------------------------------------------------------
    // Custom updates
    // merged0
    if(id < 64) {
        struct MergedCustomUpdateGroup0 *group = &d_mergedCustomUpdateGroup0[0]; 
        const unsigned int lid = id - 0;
        // only do this for existing neurons
        if(lid < group->size) {
            const scalar lVal = group->Val[lid];
            const scalar lMaxVal = group->MaxVal[0];
            const scalar lSumExpVal = group->SumExpVal[0];
            scalar lSoftmaxVal = group->SoftmaxVal[lid];
            
            lSoftmaxVal = exp(lVal - lMaxVal) / lSumExpVal;
            group->SoftmaxVal[lid] = lSoftmaxVal;
        }
    }
    // ------------------------------------------------------------------------
    // Custom WU updates
}
void updatesoftmax3() {
     {
        const dim3 threads(64, 1);
        const dim3 grid(1, 1);
        customUpdatesoftmax3<<<grid, threads>>>(t);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}
