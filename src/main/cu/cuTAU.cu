/*
 *  Copyright 2016 Alexander Terenin
 *
 *  Licensed under the Apache License, Version 2.0 (the "License")
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 * /
 */

#include <curand_kernel.h>

/*
 * Function             : cuda_tauSqInv
 * Purpose              : calculates the relevant parameters for tau^-2, and
                          draws one sample from G(a,b) distribution with a large by
                          implementing the rejection sampler in Cheng (1977) for a set of
                          parallel threads in one block and taking one of the samples
                          that wasn't rejected. Note that to work properly,
                          blockDim.x and size of shared memory should be equal.
 * Argument *state      : pointer to random number generator
 * Argument *tauSqInv   : pointer to tauSqInv output and LB input (used in calculating rate)
 * Argument *alphaG     : pointer to shape parameter
 * Argument *xiInv      : pointer to xiInv (used in calculating rate)
 * Output               : mutates tauSqInv and stores result in its place
 */
extern "C"
__global__ void cuda_tauSqInv(curandStatePhilox4_32_10_t *globalState, float *tauSqInv, float *alphaG, float *xiInv) {
  extern __shared__ float acc[]; //store accepted proposals in shared memory
  __shared__ int success[1]; //store flag value indicating whether proposal was accepted in shared memory
  if(threadIdx.x == 0)
    success[1] = 0; //initialize success

  if(threadIdx.x < blockDim.x && blockIdx.x == 0) {
    acc[threadIdx.x] = 0.0f;
    //copy parameters to local memory
    float alpha = alphaG[0];
    float LB = tauSqInv[0];

    //copy RNG to local memory, skip to new sequence (avoid overlap with z), and
    curandStatePhilox4_32_10_t state = globalState[0]; //copy random number generator state to local memory
    skipahead((unsigned long long) (6*threadIdx.x), &state); //give each thread its own pseudorandom subsequence

    //compute rate parameter
    float beta = xiInv[0] + (0.5 * LB);

    //compute constants
    float a = rsqrtf(2.0f * alpha - 1.0f);
    float b = alpha - 1.3862944f; //log(4) = 1.3862944f
    float c = alpha + (1.0f / a);

    //perform rejection sampling
    while(success[0] == 0) {
      //compute uniforms
      float u1 = curand_uniform(&state); //one uniform for proposal
      float u2 = curand_uniform(&state); //one uniform for accept/reject step

      //compute proposal-dependant constants
      float v = a * logf(u1 / (1.0f - u1));
      float x = alpha * expf(v);

      //perform accept/reject
      if( (b + (c*v) - x) > logf(u1 * u1 * u2) ) {
        acc[threadIdx.x] = x;
      }

      __syncthreads();

      //find accepted value on thread 0
      if(threadIdx.x == 0) {
        for(int j=0; j < blockDim.x; j++) {
          float stdGamma = acc[j];
          if(stdGamma > 0.0f) { //thread accepted its proposal
            tauSqInv[0] = stdGamma / beta; //write accepted proposal back to global memory
            success[0] = 1; //tell other threads to stop
            break; //stop checking for accepted proposals
          }
        }
      }
    }

    __syncthreads();

    //last thread: copy curand state back to global memory
    if(threadIdx.x == blockDim.x - 1)
      globalState[0] = state;
  }
}