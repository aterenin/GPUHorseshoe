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



/*
 * Function         : cuda_tauSqInv
 * Purpose          : calculates the relevant parameters for tau^-2, and
                      draws one sample from G(a,b) distribution with a large by
                      implementing the rejection sampler in Cheng (1977) for a set of
                      parallel threads in one block and taking one of the samples
                      that wasn't rejected (best when run on a 32-thread warp)
 * Argument n       : size of parallel proposal
 * Argument *u      : pointer to array of blockDim.x * x uniform random variables
 * Argument *k      : pointer to shape parameter
 * Argument *xiInv  : pointer to xiInv (used in calculating scale)
 * Argument *LB     : pointer to LB (used in calculating scale)
 * Argument *success: pointer to flag indicating whether kernel succeeded or not
 * Output           : mutates u[0] and stores result in its place, and mutates success
                      and stores 1 if sample was accepted and 0 otherwise
 */
extern "C"
__global__ void cuda_tauSqInv(int n, float *u, float *k, float *xiInv, float *LB, int* success) {
  __shared__ float acc[32]; //hardcode to 32 for now, fix later
  if(threadIdx.x < 32 && blockIdx.x == 0) {
    acc[threadIdx.x] = 0.0f;
    //copy parameters and uniforms to local memory
    float u1 = u[threadIdx.x * 2];
    float u2 = u[threadIdx.x * 2 + 1];
    float alpha = k[0];

    //compute rate parameter
    float theta = xiInv[0] + (0.5 * LB[0]);

    //compute constants
    float a = 1.0f/sqrtf(2.0f * alpha - 1.0f);
    float b = alpha - 1.3862944f; //log(4) = 1.3862944f
    float c = alpha + (1.0f / a);
    float v = a * logf(u1 / (1.0f - u1));
    float x = alpha * expf(v);

    //perform accept/reject
    if( (b + (c*v) - x) > logf(u1 * u1 * u2) ) {
      acc[threadIdx.x] = x;
      u[threadIdx.x] = x;
    }


    __syncthreads();

    //find accepted value on thread 0
    if(threadIdx.x == 0) {
      for(int j=0; j < 32; j++) {
        float stdGamma = acc[j];
        if(stdGamma > 0.0f) { //thread accepted its proposal
          u[0] = stdGamma / theta;
          success[0] = 1;
          break;
        }
        if(j == blockDim.x - 1) { //previous for loop maxed out, so all samplers rejected
          success[0] = 0;
          break;
        }
      }
    }
  }
}