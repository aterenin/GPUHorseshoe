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
 * Function         : cuda_onesided_unitvar_tnorm
 * Purpose          : draws samples from independent truncated normals with mean vector mu,
                      std. dev. 1, truncated from zero to positive infinity if y = 1 and
                      negative infinity if y = -1, using inversion method, which is reasonable
                      in floating point precision as long as mean vector mu is not too far
                      away from 0.
 * Argument n       : size of sample
 * Argument *state  : pointer to random number generator state
 * Argument *mu     : pointer to mean vector
 * Argument *y      : pointer to truncation vector, 1 if positive, 0 if negative
 * Output           : mutates mu and stores result in its place
 */
extern "C"
__global__ void cuda_onesided_unitvar_tnorm(int n, curandStatePhilox4_32_10_t *globalState, float *mu, int *y) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state = globalState[0]; //copy random number generator state to local memory

  if(i < n) {
    //combined rejection sampler version
    float ystar = (float) (y[i] * 2 - 1); //transform from {0,1} to {-1.0f, 1.0f}
    float mustar = ystar * mu[i]; //always positive

    skipahead((unsigned long long) (6*i), &state); //give each thread its own pseudorandom subsequence with spacing 2^67
    //skipahead_sequence overflows somewhere, so use standard skipahead with spacing 3.

    if(!isfinite(mustar))
      mu[i] = 0.0f;
    else if(mustar < 0.47f) { //magic number to lower bound acceptance probability at around 2/3
      //upper tail: use exponential rejection sampler
      while(true) {
        float u = curand_uniform(&state); //one uniform for proposal
        float u2 = curand_uniform(&state); //one uniform for accept/reject step
        float alpha = (-mustar + sqrtf(mustar * mustar + 4.0f))/2.0f; //optimal scaling factor
        float prop = -logf(u) / alpha; //generate translated exponential(alpha, mu-)
        float rho = expf((prop - mustar - alpha) * (prop - mustar - alpha) / -2.0f); //compute acceptance probability
        if(u2 < rho) {
          mu[i] = ystar * prop;
          break;
        }
      }
    } else {
      //lower tail: use Gaussian rejection sampler
      while(true) {
        //float prop = curand_normal(&state) + mustar; //BROKEN: use inverse transform method instead
        float u = curand_uniform(&state);
        float prop = normcdfinvf(u) + mustar;
        if(isinf(prop))
          prop = 5.0f + mustar; //edge case, make sure computation doesn't stop if u == 1.0f
        if(prop > 0.0f) {
          mu[i] = ystar * prop;
          break;
        }
      }
    }
  }

  __syncthreads();

  //last thread: copy curand state back to global memory
  if(i == n-1)
    globalState[0] = state;
}