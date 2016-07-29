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
 * Function         : cuda_lambdaSqInv
 * Purpose          : draws Exp(1 + theta^2) random variables
 * Argument n       : size of sampler
 * Argument *u      : pointer to array of uniforms
 * Argument *beta   : pointer to beta vector
 * Argument *nuInv  : pointer to nuInv vector
 * Argument *tSqInv : pointer to tauSqInv constant
 * Output           : mutates u and stores result in its place
 */
extern "C"
__global__ void cuda_lambdaSqInv(int n, float *u, float* beta, float *nuInv, float* tSqInv) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < n) {
    u[i] = (-1.0f/(nuInv[i] + (0.5f * tSqInv[0] * beta[i] * beta[i]) )) * logf(u[i]);
  }
}