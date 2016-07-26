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
 * Function     : cuda_draw_y
 * Purpose      : draws samples from Bernoulli distribution with parameter Phi(Xbeta)
 * Argument n   : size of sample
 * Argument *xb : pointer to array of uniform random numbers
 * Argument *xb : pointer to array of parameter Xbeta
 * Argument *y  : pointer to output vector (int)
 * Output       : mutates y and stores result in its place
 */
extern "C"
__global__ void cuda_draw_y(int n, float* u, float *xb, int *y) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < n) {
    float phi = normcdf(xb[i]);
    y[i] = (u[i] < phi) ? 1 : 0;
  }
}