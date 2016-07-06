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
 * Function     : cuda_positive_tnorm
 * Purpose      : draws samples from independent truncated normals with mean vector mu,
                  std. dev. 1, truncated up from zero to infinity, using inversion method,
                  which is reasonable in floating point precision as long as mean vector mu
                  is not too far away from 0.
 * Argument n   : size of sample
 * Argument z*  : pointer to array of uniform random variables
 * Argument mu* : pointer to mean vector
 * Output       : mutates z and stores result in its place
 */
extern "C"
__global__ void cuda_positive_tnorm(int n, float *z, float *mu) {
  int i = blockIdx.z*blockDim.z + threadIDX.z
  if(i < n)
    z[i] = mu[i] + normcdfinvf(normcdff(-mu[i]) + (z[i] * (1.0f - normcdff(-mu[i]))))
}