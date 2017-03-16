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
 * Function         : cuda_RL
 * Purpose          : adds sqrt(L) to lower diagonal of 2*n*n matrix R
 * Argument n       : size of L, R
 * Argument *L      : pointer to L vector
 * Argument *R      : pointer to R matrix
 * Output           : mutates R and stores result in its place
 */
extern "C"
__global__ void cuda_RL(int n, float *L, float* R) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < n)
    R[(2*i+1)*n + i] = sqrtf(L[i]);
}