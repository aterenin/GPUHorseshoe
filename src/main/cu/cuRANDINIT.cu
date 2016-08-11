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
 * Function         : cuda_rand_init
 * Purpose          : initializes random number generator
 * Argument state   : random number generator state
 * Output           : mutates state and stores result in its place
 */
extern "C"
__global__ void cuda_rand_init(int seed, curandStatePhilox4_32_10_t *state) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i == 0)
    curand_init(seed, 0, 0, &state[0]);
}