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

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>

// utility script to print sizeof(curandState), which is nowhere to be found in JCuda
int main() {
    curandState *states;
    curandStatePhilox4_32_10_t *philox;
    curandStateMRG32k3a *mrg;
    cudaMalloc((void **)&states, 64 * 64 * sizeof(curandState));
    cudaMalloc((void **)&philox, 64 * 64 * sizeof(curandStatePhilox4_32_10_t));
    cudaMalloc((void **)&mrg, 64 * 64 * sizeof(curandStateMRG32k3a));
    printf("sizeof(curandState) %lu\n",sizeof(curandState));
    printf("sizeof(curandStatePhilox4_32_10_t) %lu\n",sizeof(curandStatePhilox4_32_10_t));
    printf("sizeof(curandStateMRG32k3a) %lu\n",sizeof(curandStateMRG32k3a));
}


