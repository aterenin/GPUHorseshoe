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
 * Function     : cuda_onesided_unitvar_tnorm
 * Purpose      : draws samples from independent truncated normals with mean vector mu,
                  std. dev. 1, truncated from zero to positive infinity if y = 1 and
                  negative infinity if y = -1, using inversion method, which is reasonable
                  in floating point precision as long as mean vector mu is not too far
                  away from 0.
 * Argument n   : size of sample
 * Argument z*  : pointer to array of uniform random variables
 * Argument mu* : pointer to mean vector
 * Argument y*  : pointer to truncation vector, 1.0f if positive, 0.0f if negative
 * Output       : mutates z and stores result in its place
 */
extern "C"
__global__ void cuda_onesided_unitvar_tnorm(int n, float *z, float *mu, int *y) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < n) {
    //single precision version
    float ystar = (y[i] == 1) ? 1.0f : -1.0f; //faster than if/else
    float phi = normcdff(-mu[i]*ystar);
    z[i] = ystar * ((mu[i]*ystar) + normcdfinvf(phi + (z[i] * (1.0f - phi))));

    //double precision version
//    float ystar = (y[i] == 1) ? 1.0f : -1.0f; //faster than if/else
//    double phi = normcdf((double) (-mu[i]*ystar));
//    double zdbl = (double) z[i];
//    double phiinv = normcdfinv(phi + (zdbl * (1.0 - phi)));
//    float phiinvf = (float) phiinv;
//    z[i] = ystar * ((mu[i]*ystar) + phiinvf
  }
}