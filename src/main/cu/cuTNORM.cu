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
 * Function         : cuda_onesided_unitvar_tnorm
 * Purpose          : draws samples from independent truncated normals with mean vector mu,
                      std. dev. 1, truncated from zero to positive infinity if y = 1 and
                      negative infinity if y = -1, using inversion method, which is reasonable
                      in floating point precision as long as mean vector mu is not too far
                      away from 0.
 * Argument n       : size of sample
 * Argument *z      : pointer to array of Gaussian random variables
 * Argument *mu     : pointer to mean vector
 * Argument *y      : pointer to truncation vector, 1 if positive, 0 if negative
 * Output           : mutates z and stores result in its place
 */
extern "C"
__global__ void cuda_onesided_unitvar_tnorm(int n, float *z, float *mu, int *y) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < n) {
    //combined rejection sampler version
    float ystar = (float) (y[i] * 2 - 1); //transform from {0,1} to {-1.0f, 1.0f}
    float mustar = ystar * mu[i]; //always positive

    //try to just take the provided Gaussian
    float localz = z[i];
    float prop = localz + mustar;
    if(prop > 0.0f) {
      z[i] = ystar * prop;
    } else {
      //try to use inversion
      float phi = normcdff(mustar);
      float u = normcdff(localz) * phi; //1.0 - phi = normcdff(mustar);
      float out = ystar * (mustar + normcdfinvf((1.0f - phi) + (u * phi)));
      //check for infinite
      if(!isinf(out))
        z[i] = out;
      else
        z[i] = mustar * ystar;
    }
  }
}
    //single precision version
//    float ystar = (y[i] == 1) ? 1.0f : -1.0f; //faster than if/else
//    float phi = normcdff(-mu[i]*ystar);
//    z[i] = ystar * ((mu[i]*ystar) + normcdfinvf(phi + (z[i] * (1.0f - phi))));

    //double precision version
//    float ystar = (y[i] == 1) ? 1.0f : -1.0f; //faster than if/else
//    double phi = normcdf((double) (-mu[i]*ystar));
//    double zdbl = (double) z[i];
//    double phiinv = normcdfinv(phi + (zdbl * (1.0 - phi)));
//    float phiinvf = (float) phiinv;
//    z[i] = ystar * ((mu[i]*ystar) + phiinvf);

//    if(mustar < -5.0f) {
//      //upper tail: use exponential rejection sampler
//      while(true) {
//        float u = curand_uniform(state);
//        float alpha = -mustar + sqrtf(mustar * mustar + 4.0f); //optimal scaling factor
//        float prop = (-logf(u) / alpha) - mustar; //generate translated exponential(alpha, mu-)
//        float rho = expf((prop - alpha) * (prop - alpha) / -2.0f); //compute acceptance probability
//        if(u < rho) {
//          z[i] = ystar * prop;
//          break;
//        }
//      }
//    } else if(mustar < 5.0f) {
//      //middle: use inversion method
//      float u = curand_uniform(state);
//      float phi = normcdff(-mustar);
//      z[i] = ystar * (mustar + normcdfinvf(phi + (u * (1.0f - phi))));
//    } else {
//      //lower tail: use Gaussian rejection sampler
//      while(true) {
//        float prop = curand_normal(state) + mustar;
//        if(prop > 0.0f) {
//          z[i] = ystar * prop;
//          break;
//        }
//      }
//    }
//    //copy curand state back to global memory
//    state[i] = localState;