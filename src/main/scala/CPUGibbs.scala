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

import breeze.linalg.{DenseMatrix, DenseVector, cholesky, max}
import breeze.stats.distributions._

import scala.collection.mutable
import com.github.fommil.netlib.BLAS.{getInstance => blas}
import com.github.fommil.netlib.LAPACK.{getInstance => lapack}



object CPUGibbs extends App {
  val (nMC, n, p, thinning) = args match {
    case Array(a1) => (a1.toInt, 1000, 100, 1)
    case Array(a1, a2, a3) => (a1.toInt, a2.toInt, a3.toInt, 1)
    case Array(a1, a2, a3, a4) => (a1.toInt, a2.toInt, a3.toInt, a4.toInt)
    case Array() => (10000, 1000, 100, 1)
  }
  val nonZeroBeta = Array(1.3,4.0,-1.0,1.6,5.0,-2.0)
  val numStoredVariables = math.min(p,143)
  val seed = 1

  println(s"initializing")

  val X = DenseMatrix.rand(n, p, Rand.gaussian)
  val betaCorrect = DenseVector(nonZeroBeta ++ Array.fill(p-nonZeroBeta.length)(0.0))
  val y = ((X * betaCorrect) + DenseVector.rand(n, Rand.gaussian)).map(v => math.round(Gaussian(0,1).cdf(v)).toInt)
  val XtX = X.t * X

  val Sigma = XtX.copy
  val mu = DenseVector.zeros[Double](n)

  val beta = DenseVector.zeros[Double](p)
  val z = DenseVector.zeros[Double](n)
  val lambdaSqInv = DenseVector.ones[Double](p)
  val nuInv = DenseVector.ones[Double](p)
  var tauSqInv = 1.0
  var xiInv = 1.0


  val betaOut = new mutable.Queue[DenseVector[Double]]
  val lambdaOut = new mutable.Queue[DenseVector[Double]]()
  val nuOut = new mutable.Queue[DenseVector[Double]]()
  val tauOut = new mutable.Queue[DenseVector[Double]]()
  val xiOut = new mutable.Queue[DenseVector[Double]]()

  println("starting MCMC")
  val time = System.nanoTime()

  for(i <- 0 until nMC) {
    // sample lambdaSqInv
    (0 until p).par.foreach(j => lambdaSqInv(j) = Exponential(nuInv(j) + (tauSqInv * beta(j) * beta(j))/2.0).draw())

    // sample nuInv
    (0 until p).par.foreach(j => nuInv(j) = Exponential(1.0 + lambdaSqInv(j)).draw())

    // sample tauSqInv
    tauSqInv = Gamma((p.toDouble+1.0)/2.0, xiInv + (0.5 * lambdaSqInv.data.zip(beta.data).par.map{case (l,b) => l * b * b}.sum)).draw()

    // sample xiInv
    xiInv = Exponential(1.0 + tauSqInv).draw()

    // sample beta
    // compute XtX + L, store in Sigma
    (0 until p).par.foreach(j => Sigma(j,j) = XtX(j,j) + (tauSqInv * lambdaSqInv(j)))
    // copy Sigma to R and compute R = chol(XtX + L)
    val R = cholesky(Sigma)
    // draw s, a vector of standard normals, store in beta
    (0 until p).par.foreach(j => beta(j) = Rand.gaussian.draw())
    // compute R^-1 s by solving triangular system Rv = s for v, store in beta
    blas.dtrsv("L", "T", "N", p, R.data, p, beta.data, 1)
    // compute Xtz, store in Xtz
    val Xtz = X.t * z
    // compute mu = (XtX)^-1 Xtz by solving the system (XtX) mu = Xtz for mu, store in Xtz
    blas.dtrsv("L", "N", "N", p, R.data, p, Xtz.data, 1)
    blas.dtrsv("L", "T", "N", p, R.data, p, Xtz.data, 1)
    // compute beta = Sigma s + mu, store in beta
    (0 until p).par.foreach(j => beta(j) = beta(j) + Xtz(j))

    // sample z
    // compute Xbeta, store in z
    blas.dgemv("N",n,p,1.0,X.data,n,beta.data,1,0.0,z.data,1)
    // draw z ~ TN
    (0 until n).par.foreach(j => z(j) = tn(z(j),y(j)))

    betaOut.enqueue(beta.copy)
    lambdaOut.enqueue(lambdaSqInv.copy)
    nuOut.enqueue(nuInv.copy)
    tauOut.enqueue(DenseVector(tauSqInv).copy)
    xiOut.enqueue(DenseVector(xiInv).copy)

    if (i % max(nMC / 100, 1) == 0) println(s"total samples: $i")
  }

  println(beta(0 until 10))

  println(s"finished, total run time in minutes: ${(System.nanoTime() - time).toDouble / 60000000000.0}")

  def tn(z: Double, y: Int): Double = {
    val ystar = (y * 2 - 1).toDouble
    val mustar = ystar * z
    if(mustar < 0.47) { // magic number to lower bound acceptance probability at around 2/3
      // upper tail: use exponential rejection sampler
      while(true) {
        val u = Rand.generator.nextDouble() // one uniform for proposal
        val u2 = Rand.generator.nextDouble() // one uniform for accept/reject step
        val alpha = (-mustar + math.sqrt(mustar * mustar + 4.0)) / 2.0 // optimal scaling factor
        val prop = -math.log(u) / alpha // generate truncated exponential
        val rho = math.exp((prop - mustar - alpha) * (prop - mustar - alpha) / -2.0) // compute acceptance probability
        if(u2 < rho) {
          return ystar * prop
        }
      }
    } else {
      // lower tail: use Gaussian rejection sampler
      while(true) {
        val prop = Rand.generator.nextGaussian() + mustar
        if(prop > 0.0) {
          return ystar * prop
        }
      }
    }
    throw new Exception
  }
}
