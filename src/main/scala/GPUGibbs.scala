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

import java.io.{BufferedWriter, FileOutputStream, OutputStreamWriter}
import scala.io.Source._
import scala.collection.mutable

import jcuda._
import jcuda.driver._
import jcuda.driver.JCudaDriver._
import jcuda.jcurand._
import jcuda.jcurand.JCurand._
import jcuda.jcurand.curandRngType._
import jcuda.runtime.JCuda._
import jcuda.runtime.cudaMemcpyKind._
import jcuda.runtime.cudaStream_t
import jcuda.jcublas._
import jcuda.jcublas.JCublas2._
import jcuda.jcublas.cublasSideMode._
import jcuda.jcublas.cublasOperation._
import jcuda.jcublas.cublasPointerMode._
import jcuda.jcusolver._
import jcuda.jcusolver.JCusolverDn._
import jcuda.jcublas.cublasFillMode._
import jcuda.jcublas.cublasDiagType._
import jcuda.jcublas.cublasStatus._
import jcuda.jcusolver.cusolverStatus._
import jcuda.jcurand.curandStatus._
import jcuda.driver.CUresult._



object GPUGibbs extends App {
  val (nMC, n, p, thinning) = args match {
    case Array(a1) => (a1.toInt, 1000, 100, 1)
    case Array(a1, a2, a3) => (a1.toInt, a2.toInt, a3.toInt, 1)
    case Array(a1, a2, a3, a4) => (a1.toInt, a2.toInt, a3.toInt, a4.toInt)
    case Array() => (100, 1000, 100, 1)
  }
  val nonZeroBeta = Array(1.3f,4.0f,-1.0f,1.6f,5.0f,-2.0f)
  val numStoredVariables = 87
  val seed = 1

  println(s"initializing")
  printMemInfo()

  val sizeOfCurandStateXORWOW = 48L //somehow, nowhere to be found in JCuda
  val sizeOfCurandStatePhilox4_32_10_t = 64L
  val sizeOfCurandStateMRG32k3a = 72L

  // function to generate data
  def genData(): Unit = {
    // draw a vector of standard normals, store in X
    curandSetPseudoRandomGeneratorSeed(curandGenerator, seed)
    curandGenerateNormal(curandGenerator, X, n.toLong*p.toLong, 0.0f, 1.0f)

    // compute R and draw X again
    cusolverDnSgeqrf(cusolverHandle, n, p, X, n, qrXTau, qrXWorkspace, qrXWorkspaceNumBytes, cusolverDevInfo)
    for(i <- 0L until p.toLong) { // copy upper triangle
      cudaMemcpy(R.withByteOffset(i*2L*p*Sizeof.FLOAT.toLong), X.withByteOffset(i * n.toLong * Sizeof.FLOAT.toLong), (i+1L)*Sizeof.FLOAT.toLong, cudaMemcpyDeviceToDevice)
    }
    curandSetPseudoRandomGeneratorSeed(curandGenerator, seed)
    curandGenerateNormal(curandGenerator, X, n.toLong*p.toLong, 0.0f, 1.0f)

    // copy nonZeroBeta to beta
    cudaMemcpy(beta, Pointer.to(nonZeroBeta), nonZeroBeta.length * Sizeof.FLOAT, cudaMemcpyHostToDevice)

    // compute Xbeta, store in muScratch
    cublasSgemv(cublasHandle,CUBLAS_OP_N,n,nonZeroBeta.length,ptrOnef,X,n,beta,1,ptrZerof,muScratch,1)

    // draw u, a vector of uniforms, store in z
    curandGenerateUniform(curandGenerator, z, n)

    // draw y ~ Ber(phi(Xbeta))
    // load Y kernel
    val berModule = new CUmodule()
    cuModuleLoad(berModule, "cuY.ptx")
    val berFunction = new CUfunction
    cuModuleGetFunction(berFunction,berModule,"cuda_draw_y")
    val blockSizeX = 256
    val gridSizeX = math.ceil(n.toDouble / blockSizeX.toDouble).toInt
    val berInputPointer = Pointer.to(
      Pointer.to(Array(n)),
      Pointer.to(z),
      Pointer.to(muScratch),
      Pointer.to(y)
    )
    // call custom kernel cuda_draw_y
    cuLaunchKernel(berFunction, gridSizeX,1,1, blockSizeX,1,1, 0, null, berInputPointer, null)

    //cleanup
    cuModuleUnload(berModule)
    cudaMemcpy(beta, Pointer.to(Array.fill(p)(0.0f)), p*Sizeof.FLOAT, cudaMemcpyHostToDevice)
  }

  def loadData() = {
//    val interactionIdxs = (1 to 72).toArray // exclude intercept
    // load data from disk
    val localData =
      fromFile("data/kaiser.csv") // 372295 x 87, 1295848 x 143
        .getLines()
        .map{
          line =>
            val y = line.split(",").head.toInt
            val x = line.split(",").drop(1).map(_.toFloat)
            (y, x)
        }
        .toArray

    val yLocal = localData.map(_._1)
    val mainEffectsUnscaled = localData.map(_._2).transpose //transpose because Scala uses row major format
    val mainEffects = mainEffectsUnscaled
//      .map(
//      col => {
//        val mean = col.sum / col.length.toFloat
//        val variance = col.map(v => (v - mean)*(v - mean)).sum / col.length.toFloat
//        val sd = math.sqrt(variance.toDouble).toFloat
//        col.map(v => (v - mean) / sd)
//      }
//    )

    val intercept = Array(Array.fill(mainEffects.head.length)(1.0f))

//    val interactions = interactionIdxs
//      .par
//      .flatMap(
//        // create list of crossed indices, avoiding square terms and duplicates
//        v =>
//          interactionIdxs
//            .filter(w => w > v)
//            .map(w => (v,w))
//      )
//      .map{
//        case (i, j) =>
////          println(s"creating interaction effect for columns ($i,$j)")
//          val iColumn = mainEffects(i)
//          val jColumn = mainEffects(j)
//          val ijColumn = iColumn.zip(jColumn).map{case (a,b) => a*b}
//          val mean = ijColumn.sum / ijColumn.length.toFloat
//          val variance = ijColumn.map(v => (v - mean)*(v - mean)).sum / ijColumn.length.toFloat
//          val sd = math.sqrt(variance.toDouble).toFloat
//          ijColumn.map(v => (v - mean) / sd)
//      }
    val interactions = Array.empty[Array[Float]]

    val XLocal = (intercept ++ mainEffects ++ interactions).flatten

    val nLoaded = yLocal.length
    val pLoaded = (XLocal.length.toDouble / yLocal.length.toDouble).toInt

    println(s"loaded data, size ($n,$p), size check ($nLoaded,$pLoaded)")
    if(n != nLoaded || p != pLoaded) throw new Exception("mismatched n,p in csv file")

    cudaMemcpy(X, Pointer.to(XLocal), n.toLong * p.toLong * Sizeof.FLOAT.toLong, cudaMemcpyHostToDevice) //Xlocal is column major
    cudaMemcpy(y, Pointer.to(yLocal), n.toLong * Sizeof.INT.toLong, cudaMemcpyHostToDevice)

    // compute R and load X again
    cusolverDnSgeqrf(cusolverHandle, n, p, X, n, qrXTau, qrXWorkspace, qrXWorkspaceNumBytes, cusolverDevInfo)
    for(i <- 0L until p.toLong) { // copy upper triangle
      cudaMemcpy(R.withByteOffset(i*2L*p*Sizeof.FLOAT.toLong), X.withByteOffset(i * n.toLong * Sizeof.FLOAT.toLong), (i+1L)*Sizeof.FLOAT.toLong, cudaMemcpyDeviceToDevice)
    }
    cudaMemcpy(X, Pointer.to(XLocal), n.toLong * p.toLong * Sizeof.FLOAT.toLong, cudaMemcpyHostToDevice) //Xlocal is column major
  }

  // initialize cuRAND, cuBLAS
  val curandGenerator = new curandGenerator()
  curandCreateGenerator(curandGenerator, CURAND_RNG_PSEUDO_PHILOX4_32_10)

  val cublasHandle = new cublasHandle()
  cublasCreate(cublasHandle)
  cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE)

  val cusolverHandle = new cusolverDnHandle
  cusolverDnCreate(cusolverHandle)


  // create device pointers and copy data
  val y = new Pointer
  cudaMalloc(y, n*Sizeof.INT)
//  cudaMemcpy(y, Pointer.to(localY), n*Sizeof.INT, cudaMemcpyHostToDevice)

  val z = new Pointer
  cudaMalloc(z, n*Sizeof.FLOAT)
  cudaMemcpy(z, Pointer.to(Array.fill(n)(0.0f)), n*Sizeof.FLOAT, cudaMemcpyHostToDevice)

  val X = new Pointer
  cudaMalloc(X, n.toLong*p.toLong*Sizeof.FLOAT.toLong) //must be long to avoid integer overflow
    .checkJCudaStatus() //check here for out of memory error

  val beta = new Pointer
  cudaMalloc(beta, (p + (p % 2))*Sizeof.FLOAT)
  cudaMemcpy(beta, Pointer.to(Array.fill(p + (p % 2))(0.0f)), (p + (p % 2))*Sizeof.FLOAT, cudaMemcpyHostToDevice)


  val lambdaSqInv = new Pointer
  cudaMalloc(lambdaSqInv, p*Sizeof.FLOAT)
  cudaMemcpy(lambdaSqInv, Pointer.to(Array.fill(p)(1.0f)), p*Sizeof.FLOAT, cudaMemcpyHostToDevice)

  val nuInv = new Pointer
  cudaMalloc(nuInv, p*Sizeof.FLOAT)
  cudaMemcpy(nuInv, Pointer.to(Array.fill(p)(1.0f)), p*Sizeof.FLOAT, cudaMemcpyHostToDevice)

  val xiInv = new Pointer
  cudaMalloc(xiInv, Sizeof.FLOAT)
  cudaMemcpy(xiInv, Pointer.to(Array(1.0f)), Sizeof.FLOAT, cudaMemcpyHostToDevice)

  val tauSqInv = new Pointer
  cudaMalloc(tauSqInv, Sizeof.FLOAT)
  cudaMemcpy(tauSqInv, Pointer.to(Array(1.0f)), Sizeof.FLOAT, cudaMemcpyHostToDevice)

  val tauSqInvShape = new Pointer
  cudaMalloc(tauSqInvShape, Sizeof.FLOAT)
  cudaMemcpy(tauSqInvShape, Pointer.to(Array((p.toFloat + 1.0f) / 2.0f)), Sizeof.FLOAT, cudaMemcpyHostToDevice)


  val XtX = new Pointer
  cudaMalloc(XtX, p*p*Sizeof.FLOAT)

  val R = new Pointer
  cudaMalloc(R, 2*p*p*Sizeof.FLOAT)

  val SigmaScratch = new Pointer
  cudaMalloc(SigmaScratch, 2*p*p*Sizeof.FLOAT)

  val muScratch = new Pointer
  cudaMalloc(muScratch, n*Sizeof.FLOAT) // needs to be n for genData

  val betaSqScratch = new Pointer
  cudaMalloc(betaSqScratch, p*Sizeof.FLOAT)


  val cholWorkspaceSize = Array(0)
  cusolverDnSpotrf_bufferSize(cusolverHandle, CUBLAS_FILL_MODE_UPPER, p, SigmaScratch, 1, cholWorkspaceSize)
  val cholWorkspaceNumBytes = cholWorkspaceSize.head * Sizeof.FLOAT
  val cholWorkspace = new Pointer
  cudaMalloc(cholWorkspace, cholWorkspaceNumBytes)


  val qrXTau = new Pointer
  cudaMalloc(qrXTau, n*Sizeof.INT)
  val qrXWorkspaceSize = Array(0)
  cusolverDnSgeqrf_bufferSize(cusolverHandle, n, p, X, n, qrXWorkspaceSize)
  val qrXWorkspaceNumBytes = qrXWorkspaceSize.head * Sizeof.FLOAT
  val qrXWorkspace = new Pointer
  cudaMalloc(qrXWorkspace, qrXWorkspaceNumBytes)


  val qrTau = new Pointer
  cudaMalloc(qrTau, p*Sizeof.INT)
  val qrWorkspaceSize = Array(0)
  cusolverDnSgeqrf_bufferSize(cusolverHandle, 2*p, p, SigmaScratch, 2*p, qrWorkspaceSize)
  val qrWorkspaceNumBytes = qrWorkspaceSize.head * Sizeof.FLOAT
  val qrWorkspace = new Pointer
  cudaMalloc(qrWorkspace, qrWorkspaceNumBytes)
  val cusolverDevInfo = new Pointer
  cudaMalloc(cusolverDevInfo, Sizeof.INT)


  val ptrOnef = new Pointer
  cudaMalloc(ptrOnef, Sizeof.FLOAT)
  cudaMemcpy(ptrOnef, Pointer.to(Array(1.0f)), Sizeof.FLOAT, cudaMemcpyHostToDevice)

  val ptrZerof = new Pointer
  cudaMalloc(ptrZerof, Sizeof.FLOAT)
  cudaMemcpy(ptrZerof, Pointer.to(Array(0.0f)), Sizeof.FLOAT, cudaMemcpyHostToDevice)


  val curandStateZ = new Pointer
  cudaMalloc(curandStateZ, sizeOfCurandStatePhilox4_32_10_t)
  val curandStateTau = new Pointer
  cudaMalloc(curandStateTau, sizeOfCurandStatePhilox4_32_10_t)

  // initialize device RNG
  val randInitModule = new CUmodule()
  cuModuleLoad(randInitModule, "cuRANDINIT.ptx")
  val randInitFunction = new CUfunction
  cuModuleGetFunction(randInitFunction, randInitModule, "cuda_rand_init")
  // create the device RNG state
  cuLaunchKernel(randInitFunction, 1,1,1, 1,1,1, 0, null, Pointer.to(Pointer.to(Array(seed + 1000)), Pointer.to(curandStateZ)), null)
  cuLaunchKernel(randInitFunction, 1,1,1, 1,1,1, 0, null, Pointer.to(Pointer.to(Array(seed + 2000)), Pointer.to(curandStateTau)), null)


  // load tnorm kernel
  val tnormModule = new CUmodule()
  cuModuleLoad(tnormModule, "cuTNORM.ptx")
  val tnormFunction = new CUfunction
  cuModuleGetFunction(tnormFunction,tnormModule,"cuda_onesided_unitvar_tnorm")
  def tnormGetBlockAndGridSize(n: Int) = {
    val blockSizeX = 256
    val gridSizeX = math.ceil(n.toDouble / blockSizeX.toDouble).toInt
    (blockSizeX, gridSizeX)
  }
  val tnormInputPointer = Pointer.to(
    Pointer.to(Array(n)),
    Pointer.to(curandStateZ),
    Pointer.to(z),
    Pointer.to(y)
  )

  //load lambda kernel
  val lambdaModule = new CUmodule()
  cuModuleLoad(lambdaModule, "cuLAMBDA.ptx")
  val lambdaFunction = new CUfunction
  cuModuleGetFunction(lambdaFunction,lambdaModule,"cuda_lambdaSqInv")
  def lambdaGetBlockAndGridSize(n: Int) = {
    val blockSizeX = 256
    val gridSizeX = math.ceil(n.toDouble / blockSizeX.toDouble).toInt
    (blockSizeX, gridSizeX)
  }
  val lambdaInputPointer = Pointer.to(
    Pointer.to(Array(p)),
    Pointer.to(lambdaSqInv),
    Pointer.to(beta),
    Pointer.to(nuInv),
    Pointer.to(tauSqInv)
  )

  //load nu, xi kernel
  val nuXiModule = new CUmodule()
  cuModuleLoad(nuXiModule, "cuNUXI.ptx")
  val nuXiFunction = new CUfunction
  cuModuleGetFunction(nuXiFunction,nuXiModule,"cuda_nuInvXiInv")
  def nuGetBlockAndGridSize(n: Int) = {
    val blockSizeX = 256
    val gridSizeX = math.ceil(n.toDouble / blockSizeX.toDouble).toInt
    (blockSizeX, gridSizeX)
  }
  val nuInputPointer = Pointer.to(
    Pointer.to(Array(p)),
    Pointer.to(nuInv),
    Pointer.to(lambdaSqInv)
  )
  def xiGetBlockAndGridSize() = (1,1)
  val xiInputPointer = Pointer.to(
    Pointer.to(Array(1)),
    Pointer.to(xiInv),
    Pointer.to(tauSqInv)
  )

  //load tau kernel
  val tauModule = new CUmodule()
  cuModuleLoad(tauModule, "cuTAU.ptx")
  val tauFunction = new CUfunction
  cuModuleGetFunction(tauFunction,tauModule,"cuda_tauSqInv")
  val tauInputPointer = Pointer.to(
    Pointer.to(curandStateTau),
    Pointer.to(tauSqInv),
    Pointer.to(tauSqInvShape),
    Pointer.to(xiInv)
  )

  //load betaSq kernel
  val betaModule = new CUmodule()
  cuModuleLoad(betaModule, "cuBETA.ptx")
  val betaFunction = new CUfunction
  cuModuleGetFunction(betaFunction,betaModule,"cuda_betaSq")
  def betaGetBlockAndGridSize(n: Int) = {
    val blockSizeX = 256
    val gridSizeX = math.ceil(n.toDouble / blockSizeX.toDouble).toInt
    (blockSizeX, gridSizeX)
  }
  val betaInputPointer = Pointer.to(
    Pointer.to(Array(p)),
    Pointer.to(beta),
    Pointer.to(betaSqScratch)
  )

  //load RL kernel
  val RLModule = new CUmodule()
  cuModuleLoad(RLModule, "cuRL.ptx")
  val RLFunction = new CUfunction
  cuModuleGetFunction(RLFunction,RLModule,"cuda_RL")
  def RLGetBlockAndGridSize(n: Int) = {
    val blockSizeX = 256
    val gridSizeX = math.ceil(n.toDouble / blockSizeX.toDouble).toInt
    (blockSizeX, gridSizeX)
  }
  val RLInputPointer = Pointer.to(
    Pointer.to(Array(p)),
    Pointer.to(R),
    Pointer.to(lambdaSqInv)
  )

  // create streams
  val betaStream_t = new cudaStream_t
  val betaCUstream = new CUstream(betaStream_t)
  cudaStreamCreate(betaStream_t)

  val zStream_t = new cudaStream_t
  val zCUstream = new CUstream(zStream_t)
  cudaStreamCreate(zStream_t)

  val lambdaTauStream_t = new cudaStream_t
  val lambdaTauCUstream = new CUstream(lambdaTauStream_t)
  cudaStreamCreate(lambdaTauStream_t)

  val nuStream_t = new cudaStream_t
  val nuCUstream = new CUstream(nuStream_t)
  cudaStreamCreate(nuStream_t)

  val xiStream_t = new cudaStream_t
  val xiCUstream = new CUstream(xiStream_t)
  cudaStreamCreate(xiStream_t)

  // generate data
//  genData(); def updateBeta(): Unit = { updateBetaChol() }
  loadData(); def updateBeta(): Unit = { updateBetaQR() }

  // precompute XtX
  cublasSgemm(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_N,p,p,n,ptrOnef,X,n,X,n,ptrZerof,XtX,p)

  // create output queue
  val betaOut = new mutable.Queue[Array[Float]]()
  val zOut = new mutable.Queue[Array[Float]]()
  val lambdaOut = new mutable.Queue[Array[Float]]()
  val nuOut = new mutable.Queue[Array[Float]]()
  val tauOut = new mutable.Queue[Array[Float]]()
  val xiOut = new mutable.Queue[Array[Float]]()

  println("data loaded into GPU, starting MCMC")
  printMemInfo()
  val time = System.nanoTime()


  // run MCMC
  for(i <- 0 until nMC) {
    if(i % thinning == 0) {
      downloadOutput(z, zOut, zStream_t, numStoredVariables)
      downloadOutput(lambdaSqInv, lambdaOut, lambdaTauStream_t, numStoredVariables)
      downloadOutput(tauSqInv, tauOut, lambdaTauStream_t, 1)
    }
    updateBeta()
    updateNu()
    updateXi()
    syncAllStreams()

    if(i % thinning == 0) {
      downloadOutput(beta, betaOut, betaStream_t, numStoredVariables)
      downloadOutput(nuInv, nuOut, nuStream_t, numStoredVariables)
      downloadOutput(xiInv, xiOut, xiStream_t, 1)
    }
    updateZ()
    updateLambda()
    updateTau()
    syncAllStreams()

    if (i % math.max(nMC / 100, 1) == 0) println(s"total samples: $i")
  }

  println(s"finished, total run time in minutes: ${(System.nanoTime() - time).toDouble / 60000000000.0}")

  val fInvSqrt = { v: Float => math.sqrt(1.0 / v.toDouble).toFloat }
  val fInv = { v: Float => 1.0f / v }
  val fIdentity = { v: Float => v }
  // write output
  for {
    ((out, name), f) <- Array(betaOut, zOut, lambdaOut, nuOut, tauOut, xiOut)
      .zip(Array("beta", "z", "lambda", "nu", "tau", "xi"))
      .zip(Array(fIdentity, fIdentity, fInvSqrt, fInv, fInvSqrt, fInv))
  } {
    val fileName = s"output/out-GPU-$name.csv"
    val writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(fileName)))
    println(s"writing output of length ${out.size} to $fileName")
    out.foreach { outRow => writer.write(s"${outRow.map(f).mkString(",")}\n") }
    writer.close()
  }


  cleanup() //free memory and destroy


  def syncAllStreams(): Unit = {
    cudaStreamSynchronize(betaStream_t)
    cudaStreamSynchronize(zStream_t)
    cudaStreamSynchronize(lambdaTauStream_t)
    cudaStreamSynchronize(nuStream_t)
    cudaStreamSynchronize(xiStream_t)
  }

  def downloadOutput(p: Pointer, o: mutable.Queue[Array[Float]], s: cudaStream_t, numV: Int): Unit = {
    val nv = Array.ofDim[Float](numV)
    cudaMemcpyAsync(Pointer.to(nv), p, numV * Sizeof.FLOAT, cudaMemcpyDeviceToHost, s)
    o.enqueue(nv) // copy reference to Monte Carlo output without blocking
  }

  def updateLambda(): Unit = {
    curandSetStream(curandGenerator, lambdaTauStream_t)

    // draw u, a vector of uniforms, store in lambda
    curandGenerateUniform(curandGenerator, lambdaSqInv, p)

    // transform u into lambdaSqInv ~ Exp via custom kernel cuda_lambdaSqInv
    val (blockSizeX, gridSizeX) = lambdaGetBlockAndGridSize(p)
    cuLaunchKernel(lambdaFunction, gridSizeX,1,1, blockSizeX,1,1, 0, lambdaTauCUstream, lambdaInputPointer, null)
  }

  def updateNu(): Unit = {
    curandSetStream(curandGenerator, nuStream_t)

    // draw u, a vector of uniforms, store in nu
    curandGenerateUniform(curandGenerator, nuInv, p)

    // transform u into lambdaSqInv ~ Exp via custom kernel cuda_lambdaSqInv
    val (blockSizeX, gridSizeX) = nuGetBlockAndGridSize(p)
    cuLaunchKernel(nuXiFunction, gridSizeX,1,1, blockSizeX,1,1, 0, nuCUstream, nuInputPointer, null)
  }

  def updateXi(): Unit = {
    curandSetStream(curandGenerator, xiStream_t)

    // draw u, a vector of uniforms, store in lambda
    curandGenerateUniform(curandGenerator, xiInv, 1)

    // transform u into lambdaSqInv ~ Exp via custom kernel cuda_lambdaSqInv
    val (blockSizeX, gridSizeX) = xiGetBlockAndGridSize()
    cuLaunchKernel(nuXiFunction, gridSizeX,1,1, blockSizeX,1,1, 0, xiCUstream, xiInputPointer, null)
  }


  def updateTau(): Unit = {
    cublasSetStream(cublasHandle, lambdaTauStream_t)

    // compute lambdaSqInv . betaSq
    val (blockSizeX, gridSizeX) = betaGetBlockAndGridSize(p)
    cuLaunchKernel(betaFunction, gridSizeX,1,1, blockSizeX,1,1, 0, lambdaTauCUstream, betaInputPointer, null)

    cublasSdot(cublasHandle, p, lambdaSqInv, 1, betaSqScratch, 1, tauSqInv)

    cuLaunchKernel(tauFunction, 1,1,1, 32,1,1, 32*Sizeof.FLOAT, lambdaTauCUstream, tauInputPointer, null)
  }

  def updateBetaChol(): Unit = {
    cublasSetStream(cublasHandle, betaStream_t)
    curandSetStream(curandGenerator, betaStream_t)
    cusolverDnSetStream(cusolverHandle, betaStream_t)
    // draws beta ~ N_p((XtX)^-1 Xtz, (XtX)^-1)

    // store XtX in SigmaScratch
    cudaMemcpyAsync(SigmaScratch, XtX, p*p*Sizeof.FLOAT, cudaMemcpyDeviceToDevice, betaStream_t)

    // compute XtX + L, store in SigmaScratch
    cublasSaxpy(cublasHandle, p, tauSqInv, lambdaSqInv, 1, SigmaScratch, p+1) // p+1 adds to diagonal

    // compute R = chol(XtX + L), store in SigmaScratch
    cusolverDnSpotrf(cusolverHandle, CUBLAS_FILL_MODE_UPPER, p, SigmaScratch, p, cholWorkspace, cholWorkspaceNumBytes, cusolverDevInfo)

    // draw s, a vector of standard normals, store in beta
    curandGenerateNormal(curandGenerator, beta, p + (p % 2), 0.0f, 1.0f)

    // compute R^-1 s by solving triangular system Rv = s for v, store in beta
    cublasStrsv(cublasHandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, p, SigmaScratch, p, beta, 1)

    // compute Xtz, store in muScratch
    cublasSgemv(cublasHandle,CUBLAS_OP_T,n,p,ptrOnef,X,n,z,1,ptrZerof,muScratch,1)

    // compute mu = (XtX)^-1 Xtz by solving the system (XtX) mu = Xtz for mu, store in muScratch
    cusolverDnSpotrs(cusolverHandle, CUBLAS_FILL_MODE_UPPER, p, 1, SigmaScratch, p, muScratch, p, cusolverDevInfo)

    // compute beta = Sigma s + mu, store in beta
    cublasSaxpy(cublasHandle, p, ptrOnef, muScratch, 1, beta, 1)
  }

  def updateBetaQR(): Unit = {
    cublasSetStream(cublasHandle, betaStream_t)
    curandSetStream(curandGenerator, betaStream_t)
    cusolverDnSetStream(cusolverHandle, betaStream_t)
    // draws beta ~ N_p((XtX)^-1 Xtz, (XtX)^-1)

    // store R in upper part SigmaScratch, overwrite lower part with zeros
    cudaMemcpyAsync(SigmaScratch, R, 2*p*p*Sizeof.FLOAT, cudaMemcpyDeviceToDevice, betaStream_t)

    // store sqrt(L) in lower part of SigmaScratch
    val (blockSizeX, gridSizeX) = RLGetBlockAndGridSize(p)
    cuLaunchKernel(RLFunction, gridSizeX,1,1, blockSizeX,1,1, 0, betaCUstream, RLInputPointer, null)

    // compute R = QR(R_x, R_L), store in SigmaScratch
    cusolverDnSgeqrf(cusolverHandle, 2*p, p, SigmaScratch, 2*p, qrTau, qrWorkspace, qrWorkspaceNumBytes, cusolverDevInfo)

    // draw s, a vector of standard normals, store in beta
    curandGenerateNormal(curandGenerator, beta, p + (p % 2), 0.0f, 1.0f)

    // compute R^-1 s by solving triangular system Rv = s for v, store in beta
    cublasStrsv(cublasHandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, p, SigmaScratch, 2*p, beta, 1)

    // compute Xtz, store in muScratch
    cublasSgemv(cublasHandle,CUBLAS_OP_T,n,p,ptrOnef,X,n,z,1,ptrZerof,muScratch,1)

    // compute mu = (XtX)^-1 Xtz by solving the system (XtX) mu = Xtz for mu, store in muScratch
    cublasStrsv(cublasHandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, p, SigmaScratch, 2*p, muScratch, 1)
    cublasStrsv(cublasHandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, p, SigmaScratch, 2*p, muScratch, 1)

    // compute beta = Sigma s + mu, store in beta
    cublasSaxpy(cublasHandle, p, ptrOnef, muScratch, 1, beta, 1)
  }

  def updateZ(): Unit = {
    cublasSetStream(cublasHandle, zStream_t)
    // draws z ~ TN(Xbeta,1), positive truncated if y=1, negative truncated if y=0 via inversion

    // compute Xbeta, store in z
    cublasSgemv(cublasHandle,CUBLAS_OP_N,n,p,ptrOnef,X,n,beta,1,ptrZerof,z,1)

    // draw z ~ TN via rejection sampler, passing in device pseudorandom number generator
    // call custom kernel cuda_positive_tnorm
    val (blockSizeX, gridSizeX) = tnormGetBlockAndGridSize(n)
    cuLaunchKernel(tnormFunction, gridSizeX,1,1, blockSizeX,1,1, 0, zCUstream, tnormInputPointer, null)
  }

  def cleanup(): Unit = {
    val gpuVariables = Array(y, z, X, beta,
      lambdaSqInv, nuInv, tauSqInv, tauSqInvShape,
      SigmaScratch, muScratch, betaSqScratch,
      curandStateZ, ptrOnef, ptrZerof,
      qrWorkspace, qrXWorkspace,
      cholWorkspace, cusolverDevInfo)
    gpuVariables.foreach(cudaFree)
    cublasDestroy(cublasHandle)
    curandDestroyGenerator(curandGenerator)
    cusolverDnDestroy(cusolverHandle)
    val modules = Array(betaModule, lambdaModule, nuXiModule, randInitModule, tauModule, tnormModule, RLModule) //svdModule
    modules.foreach(cuModuleUnload)
  }

  implicit class JCudaImplicits(i: Int) {
    def checkJCublasStatus() = {
      if(i != CUBLAS_STATUS_SUCCESS) {
        val failType = i match {
          case CUBLAS_STATUS_ALLOC_FAILED => "CUBLAS_STATUS_ALLOC_FAILED"
          case CUBLAS_STATUS_ARCH_MISMATCH => "CUBLAS_STATUS_ARCH_MISMATCH"
          case CUBLAS_STATUS_EXECUTION_FAILED => "CUBLAS_STATUS_EXECUTION_FAILED"
          case CUBLAS_STATUS_INTERNAL_ERROR => "CUBLAS_STATUS_INTERNAL_ERROR"
          case CUBLAS_STATUS_INVALID_VALUE => "CUBLAS_STATUS_INVALID_VALUE"
          case CUBLAS_STATUS_MAPPING_ERROR => "CUBLAS_STATUS_MAPPING_ERROR"
          case CUBLAS_STATUS_NOT_INITIALIZED => "CUBLAS_STATUS_NOT_INITIALIZED"
          case CUBLAS_STATUS_NOT_SUPPORTED => "CUBLAS_STATUS_NOT_SUPPORTED"
          case JCUBLAS_STATUS_INTERNAL_ERROR => "JCUBLAS_STATUS_INTERNAL_ERROR"
        }
        throw new Exception(failType)
      }
    }

    def checkJCusolverStatus() = {
      if(i != CUSOLVER_STATUS_SUCCESS) {
        val failType = i match {
          case CUSOLVER_STATUS_ALLOC_FAILED => "CUSOLVER_STATUS_ALLOC_FAILED"
          case CUSOLVER_STATUS_ARCH_MISMATCH => "CUSOLVER_STATUS_ARCH_MISMATCH"
          case CUSOLVER_STATUS_EXECUTION_FAILED => "CUSOLVER_STATUS_EXECUTION_FAILED"
          case CUSOLVER_STATUS_INTERNAL_ERROR => "CUSOLVER_STATUS_INTERNAL_ERROR"
          case CUSOLVER_STATUS_INVALID_LICENSE => "CUSOLVER_STATUS_INVALID_LICENSE"
          case CUSOLVER_STATUS_INVALID_VALUE => "CUSOLVER_STATUS_INVALID_VALUE"
          case CUSOLVER_STATUS_MAPPING_ERROR => "CUSOLVER_STATUS_MAPPING_ERROR"
          case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED => "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED"
          case CUSOLVER_STATUS_NOT_INITIALIZED => "CUSOLVER_STATUS_NOT_INITIALIZED"
          case CUSOLVER_STATUS_NOT_SUPPORTED => "CUSOLVER_STATUS_NOT_SUPPORTED"
          case CUSOLVER_STATUS_ZERO_PIVOT => "CUSOLVER_STATUS_ZERO_PIVOT"
        }
        throw new Exception(failType)
      }
    }

    def checkJCurandStatus() = {
      if(i != CURAND_STATUS_SUCCESS) {
        val failType = i match {
          case CURAND_STATUS_ALLOCATION_FAILED => "CURAND_STATUS_ALLOCATION_FAILED"
          case CURAND_STATUS_ARCH_MISMATCH => "CURAND_STATUS_ARCH_MISMATCH"
          case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED => "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED"
          case CURAND_STATUS_INITIALIZATION_FAILED => "CURAND_STATUS_INITIALIZATION_FAILED"
          case CURAND_STATUS_INTERNAL_ERROR => "CURAND_STATUS_INTERNAL_ERROR"
          case CURAND_STATUS_LAUNCH_FAILURE => "CURAND_STATUS_LAUNCH_FAILURE"
          case CURAND_STATUS_LENGTH_NOT_MULTIPLE => "CURAND_STATUS_LENGTH_NOT_MULTIPLE"
          case CURAND_STATUS_NOT_INITIALIZED => "CURAND_STATUS_NOT_INITIALIZED"
          case CURAND_STATUS_OUT_OF_RANGE => "CURAND_STATUS_OUT_OF_RANGE"
          case CURAND_STATUS_PREEXISTING_FAILURE => "CURAND_STATUS_PREEXISTING_FAILURE"
          case CURAND_STATUS_TYPE_ERROR => "CURAND_STATUS_TYPE_ERROR"
          case CURAND_STATUS_VERSION_MISMATCH => "CURAND_STATUS_VERSION_MISMATCH"
        }
        throw new Exception(failType)
      }
    }

    def checkJCudaStatus() = {
      if(i != CUDA_SUCCESS){
        val failType = i match {
          case CUDA_ERROR_INVALID_VALUE => "CUDA_ERROR_INVALID_VALUE"
          case CUDA_ERROR_OUT_OF_MEMORY => "CUDA_ERROR_OUT_OF_MEMORY"
          case CUDA_ERROR_NOT_INITIALIZED => "CUDA_ERROR_NOT_INITIALIZED"
          case CUDA_ERROR_DEINITIALIZED => "CUDA_ERROR_DEINITIALIZED"
          case CUDA_ERROR_PROFILER_DISABLED => "CUDA_ERROR_PROFILER_DISABLED"
          case CUDA_ERROR_NO_DEVICE => "CUDA_ERROR_NO_DEVICE"
          case CUDA_ERROR_INVALID_DEVICE => "CUDA_ERROR_INVALID_DEVICE"
          case CUDA_ERROR_INVALID_IMAGE => "CUDA_ERROR_INVALID_IMAGE"
          case CUDA_ERROR_INVALID_CONTEXT => "CUDA_ERROR_INVALID_CONTEXT"
          case CUDA_ERROR_CONTEXT_ALREADY_CURRENT => "CUDA_ERROR_CONTEXT_ALREADY_CURRENT"
          case CUDA_ERROR_MAP_FAILED => "CUDA_ERROR_MAP_FAILED"
          case CUDA_ERROR_UNMAP_FAILED => "CUDA_ERROR_UNMAP_FAILED"
          case CUDA_ERROR_ARRAY_IS_MAPPED => "CUDA_ERROR_ARRAY_IS_MAPPED"
          case CUDA_ERROR_ALREADY_MAPPED => "CUDA_ERROR_ALREADY_MAPPED"
          case CUDA_ERROR_NO_BINARY_FOR_GPU => "CUDA_ERROR_NO_BINARY_FOR_GPU"
          case CUDA_ERROR_ALREADY_ACQUIRED => "CUDA_ERROR_ALREADY_ACQUIRED"
          case CUDA_ERROR_NOT_MAPPED => "CUDA_ERROR_NOT_MAPPED"
          case CUDA_ERROR_NOT_MAPPED_AS_ARRAY => "CUDA_ERROR_NOT_MAPPED_AS_ARRAY"
          case CUDA_ERROR_NOT_MAPPED_AS_POINTER => "CUDA_ERROR_NOT_MAPPED_AS_POINTER"
          case CUDA_ERROR_ECC_UNCORRECTABLE => "CUDA_ERROR_ECC_UNCORRECTABLE"
          case CUDA_ERROR_UNSUPPORTED_LIMIT => "CUDA_ERROR_UNSUPPORTED_LIMIT"
          case CUDA_ERROR_CONTEXT_ALREADY_IN_USE => "CUDA_ERROR_CONTEXT_ALREADY_IN_USE"
          case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED => "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED"
          case CUDA_ERROR_INVALID_PTX => "CUDA_ERROR_INVALID_PTX"
          case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT => "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT";
          case CUDA_ERROR_INVALID_SOURCE => "CUDA_ERROR_INVALID_SOURCE"
          case CUDA_ERROR_FILE_NOT_FOUND => "CUDA_ERROR_FILE_NOT_FOUND"
          case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND => "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND"
          case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED => "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED"
          case CUDA_ERROR_OPERATING_SYSTEM => "CUDA_ERROR_OPERATING_SYSTEM"
          case CUDA_ERROR_INVALID_HANDLE => "CUDA_ERROR_INVALID_HANDLE"
          case CUDA_ERROR_NOT_FOUND => "CUDA_ERROR_NOT_FOUND"
          case CUDA_ERROR_NOT_READY => "CUDA_ERROR_NOT_READY"
          case CUDA_ERROR_ILLEGAL_ADDRESS => "CUDA_ERROR_ILLEGAL_ADDRESS"
          case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES"
          case CUDA_ERROR_LAUNCH_TIMEOUT => "CUDA_ERROR_LAUNCH_TIMEOUT"
          case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING => "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING"
          case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED"
          case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED => "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED"
          case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE => "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE"
          case CUDA_ERROR_CONTEXT_IS_DESTROYED => "CUDA_ERROR_CONTEXT_IS_DESTROYED"
          case CUDA_ERROR_ASSERT => "CUDA_ERROR_ASSERT"
          case CUDA_ERROR_TOO_MANY_PEERS => "CUDA_ERROR_TOO_MANY_PEERS"
          case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED => "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED"
          case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED => "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED"
          case CUDA_ERROR_HARDWARE_STACK_ERROR => "CUDA_ERROR_HARDWARE_STACK_ERROR"
          case CUDA_ERROR_ILLEGAL_INSTRUCTION => "CUDA_ERROR_ILLEGAL_INSTRUCTION"
          case CUDA_ERROR_MISALIGNED_ADDRESS => "CUDA_ERROR_MISALIGNED_ADDRESS"
          case CUDA_ERROR_INVALID_ADDRESS_SPACE => "CUDA_ERROR_INVALID_ADDRESS_SPACE"
          case CUDA_ERROR_INVALID_PC => "CUDA_ERROR_INVALID_PC"
          case CUDA_ERROR_LAUNCH_FAILED => "CUDA_ERROR_LAUNCH_FAILED"
          case CUDA_ERROR_NOT_PERMITTED => "CUDA_ERROR_NOT_PERMITTED"
          case CUDA_ERROR_NOT_SUPPORTED => "CUDA_ERROR_NOT_SUPPORTED"
          case CUDA_ERROR_UNKNOWN => "CUDA_ERROR_UNKNOWN"
        }
        throw new Exception(failType)
      }
    }
  }

  def sanityCheckInt(p: Pointer, dim: Int = 1): Unit = {
    val local = Array.ofDim[Int](dim)
    cudaMemcpy(Pointer.to(local), p, dim*Sizeof.INT, cudaMemcpyDeviceToHost)
    println(s"sanity check: ${local.mkString(",")}")
  }

  def sanityCheckFloat(p: Pointer, dim: Int = 1): Unit = {
    val local = Array.ofDim[Float](dim)
    cudaMemcpy(Pointer.to(local), p, dim*Sizeof.FLOAT, cudaMemcpyDeviceToHost)
    println(s"sanity check: ${local.mkString(",")}")
  }

  def printMemInfo(): Unit = {
    val cudaMemInfo = (Array(0L), Array(0L))
    cudaMemGetInfo(cudaMemInfo._1, cudaMemInfo._2)
    println(s"available memory ${cudaMemInfo._1.head}, total memory ${cudaMemInfo._2.head}")
  }
}
