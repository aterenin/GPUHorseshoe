# GPUHorseshoe

This is an implementation of a Gibbs sampler for the Horseshoe Probit model using CUDA.
It is written in Scala using JCuda as an interface.
It also includes a CPU implementation for the same model.
This algorithm was showcased in *[GPU-accelerated Gibbs Sampling: a case study of the Horseshoe Probit model](https://arxiv.org/abs/1608.04329)*.

To compile, you will first need to download [JCuda](http://www.jcuda.org) for your platform.
Create a directory called `lib` inside the project's repository and place the `jar` files for JCuda there. Ensure that your system's CUDA directory is present in its `PATH` variable.

To compile the Scala code, run `sbt assembly`, which will create a `jar` file that can run on the JVM.
To compile device code, use `nvcc`, with the `ptx` flag to create a suitable PTX file for your GPU, and ensure these are placed in your current directory so that the program can load them at runtime.

Once compiled, the program can be executing using `java -jar /path/to/jar`. If you get an out of memory error, you may need to increase maximum memory using for example `-xmX8g`.
