source("../TeX/AVTfunctions.R")

generate.probit.data = function(n, beta, id=0, write=FALSE) { 
  beta = matrix(beta, ncol=1)
  p = nrow(beta)
  x = matrix(rnorm(n*p),nrow=n,ncol=p)
  y = pnorm(x %*% beta + rnorm(n,0,1)) %>% round
  d = cbind(y,x) %>% data.frame %>% set_colnames(c("y", paste0("x",1:p))) 
  if(write) {
    write.table(d, paste0("data/probit-",id,".csv"), sep = ",", row.names = FALSE) 
    #write.table(t(x) %*% x, paste0("data/probit-xtx-",id,".csv"), sep = ",", row.names = FALSE) 
  } else
    return(d) 
}

beta = c(1.3,4,-1,1.6,5,-2,rep(0,100-6))

generate.probit.data(1000,beta,1,TRUE)

n=1000
p=100
require(truncnorm)
d = read.csv("data/probit-1.csv")
X = d[,-1] %>% as.matrix
y = d[,1] %>% as.matrix
XtX = t(X) %*% X
Xt = t(X)
lowertrunc = y %>% sapply(function(i) {
  if(i == 1) 0 else -Inf
})
uppertrunc = y %>% sapply(function(i) {
  if(i == 1) Inf else 0
})

nMC=10000

beta = matrix(0, nrow=nMC, ncol=p)
z = matrix(0, nrow=nMC,ncol=n)
lambdaSqInv = matrix(1, nrow=nMC, ncol=p)
nuInv = matrix(1, nrow=nMC, ncol=p)
tauSqInv = matrix(1, nrow=nMC, ncol=1)
xiInv = matrix(1, nrow=nMC, ncol=1)

for(i in 2:nMC) {
  #sample lambdaSqInv
  lambdaSqInv[i,] = rexp(p, rate = nuInv[i-1,] + (tauSqInv[i-1,] * (beta[i-1,]^2) / 2))
  
  #sample nuInv
  nuInv[i,] = rexp(p, rate = 1 + lambdaSqInv[i,])
  
  #sample tauSqInv
  tauSqInv[i,] = rgamma(1, (p+1)/2, rate = xiInv[i-1,] + (0.5 * crossprod(lambdaSqInv[i,], beta[i-1,]^2)))
  
  #sample xiInv
  xiInv[i,] = rexp(1, rate = 1 + tauSqInv[i,])
  
  #sample beta
  Sigma = XtX + (tauSqInv[i,] + diag(lambdaSqInv[i,]))
  R = chol(Sigma)
  s = rnorm(p,0,1)
  Sigma.s = backsolve(R, s)
  Xtz = Xt %*% z[i-1,]
  mu = solve(Sigma, Xtz)
  beta[i,] = Sigma.s + mu
  # beta[i,] = mvrnorm(1, SigmaInvXt %*% z[i-1,], SigmaInv)
  
  #sample z
  mu = X %*% beta[i,]
  z[i,] = rtruncnorm(n, a=lowertrunc, b=uppertrunc, mean=mu, sd=1)
  
  print.iteration(i, nMC)
}

round(colMeans(beta[,1:20]),2)
plot(beta[,1])
plot(beta[,2])
plot(beta[,3])
plot(beta[,4])
plot(beta[,5])
plot(beta[,6])
plot(z[,1])

out.gpu.beta = read.csv("~/Git/AsyncGibbsMPI/output/out-GPU-beta", header=FALSE)
out.gpu.z = read.csv("~/Git/AsyncGibbsMPI/output/out-GPU-z", header=FALSE)
out.gpu.lambda = read.csv("~/Git/AsyncGibbsMPI/output/out-GPU-lambda", header=FALSE)
out.gpu.nu = read.csv("~/Git/AsyncGibbsMPI/output/out-GPU-nu", header=FALSE)
out.gpu.tau = read.csv("~/Git/AsyncGibbsMPI/output/out-GPU-tau", header=FALSE)
out.gpu.xi = read.csv("~/Git/AsyncGibbsMPI/output/out-GPU-xi", header=FALSE)
round(colMeans(out.gpu.beta[,1:20]),2)
plot(out.gpu.beta[,1],plot=3)
plot(out.gpu.beta[,2],plot=3)
plot(out.gpu.beta[,3],plot=3)
plot(out.gpu.beta[,4],plot=3)
plot(out.gpu.beta[,5],plot=3)
plot(out.gpu.beta[,6],plot=3)
plot(out.gpu.beta[,7],plot=3)
plot(out.gpu.z[,1],plot=3)
plot(out.gpu.lambda[,1],plot=3)
plot(out.gpu.nu[,1],plot=3)
plot(out.gpu.tau[,1],plot=3)
plot(out.gpu.xi[,1],plot=3)
