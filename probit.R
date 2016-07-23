source("../TeX/AVTfunctions.R")

generate.probit.data = function(n, beta, id=0, write=FALSE) { 
  beta = matrix(beta, ncol=1)
  p = nrow(beta)
  x = matrix(rnorm(n*p),nrow=n,ncol=p)
  y = pnorm(x %*% beta) %>% round
  d = cbind(y,x) %>% data.frame %>% set_colnames(c("y", paste0("x",1:p))) 
  if(write) {
    write.table(d, paste0("data/probit-",id,".csv"), sep = ",", row.names = FALSE) 
    write.table(t(x) %*% x, paste0("data/probit-xtx-",id,".csv"), sep = ",", row.names = FALSE) 
  } else
    return(d) 
}

beta = c(1.3,4,-1,1.6,5,-2,rep(0,394))

generate.probit.data(1000,beta,1,TRUE)

n=1000
p=400
require(truncnorm)
d = read.csv("data/probit-1.csv")
XtX = read.csv("data/probit-xtx-1.csv")
X = d[,-1] %>% as.matrix
y = d[,1] %>% as.matrix
which(round(t(X) %*% X,5) != round(XtX,5)) %>% length
Xt = t(X)
lowertrunc = y %>% sapply(function(i) {
  if(i == 1) 0 else -Inf
})
uppertrunc = y %>% sapply(function(i) {
  if(i == 1) Inf else 0
})
R = chol(XtX)
XtXinv = chol2inv(R)
XtXinvXt = XtXinv %*% Xt

nMC=10000

beta = matrix(0, nrow=nMC, ncol=p)
z = matrix(0, nrow=nMC,ncol=n)

for(i in 2:nMC) {
  #sample beta
  # R = chol(XtX) #precomputed
  s = rnorm(p,0,1)
  Sigma.s = backsolve(R, s)
  Xtz = Xt %*% z[i-1,]
  mu = solve(XtX, Xtz)
  beta[i,] = Sigma.s + mu
  # beta[i,] = mvrnorm(1, XtXinvXt %*% z[i-1,], XtXinv)
  
  #sample z
  mu = X %*% beta[i,]
  z[i,] = rtruncnorm(n, a=lowertrunc, b=uppertrunc, mean=mu, sd=1)
  
  print.iteration(i, nMC)
}

plot(beta[,1])
plot(beta[,2])
plot(beta[,3])
plot(beta[,4])
plot(beta[,5])
plot(beta[,6])
round(colMeans(beta[,1:20]),2)
plot(z[,1])

out.beta = read.csv("~/Git/AsyncGibbsMPI/output/out-0-beta", header=FALSE)
out.z0 = read.csv("~/Git/AsyncGibbsMPI/output/out-0-z0", header=FALSE)
plot(out.beta[,1])
plot(out.beta[,2])
plot(out.beta[,3])
plot(out.beta[,4])
plot(out.beta[,5])
plot(out.beta[,6])
round(colMeans(out.beta[,1:20]),2)
plot(out.z0)

out.gpu.beta = read.csv("~/Git/AsyncGibbsMPI/output/out-GPU-beta", header=FALSE)
out.gpu.z = read.csv("~/Git/AsyncGibbsMPI/output/out-GPU-z", header=FALSE)
plot(out.gpu.beta[,1])
plot(out.gpu.beta[,2])
plot(out.gpu.beta[,3])
plot(out.gpu.beta[,4])
plot(out.gpu.beta[,5])
plot(out.gpu.beta[,6])
round(colMeans(out.gpu.beta[,1:20]),2)
plot(out.gpu.z[,1])
