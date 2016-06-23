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

d = read.csv("data/probit-1.csv")
xtx = read.csv("data/probit-xtx-1.csv")

out.beta = read.csv("~/Git/AsyncGibbsMPI/output/out-beta", header=FALSE)
out.z0 = read.csv("~/Git/AsyncGibbsMPI/output/out-z0", header=FALSE)
plot(out.beta[,1])
plot(out.beta[,2])
plot(out.beta[,3])
plot(out.beta[,4])
plot(out.beta[,5])
plot(out.beta[,6])
round(colMeans(out.beta[,1:20]),2)
plot(out.z0)
