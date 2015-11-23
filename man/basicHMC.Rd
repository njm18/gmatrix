\name{basicHMC}
\alias{basicHMC}

\title{
Performing Hamiltonian MCMC
}
\description{
This function performs Hamiltonian MCMC for continuous distributions.
}
\usage{
basicHMC(lpgr, initial, nsims, nsteps, step,
			burnin = 1, nstepsburnin = nsteps, stepburnin = step, Tstart = 1, r = 1,  
			keep=function(q) lapply(q, function(x) if(any(class(x) %in% c("gmatrix","gvector"))) g(x) else x),
			thin = 1, report = 100)
}

\arguments{
  \item{lpgr}{
This \code{lpgrf} input must be a function which takes as its input a list of parameters and gives as its output the log probability (\code{lp}) excluding the normalization constant and a first derivative of the log probability (\code{gr}). Both the \code{lp} and \code{gr} information must be returned as slots in an object of class \code{lpgr}. See details.
}
  \item{initial}{
Starting values as a list. Again each element of the starting value must be a matrix on the CPU or GPU. The number columns of each matrix is the number of parrallel runs. 
}
  \item{nsims}{
Total number of simulations includeing burnin.
}
  \item{nsteps}{
	Tuning parameter for HMC representing the number of leepfrog steps for each iteration. 
}
  \item{step}{
	Tuning parameter for HMC controls the length of each leepfrog steps for each iteration. 
}
  \item{nstepsburnin}{
	Tuning parameter for HMC representing the number of leepfrog steps for each iteration. This parameter is used only during burnin.
}
  \item{stepburnin}{
	Tuning parameter for HMC controls the length of each leepfrog steps for each iteration. This parameter is used only during burnin.
}
  \item{Tstart}{
	During the burnin phase only tempering is used to get the simulation moving. Tstart is the starting temperature. 
}
  \item{r}{
	The temperature during the burning phase is exponentially decreased at a rate \code{r}. The temperature is set to 1 if it drops below 1. Also, the temperature is set to 1 once the burnin phas is done.
}
  \item{burnin}{
	The number of iterations used for burnin. All burnin samples are discarded.
}
  \item{keep}{
	Function to extract samples to keep.
}
  \item{thin}{
	Selects the number of samples to keep. \code{thin=1} keeps all the samples. \code{thin=5} keeps only ever fifth sample.
}
  \item{report}{
	Controls how often an update of the progress is printed out.
}
}
\details{
 The most important input of \code{basicHMC()} is the \code{lpgrf} parameter. This \code{lpgrf} input must be a function which takes as its input a list of parameters and gives as its output the log probability (\code{lp}) excluding the normalization constant and a first derivative of the log probability (\code{gr}). Both the \code{lp} and \code{gr} information must be returned as slots in an object of class \code{lpgr}. The \code{lp} slot must be a numberic CPU vector while \code{gr} slot must be a list with elements either on the CPU or GPU.The function \code{basicHMC()} is designed to run multiple chains simultaneously. Thus each element in the input is expected to be a matrix where the number of columns is the number of parallel chains and the number of rows is the dimension of the random variable being simulated. The input is assumed to be a list of parameters because in many cases there are different types of parameters such as hyper-parameters. Tracking the current stated of the chain using a list is therefore often quite natural. As a result of this convention, the  \code{gr} slot for \code{lpgr} object returned by the  \code{lpgrf} function must also be a list.
}
\value{
Output is a list:
 \item{sims }{A list with the simulation results from each iteration as elements.}
 \item{lp }{A matrix with the simulation log probabilities for all simulations.}
 \item{AcceptanceRate }{Acceptence rate for each parallel chain.}
 \item{BurninAcceptanceRate }{Acceptence rate during the burnin phase for each parallel chain.} 
}
\references{
Neal, Radford M. "MCMC using Hamiltonian dynamics." Handbook of Markov Chain Monte Carlo 2 (2011).

Beam, Andrew L., Sujit K. Ghosh, and Jon Doyle. "Fast hamiltonian monte carlo using gpu computing." Journal of Computational and Graphical Statistics just-accepted (2015): 00-00.
}
\author{
Nathan Morris (this is actually a significantly modified version of an HMC implementation created by Radford Neal: http://www.cs.utoronto.ca/~radford/GRIMS.html)
}


\seealso{
\link{lpgr-class}
}
\examples{
Test
}

\keyword{ MCMC }
\keyword{ HMC }
