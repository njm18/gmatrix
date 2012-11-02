\name{normal}
\alias{gdnorm}
\alias{grnorm}
\alias{gqnorm}
\alias{gpnorm}

\title{
GPU Normal Distribution
}
\description{
Density, distribution function, quantile function and random generation for the normal distribution with mean equal to mean and standard deviation equal to sd. All functions performed on the GPU.
}
\usage{
gdnorm(x, mean = 0, sd = 1, log = FALSE, type = "d")
grnorm(n, mean = 0, sd = 1, type = "d")
gqnorm(q, mean = 0, sd = 1, lower.tail = TRUE, log.p = FALSE,
       warn = TRUE, type = "d")
gpnorm(p, mean = 0, sd = 1, lower.tail = TRUE, log.p = FALSE,
       warn = TRUE, type = "d")
}

\arguments{
  \item{x,q}{vector of quantiles}
  \item{p}{vector of probabilities}
  \item{n}{number of observations}
  \item{mean}{vector of means}
  \item{sd}{vector of standard deviations}
  \item{log, log.p}{logical; if TRUE, probabilities p are given as log(p)}
  \item{lower.tail}{logical; if TRUE (default), probabilities are P[X < x] otherwise, P[X > x].}
  \item{warn}{logical; if FALSE then additional warnings are disabled.}
  \item{type}{specify the type; may be \code{"double"}, \code{"single"} (or short form \code{"d"} or \code{"s"}) }
}
\details{
Very similar to the R stats functionality. However, \code{gqnorm} and \code{gpnorm} are implement currently in a way that is NOT numerically precise in the deep tails. Future releases may fix this problem.
}
\value{
\code{gdnorm} gives the density, \code{gpnorm} gives the distribution function, \code{gqnorm} gives the quantile function, and \code{grnorm} generates random deviates. 
}

\author{
Nathan Morris
}
\seealso{
\code{\link{dnorm}}
}
\examples{
grnorm(100) #standard normals
grnorm(100, mean=c(1,2), sd=c(1,2,4,6))
}
\keyword{Normal}