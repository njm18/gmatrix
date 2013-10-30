\name{gdpois}
\alias{gdpois}
\alias{grpois}
\title{
The Poisson Distribution
}
\description{
Density and random generation for the Poisson distribution with parameter lambda. 
}
\usage{
gdpois(x, lambda, log = FALSE, type = "d")
grpois(n, lambda)
}
\arguments{
  \item{x}{
vector of (non-negative integer) quantiles
}
  \item{n}{
number of random values to return
}
  \item{lambda}{
vector of (non-negative) means
}
  \item{log}{
logical; if TRUE, probabilities p are given as log(p)
}

 \item{type}{specify the type; may be \code{"double"}, \code{"single"} (or short form \code{"d"} or \code{"s"}) }
}
\details{
Very similar to the R stats functionality.
}
\value{
\code{gdpois} gives the (log) density, and \code{grpois} generates random deviates. 
}

\author{
Nathan Morris
}
\note{
The algorithm for simulating from the poisson distribution was modified from the GSL library which uses the Knuth method.
}

\seealso{
\code{\link{dpois}}
}
\examples{
myRndNums = grpois(10,4)
}

\keyword{poisson }

