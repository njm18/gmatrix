\name{gdbinom}
\alias{gdbinom}
\alias{grbinom}
\title{The Binomial Distribution}
\description{
Density and random generation for the binomial distribution with parameters \code{size} and \code{prob}. 
}
\usage{
gdbinom(x, size, prob, log = FALSE, type = "d")
grbinom(n, size, prob)
}

\arguments{
  \item{x}{vector of quantiles.}
  \item{n}{number of observations}
  \item{size}{number of trials (zero or more)}
  \item{prob}{probability of success on each trial}
  \item{log}{logical; if TRUE, probabilities p are given as log(p).}
  \item{type}{specify the type; may be \code{"double"}, \code{"single"} (or short form \code{"d"} or \code{"s"}) }
}
\details{
Very similar to the R stats functionality.
}
\value{
\code{gdbinom} gives the density and \code{grbinom} generates random deviates
}

\author{
Nathan Morris
}
\note{
The algorythm for simulating from the poisson distribution was modified from the GSL library which uses the modified from the GSL library which uses the Knuth method.
}

\seealso{
\code{\link{dbinom}}
}
\examples{
myRndNums = grbinom(10,10,.6)
}
\keyword{binomial}

