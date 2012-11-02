\name{grunif}
\alias{grunif}
\alias{gdunif}
\title{
The Uniform Distribution
}
\description{
These functions provide information about the uniform distribution on the interval from \code{min} to \code{max}. dunif gives the density and runif generates random deviates.
}
\usage{
gdunif(x, min = 0, max = 1, log = FALSE, type = "d")
grunif(n, min = 0, max = 1, type = "d")

}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{vector of quantiles.}
  \item{n}{number of observations}
  \item{min, max}{lower and upper limits of the distribution}
  \item{log}{logical; if TRUE, probabilities/densities p are returned as log(p)}
  \item{type}{specify the type; may be \code{"double"}, \code{"single"} (or short form \code{"d"} or \code{"s"}) }
}
\details{
If min or max are not specified they assume the default values of 0 and 1 respectively. Works very similarly to the R stats functions.
}

\author{
Nathan Morris
}
\note{
Random number generation implemented with the cuRAND library.
}

\seealso{
\code{\link{runif}}
}
\examples{
grunif(10)

}
\keyword{ uniform }

