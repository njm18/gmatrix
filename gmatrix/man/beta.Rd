\name{gdbeta}
\alias{gdbeta}
\alias{grbeta}

\title{
The Beta Distribution
}
\description{
Density and random generation for the Beta distribution with parameters \code{shape1} and \code{shape2}.
}
\usage{
gdbeta(x, shape1, shape2, ncp = 0, log = FALSE, type = "d")
grbeta(n, shape1, shape2, ncp = 0, type = "d")
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{vector of quantiles}
  \item{n}{number of random values to return}
  \item{shape1, shape2}{positive parameters of the Beta distribution}
  \item{ncp}{must be 0 at this point}
  \item{log}{logical; if TRUE, probabilities p are given as log(p)}
  \item{type}{specify the type; may be \code{"double"}, \code{"single"} (or short form \code{"d"} or \code{"s"}) }
}
\details{
Very similar to the R stats functionality.
}
\value{
\code{gdbeta} gives the density and \code{grbeta} generates random deviates. 
}

\author{
Nathan Morris
}
\seealso{
\code{\link{dbeta}}
}
\examples{
myRndNums = grbeta(10,c(1,2), c(1,2,3,4,5))
}

\keyword{beta}

