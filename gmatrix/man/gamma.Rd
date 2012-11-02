\name{gdgamma}
\alias{gdgamma}
\alias{grgamma}

\title{
The Gamma Distribution
}
\description{
Density, and random generation for the Gamma distribution with parameters shape and scale. 
}
\usage{
gdgamma(x, shape, rate = 1, scale = 1/rate, log = FALSE, type = "d")
grgamma(n, shape, rate = 1, scale = 1/rate, type = "d")
}

\arguments{
  \item{x}{vector of quantiles}
  \item{n}{number of random values to return}
  \item{shape,scale}{shape and scale parameters. Must be positive, \code{scale} strictly}
  \item{rate}{an alternative way to specify the scale}
  \item{log}{logical; if TRUE, probabilities/densities p are returned as log(p)}
  \item{type}{specify the type; may be \code{"double"}, \code{"single"} (or short form \code{"d"} or \code{"s"}) }
}
\details{
Works in much the same way as the R base functions.
}
\value{
\code{gdgamma} gives the density and \code{grgamma} generates random deviates.
}
\references{
Random gamma simulation was implemented using the algorithm described in:
Marsaglia and Tsang (2000) A Simple Method for Generating Gamma Variables. ACM Transactions on Mathematical Software. Volume 26 Issue 3, Pages 363-372.
}
\author{
Nathan Morris
}
\seealso{
 \code{\link{dgamma}}
}
\examples{
grgamma(10, c(1,2))
}

\keyword{gamma}

