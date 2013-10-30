\name{gouter}
\alias{gouter}
\alias{\%o\%}
\alias{\%o\%-methods}
\alias{\%o\%,ANY,ANY-method}
\alias{\%o\%,gvector,gvector-method}
\alias{\%o\%,gvector,logical-method}
\alias{\%o\%,gvector,numeric-method}
\alias{\%o\%,logical,gvector-method}
\alias{\%o\%,numeric,gvector-method}
\title{
GPU generalized outer product.
}
\description{
This function calculates the outer product of two gvectors.
}
\usage{
gouter(x, y, FUN = "*", normalOrder = TRUE)
}

\arguments{
  \item{x}{
	a 'vector' or \code{gvector} object
}
  \item{y}{
	a 'vector' or \code{gvector} object
}
  \item{FUN}{
	one of the following characters '*', '+', '-', '/' or '^'
}
  \item{normalOrder}{
	logical input determining which order the operation should be performed in.
}
}
\details{
This function is similar to the \code{\link{outer}} function in R base. It performs the specified operation on all posible combinations of elements from the first vector with the second. This function does not behave the same as R base when operating on matrices. Also, the operator \%o\% may be used as in the R base package.
}
\value{
returns a \code{gmatrix} with the specified operation performed
}

\author{
Nathan Morris
}
\seealso{
\code{\link{outer}}, \code{\link{\%o\%}}
}
\examples{
gouter(grnorm(10), grnorm(10), FUN="-")
}

\keyword{outer}

