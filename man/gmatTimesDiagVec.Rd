\name{gmatTimesDiagVec}
\alias{gmatTimesDiagVec}
\title{
Multiply a matrix times the 'diag' of a vector quickly.
}
\description{
This function calculates \code{A \%*\% diag(v)}. It is much faster than the direct calculation as it avoids the huge matrix multiplication problem.
}
\usage{
gmatTimesDiagVec(A, v)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{A}{
	An object of class \code{gmatrix} or 'matrix'.
}
  \item{v}{
	An object of class 'gvecotr' or a 'vector'.
}
}

\value{
Returns a \code{gmatrix}
}

\author{
Nathan Morris
}
\note{
Note that to calculate the \code{diag(v) \%*\% A} one may use the efficient formula \code{v*A}.
}


\seealso{
\code{\link{diag}}
}
\examples{
v=grnorm(10)
A=gmatrix(grnorm(100),10,10,dup=FALSE)
gmatTimesDiagVec(A, v)
}
\keyword{multiplication }

