\name{gident}
\alias{gident}
\title{
Create an identity matrix on the GPU.
}
\description{Create an identity matrix on the GPU size}
\usage{
gident(n,val=1, type="d") 

}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{n}{Integer specifying the dimensions of the matrix.}
  \item{val}{The constant number to put on the diagonal. 1 leads to an identity matrix}
  \item{type}{The type of matrix to return}
}

\value{
Returns and n x n matrix with zeros on the of diagonals.
}

\author{
Nathan Morris
}

\seealso{
\code{\link{as.gmatrix}} and \code{\link{as.numeric}}
}
\examples{
ga = gident(40)
da = diag(40)
sum(h(ga) ==  da)
}

\keyword{transfer}
