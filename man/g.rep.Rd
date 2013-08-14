\name{g.rep}
\alias{g.rep}
\title{
Replicate Elements of a vector or gvector,
}
\description{
\code{g.rep} replicates the 'vector' or \code{gvector}.
}
\usage{
g.rep(x, times = 1L, each = NULL, type = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{
	a 'vector' or 'gvector.'
}
  \item{times}{
	a single integer representing the number of times the object should be replicated.
}
  \item{each}{
	a single integer representing the number of times each element should be replicated. If each is specified, the times will be ignored.
}
  \item{type}{
	coerce the output to the given type (i.e \code{"double"}, \code{"single"}, \code{"integer"} or \code{"logical"}).
}
}
\details{
\code{g.rep} is somewhat simplified from the R base \code{rep} function. It is called \code{g.rep} to avoid conflict with the \code{grep} function. It can replicate in two different ways. If times is specified then the entire vector is replicated, while when each is specified each element is replicated one at a time.
}
\value{
a \code{gvector} object with the replicated information
}

\author{
Nathan Morris
}

\seealso{
\code{\link{rep}}
}
\examples{
x=g.rep(1:2, 2) #returns (1,2,1,2)
x=g.rep(1:2, each=2) #returns (1,1,2,2)
}
\keyword{replicate}

