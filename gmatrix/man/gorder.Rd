\name{gorder}
\alias{gorder}
\title{
Ordering Permutation
}
\description{
\code{gorder} returns a permutation which rearranges its first argument into ascending or descending order.
}
\usage{
gorder(x, decreasing = FALSE, stable = TRUE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{
object to be ordered
}
  \item{decreasing}{
logical to specify sort order
}
  \item{stable}{
make sure that ties are in the original order
}
}
\details{
Unlike the base R function, only one input column may be given. The thrust library is used for this operation.
}
\value{
a \code{gvector} of type \code{"integer"} which will permute the input vector to the proper sort order.
}

\author{
Nathan Morris
}
\seealso{
\code{\link{order}}
}
\examples{
x=grnorm(10)
i=gorder(x)
x[i] #sort x
}


