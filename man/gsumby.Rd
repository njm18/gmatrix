\name{gsumby}
\alias{gsumby}
\title{
Sumby
}
\description{
Find sub-sums of a \code{gvector}
}
\usage{
gsumby(v, startPos, stopPos)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{v}{a \code{gvector}}
  \item{startPos}{a vector of start positions}
  \item{stopPos}{a vector of stop positions}
}
\details{
\code{startPos} and \code{stopPos} must be of the same length.
}
\value{
returns a \code{gvector} object of the same length as \code{startPos} and \code{stopPos}. Each element is the sum of elements in \code{v}.
}

\author{
Nathan Morris
}


\examples{
gsumby(g(1:10), c(1,6), c(5,10)) # g(c(15,40))
}
