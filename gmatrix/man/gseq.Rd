\name{gseq}
\alias{gseq}
\alias{\%to\%}
\alias{\%to\%,numeric,numeric-method}
\title{
Sequence
}
\description{
Create a sequence on the GPU.
}
\usage{
gseq(from = 1, to = 1, by = ((to - from)/(length.out - 1)),
     length.out = NULL, along.with = NULL, type = NULL)
from \%to\% to
}

\arguments{
  \item{from, to}{
the starting and (maximal) end value of the sequence.
}

  \item{by}{
number: increment of the sequence.
}
  \item{length.out}{
desired length of the sequence. A non-negative number, which for seq and seq.int will be rounded up if fractional.
}
  \item{along.with}{
take the length from the length of this argument
}
  \item{type}{
output type (i.e. \code{"double"}, \code{"single"}, \code{"integer"} or \code{"logical"})
}
}

\value{
returns a \code{gvector} with the specified sequence.
}

\author{
Nathan Morris
}

\seealso{
\code{\link{seq}}
}
\examples{
gseq(1,100)
1\%to\%10
}
