\name{rsample}
\alias{rsample}
\title{
Sample and integer for each row of a gmatrix object.
}
\description{
Sample an integer with the unormalized (log) probabilities specified for each row.
}
\usage{
rsample(P, log=TRUE)
}

\arguments{
  \item{P}{
A matrix conaining unormalizes (log) probabilities in each row.
}
  \item{log}{
Specifies if the input is on a log scale.
}
}
\details{
Conceptually, each row is first normalized to sum to one, and then an integer is sampled for each row under the specified probablility.
}
\value{
Returns a gvector with an integer value for each row.
}

\author{
Nathan Morris
}

\seealso{
\code{\link{gRowLogSums}}
}




