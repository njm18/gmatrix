\name{gRowLogSums}
\alias{gRowLogSums}
\title{
Sum the up the rows in log space.
}
\description{
Sum up the rows in log space.
}
\usage{
gRowLogSums(x, startCol=1, endCol=ncol(x))
}

\arguments{
  \item{x}{
A matrix }
  \item{startCol}{ May be used to perform the log sum on only selected columns}
  \item{endCol}{ May be used to perform the log sum on only selected columns}
}
\details{
Calculates for each row vector \code{v} the value \code{log(sum(exp(v)))} in a way that avoids overflow. This function may be useful for mixture models. This function is only efficient for a large number of rows and a smaller number of columns.
}
\value{
Returns a vector with a value for each row.
}

\author{
Nathan Morris
}

\seealso{
\code{\link{\%lgspadd\%-methods}}
}




