\name{rowLogSums}
\alias{rowLogSums}
\title{
Sum the up the rows in log space.
}
\description{
Sum up the rows in log space.
}
\usage{
rowLogSums(x)
}

\arguments{
  \item{x}{
A matrix
}
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




