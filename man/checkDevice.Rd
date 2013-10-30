\name{checkDevice}
\alias{checkDevice}
\title{
Check that the specified integer is the current device.
}
\description{
Mostly for internal uses.
}
\usage{
checkDevice(x)
} 

\arguments{
  \item{x}{
A vector of integers
}
}
\details{
Errors out if x is not the current device.
}
\value{
Returns TRUE or errors out.
}

\author{
Nathan Morris
}

\seealso{
\code{\link{getDevice}}, \code{\link{setDevice}} and \code{\link{device}}
}


\keyword{ device }

