\name{ggc}
\alias{ggc}
\title{
GPU Garbage Collection
}
\description{
Performs Garbage Collection and reports the available memory.
}
\usage{
ggc(silent=FALSE)
}
\arguments{
  \item{silent}{If silent is TRUE then nothing will be printed out.}
 }

\details{
Used for garbage collection, and also used to discover the amount of memory on the graphics card which is still available.
}
\value{
returns TRUE, but prints the amount of memory left.
}

\author{
Nathan Morris
}


\seealso{
\code{\link{gc}}
}
\examples{
ggc()
}
\keyword{Garbage}

