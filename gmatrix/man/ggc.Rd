\name{ggc}
\alias{ggc}
\title{
GPU Garbage Collection
}
\description{
Performs Garbage Collection and reports the available memory.
}
\usage{
ggc()
}

\details{
Mostly used to discover the amount of memory on the graphics card which is still available.
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

