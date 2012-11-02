\name{g}
\alias{g}
\alias{h}
\title{
Transfer and R object to and from the GPU.
}
\description{The function \code{g} transfers the object to the GPU.
 The function \code{h} transfers it back to the host.}
\usage{
g(x, type = NULL, dup = TRUE)
h(x)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{The object to be transfered.}
  \item{type}{The type to convert the object to. There are three different ways to specify this. One may use the \code{"double"}, \code{"single"}, \code{"integer"} or \code{"logical"} notation. Alternatively the short form \code{"d"}, \code{"s"}, \code{"i"} or \code{"s"} notation may be used. Finally the internal integer representation of 0L, 1L, 2L or 3L may be used.}
  \item{dup}{This may be set to 'FALSE' to avoid duplicating x in the event that it is on the GPU allready and the original type is the same as the type argument.}
}
\details{
These functions may be more convenient than functions such as 'as.gmatrix' or 'as.gvector'. In addition to having a very short name, one does not need to consider whether the object is a vector or matrix.
}
\value{
The 'h' function transfers device data back to the host and returns a matrix or vector, while the 'g' function 
returns an object of class \code{gmatrix} or \code{gvector} storing the data on the GPU.
}

\author{
Nathan Morris
}

\seealso{
\code{\link{as.gmatrix}} and \code{\link{as.numeric}}
}
\examples{
h_x=1:10
g_x=g(h_x) #Transfer to the device
g_ans=crossprod(g_x) #Do a a calculation
h_ans=h(g_ans) #Transfer back to the host
}

\keyword{transfer}
