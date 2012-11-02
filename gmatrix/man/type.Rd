\name{type}
\alias{type}
\alias{type<-}
\alias{type<--methods}
\alias{type<-,gmatrix-method}
\alias{type<-,gvector-method}
\alias{type-methods}
\alias{type,gmatrix-method}
\alias{type,gvector-method}
\title{
Get or set the type of a GPU object
}
\description{
These functions may be used to get or set the type of a GPU object.
}
\usage{
type(x)
type(x)<-value
}

\arguments{
  \item{x}{A GPU object of class \code{gvector} or \code{gmatrix}}
  \item{value}{The type to convert the object to. There are three different ways to specify this. One may use the \code{"double"}, \code{"single"}, \code{"integer"} or \code{"logical"} notation. Alternatively the short form \code{"d"}, \code{"s"}, \code{"i"} or \code{"s"} notation may be used. Finally the internal integer representation of 0L, 1L, 2L or 3L may be used.}
}

\value{
type(x) returns the type of x
type(x)<-value change the type of x
}

\author{
Nathan Morris
}

\seealso{
\code{\link{convertType}}
}
\examples{
x=g(1:4)
type(x)="d" #change the type from "integer" to "double"
}
\keyword{type }

