\name{as.gmatrix, as.gvector}
\alias{as.gmatrix}
\alias{as.gmatrix-methods}
\alias{as.gmatrix,ANY-method}
\alias{as.gmatrix,gmatrix-method}
\alias{as.gmatrix,gvector-method}
\alias{as.gmatrix,matrix-method}
\alias{as.gvector}
\alias{as.gvector-methods}
\alias{as.gvector,ANY-method}
\alias{as.gvector,gmatrix-method}
\alias{as.gvector,gvector-method}

\title{Methods for coercing an object to a GPU class}
\description{
The \code{as.gmatrix} and \code{as.gvector} methods may be used to coerce an object to a given GPU class (i.e. \code{gmatrix} or \code{gvector}).
}
%as.gmatrix(data, type=NULL, dup=TRUE)

\arguments{
  \item{x, data}{object to be coerced}
  \item{type}{The type to convert the object to. There are three different ways to specify this. One may use the \code{"double"}, \code{"single"}, \code{"integer"} or \code{"logical"} notation. Alternatively the short form \code{"d"}, \code{"s"}, \code{"i"} or \code{"s"} notation may be used. Finally the internal integer representation of 0L, 1L, 2L or 3L may be used.}
  \item{dup}{This may be set to 'FALSE' to avoid duplicating x in the event that it is on the GPU allready and the original type is the same as the type argument.}
  \item{nrow,ncol,byrow,dimnames}{these attributes should generally only be set by a direct call to \code{\link{gmatrix}}}
 
}

\details{
The \code{as.gmatrix} and \code{as.gvector} methods may be used to move an R object to the current GPU device and returns an object of class
\code{gmatrix} or \code{gvector} respectively. The methods have the following arguments:

as.gmatrix(data = NA, nrow = 1L, ncol = 1L, byrow = FALSE,
    dimnames = NULL, type = NULL, dup = TRUE)
as.gvector(x, type=NULL, dup=TRUE)

The \code{dup} argument is used only when the input argument (\code{data}) is all ready on the GPU. 
\code{dup=FALSE} will return the original data without duplicating it on the GPU.
}

\value{
An object of type \code{gmatrix} or \code{gvector}
}

\author{
Nathan Morris
}
\seealso{
\code{\link{g}}, \code{\link{gmatrix}} and \code{\link{gvector}}
}
\examples{
A=matrix(1:10,2,5)
B=as.gmatrix(A)

}
