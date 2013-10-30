\name{convertType}
\alias{convertType}

\title{Convert the type of a GPU object}
\description{
This function is used to convert the type (i.e. \code{"double"}, \code{"single"}, \code{"integer"} or \code{"logical"}) of a \code{gmatrix} or \code{gvector} object. 
}
\usage{
convertType(x, to, dup = TRUE)
}

\arguments{
  \item{x}{A GPU object of class \code{gvector} or \code{gmatrix}}
  \item{to}{The type to convert the object to. There are three different ways to specify this. One may use the \code{"double"}, \code{"single"}, \code{"integer"} or \code{"logical"} notation. Alternatively the short form \code{"d"}, \code{"s"}, \code{"i"} or \code{"s"} notation may be used. Finally the internal integer representation of 0L, 1L, 2L or 3L may be used.}
  \item{dup}{This may be set to 'FALSE' to avoid duplicating \code{x} in the event the original type and the \code{to} type are the same}
}
%\details{
%%  ~~ If necessary, more details than the description above ~~
%}
\value{
Returns the x after converting it to a new type.
}

\author{
Nathan Morris
}

\seealso{
\code{\link{type}}
}
\examples{

x=gseq(1,10)
y=convertType(x,'d')
type(y)

}
\keyword{ type }

