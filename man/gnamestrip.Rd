\name{gnamestrip}
\alias{gnamestrip}
\title{
Strip a GPU object of it names, or row/colnames.
}
\description{
Returns a GPU object without any names or row/colnames. This may speed up certian operations.
}
\usage{
gnamestrip(x, dup = TRUE)
}
\arguments{
  \item{x}{
	input object to be stripped of names
}
  \item{dup}{
	if FALSE then don't duplicate \code{x} on the GPU. Return object will point to the same GPU memory as the original object. 
}
}

\value{
Returns the original object with no names.
}
\author{
Nathan Morris
}

\seealso{
\code{\link{names}}, \code{\link{colnames}} and \code{\link{dimnames}}
}
\examples{
x=gseq(1,2)
names(x)=c("bill","joe")
gnamestrip(x)

}
\keyword{names}

