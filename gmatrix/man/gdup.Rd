\name{gdup}
\alias{gdup}
\title{
Duplicate a GPU object
}
\description{
This function duplicates a \code{gmatrix} or \code{gvector} object on the specified device.
}
\usage{
gdup(x, dev=getDevice())
}

\arguments{
  \item{x}{
	object to duplicate
}
  \item{dev}{
	device to duplicate the object on
}
}
\details{
This function is very important because the \code{gmatrix} and \code{gvector} classes store only a pointer to an object on the GPU device. Thus, a line such as \code{y<-x} will duplicate the pointer but not the actual data on the GPU. One should use instead \code{y<-gdup(x)}. The \code{gdup} function is also useful for copying the data to a different GPU device.
}
\value{
a \code{gmatrix} or \code{gvector} object.
}

\author{
Nathan Morris
}

\seealso{
To move data to a different device instead of copying it see \code{\link{device}}
}
\examples{

#problematic code
x=gseq(1,10)
y=x 
x[1]=1000 
show(y)#changes to x also change y

#correct
x=gseq(1,10)
y=gdup(x) 
x[1]=1000 
show(y) #changes to x do not change y

#copy to a different device 
#y=gdup(x, dev=2L)
}
\keyword{copy }

