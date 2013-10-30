\name{gmatrix}
\alias{gmatrix}
\title{
Create and object of class \code{gmatrix}
}
\description{
This function creates a \code{gmatrix} object from an input value.
}
\usage{
gmatrix(data = NA, nrow = 1L, ncol = 1L, byrow = FALSE,
 dimnames = NULL, type = NULL, dup = TRUE)
}

\arguments{
  \item{data}{
 The input data. Must be a \code{vector} or \code{gvector}.
}
  \item{nrow}{
the desired number of rows
}
  \item{ncol}{
the desired number of rows.
}
  \item{byrow}{
logical. If FALSE (the default) the matrix is filled by columns, otherwise the matrix is filled by rows.
}
  \item{dimnames}{
A dimnames attribute for the matrix: NULL or a list of length 2 giving the row and column names respectively. An empty list is treated as NULL.
}
  \item{type}{
The type (i.e. \code{"double"}, \code{"single"}, \code{"integer"} or \code{"logical"})
}
  \item{dup}{
\code{dup=FALSE} will return the original data without duplicating it on the GPU. The \code{dup=FALSE} should only be used if the data in the input will not be used in it's original form again.
}
}
\details{
This is implemented in a very similar fashion to the \code{matrix} function in R base.
}
\value{
returns a \code{gmatrix} object.
}

\author{
Nathan Morrris
}

\seealso{
\code{\link{matrix}}
}
\examples{
x=gmatrix(grnorm(100),10,10, type='d') #this makes an unneeded duplication
x=gmatrix(grnorm(100),10,10, dup=FALSE) #this avoids the duplication

#problematic code
x=grnorm(100)
y=as.gmatrix(x,10,10, dup=FALSE) #don't do this
y[1,1]=100 #this changes both y and x because both point to the same spot in GPU memory
x
}

\keyword{ gmatrix }

