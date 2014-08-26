\name{gmm}
\alias{gmm}
\title{
Multiply two matrices with the output matrix allready allocated.
}
\description{
In effect this function sets C <- A \%*\% B. This makes it possible to avoid repeated allocation and deallocation steps which are very computationaly costly.
}
\usage{
gmm(A, B, C, trA=FALSE, trB=FALSE) 
}

\arguments{
  \item{A, B}{
	Matrices to multiply together.
}
  \item{C}{
	Matrices where output will be saved.
}
  \item{trA, trB}{
  Should A or B be transposed.
}


}
\details{
This function takes advantage of the fact that gmatrix objects are stored as pointers making it posible modify the input to a function. Thus the area of memory on the GPU pointed to by the C object is modified to contain A \%*\% B. 
}
\value{
Returns C invisibly.
}

\author{
Nathan Morris
}


\examples{
A=gmatrix(rnorm(10), 2, 5)
B=gmatrix(rnorm(10), 5, 2)
C=gmatrix(0,2,2)
gmm(A,B,C)
gmm(B,A,C,TRUE,TRUE)

}
\keyword{multiplication}

