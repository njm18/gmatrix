\name{gmatrix-package}
\alias{gmatrix-package}
%\alias{gmatrix}
\docType{package}
\title{
Making GPU power available in R
}
\description{
This package implements a general framework for utilizing R to harness the power
of NVIDIA GPU's. The \code{gmatrix} and \code{gvector} classes allow for easy management
of the separate device and host memory spaces. Numerous numerical operations are
implemented for these objects on the GPU. These operations include matrix
multiplication, addition, subtraction, the kronecker product, the outer product,
comparison operators, logical operators, trigonometric functions, indexing, sorting,
random number generation and many more.
}
\details{
	\tabular{ll}{
		Package: \tab gmatrix\cr
		Type: \tab Package\cr
		Version: \tab 0.5\cr
		Date: \tab 2013-5-10\cr
		License: \tab GPL-2\cr
		LazyLoad: \tab yes\cr
		Depends: \tab methods\cr
	}
To move an R object to the GPU use the \code{g} function. To move the object back the the host use the \code{h} function. 
Standard operations such as matrix multiplication or addition may be performed on GPU objects as with any \code{matrix} or \code{vector} object in R. 
The \code{gmatrix} and \code{gvector} classes contain a \code{type} slot which determines if the data is stored as a \code{"double"}, \code{"single"}, \code{"integer"} or \code{"logical."}
The \code{gmatrix} and \code{gvector} objects store only a pointer to an object on the GPU device. Thus, a line such as \code{y<-x} will duplicate the pointer but not the actual data on the GPU. One should use instead \code{y<-gdup(x)}.
Numerous random number generators such as \code{grnorm} have been implemented in this package. This package is intended to work with the companion package \code{rcula} which implements matrix factorization and inversions methods.
}
\author{
Nathan Morris
}
\references{
None
}

\keyword{NVIDIA}
\keyword{GPU}

\seealso{
\code{rcula-package}, \code{\link{g}}, \code{\link{h}}, \code{\link{gmatrix-class}}, \code{\link{gvector-class}}
}
\examples{
#Look at available GPU devices (use setDevice to change the device)
listDevices()

#Creating GPU objects
g_x=g(1:10) #transfer a sequence to the GPU
g_y=gseq(1,10) #create a sequence on the GPU
g_A=gmatrix(grnorm(100),10,10,dup=FALSE) #create a random 10x10 matrix

#look at and change the properties of these objects
type(g_x)
type(g_A)
type(g_x)="d" #'d' may be used as short form of"double"
length(g_x)
dim(g_A)
nrow(g_A)

#Perform some simple calculations
g_ans=g_x \%*\% log(abs(g_A)) \%*\% sin(-g_y)
g_rowsum= rowSums(g_A)

#Move data back to the host
h_ans=h(g_x)
h_roswum =h(g_A)
}
