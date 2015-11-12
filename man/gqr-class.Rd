\name{gqr-class}
\Rdversion{1.1}
\docType{class}
\alias{gqr-class}


\alias{solve}
\alias{solve-methods}
\alias{solve,ANY-method}
\alias{solve,gmatrix-method}
\alias{solve,gqr-method}
\alias{qr-methods}
\alias{qr,ANY-method}
\alias{qr,gmatrix-method}
\alias{gqr.coef}
\alias{gqr.coef-methods}
\alias{gqr.coef,ANY-method}
\alias{gqr.coef,gqr-method}


\title{Class \code{"gqr"}}
\description{
Contains the results of a QR decomposition from using the \code{qr()} method on a \code{gmatrix} object. Note that the \code{gmatrix} \code{solve} function uses QR decompostion. Also, \code{qr.coef} can act on the \code{gqr} object. The \code{qr} method does not use pivoting or check the matrix rank. The QR and related methods are only available for cuda v7.0 or greater.
}
\section{Objects from the Class}{
Objects can be created by calls of the \code{qr()} method.
}
\section{Slots}{
  \describe{
    \item{\code{qr}:}{Information about the Q and R matrices stored as and bject of class \code{"gmatrix"}.}
    \item{\code{qraux}:}{Information about the Q matrix stored as an object of class \code{"gvector"}. }
  }
}
\section{Methods}{
  \describe{
    \item{qr.coef}{Simular to the \code{\link{qr.coef}} function. }
    \item{solve}{Simular to the \code{\link{solve.qr}} function. }
	 }
}

\author{
Nathan Morris
}


\seealso{
\code{\link{qr}}, \code{\link{qr.coef}}, \code{\link{solve.qr}}, \code{\link{solve}}
}
\examples{
M=gmatrix(grnorm(25), 5,5,dup=FALSE)
y=grnorm(5)
solve(M)
solve(M,y)
myqr =qr(M)
solve(myqr,y)
}
\keyword{classes}