\name{gsvd-class}
\Rdversion{1.1}
\docType{class}
\alias{svd}
\alias{gsvd-class}
\alias{svd-methods}
\alias{svd,ANY-method}
\alias{svd,gmatrix-method}


\title{Class \code{"gsvd"}}
\description{
Contains the results of an SVD decomposition as returned by the svd method. The SVD method is only available for cuda v7.0 or greater.
}
\section{Objects from the Class}{
Objects should be created by a call to the \code{svd()} method on a \code{gmatrix} object. The slots of this object are essentually the same as those created by the base \code{svd} function. However the \code{VT} is the transpose of the \code{v} slot in R base. The \code{gmatrix} \code{svd} function does not allow the user to select the number of singular values.
}
\section{Slots}{
  \describe{
    \item{\code{U}:}{Object of class \code{"gmatrix"} whose columns contain the left singular vectors. }
    \item{\code{S}:}{Object of class \code{"gvector"} containing the singular values.}
    \item{\code{VT}:}{Object of class \code{"gmatrix"} whose columns contain the TRANSPOSE of the left singular vectors.}
  }
}


\author{
Nathan Morris
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link{svd}}
}
\examples{
M=gmatrix(grnorm(25), 5,5,dup=FALSE)
mysvd=svd(M)
mysvd@U %*% diag(S) %*% mysvd@VT #reconstruct the matrix M
}
\keyword{classes}
