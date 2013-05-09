\name{crossprod,tcrossprod}
\docType{methods}
\alias{crossprod-methods}
\alias{crossprod,ANY,ANY-method}
\alias{crossprod,gmatrix,gmatrix-method}
\alias{crossprod,gmatrix,gvector-method}
\alias{crossprod,gmatrix,logical-method}
\alias{crossprod,gmatrix,matrix-method}
\alias{crossprod,gmatrix,missing-method}
\alias{crossprod,gmatrix,numeric-method}
\alias{crossprod,gvector,gmatrix-method}
\alias{crossprod,gvector,gvector-method}
\alias{crossprod,gvector,logical-method}
\alias{crossprod,gvector,missing-method}
\alias{crossprod,gvector,numeric-method}
\alias{crossprod,logical,gmatrix-method}
\alias{crossprod,logical,gvector-method}
\alias{crossprod,matrix,gmatrix-method}
\alias{crossprod,numeric,gmatrix-method}
\alias{crossprod,numeric,gvector-method}
\alias{crossprod,gvector,matrix-method}
\alias{crossprod,matrix,gvector-method}

\alias{tcrossprod-methods}
\alias{tcrossprod,ANY,ANY-method}
\alias{tcrossprod,gmatrix,gmatrix-method}
\alias{tcrossprod,gmatrix,gvector-method}
\alias{tcrossprod,gmatrix,logical-method}
\alias{tcrossprod,gmatrix,matrix-method}
\alias{tcrossprod,gmatrix,missing-method}
\alias{tcrossprod,gmatrix,numeric-method}
\alias{tcrossprod,gvector,gmatrix-method}
\alias{tcrossprod,gvector,gvector-method}
\alias{tcrossprod,gvector,logical-method}
\alias{tcrossprod,gvector,missing-method}
\alias{tcrossprod,gvector,numeric-method}
\alias{tcrossprod,logical,gmatrix-method}
\alias{tcrossprod,logical,gvector-method}
\alias{tcrossprod,matrix,gmatrix-method}
\alias{tcrossprod,numeric,gmatrix-method}
\alias{tcrossprod,numeric,gvector-method}
\alias{tcrossprod,gvector,matrix-method}
\alias{tcrossprod,matrix,gvector-method}
\title{Matrix Crossproduct}
\description{
 Given matrices x and y as arguments, return a matrix cross-product. This is formally equivalent to (but usually slightly faster than) the call \code{t(x) \%*\% y} (crossprod) or \code{x \%*\% t(y)} (tcrossprod). 
}
\section{Methods}{
\describe{
\item{\code{signature(x = "ANY", y = "ANY")}}{}
\item{\code{signature(x = "gmatrix", y = "gmatrix")}}{}
\item{\code{signature(x = "gmatrix", y = "gvector")}}{}
\item{\code{signature(x = "gmatrix", y = "logical")}}{}
\item{\code{signature(x = "gmatrix", y = "matrix")}}{}
\item{\code{signature(x = "gmatrix", y = "missing")}}{}
\item{\code{signature(x = "gmatrix", y = "numeric")}}{}
\item{\code{signature(x = "gvector", y = "gmatrix")}}{}
\item{\code{signature(x = "gvector", y = "gvector")}}{}
\item{\code{signature(x = "gvector", y = "logical")}}{}
\item{\code{signature(x = "gvector", y = "missing")}}{}
\item{\code{signature(x = "gvector", y = "numeric")}}{}
\item{\code{signature(x = "logical", y = "gmatrix")}}{}
\item{\code{signature(x = "logical", y = "gvector")}}{}
\item{\code{signature(x = "matrix", y = "gmatrix")}}{}
\item{\code{signature(x = "numeric", y = "gmatrix")}}{}
\item{\code{signature(x = "numeric", y = "gvector")}}{}
\item{\code{signature(x = "ANY", y = "ANY")}}{}
\item{\code{signature(x = "gmatrix", y = "gmatrix")}}{}
\item{\code{signature(x = "gmatrix", y = "gvector")}}{}
\item{\code{signature(x = "gmatrix", y = "logical")}}{}
\item{\code{signature(x = "gmatrix", y = "matrix")}}{}
\item{\code{signature(x = "gmatrix", y = "missing")}}{}
\item{\code{signature(x = "gmatrix", y = "numeric")}}{}
\item{\code{signature(x = "gvector", y = "gmatrix")}}{}
\item{\code{signature(x = "gvector", y = "gvector")}}{}
\item{\code{signature(x = "gvector", y = "logical")}}{}
\item{\code{signature(x = "gvector", y = "missing")}}{}
\item{\code{signature(x = "gvector", y = "numeric")}}{}
\item{\code{signature(x = "logical", y = "gmatrix")}}{}
\item{\code{signature(x = "logical", y = "gvector")}}{}
\item{\code{signature(x = "matrix", y = "gmatrix")}}{}
\item{\code{signature(x = "numeric", y = "gmatrix")}}{}
\item{\code{signature(x = "numeric", y = "gvector")}}{}
}}
\keyword{methods}

