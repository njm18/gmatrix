\name{as.numeric, as.integer, as.logical, as.vector and as.matrix}
\docType{methods}
\alias{as.numeric-methods}
\alias{as.numeric,gmatrix-method}
\alias{as.numeric,gvector-method}
\alias{as.integer-methods}
\alias{as.integer,gmatrix-method}
\alias{as.integer,gvector-method}
\alias{as.logical-methods}
\alias{as.logical,gmatrix-method}
\alias{as.logical,gvector-method}
\alias{as.matrix.gmatrix}
\alias{as.matrix.gvector}
\alias{as.matrix-methods}
\alias{as.matrix,ANY-method}
\alias{as.matrix,gmatrix-method}
\alias{as.matrix,gvector-method}
\alias{as.vector.gmatrix}
\alias{as.vector.gvector}
\alias{as.vector-methods}
\alias{as.vector,ANY-method}
\alias{as.vector,gmatrix-method}
\alias{as.vector,gvector-method}
\alias{as.vector.gmatrix}
\alias{as.vector.gvector}

\title{Methods to coerce  a GPU object to an R object}
\description{
 The functions \code{as.numeric}, \code{as.integer}, \code{as.matrix} and \code{as.vector} are intended to bring a GPU object to the main 
 host memory and coerce it into the given type.
}
\section{Methods}{
\describe{
\item{\code{signature(x = "gmatrix")}}{}
\item{\code{signature(x = "gvector")}}{}
}}
\keyword{methods}
\seealso{
\code{\link{h}}
}
