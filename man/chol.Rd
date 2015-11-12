\name{chol}
\docType{methods}
\alias{chol-methods}
\alias{chol,ANY-method}
\alias{chol,gmatrix-method}

\title{Method to calculate the Choleski decomposition square symmetric posative definite matrix}
\description{
Takes a square symmetric posative definite \code{gmatrix} object. Note that no checking for symmetry is done. An error will be thrown if the the matrix is not posative definite. Returns a \code{gmatrix} object that is upper triangular.
}
\section{Methods}{
\describe{
\item{\code{signature(x = "ANY")}}{}
\item{\code{signature(x = "gmatrix")}}{}
}}
\keyword{methods}

