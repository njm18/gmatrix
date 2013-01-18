\name{gkroneckerProd}
\alias{gkroneckerProd}
\alias{\%x\%-methods}
\alias{\%x\%,ANY,ANY-method}
\alias{\%x\%,gmatrix,gmatrix-method}
\alias{\%x\%,gmatrix,gvector-method}
\alias{\%x\%,gmatrix,numeric-method}
\alias{\%x\%,gvector,gmatrix-method}
\alias{\%x\%,gvector,gvector-method}
\alias{\%x\%,gvector,numeric-method}
\alias{\%x\%,numeric,gmatrix-method}
\alias{\%x\%,numeric,gvector-method}
\alias{\%x\%,gmatrix,matrix-method}
\alias{\%x\%,gvector,matrix-method}
\alias{\%x\%,matrix,gmatrix-method}
\alias{\%x\%,matrix,gvector-method}





\title{
Kronecker Product
}
\description{
Calculates the kronecker product of two matrices.
}
\usage{
gkroneckerProd(A, B)
A %x% B
}

\arguments{
  \item{A}{A \code{gmatrix}, \code{gvector}, \code{matrix} or \code{vector}}
  \item{B}{A \code{gmatrix}, \code{gvector}, \code{matrix} or \code{vector}}
}
\value{
A GPU object storing the Kronecker product.
}
\author{
Nathan Morris
}
