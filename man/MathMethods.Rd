\name{elementwise functions}
\docType{methods}
\alias{abs-methods}
\alias{abs,gmatrix-method}
\alias{abs,gvector-method}

\alias{sqrt-methods}
\alias{sqrt,gmatrix-method}
\alias{sqrt,gvector-method}

\alias{exp-methods}
\alias{exp,gmatrix-method}
\alias{exp,gvector-method}

\alias{expm1-methods}
\alias{expm1,gmatrix-method}
\alias{expm1,gvector-method}

\alias{log-methods}
\alias{log,gmatrix-method}
\alias{log,gvector-method}

\alias{log2-methods}
\alias{log2,gmatrix-method}
\alias{log2,gvector-method}

\alias{log10-methods}
\alias{log10,gmatrix-method}
\alias{log10,gvector-method}

\alias{log1p-methods}
\alias{log1p,gmatrix-method}
\alias{log1p,gvector-method}

\alias{sin-methods}
\alias{sin,gmatrix-method}
\alias{sin,gvector-method}

\alias{cos-methods}
\alias{cos,gmatrix-method}
\alias{cos,gvector-method}

\alias{tan-methods}
\alias{tan,gmatrix-method}
\alias{tan,gvector-method}

\alias{asin-methods}
\alias{asin,gmatrix-method}
\alias{asin,gvector-method}

\alias{acos-methods}
\alias{acos,gmatrix-method}
\alias{acos,gvector-method}

\alias{atan-methods}
\alias{atan,gmatrix-method}
\alias{atan,gvector-method}

\alias{sinh-methods}
\alias{sinh,gmatrix-method}
\alias{sinh,gvector-method}

\alias{cosh-methods}
\alias{cosh,gmatrix-method}
\alias{cosh,gvector-method}

\alias{tanh-methods}
\alias{tanh,gmatrix-method}
\alias{tanh,gvector-method}

\alias{asinh-methods}
\alias{asinh,gmatrix-method}
\alias{asinh,gvector-method}

\alias{acosh-methods}
\alias{acosh,gmatrix-method}
\alias{acosh,gvector-method}

\alias{atanh-methods}
\alias{atanh,gmatrix-method}
\alias{atanh,gvector-method}

\alias{abs-methods}
\alias{abs,gmatrix-method}
\alias{abs,gvector-method}

\alias{lgamma-methods}
\alias{lgamma,gmatrix-method}
\alias{lgamma,gvector-method}

\alias{gamma-methods}
\alias{gamma,gmatrix-method}
\alias{gamma,gvector-method}

\alias{sign-methods}
\alias{sign,gmatrix-method}
\alias{sign,gvector-method}

\alias{round-methods}
\alias{round,gmatrix-method}
\alias{round,gvector-method}

\alias{ceiling-methods}
\alias{ceiling,gmatrix-method}
\alias{ceiling,gvector-method}

\alias{floor-methods}
\alias{floor,gmatrix-method}
\alias{floor,gvector-method}

\alias{is.na-methods}
\alias{is.na,gmatrix-method}
\alias{is.na,gvector-method}

\alias{is.nan-methods}
\alias{is.nan,gmatrix-method}
\alias{is.nan,gvector-method}

\alias{is.finite-methods}
\alias{is.finite,gmatrix-method}
\alias{is.finite,gvector-method}

\alias{is.infinite-methods}
\alias{is.infinite,gmatrix-method}
\alias{is.infinite,gvector-method}

\alias{!-methods}
\alias{!,gmatrix-method}
\alias{!,gvector-method}

\title{Elementwise functions/operators}
\description{
 This is the documentation for the functions \code{abs}, \code{exp}, \code{expm1}, \code{log}, \code{log2}, \code{log10}, \code{log1p},
  \code{sin}, \code{cos}, \code{tan}, \code{asin}, \code{acos}, \code{atan},
  \code{sinh}, \code{cosh}, \code{tanh}, \code{asinh}, \code{acosh}, \code{atanh},
   \code{abs}, \code{lgamma}, \code{gamma}, \code{sign}, \code{round}, \code{ceiling},
    \code{is.na}, \code{is.nan}, \code{is.finite}, \code{is.infinite} and \code{!}.
     These functions are elementwise functions/operators which are defined much the same as the functions the R base package. When operation on 
 object of class gmatrix or gvector the work will be performed by the GPU and will return a GPU object of the same dimension as the input.
}
\section{Methods}{
\describe{
\item{\code{signature(x = "gmatrix")}}{ Returns an object of type gmatrix with the same dimensions as the input.}
\item{\code{signature(x = "gvector")}}{ Returns an object of type gvector with the same dimensions as the input.}

}}
\keyword{methods}
\keyword{Trig}
