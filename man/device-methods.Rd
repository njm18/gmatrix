\name{device}
\docType{methods}
\alias{device}
\alias{device<-}
\alias{device-methods}
\alias{device,gmatrix-method}
\alias{device,gvector-method}
\alias{device<--methods}
\alias{device<-,gmatrix-method}
\alias{device<-,gvector-method}
\title{Get and set methods for the device slot of a gvector or gmatrix object}
\description{
The \code{device} methods can be used to find what GPU device an object is stored on. 
The \code{device<-} can be used to move the object to a different GPU. Note that this is
 currently implemented by moving the object to host memory first.
}
\section{Methods}{
\describe{
\item{\code{signature(x = "gmatrix")}}{}
\item{\code{signature(x = "gvector")}}{}
}}

\seealso{
\code{\link{gdup}} and \code{\link{setDevice}} 
}
\keyword{methods}

