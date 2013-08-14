\name{indexing}
\docType{methods}
\alias{[-methods}
\alias{[,gmatrix,index,index-method}
\alias{[,gmatrix,index,missing-method}
\alias{[,gmatrix,missing,index-method}
\alias{[,gvector,ANY,ANY-method}

\alias{[<--methods}
\alias{[<-,gmatrix,index,index-method}
\alias{[<-,gmatrix,index,missing-method}
\alias{[<-,gvector,ANY,ANY-method}
\alias{[<-,gmatrix,missing,index-method}

\title{Indexing Methods}
\description{
These methods are used to perform indexing operations. They work in much the same way as the base R functions. However, if the indexes involve very random numbers, it maybe very difficult for the GPU to coalesce the memory operations. Therefore it may be faster to move the object back to the host befor the indexing operations are performed.
}
\section{Methods}{
\describe{
\item{\code{signature(x = "gmatrix", i = "gvector", j = "gvector")}}{}
\item{\code{signature(x = "gmatrix", i = "gvector", j = "missing")}}{}
\item{\code{signature(x = "gmatrix", i = "gvector", j = "numeric")}}{}
\item{\code{signature(x = "gmatrix", i = "numeric", j = "gvector")}}{}
\item{\code{signature(x = "gmatrix", i = "numeric", j = "missing")}}{}
\item{\code{signature(x = "gmatrix", i = "numeric", j = "numeric")}}{}
\item{\code{signature(x = "gvector", i = "ANY", j = "ANY")}}{}
}}
\keyword{methods}
