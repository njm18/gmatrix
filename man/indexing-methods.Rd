\name{indexing}
\docType{methods}

\alias{[-methods}
\alias{[,ANY,ANY,ANY-method}
\alias{[,gmatrix,index,index-method}
\alias{[,gmatrix,index,missing-method}
\alias{[,gmatrix,missing,index-method}
\alias{[,gmatrix,missing,missing-method}
\alias{[,gvector,ANY,ANY-method}
\alias{[,nonStructure,ANY,ANY-method}


\alias{[<--methods}
\alias{[<-,ANY,ANY,ANY-method}
\alias{[<-,gmatrix,index,index-method}
\alias{[<-,gmatrix,index,missing-method}
\alias{[<-,gmatrix,missing,index-method}
\alias{[<-,gmatrix,missing,missing-method}
\alias{[<-,gvector,ANY,ANY-method}


\title{Indexing Methods}
\description{
These methods are used to perform indexing operations. They work in much the same way as the base R functions. However, if the indexes involve very random numbers, it maybe very difficult for the GPU to coalesce the memory operations. Therefore it may be faster to move the object back to the host befor the indexing operations are performed.
}

\section{Methods}{
\describe{

\item{\code{signature(x = "gvector", i = "ANY", j = "ANY")}}{}
\item{\code{signature(x = "ANY", i = "ANY", j = "ANY")}}{}
\item{\code{signature(x = "gmatrix", i = "index", j = "index")}}{}
\item{\code{signature(x = "gmatrix", i = "index", j = "missing")}}{}
\item{\code{signature(x = "gmatrix", i = "missing", j = "index")}}{}
\item{\code{signature(x = "gmatrix", i = "missing", j = "missing")}}{}
\item{\code{signature(x = "gvector", i = "ANY", j = "ANY")}}{}
\item{\code{signature(x = "nonStructure", i = "ANY", j = "ANY")}}{}


}
}
\keyword{methods}
