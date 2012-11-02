\name{gvector-class}
\Rdversion{1.1}
\docType{class}
\alias{gvector-class}


\title{Class "gvector"}
\description{
This object contains a pointer to a vector on a NVIDIA GPU.
}
\section{Objects from the Class}{
Objects should NOT be created by calls of the form \code{new("gmatrix", ...)}. Instead use the \code{gvector}, \code{as.gvector} or \code{g} functions.
}
\section{Slots}{
  \describe{
    \item{\code{ptr}:}{Represents an external pointer to the data on the GPU}
    \item{\code{length}:}{Object of class \code{"integer"} representing the number of elements in the vector }
    \item{\code{names}:}{Should be a character vector with length equal nrow }
    \item{\code{type}:}{Must be 0L, 1L, 2L or 3L. These integers represent the types \code{"double"}, \code{"single"}, \code{"integer"} or \code{"logical"} respectively.}
    \item{\code{device}:}{Object of class \code{"integer"} representing the device}
  }
}
\section{Methods}{
  \describe{
    \item{-}{\code{signature(e1 = "gmatrix", e2 = "gvector")}: subtraction operator }
    \item{-}{\code{signature(e1 = "gvector", e2 = "gmatrix")}: subtraction operator }
    \item{-}{\code{signature(e1 = "gvector", e2 = "gvector")}: subtraction operator }
    \item{-}{\code{signature(e1 = "gvector", e2 = "logical")}: subtraction operator }
    \item{-}{\code{signature(e1 = "gvector", e2 = "matrix")}: subtraction operator }
    \item{-}{\code{signature(e1 = "gvector", e2 = "missing")}: subtraction operator }
    \item{-}{\code{signature(e1 = "gvector", e2 = "numeric")}: subtraction operator }
    \item{-}{\code{signature(e1 = "logical", e2 = "gvector")}: subtraction operator }
    \item{-}{\code{signature(e1 = "matrix", e2 = "gvector")}: subtraction operator. }
    \item{-}{\code{signature(e1 = "numeric", e2 = "gvector")}:subtraction operator }
    \item{!}{\code{signature(x = "gvector")}: not operator }
    \item{!=}{\code{signature(e1 = "gmatrix", e2 = "gvector")}: not equal operator }
    \item{!=}{\code{signature(e1 = "gvector", e2 = "gmatrix")}: not equal operator }
    \item{!=}{\code{signature(e1 = "gvector", e2 = "gvector")}: not equal operator }
    \item{!=}{\code{signature(e1 = "gvector", e2 = "logical")}: not equal operator }
    \item{!=}{\code{signature(e1 = "gvector", e2 = "matrix")}: not equal operator }
    \item{!=}{\code{signature(e1 = "gvector", e2 = "numeric")}: not equal operator }
    \item{!=}{\code{signature(e1 = "logical", e2 = "gvector")}: not equal operator }
    \item{!=}{\code{signature(e1 = "matrix", e2 = "gvector")}: not equal operator }
    \item{!=}{\code{signature(e1 = "numeric", e2 = "gvector")}: not equal operator }
    \item{\%\%}{\code{signature(e1 = "gmatrix", e2 = "gvector")}: mod operator }
    \item{\%\%}{\code{signature(e1 = "gvector", e2 = "gmatrix")}: mod operator }
    \item{\%\%}{\code{signature(e1 = "gvector", e2 = "gvector")}: mod operator }
    \item{\%\%}{\code{signature(e1 = "gvector", e2 = "logical")}: mod operator }
    \item{\%\%}{\code{signature(e1 = "gvector", e2 = "matrix")}: mod operator }
    \item{\%\%}{\code{signature(e1 = "gvector", e2 = "numeric")}: mod operator }
    \item{\%\%}{\code{signature(e1 = "logical", e2 = "gvector")}: mod operator }
    \item{\%\%}{\code{signature(e1 = "matrix", e2 = "gvector")}: mod operator }
    \item{\%\%}{\code{signature(e1 = "numeric", e2 = "gvector")}: mod operator }
    \item{\%*\%}{\code{signature(x = "gmatrix", y = "gvector")}: matrix multiplication operator  }
    \item{\%*\%}{\code{signature(x = "gvector", y = "gmatrix")}: matrix multiplication operator  }
    \item{\%*\%}{\code{signature(x = "gvector", y = "gvector")}: matrix multiplication operator  }
    \item{\%*\%}{\code{signature(x = "gvector", y = "logical")}: matrix multiplication operator  }
    \item{\%*\%}{\code{signature(x = "gvector", y = "matrix")}: matrix multiplication operator  }
    \item{\%*\%}{\code{signature(x = "gvector", y = "numeric")}: matrix multiplication operator }
    \item{\%*\%}{\code{signature(x = "logical", y = "gvector")}: matrix multiplication operator }
    \item{\%*\%}{\code{signature(x = "matrix", y = "gvector")}: matrix multiplication operator }
    \item{\%*\%}{\code{signature(x = "numeric", y = "gvector")}: matrix multiplication operator  }
    \item{\%o\%}{\code{signature(X = "gvector", Y = "gvector")}: outer product }
    \item{\%o\%}{\code{signature(X = "gvector", Y = "logical")}: outer product }
    \item{\%o\%}{\code{signature(X = "gvector", Y = "numeric")}: outer product }
    \item{\%o\%}{\code{signature(X = "logical", Y = "gvector")}: outer product }
    \item{\%o\%}{\code{signature(X = "numeric", Y = "gvector")}: outer product }
    \item{\%x\%}{\code{signature(X = "gmatrix", Y = "gvector")}: kronecker product }
    \item{\%x\%}{\code{signature(X = "gvector", Y = "gmatrix")}: kronecker product. }
    \item{\%x\%}{\code{signature(X = "gvector", Y = "gvector")}: kronecker product }
    \item{\%x\%}{\code{signature(X = "gvector", Y = "numeric")}: kronecker product }
    \item{\%x\%}{\code{signature(X = "numeric", Y = "gvector")}: kronecker product }
    \item{&}{\code{signature(e1 = "gmatrix", e2 = "gvector")}: and operator }
    \item{&}{\code{signature(e1 = "gvector", e2 = "gmatrix")}: and operator }
    \item{&}{\code{signature(e1 = "gvector", e2 = "gvector")}: and operator }
    \item{&}{\code{signature(e1 = "gvector", e2 = "logical")}: and operator }
    \item{&}{\code{signature(e1 = "gvector", e2 = "matrix")}: and operator }
    \item{&}{\code{signature(e1 = "gvector", e2 = "numeric")}: and operator }
    \item{&}{\code{signature(e1 = "logical", e2 = "gvector")}: and operator }
    \item{&}{\code{signature(e1 = "matrix", e2 = "gvector")}: and operator }
    \item{&}{\code{signature(e1 = "numeric", e2 = "gvector")}: and operator }
    \item{*}{\code{signature(e1 = "gmatrix", e2 = "gvector")}: elementwise multiplication }
    \item{*}{\code{signature(e1 = "gvector", e2 = "gmatrix")}: elementwise multiplication }
    \item{*}{\code{signature(e1 = "gvector", e2 = "gvector")}: elementwise multiplication }
    \item{*}{\code{signature(e1 = "gvector", e2 = "logical")}: elementwise multiplication }
    \item{*}{\code{signature(e1 = "gvector", e2 = "matrix")}: elementwise multiplication }
    \item{*}{\code{signature(e1 = "gvector", e2 = "numeric")}: elementwise multiplication }
    \item{*}{\code{signature(e1 = "logical", e2 = "gvector")}: elementwise multiplication }
    \item{*}{\code{signature(e1 = "matrix", e2 = "gvector")}: elementwise multiplication }
    \item{*}{\code{signature(e1 = "numeric", e2 = "gvector")}: elementwise multiplication }
    \item{/}{\code{signature(e1 = "gmatrix", e2 = "gvector")}: division operator }
    \item{/}{\code{signature(e1 = "gvector", e2 = "gmatrix")}: division operator }
    \item{/}{\code{signature(e1 = "gvector", e2 = "gvector")}: division operator }
    \item{/}{\code{signature(e1 = "gvector", e2 = "logical")}: division operator }
    \item{/}{\code{signature(e1 = "gvector", e2 = "matrix")}: division operator }
    \item{/}{\code{signature(e1 = "gvector", e2 = "numeric")}: division operator }
    \item{/}{\code{signature(e1 = "logical", e2 = "gvector")}: division operator }
    \item{/}{\code{signature(e1 = "matrix", e2 = "gvector")}: division operator }
    \item{/}{\code{signature(e1 = "numeric", e2 = "gvector")}: division operator }
    \item{[}{\code{signature(x = "gmatrix", i = "gvector", j = "gvector")}: indexing }
    \item{[}{\code{signature(x = "gmatrix", i = "gvector", j = "missing")}: indexing }
    \item{[}{\code{signature(x = "gmatrix", i = "gvector", j = "numeric")}: indexing }
    \item{[}{\code{signature(x = "gmatrix", i = "numeric", j = "gvector")}: indexing }
    \item{[}{\code{signature(x = "gvector", i = "ANY", j = "ANY")}: indexing }
    \item{[<-}{\code{signature(x = "gmatrix", i = "gvector", j = "gvector")}: indexing }
    \item{[<-}{\code{signature(x = "gmatrix", i = "gvector", j = "missing")}: indexing }
    \item{[<-}{\code{signature(x = "gmatrix", i = "gvector", j = "numeric")}: indexing }
    \item{[<-}{\code{signature(x = "gmatrix", i = "numeric", j = "gvector")}: indexing }
    \item{[<-}{\code{signature(x = "gvector", i = "ANY", j = "ANY")}: indexing }
    \item{^}{\code{signature(e1 = "gmatrix", e2 = "gvector")}: power operator }
    \item{^}{\code{signature(e1 = "gvector", e2 = "gmatrix")}: power operator }
    \item{^}{\code{signature(e1 = "gvector", e2 = "gvector")}: power operator }
    \item{^}{\code{signature(e1 = "gvector", e2 = "logical")}: power operator }
    \item{^}{\code{signature(e1 = "gvector", e2 = "matrix")}: power operator }
    \item{^}{\code{signature(e1 = "gvector", e2 = "numeric")}: power operator }
    \item{^}{\code{signature(e1 = "logical", e2 = "gvector")}: power operator }
    \item{^}{\code{signature(e1 = "matrix", e2 = "gvector")}: power operator }
    \item{^}{\code{signature(e1 = "numeric", e2 = "gvector")}: power operator }
    \item{|}{\code{signature(e1 = "gmatrix", e2 = "gvector")}: or operator }
    \item{|}{\code{signature(e1 = "gvector", e2 = "gmatrix")}: or operator }
    \item{|}{\code{signature(e1 = "gvector", e2 = "gvector")}: or operator }
    \item{|}{\code{signature(e1 = "gvector", e2 = "logical")}: or operator }
    \item{|}{\code{signature(e1 = "gvector", e2 = "matrix")}: or operator }
    \item{|}{\code{signature(e1 = "gvector", e2 = "numeric")}: or operator }
    \item{|}{\code{signature(e1 = "logical", e2 = "gvector")}: or operator }
    \item{|}{\code{signature(e1 = "matrix", e2 = "gvector")}: or operator }
    \item{|}{\code{signature(e1 = "numeric", e2 = "gvector")}: or operator }
    \item{+}{\code{signature(e1 = "gmatrix", e2 = "gvector")}: plus operator }
    \item{+}{\code{signature(e1 = "gvector", e2 = "gmatrix")}: plus operator }
    \item{+}{\code{signature(e1 = "gvector", e2 = "gvector")}: plus operator }
    \item{+}{\code{signature(e1 = "gvector", e2 = "logical")}: plus operator }
    \item{+}{\code{signature(e1 = "gvector", e2 = "matrix")}: plus operator }
    \item{+}{\code{signature(e1 = "gvector", e2 = "missing")}: plus operator }
    \item{+}{\code{signature(e1 = "gvector", e2 = "numeric")}: plus operator }
    \item{+}{\code{signature(e1 = "logical", e2 = "gvector")}: plus operator }
    \item{+}{\code{signature(e1 = "matrix", e2 = "gvector")}: plus operator }
    \item{+}{\code{signature(e1 = "numeric", e2 = "gvector")}: plus operator }
    \item{<}{\code{signature(e1 = "gmatrix", e2 = "gvector")}: less than operator }
    \item{<}{\code{signature(e1 = "gvector", e2 = "gmatrix")}: less than operator }
    \item{<}{\code{signature(e1 = "gvector", e2 = "gvector")}: less than operator }
    \item{<}{\code{signature(e1 = "gvector", e2 = "logical")}: less than operator }
    \item{<}{\code{signature(e1 = "gvector", e2 = "matrix")}: less than operator }
    \item{<}{\code{signature(e1 = "gvector", e2 = "numeric")}: less than operator }
    \item{<}{\code{signature(e1 = "logical", e2 = "gvector")}: less than operator }
    \item{<}{\code{signature(e1 = "matrix", e2 = "gvector")}: less than operator }
    \item{<}{\code{signature(e1 = "numeric", e2 = "gvector")}: less than operator }
    \item{<=}{\code{signature(e1 = "gmatrix", e2 = "gvector")}: less than or equal operator }
    \item{<=}{\code{signature(e1 = "gvector", e2 = "gmatrix")}: less than or equal operator }
    \item{<=}{\code{signature(e1 = "gvector", e2 = "gvector")}: less than or equal operator }
    \item{<=}{\code{signature(e1 = "gvector", e2 = "logical")}: less than or equal operator }
    \item{<=}{\code{signature(e1 = "gvector", e2 = "matrix")}: less than or equal operator }
    \item{<=}{\code{signature(e1 = "gvector", e2 = "numeric")}: less than or equal operator }
    \item{<=}{\code{signature(e1 = "logical", e2 = "gvector")}: less than or equal operator }
    \item{<=}{\code{signature(e1 = "matrix", e2 = "gvector")}: less than or equal operator }
    \item{<=}{\code{signature(e1 = "numeric", e2 = "gvector")}: less than or equal operator }
    \item{==}{\code{signature(e1 = "gmatrix", e2 = "gvector")}: equal operator }
    \item{==}{\code{signature(e1 = "gvector", e2 = "gmatrix")}: equal operator }
    \item{==}{\code{signature(e1 = "gvector", e2 = "gvector")}: equal operator }
    \item{==}{\code{signature(e1 = "gvector", e2 = "logical")}: equal operator }
    \item{==}{\code{signature(e1 = "gvector", e2 = "matrix")}: equal operator }
    \item{==}{\code{signature(e1 = "gvector", e2 = "numeric")}: equal operator }
    \item{==}{\code{signature(e1 = "logical", e2 = "gvector")}: equal operator }
    \item{==}{\code{signature(e1 = "matrix", e2 = "gvector")}: equal operator }
    \item{==}{\code{signature(e1 = "numeric", e2 = "gvector")}: equal operator }
    \item{>}{\code{signature(e1 = "gmatrix", e2 = "gvector")}: greater than operator }
    \item{>}{\code{signature(e1 = "gvector", e2 = "gmatrix")}: greater than operator }
    \item{>}{\code{signature(e1 = "gvector", e2 = "gvector")}: greater than operator }
    \item{>}{\code{signature(e1 = "gvector", e2 = "logical")}: greater than operator }
    \item{>}{\code{signature(e1 = "gvector", e2 = "matrix")}: greater than operator }
    \item{>}{\code{signature(e1 = "gvector", e2 = "numeric")}: greater than operator }
    \item{>}{\code{signature(e1 = "logical", e2 = "gvector")}: greater than operator}
    \item{>}{\code{signature(e1 = "matrix", e2 = "gvector")}: greater than operator }
    \item{>}{\code{signature(e1 = "numeric", e2 = "gvector")}: greater than operator }
    \item{>=}{\code{signature(e1 = "gmatrix", e2 = "gvector")}: greater than or equal operator }
    \item{>=}{\code{signature(e1 = "gvector", e2 = "gmatrix")}: greater than or equal operator }
    \item{>=}{\code{signature(e1 = "gvector", e2 = "gvector")}: greater than or equal operator }
    \item{>=}{\code{signature(e1 = "gvector", e2 = "logical")}: greater than or equal operator }
    \item{>=}{\code{signature(e1 = "gvector", e2 = "matrix")}: greater than or equal operator }
    \item{>=}{\code{signature(e1 = "gvector", e2 = "numeric")}: greater than or equal operator }
    \item{>=}{\code{signature(e1 = "logical", e2 = "gvector")}: greater than or equal operator }
    \item{>=}{\code{signature(e1 = "matrix", e2 = "gvector")}: greater than or equal operator }
    \item{>=}{\code{signature(e1 = "numeric", e2 = "gvector")}: greater than or equal operator }
    \item{abs}{\code{signature(x = "gvector")}: absolute value }
    \item{acos}{\code{signature(x = "gvector")}: inverse cosine }
    \item{acosh}{\code{signature(x = "gvector")}: inverse hyperbolic cosine }
    \item{as.gmatrix}{\code{signature(data = "gvector")}:coersion function }
    \item{as.gvector}{\code{signature(x = "gvector")}: coersion function }
    \item{as.integer}{\code{signature(x = "gvector")}: coersion function }
    \item{as.logical}{\code{signature(x = "gvector")}: coersion function }
    \item{as.matrix}{\code{signature(x = "gvector")}: coersion function }
    \item{as.numeric}{\code{signature(x = "gvector")}: coersion function }
    \item{as.vector}{\code{signature(x = "gvector")}: coersion function}
    \item{asin}{\code{signature(x = "gvector")}: inverse sine function }
    \item{asinh}{\code{signature(x = "gvector")}: inverse hyperbolic sine function }
    \item{atan}{\code{signature(x = "gvector")}: inverse tangent function }
    \item{atanh}{\code{signature(x = "gvector")}: inverse hyperbolic tangent function }
    \item{ceiling}{\code{signature(x = "gvector")}: ceiling function }
    \item{cos}{\code{signature(x = "gvector")}: cosine function }
    \item{cosh}{\code{signature(x = "gvector")}: hyperbolic cosine }
    \item{crossprod}{\code{signature(x = "gmatrix", y = "gvector")}: calculates: \code{t(x) \%*\% y} }
    \item{crossprod}{\code{signature(x = "gvector", y = "gmatrix")}: calculates: \code{t(x) \%*\% y} }
    \item{crossprod}{\code{signature(x = "gvector", y = "gvector")}: calculates: \code{t(x) \%*\% y} }
    \item{crossprod}{\code{signature(x = "gvector", y = "logical")}: calculates: \code{t(x) \%*\% y} }
    \item{crossprod}{\code{signature(x = "gvector", y = "missing")}: calculates: \code{t(x) \%*\% y} }
    \item{crossprod}{\code{signature(x = "gvector", y = "numeric")}: calculates: \code{t(x) \%*\% y} }
    \item{crossprod}{\code{signature(x = "logical", y = "gvector")}: calculates: \code{t(x) \%*\% y} }
    \item{crossprod}{\code{signature(x = "numeric", y = "gvector")}: calculates: \code{t(x) \%*\% y} }
    \item{device}{\code{signature(x = "gvector")}: get device }
    \item{device<-}{\code{signature(x = "gvector")}: set device }
    \item{exp}{\code{signature(x = "gvector")}: exponential function }
    \item{expm1}{\code{signature(x = "gvector")}: computes  \code{exp(x) - 1}  }
    \item{floor}{\code{signature(x = "gvector")}: floor function }
    \item{gamma}{\code{signature(x = "gvector")}: gamma function }
    \item{ifelse}{\code{signature(test = "gvector")}: conditional selection  }
    \item{is.finite}{\code{signature(x = "gvector")}: finite status  }
    \item{is.infinite}{\code{signature(x = "gvector")}: infinite status  }
    \item{is.na}{\code{signature(x = "gvector")}: missing status  }
    \item{is.nan}{\code{signature(x = "gvector")}: not a number status }
    \item{length}{\code{signature(x = "gvector")}:  total number of elements in object }
    \item{lgamma}{\code{signature(x = "gvector")}: log gamma function }
    \item{log}{\code{signature(x = "gvector")}: logarithm function  }
    \item{log10}{\code{signature(x = "gvector")}: base 10 log }
    \item{log1p}{\code{signature(x = "gvector")}: \code{log(1+x)} }
    \item{log2}{\code{signature(x = "gvector")}: base 2 log }
    \item{max}{\code{signature(x = "gvector")}: maximum of all elements }
    \item{mean}{\code{signature(x = "gvector")}: mean of all elements}
    \item{min}{\code{signature(x = "gvector")}: minimum of all elements }
    \item{names}{\code{signature(x = "gvector")}: get element names }
    \item{names<-}{\code{signature(x = "gvector")}: set element names }
    \item{round}{\code{signature(x = "gvector")}: round to neares integer }
    \item{show}{\code{signature(object = "gvector")}: display object }
    \item{sign}{\code{signature(x = "gvector")}: sign }
    \item{sin}{\code{signature(x = "gvector")}: sine function }
    \item{sinh}{\code{signature(x = "gvector")}: hyperbolic sine function }
    \item{sort}{\code{signature(x = "gvector")}: sort }
    \item{sqrt}{\code{signature(x = "gvector")}: square root }
    \item{sum}{\code{signature(x = "gvector")}: sum of all elements }
    \item{t}{\code{signature(x = "gvector")}: transpose }
    \item{tan}{\code{signature(x = "gvector")}: tangent function}
    \item{tanh}{\code{signature(x = "gvector")}: hyperbolic tangent function }
    \item{tcrossprod}{\code{signature(x = "gmatrix", y = "gvector")}: calculates: \code{x \%*\% t(y)}}
    \item{tcrossprod}{\code{signature(x = "gvector", y = "gmatrix")}: calculates: \code{x \%*\% t(y)} }
    \item{tcrossprod}{\code{signature(x = "gvector", y = "gvector")}: calculates: \code{x \%*\% t(y)} }
    \item{tcrossprod}{\code{signature(x = "gvector", y = "logical")}: calculates: \code{x \%*\% t(y)} }
    \item{tcrossprod}{\code{signature(x = "gvector", y = "missing")}: calculates: \code{x \%*\% t(y)} }
    \item{tcrossprod}{\code{signature(x = "gvector", y = "numeric")}: calculates: \code{x \%*\% t(y)} }
    \item{tcrossprod}{\code{signature(x = "logical", y = "gvector")}: calculates: \code{x \%*\% t(y)} }
    \item{tcrossprod}{\code{signature(x = "numeric", y = "gvector")}: calculates: \code{x \%*\% t(y)} }
    \item{type}{\code{signature(x = "gvector")}: get type }
    \item{type<-}{\code{signature(x = "gvector")}: set type }
    \item{which}{\code{signature(x = "gvector")}: index list of TRUE positions}
	 }
}

\author{
Nathan Morris
}

\seealso{
\code{\linkS4class{gmatrix}}
}
\examples{
showClass("gvector")
}
\keyword{classes}
