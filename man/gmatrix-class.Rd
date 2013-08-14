\name{gmatrix-class}
\Rdversion{1.1}
\docType{class}
\alias{gmatrix-class}

\title{Class "gmatrix"}
\description{
This object contains a pointer to a matrix on a NVIDIA GPU.
}
\section{Objects from the Class}{
Objects should NOT be created by calls of the form \code{new("gmatrix", ...)}. Instead use the \code{gmatrix}, 'as.gmatrix' or 'g' functions.
}
\section{Slots}{
  \describe{
    \item{\code{ptr}:}{Represents an external pointer to the data on the GPU}
    \item{\code{nrow}:}{Object of class \code{"integer"} representing the number of rows for the matrix}
    \item{\code{ncol}:}{Object of class \code{"integer"} representing the number of cols for the matrix}
    \item{\code{rownames}:}{Should be a character vector with length equal nrow }
    \item{\code{colnames}:}{Should be a character vector with length equal nrow }
    \item{\code{type}:}{Must be 0L, 1L, 2L or 3L. These integers represent the types \code{"double"}, \code{"single"}, \code{"integer"} or \code{"logical"} respectively.}
    \item{\code{device}:}{Object of class \code{"integer"} representing the device}
  }
}
\section{Methods}{
  \describe{
    \item{-}{\code{signature(e1 = "gmatrix", e2 = "gmatrix")}: subtraction operator}
    \item{-}{\code{signature(e1 = "gmatrix", e2 = "gvector")}: subtraction operator}
    \item{-}{\code{signature(e1 = "gmatrix", e2 = "logical")}: subtraction operator}
    \item{-}{\code{signature(e1 = "gmatrix", e2 = "matrix")}: subtraction operator}
    \item{-}{\code{signature(e1 = "gmatrix", e2 = "missing")}: subtraction operator}
    \item{-}{\code{signature(e1 = "gmatrix", e2 = "numeric")}: subtraction operator}
    \item{-}{\code{signature(e1 = "gvector", e2 = "gmatrix")}: subtraction operator}
    \item{-}{\code{signature(e1 = "logical", e2 = "gmatrix")}: subtraction operator}
    \item{-}{\code{signature(e1 = "matrix", e2 = "gmatrix")}: subtraction operator}
    \item{-}{\code{signature(e1 = "numeric", e2 = "gmatrix")}: subtraction operator}
    \item{!}{\code{signature(x = "gmatrix")}: not operator }
    \item{!=}{\code{signature(e1 = "gmatrix", e2 = "gmatrix")}: not equal operator}
    \item{!=}{\code{signature(e1 = "gmatrix", e2 = "gvector")}: not equal operator}
    \item{!=}{\code{signature(e1 = "gmatrix", e2 = "logical")}: not equal operator}
    \item{!=}{\code{signature(e1 = "gmatrix", e2 = "matrix")}:  not equal operator}
    \item{!=}{\code{signature(e1 = "gmatrix", e2 = "numeric")}: not equal operator}
    \item{!=}{\code{signature(e1 = "gvector", e2 = "gmatrix")}: not equal operator}
    \item{!=}{\code{signature(e1 = "logical", e2 = "gmatrix")}: not equal operator}
    \item{!=}{\code{signature(e1 = "matrix", e2 = "gmatrix")}: not equal operator}
    \item{!=}{\code{signature(e1 = "numeric", e2 = "gmatrix")}: not equal operator}
    \item{\%\%}{\code{signature(e1 = "gmatrix", e2 = "gmatrix")}: mod operator  }
    \item{\%\%}{\code{signature(e1 = "gmatrix", e2 = "gvector")}: mod operator}
    \item{\%\%}{\code{signature(e1 = "gmatrix", e2 = "logical")}: mod operator }
    \item{\%\%}{\code{signature(e1 = "gmatrix", e2 = "matrix")}: mod operator }
    \item{\%\%}{\code{signature(e1 = "gmatrix", e2 = "numeric")}: mod operator }
    \item{\%\%}{\code{signature(e1 = "gvector", e2 = "gmatrix")}: mod operator }
    \item{\%\%}{\code{signature(e1 = "logical", e2 = "gmatrix")}: mod operator }
    \item{\%\%}{\code{signature(e1 = "matrix", e2 = "gmatrix")}: mod operator }
    \item{\%\%}{\code{signature(e1 = "numeric", e2 = "gmatrix")}: mod operator }
    \item{\%*\%}{\code{signature(x = "gmatrix", y = "gmatrix")}: matrix multiplication operator }
    \item{\%*\%}{\code{signature(x = "gmatrix", y = "gvector")}: matrix multiplication  operator }
    \item{\%*\%}{\code{signature(x = "gmatrix", y = "logical")}: matrix multiplication  operator }
    \item{\%*\%}{\code{signature(x = "gmatrix", y = "matrix")}: matrix multiplication  operator  }
    \item{\%*\%}{\code{signature(x = "gmatrix", y = "numeric")}: matrix multiplication  operator }
    \item{\%*\%}{\code{signature(x = "gvector", y = "gmatrix")}: matrix multiplication  operator  }
    \item{\%*\%}{\code{signature(x = "logical", y = "gmatrix")}: matrix multiplication  operator  }
    \item{\%*\%}{\code{signature(x = "matrix", y = "gmatrix")}: matrix multiplication  operator  }
    \item{\%*\%}{\code{signature(x = "numeric", y = "gmatrix")}: matrix multiplication  operator  }
    \item{\%x\%}{\code{signature(X = "gmatrix", Y = "gmatrix")}: kronecker product }
    \item{\%x\%}{\code{signature(X = "gmatrix", Y = "gvector")}: kronecker product }
    \item{\%x\%}{\code{signature(X = "gmatrix", Y = "numeric")}: kronecker product }
    \item{\%x\%}{\code{signature(X = "gvector", Y = "gmatrix")}: kronecker product }
    \item{\%x\%}{\code{signature(X = "numeric", Y = "gmatrix")}: kronecker product }
    \item{&}{\code{signature(e1 = "gmatrix", e2 = "gmatrix")}: and operator }
    \item{&}{\code{signature(e1 = "gmatrix", e2 = "gvector")}: and operator }
    \item{&}{\code{signature(e1 = "gmatrix", e2 = "logical")}: and operator }
    \item{&}{\code{signature(e1 = "gmatrix", e2 = "matrix")}: and operator }
    \item{&}{\code{signature(e1 = "gmatrix", e2 = "numeric")}: and operator }
    \item{&}{\code{signature(e1 = "gvector", e2 = "gmatrix")}: and operator }
    \item{&}{\code{signature(e1 = "logical", e2 = "gmatrix")}: and operator }
    \item{&}{\code{signature(e1 = "matrix", e2 = "gmatrix")}: and operator. }
    \item{&}{\code{signature(e1 = "numeric", e2 = "gmatrix")}: and operator. }
    \item{*}{\code{signature(e1 = "gmatrix", e2 = "gmatrix")}: elementwise multiplication}
    \item{*}{\code{signature(e1 = "gmatrix", e2 = "gvector")}: elementwise multiplication}
    \item{*}{\code{signature(e1 = "gmatrix", e2 = "logical")}: elementwise multiplication}
    \item{*}{\code{signature(e1 = "gmatrix", e2 = "matrix")}: elementwise multiplication}
    \item{*}{\code{signature(e1 = "gmatrix", e2 = "numeric")}: elementwise multiplication}
    \item{*}{\code{signature(e1 = "gvector", e2 = "gmatrix")}: elementwise multiplication}
    \item{*}{\code{signature(e1 = "logical", e2 = "gmatrix")}: elementwise multiplication}
    \item{*}{\code{signature(e1 = "matrix", e2 = "gmatrix")}: elementwise multiplication}
    \item{*}{\code{signature(e1 = "numeric", e2 = "gmatrix")}: elementwise multiplication}
    \item{/}{\code{signature(e1 = "gmatrix", e2 = "gmatrix")}: division operator }
    \item{/}{\code{signature(e1 = "gmatrix", e2 = "gvector")}: division operator  }
    \item{/}{\code{signature(e1 = "gmatrix", e2 = "logical")}: division operator  }
    \item{/}{\code{signature(e1 = "gmatrix", e2 = "matrix")}: division operator  }
    \item{/}{\code{signature(e1 = "gmatrix", e2 = "numeric")}: division operator  }
    \item{/}{\code{signature(e1 = "gvector", e2 = "gmatrix")}: division operator  }
    \item{/}{\code{signature(e1 = "logical", e2 = "gmatrix")}: division operator  }
    \item{/}{\code{signature(e1 = "matrix", e2 = "gmatrix")}: division operator  }
    \item{/}{\code{signature(e1 = "numeric", e2 = "gmatrix")}: division operator  }
    \item{[}{\code{signature(x = "gmatrix", i = "gvector", j = "gvector")}: indexing }
    \item{[}{\code{signature(x = "gmatrix", i = "gvector", j = "missing")}: indexing  }
    \item{[}{\code{signature(x = "gmatrix", i = "gvector", j = "numeric")}: indexing  }
    \item{[}{\code{signature(x = "gmatrix", i = "numeric", j = "gvector")}: indexing  }
    \item{[}{\code{signature(x = "gmatrix", i = "numeric", j = "missing")}: indexing  }
    \item{[}{\code{signature(x = "gmatrix", i = "numeric", j = "numeric")}: indexing }
    \item{[<-}{\code{signature(x = "gmatrix", i = "gvector", j = "gvector")}: indexing  }
    \item{[<-}{\code{signature(x = "gmatrix", i = "gvector", j = "missing")}: indexing  }
    \item{[<-}{\code{signature(x = "gmatrix", i = "gvector", j = "numeric")}: indexing . }
    \item{[<-}{\code{signature(x = "gmatrix", i = "numeric", j = "gvector")}: indexing  }
    \item{[<-}{\code{signature(x = "gmatrix", i = "numeric", j = "missing")}: indexing  }
    \item{[<-}{\code{signature(x = "gmatrix", i = "numeric", j = "numeric")}: indexing  }
    \item{^}{\code{signature(e1 = "gmatrix", e2 = "gmatrix")}: power operator }
    \item{^}{\code{signature(e1 = "gmatrix", e2 = "gvector")}: power operator}
    \item{^}{\code{signature(e1 = "gmatrix", e2 = "logical")}: power operator }
    \item{^}{\code{signature(e1 = "gmatrix", e2 = "matrix")}: power operator }
    \item{^}{\code{signature(e1 = "gmatrix", e2 = "numeric")}: power operator}
    \item{^}{\code{signature(e1 = "gvector", e2 = "gmatrix")}: power operator }
    \item{^}{\code{signature(e1 = "logical", e2 = "gmatrix")}: power operator}
    \item{^}{\code{signature(e1 = "matrix", e2 = "gmatrix")}: power operator }
    \item{^}{\code{signature(e1 = "numeric", e2 = "gmatrix")}: power operator }
    \item{|}{\code{signature(e1 = "gmatrix", e2 = "gmatrix")}: or operator }
    \item{|}{\code{signature(e1 = "gmatrix", e2 = "gvector")}: or operator }
    \item{|}{\code{signature(e1 = "gmatrix", e2 = "logical")}: or operator }
    \item{|}{\code{signature(e1 = "gmatrix", e2 = "matrix")}: or operator }
    \item{|}{\code{signature(e1 = "gmatrix", e2 = "numeric")}: or operator }
    \item{|}{\code{signature(e1 = "gvector", e2 = "gmatrix")}: or operator }
    \item{|}{\code{signature(e1 = "logical", e2 = "gmatrix")}: or operator }
    \item{|}{\code{signature(e1 = "matrix", e2 = "gmatrix")}: or operator}
    \item{|}{\code{signature(e1 = "numeric", e2 = "gmatrix")}: or operator }
    \item{+}{\code{signature(e1 = "gmatrix", e2 = "gmatrix")}: plus operator }
    \item{+}{\code{signature(e1 = "gmatrix", e2 = "gvector")}: plus operator }
    \item{+}{\code{signature(e1 = "gmatrix", e2 = "logical")}: plus operator }
    \item{+}{\code{signature(e1 = "gmatrix", e2 = "matrix")}: plus operator }
    \item{+}{\code{signature(e1 = "gmatrix", e2 = "missing")}: plus operator }
    \item{+}{\code{signature(e1 = "gmatrix", e2 = "numeric")}: plus operator }
    \item{+}{\code{signature(e1 = "gvector", e2 = "gmatrix")}: plus operator }
    \item{+}{\code{signature(e1 = "logical", e2 = "gmatrix")}: plus operator }
    \item{+}{\code{signature(e1 = "matrix", e2 = "gmatrix")}: plus operator }
    \item{+}{\code{signature(e1 = "numeric", e2 = "gmatrix")}: plus operator}
    \item{<}{\code{signature(e1 = "gmatrix", e2 = "gmatrix")}: less than operator }
    \item{<}{\code{signature(e1 = "gmatrix", e2 = "gvector")}: less than operator }
    \item{<}{\code{signature(e1 = "gmatrix", e2 = "logical")}: less than operator }
    \item{<}{\code{signature(e1 = "gmatrix", e2 = "matrix")}: less than operator }
    \item{<}{\code{signature(e1 = "gmatrix", e2 = "numeric")}: less than operator }
    \item{<}{\code{signature(e1 = "gvector", e2 = "gmatrix")}: less than operator }
    \item{<}{\code{signature(e1 = "logical", e2 = "gmatrix")}: less than operator }
    \item{<}{\code{signature(e1 = "matrix", e2 = "gmatrix")}: less than operator }
    \item{<}{\code{signature(e1 = "numeric", e2 = "gmatrix")}: less than operator}
    \item{<=}{\code{signature(e1 = "gmatrix", e2 = "gmatrix")}: less than or equal operator }
    \item{<=}{\code{signature(e1 = "gmatrix", e2 = "gvector")}: less than or equal operator }
    \item{<=}{\code{signature(e1 = "gmatrix", e2 = "logical")}: less than or equal operator }
    \item{<=}{\code{signature(e1 = "gmatrix", e2 = "matrix")}: less than or equal operator }
    \item{<=}{\code{signature(e1 = "gmatrix", e2 = "numeric")}: less than or equal operator }
    \item{<=}{\code{signature(e1 = "gvector", e2 = "gmatrix")}: less than or equal operator }
    \item{<=}{\code{signature(e1 = "logical", e2 = "gmatrix")}: less than or equal operator }
    \item{<=}{\code{signature(e1 = "matrix", e2 = "gmatrix")}: less than or equal operator }
    \item{<=}{\code{signature(e1 = "numeric", e2 = "gmatrix")}: less than or equal operator }
    \item{==}{\code{signature(e1 = "gmatrix", e2 = "gmatrix")}: equal operator }
    \item{==}{\code{signature(e1 = "gmatrix", e2 = "gvector")}: equal operator }
    \item{==}{\code{signature(e1 = "gmatrix", e2 = "logical")}: equal operator }
    \item{==}{\code{signature(e1 = "gmatrix", e2 = "matrix")}: equal operator }
    \item{==}{\code{signature(e1 = "gmatrix", e2 = "numeric")}: equal operator }
    \item{==}{\code{signature(e1 = "gvector", e2 = "gmatrix")}: equal operator }
    \item{==}{\code{signature(e1 = "logical", e2 = "gmatrix")}: equal operator }
    \item{==}{\code{signature(e1 = "matrix", e2 = "gmatrix")}: equal operator }
    \item{==}{\code{signature(e1 = "numeric", e2 = "gmatrix")}: equal operator }
    \item{>}{\code{signature(e1 = "gmatrix", e2 = "gmatrix")}: greater than operator }
    \item{>}{\code{signature(e1 = "gmatrix", e2 = "gvector")}: greater than operator }
    \item{>}{\code{signature(e1 = "gmatrix", e2 = "logical")}: greater than operator }
    \item{>}{\code{signature(e1 = "gmatrix", e2 = "matrix")}: greater than operator }
    \item{>}{\code{signature(e1 = "gmatrix", e2 = "numeric")}: greater than operator }
    \item{>}{\code{signature(e1 = "gvector", e2 = "gmatrix")}: greater than operator }
    \item{>}{\code{signature(e1 = "logical", e2 = "gmatrix")}: greater than operator }
    \item{>}{\code{signature(e1 = "matrix", e2 = "gmatrix")}: greater than operator }
    \item{>}{\code{signature(e1 = "numeric", e2 = "gmatrix")}: greater than operator }
    \item{>=}{\code{signature(e1 = "gmatrix", e2 = "gmatrix")}: greater than or equal operator }
    \item{>=}{\code{signature(e1 = "gmatrix", e2 = "gvector")}: greater than or equal operator }
    \item{>=}{\code{signature(e1 = "gmatrix", e2 = "logical")}: greater than or equal operator }
    \item{>=}{\code{signature(e1 = "gmatrix", e2 = "matrix")}: greater than or equal operator }
    \item{>=}{\code{signature(e1 = "gmatrix", e2 = "numeric")}: greater than or equal operator }
    \item{>=}{\code{signature(e1 = "gvector", e2 = "gmatrix")}: greater than or equal operator }
    \item{>=}{\code{signature(e1 = "logical", e2 = "gmatrix")}: greater than or equal operator }
    \item{>=}{\code{signature(e1 = "matrix", e2 = "gmatrix")}: greater than or equal operator }
    \item{>=}{\code{signature(e1 = "numeric", e2 = "gmatrix")}: greater than or equal operator }
    \item{as.gmatrix}{\code{signature(data = "gmatrix")}: coersion function }
    \item{as.gvector}{\code{signature(x = "gmatrix")}: coersion function }
    \item{as.integer}{\code{signature(x = "gmatrix")}: coersion function }
    \item{as.logical}{\code{signature(x = "gmatrix")}: coersion function }
    \item{as.matrix}{\code{signature(x = "gmatrix")}: coersion function }
    \item{as.numeric}{\code{signature(x = "gmatrix")}: coersion function }
    \item{as.vector}{\code{signature(x = "gmatrix")}: coersion function }
    \item{abs}{\code{signature(x = "gmatrix")}: absolute value }
    \item{acos}{\code{signature(x = "gmatrix")}: inverse cosine }
    \item{acosh}{\code{signature(x = "gmatrix")}: inverse hyperbolic cosine }
    \item{asin}{\code{signature(x = "gmatrix")}: inverse sin }
    \item{asinh}{\code{signature(x = "gmatrix")}: inverse hyperbolic sin }
    \item{atan}{\code{signature(x = "gmatrix")}: inverse tangent }
    \item{atanh}{\code{signature(x = "gmatrix")}: inverse hyperbolic tangent }
    \item{ceiling}{\code{signature(x = "gmatrix")}: ceiling function }
    \item{colMeans}{\code{signature(x = "gmatrix")}: mean of each column }
    \item{colnames}{\code{signature(x = "gmatrix")}: column names }
    \item{colnames<-}{\code{signature(x = "gmatrix")}: set column names }
    \item{colSums}{\code{signature(x = "gmatrix")}: sum of each column }
    \item{cos}{\code{signature(x = "gmatrix")}: cos function }
    \item{cosh}{\code{signature(x = "gmatrix")}: hyperbolic cos }
    \item{crossprod}{\code{signature(x = "gmatrix", y = "gmatrix")}: calculates: \code{t(x) \%*\% y} }
    \item{crossprod}{\code{signature(x = "gmatrix", y = "gvector")}: calculates: \code{t(x) \%*\% y} }
    \item{crossprod}{\code{signature(x = "gmatrix", y = "logical")}: calculates: \code{t(x) \%*\% y} }
    \item{crossprod}{\code{signature(x = "gmatrix", y = "matrix")}: calculates: \code{t(x) \%*\% y} }
    \item{crossprod}{\code{signature(x = "gmatrix", y = "missing")}: calculates: \code{t(x) \%*\% y} }
    \item{crossprod}{\code{signature(x = "gmatrix", y = "numeric")}: calculates: \code{t(x) \%*\% y} }
    \item{crossprod}{\code{signature(x = "gvector", y = "gmatrix")}: calculates: \code{t(x) \%*\% y} }
    \item{crossprod}{\code{signature(x = "logical", y = "gmatrix")}: calculates: \code{t(x) \%*\% y} }
    \item{crossprod}{\code{signature(x = "matrix", y = "gmatrix")}: calculates: \code{t(x) \%*\% y} }
    \item{crossprod}{\code{signature(x = "numeric", y = "gmatrix")}: calculates: \code{t(x) \%*\% y} }
    \item{tcrossprod}{\code{signature(x = "gmatrix", y = "gmatrix")}: calculates: \code{x \%*\% t(y)} }
    \item{tcrossprod}{\code{signature(x = "gmatrix", y = "gvector")}: calculates: \code{x \%*\% t(y)} }
    \item{tcrossprod}{\code{signature(x = "gmatrix", y = "logical")}: calculates: \code{x \%*\% t(y)} }
    \item{tcrossprod}{\code{signature(x = "gmatrix", y = "matrix")}: calculates: \code{x \%*\% t(y)} }
    \item{tcrossprod}{\code{signature(x = "gmatrix", y = "missing")}: calculates: \code{x \%*\% t(y)} }
    \item{tcrossprod}{\code{signature(x = "gmatrix", y = "numeric")}: calculates: \code{x \%*\% t(y)} }
    \item{tcrossprod}{\code{signature(x = "gvector", y = "gmatrix")}: calculates: \code{x \%*\% t(y)} }
    \item{tcrossprod}{\code{signature(x = "logical", y = "gmatrix")}: calculates: \code{x \%*\% t(y)} }
    \item{tcrossprod}{\code{signature(x = "matrix", y = "gmatrix")}: calculates: \code{x \%*\% t(y)} }
    \item{tcrossprod}{\code{signature(x = "numeric", y = "gmatrix")}: calculates: \code{x \%*\% t(y)} }
    \item{device}{\code{signature(x = "gmatrix")}: get device property }
    \item{device<-}{\code{signature(x = "gmatrix")}: set device property }
    \item{diag}{\code{signature(x = "gmatrix")}: diagonal }
    \item{diag<-}{\code{signature(x = "gmatrix")}: set the diagonal of a matrix }
    \item{dim}{\code{signature(x = "gmatrix")}: dimension}
    \item{dim<-}{\code{signature(x = "gmatrix")}: set dimension}
    \item{dimnames}{\code{signature(x = "gmatrix")}: get dimnames}
    \item{dimnames<-}{\code{signature(x = "gmatrix")}: set dimnames }
    \item{exp}{\code{signature(x = "gmatrix")}: exponentiate function}
    \item{expm1}{\code{signature(x = "gmatrix")}: computes  \code{exp(x) - 1}  }
    \item{floor}{\code{signature(x = "gmatrix")}: floor function }
    \item{gamma}{\code{signature(x = "gmatrix")}: gamma function }
    \item{ifelse}{\code{signature(test = "gmatrix")}: conditional selection }
    \item{is.finite}{\code{signature(x = "gmatrix")}: finite status }
    \item{is.infinite}{\code{signature(x = "gmatrix")}: infinite status }
    \item{is.na}{\code{signature(x = "gmatrix")}: missing status }
    \item{is.nan}{\code{signature(x = "gmatrix")}: not a number status}
    \item{length}{\code{signature(x = "gmatrix")}: total number of elements in object}
    \item{lgamma}{\code{signature(x = "gmatrix")}: log gamma function}
    \item{log}{\code{signature(x = "gmatrix")}: logarithm function  }
    \item{log10}{\code{signature(x = "gmatrix")}: base 10 log }
    \item{log1p}{\code{signature(x = "gmatrix")}: \code{log(1+x)} }
    \item{log2}{\code{signature(x = "gmatrix")}: base 2 log }
    \item{max}{\code{signature(x = "gmatrix")}: maximum of all elements}
    \item{mean}{\code{signature(x = "gmatrix")}: mean of all elements}
    \item{min}{\code{signature(x = "gmatrix")}: min of all elements }
    \item{ncol}{\code{signature(x = "gmatrix")}: number of columns }
    \item{nrow}{\code{signature(x = "gmatrix")}: number of rows}
    \item{round}{\code{signature(x = "gmatrix")}: round function (only to nearest whole number) }
    \item{rowMeans}{\code{signature(x = "gmatrix")}: mean of each row}
    \item{rownames}{\code{signature(x = "gmatrix")}: names of rows}
    \item{rownames<-}{\code{signature(x = "gmatrix")}: set row names}
    \item{rowSums}{\code{signature(x = "gmatrix")}: sum of each row}
    \item{show}{\code{signature(object = "gmatrix")}: show the data }
    \item{sign}{\code{signature(x = "gmatrix")}: return the sign }
    \item{sin}{\code{signature(x = "gmatrix")}: sin function}
    \item{sinh}{\code{signature(x = "gmatrix")}: hyperbolic sin }
    \item{sqrt}{\code{signature(x = "gmatrix")}: square root }
    \item{sum}{\code{signature(x = "gmatrix")}: sum }
    \item{t}{\code{signature(x = "gmatrix")}: transpose }
    \item{tan}{\code{signature(x = "gmatrix")}: tangent }
    \item{tanh}{\code{signature(x = "gmatrix")}: hyperbolic tangent }
    \item{type}{\code{signature(x = "gmatrix")}: get type (e.g \code{"double"}) }
    \item{type<-}{\code{signature(x = "gmatrix")}: set type }
	 }
}

\author{
Nathan Morris
}
\seealso{
\code{\linkS4class{gvector}}
}
\examples{
showClass("gmatrix")
}
\keyword{classes}
