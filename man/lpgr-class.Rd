\name{lpgr-class}
\Rdversion{1.1}
\docType{class}
\alias{lpgr-class}

\title{Class \code{"lpgr"}}
\description{
Objects of this class must be returned by any " \code{lprf}" function which is used by the \code{basicHMC()} function.
}
\section{Objects from the Class}{
Objects can be created by calls of the form \code{new("lpgr", ...)}.
}
\section{Slots}{
  \describe{
    \item{\code{gr}:}{Object of class \code{"list"}. Must contain a list of matrices representing the first derivative of the log probility of a distribution with respect to the random variables. Each column of the matrices is for a different parrallel run. All matrices may be either on the gpu or cpu}
    \item{\code{lp}:}{Object of class \code{"numeric"}. Represent the log probability (without the normalizing constan) of a distribution. MUST be on the CPU.}
  }
}
\section{Methods}{
No methods defined with class "lpgr" in the signature.
}
\seealso{
\code{\link{basicHMC}}
}
\author{
Nathan Morris
}
\note{
%%  ~~further notes~~
}
