\name{setTuningPameters}
\alias{setTuningPameters}
\title{
Set tuning parameters
}
\description{
These methods tune the kernal calls to a specific GPU. At this point, these parameters are not well documented and should only be changed by users who can read the source code.
}
\usage{
setTuningPameters(force = TRUE, threads_per_block = as.integer(2^8), 
     total_states = as.integer(32 * 14 * 16), state = sample.int(10^6, 1))
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{force}{
	logical - force the change even if the parameters have been previously set.
}
  \item{threads_per_block}{
	number of threads per block
}
  \item{total_states}{
	total states for random number generations
}
  \item{state}{
	starting state for random number generation
}
}

\author{
Nathan Morris
}

