\name{setTuningPameters}
\alias{setTuningPameters}
\title{
Set tuning parameters
}
\description{
This function tunes the kernal calls to a specific GPU if the user is working with CUDA versions older than 6.5. At this point, these parameters are not well documented and should only be changed by users who can read the source code.
}
\usage{
setTuningPameters(force = TRUE, threads_per_block = as.integer(2^8),
                 total_states = as.integer(32 * 14 * 16), state =
                 unclass(Sys.time()))
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

