\name{gset.seed}
\alias{gset.seed}
\title{
Set the seed values, and number of states for the GPU.
}
\description{
This function enables the user to set a seed value for reproducible random number generation. It also allows the user to set the number of states to tune the efficiency of random number generator for the GPU. 
}
\usage{
gset.seed(seed=unclass(Sys.time()), total_states=as.integer(32*14*16), silent=TRUE)
}

\arguments{
  \item{seed}{
	Must be a positive integer.
}
  \item{total_states}{
	Random numbers are generated from separate streams in parallel with each stream having a current state. This variable must be a positive integer. It is recommended that the integer be a multiple of 32. If the number of states are two small or two large the GPU efficiency may be poor. 
}
  \item{silent}{
	Don't print any messages besides errors.
}
}

\author{
Nathan Morris
}



