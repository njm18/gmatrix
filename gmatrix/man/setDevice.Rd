\name{setDevice}
\alias{setDevice}
\alias{getDevice}
\alias{listDevices}
\title{
Get and set the current device.
}
\description{
On systems with multiple GPUs the current device may be set or investigated with these functions.
}
\usage{
setDevice(device, force = FALSE, ...)
getDevice()
listDevices()
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{device}{
	Device number to set the device to
}
  \item{force}{
	Force reset of the tuning parameters even if they have already been set
}
  \item{\dots}{
	Arguments to set the tuning parameters. It's not a good idea to mess with these at this point.
}
}
\details{
All operations are performed on the current GPU device only.
An error will be returned if the user attempts to perform an operation on a device that is not the current device. Details of the available devices can be obtained with the 'listDevices' function.
}
\value{
\code{setDevice} returns TRUE. \code{getDevice} returns the current device number. \code{listDevices} returns a \code{data.frame} with information about the available devices.
}

\author{
Nathan Morris
}

\seealso{
\code{\link{device}}, \code{\link{gdup}}
}
\examples{
listDevices()
getDevice()
#setDevice(1)
}
\keyword{device}

