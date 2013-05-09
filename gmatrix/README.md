The "gmatrix" Package
=================================================

This package implements a general framework for utilizing R to harness the power of NVIDIA GPU's. The "gmatrix" and "gvector" classes allow for easy management of the separate device and host memory spaces. Numerous numerical operations are implemented for these objects on the GPU. These operations include matrix multiplication, addition, subtraction, the kronecker product, the outer product, comparison operators, logical operators, trigonometric functions, indexing, sorting, random number generation and many more.
The "gmatrix" package has only been tested and compiled for linux machines. It would certainly be nice of someone to get it working in Windows. Until then, Windows is not supported. 
In addition we assume that the divice is at least of NVIDIA(R) compute capibility 2.0, so this package may not work with older devices.

Installation Instructions
-------------------------
1. Install the the CUDA Toolkit. The current version of 'gmatix' has been tested for CUDA Toolkit 4.0 and 5.0. 
2. Install R. The current version of 'gmatrix' has been tested under R 2.15.0.
3. Start R and then install the 'gmatrix' package with the following commands. Package compilation may take 5-10 minutes.

'''
download.file("http://solomon.case.edu/gmatrix/gmatrix.tar.gz", "gmatrix.tar.gz")
install.packages("gmatrix.tar.gz", repos = NULL)
file.remove("gmatrix.tar.gz")
'''
	 
Installation Note
-----------------
By default, when compiling, the makefile assumes that
+ The the CUDA library files are located in the folder /usr/local/cuda/lib64.
+ The R libraries are located in the folder /usr/include/R.
+ The compute capibility of the target device is 2.0.

If these are incorrect assumptions, the user may set these values and install using the follwing R commands as an example.
First set the environmental variables:

    Sys.setenv(CUDA_LIB_PATH="/usr/include/cuda-5.0/lib64")
    Sys.setenv(R_INC_PATH="/usr/local/R/R-2.15.0/lib64/R/include")
    Sys.setenv(NVCC_ARCH="-gencode arch=compute_30,code=sm_30")
    
Next install the package as above:

    download.file("http://solomon.case.edu/gmatrix/gmatrix.tar.gz", "gmatrix.tar.gz")
    install.packages("gmatrix.tar.gz", repos = NULL)
    file.remove("gmatrix.tar.gz")
	    
Testing the Installation
-------------------------
We recoment that the user test the installation using the following commands:

    library(gmatrix)
    gtest()
    
Please report any errors to the package maintainer.

Getting Started
---------------
+ To list available gpu devices use: listDevices()
+ To set the device use: setDevice()
+ To move object to the device use: g()
+ To move object to the host use: h()
+ Object on the device can be manipulated in much the same way other R objects can.