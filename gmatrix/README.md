Notes and Installation Instructions for "gmatrix"
=================================================

This package implements a general framework for utilizing R to harness the power of NVIDIA GPU's. The "gmatrix" and "gvector" classes allow for easy management of the separate device and host memory spaces. Numerous numerical operations are implemented for these objects on the GPU. These operations include matrix multiplication, addition, subtraction, the kronecker product, the outer product, comparison operators, logical operators, trigonometric functions, indexing, sorting, random number generation and many more.
The "gmatrix" package has only been tested and compiled for linux machines. It would certainly be nice of someone to get it working in Windows. Until then, Windows is not supported. 

Installation Instructions
-------------------------
1. Install the the CUDA Toolkit. The current version of 'gmatix' has been tested for CUDA Toolkit 4.0. At the time this was written version 4.0 was located at https://developer.nvidia.com/cuda-toolkit-40.
2. Install R. The current version of 'gmatrix' has been tested under R 2.15.0.
3. Start R and install the 'gmatrix' package with the command: '''install.packages("gmatrix")'''

Note
----
By default the makefile assumes that
+ The the CUDA library files are located in the folder /usr/local/cuda/lib64.
+ The R libraries are located in the folder /usr/include/R.
If this is an incorrect assumption the user may set these values and install using the follwing R commands:
'''R
Sys.setenv(CUDA_LIB_PATH="/usr/include/cuda/lib64") #set the cuda library path
Sys.setenv(R_INC_PATH="/usr/local/R/R-2.15.0/lib64/R/include") #set the R library path
install.packages("gmatrix")
'''
	    
Getting Started
---------------
+ To list available gpu devices use: listDevices()
+ To set the device use: setDevice()
+ To move object to the device use: g()
+ To move object to the host use: h()
+ Object on the device can be manipulated in much the same way other R objects can.