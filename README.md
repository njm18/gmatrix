The "gmatrix" Package
=================================================

This package implements a general framework for utilizing R to harness the power of NVIDIA GPU's. The "gmatrix" and "gvector" classes allow for easy management of the separate device and host memory spaces. Numerous numerical operations are implemented for these objects on the GPU. These operations include matrix multiplication, addition, subtraction, the kronecker product, the outer product, comparison operators, logical operators, trigonometric functions, indexing, sorting, random number generation and many more.
The "gmatrix" package has only been tested and compiled for linux machines. It would certainly be nice of someone to get it working in Windows. Until then, Windows is not supported. 
In addition we assume that the divice is at least of NVIDIA(R) compute capibility 2.0, so this package may not work with older devices. There is a companion package [rcula](https://github.com/njm18/rcula/tree/master/rcula) which allows the user to perform eigen decomposition and solve linear equations.

Installation Instructions
-------------------------
1. Install the the CUDA Toolkit. The current version of 'gmatix' has been tested for CUDA Toolkit 5.0. 
2. Install R. The current version of 'gmatrix' has been tested under R 3.0.2.
3. Start R and then install the 'gmatrix' package with the following commands. Package compilation may take 5-10 minutes.

```
install.packages("gmatrix")
```

Alternatively, if you would like to install the developmental version, the following from the linux command line may be used:

	git clone https://github.com/njm18/gmatrix.git
	MAKE="make -j7" #note this make the compile process use 7 threads 
	R CMD build gmatrix
	R CMD INSTALL gmatrix_0.3.tar.gz --no-test-load --configure-args="--with-arch=sm_50"

	 
Installation Note
-----------------
By default, when compiling, the build process assumes that
+ The nvcc compiler is in the PATH, and that the the CUDA library files may be located based on the location of nvcc.
+ R is located in the PATH, and that the R libraries may be located using this information.
+ The compute capability of the target device is 2.0.

If these are incorrect assumptions, the user may set these values and install using the following R command as an example.

```
install.packages("gmatrix" ,  
   configure.args = "
      --with-arch=sm_30
      --with-cuda-home=/opt/cuda
      --with-r-home==/opt/R"
)
```

	    
Testing the Installation
-------------------------
We recoment that the user test the installation using the following commands:

    library(gmatrix)
    gtest()
    
Please report any errors to the package maintainer.

Getting Started
---------------
+ Load the library for each sessesion using: library(gmatrix)
+ To list available gpu devices use: listDevices()
+ To set the device use: setDevice()
+ To move object to the device use: g()
+ To move object to the host use: h()
+ Object on the device can be manipulated in much the same way other R objects can.
+ A list of help topics may be optained using: help(package="gmatrix")