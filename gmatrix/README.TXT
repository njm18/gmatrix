**********************************************************************************************************
                            Notes and Installation Instructions for 'gmatrix'
**********************************************************************************************************
NOTE:
The 'gmatrix' package has only been tested and compiled for linux machines. It would certainly be nice
of some well intentioned individual to get it working in Windows. Until then, Windows is not supported. 

INSTALATION INSTRUCTIONS:
1.  Install the the CUDA Toolkit. The current version of 'gmatix' has been tested for CUDA Toolkit 4.0.
    At the time this was written version 4.0 was located at https://developer.nvidia.com/cuda-toolkit-40.
    
2.  Install R. The current version of 'gmatrix' has been tested under R 2.15.0.

3.  Start R and install the 'gmatrix' package with the command:
        install.packages("gmatrix")
    By default the makefile assumes that
	    a. The the CUDA library files are located in the folder /usr/local/cuda/lib64.
	    b. The R libraries are located in the folder /usr/include/R.
	If this is an incorrect assumption the user may set these values and install using the follwing R 
	commands:
	    Sys.setenv(CUDA_LIB_PATH="/usr/include/cuda/lib64") #set the cuda library path
	    Sys.setenv(R_INC_PATH="/usr/local/R/R-2.15.0/lib64/R/include") #set the R library path
	    install.packages("gmatrix")
	    
GETTING STARTED:
    To list available gpu devices use: listDevices()
    To set the device use: setDevice()
    To move object to the device use: g()
    To move object to the host use: h()
    Object on the device can be manipulated in much the same way other R objects can.