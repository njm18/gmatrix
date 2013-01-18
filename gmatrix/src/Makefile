#
# Makefile
# --------------
# by Nathan Morris
#          (Modified from the R/GPU Makefile by M. Kempenaar)
#


#to debug... make sure nvcc_flags includes -g -G and shlib flag includes -d
#            R -d cuda-gdb
#            run --vanilla
#            Run some code to load in the .so (dyn.load)
#            ctr + c to get back to debugger
#            b local.cu:[line#]
#            c to continue
#            s to step
#            d b to delet al breakpoints


# LIB / INC locations (default, change these using 'export' in the terminal, please refer
# to the manual on the gmatrix install instructions)
NVCC_ARCH     ?= sm_20
CUDA_LIB_PATH ?= /usr/local/cuda/lib64
R_INC_PATH    ?= /usr/include/R


# Nvidia CUDA Compiler
NVCC = nvcc #/usr/local/cuda/bin/nvcc

# R gcc build script
R_CC = R CMD SHLIB


# Output Files:
#   CUDA output
OUT_OBJ =  dist.o conversions.o general.o ops.o manipulation.o  matrix.o
#   R output
OUT_SO = gmatrix.so

# CUDA Libraries -d
LIBS = -lcudart -lcublas 

# CUDA Flags (-g -G removed)
NVCC_FLAGS = --shared -Xcompiler "-fPIC"  -arch=${NVCC_ARCH}



# Build rules
build: $(OUT_OBJ)
	$(R_CC) $(OUT_OBJ) -o $(OUT_SO) $(LIBS) -L$(CUDA_LIB_PATH)
	
conversions.o: 
	$(NVCC) -c conversions.cu -o conversions.o $(NVCC_FLAGS)  -I$(R_INC_PATH)
	
manipulation.o: 
	$(NVCC) -c manipulation.cu -o manipulation.o $(NVCC_FLAGS)  -I$(R_INC_PATH)

general.o:
	$(NVCC) -c general.cu -o general.o $(NVCC_FLAGS) -I$(R_INC_PATH)

matrix.o: 
	$(NVCC) -c matrix.cu -o matrix.o $(NVCC_FLAGS) -I$(R_INC_PATH)
 	
ops.o: 
	$(NVCC) -c ops.cu -o ops.o $(NVCC_FLAGS) -I$(R_INC_PATH)
	
dist.o: 
	$(NVCC) -c dist.cu -o dist.o $(NVCC_FLAGS) -I$(R_INC_PATH)
 	

clean:
	-rm -f *.o core
	-rm -f *.so core
	-rm -f *.out
	-rm -f *.linkinfo

rebuild: clean build

