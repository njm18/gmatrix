
#define DEFINEGLOBALSHERE
#include "gmatrix.h"


void initialize_globals()
{

	for(int i=0;i<MAX_DEVICE;i++) {
		total_states[i]=0;
		threads_per_block[i]=0;
		dev_state_set[i]=0;
		dev_cublas_set[i]=0;

	};
	currentDevice=0;

}


SEXP get_globals()
{
	int deviceCount = 0;
	int i;

	cudaGetDeviceCount(&deviceCount);
    SEXP ret, ret_total_states,ret_threads_per_block,ret_dev_state_set,ret_dev_cublas_set,ret_currentDevice;
	PROTECT(ret = allocVector(VECSXP, deviceCount));

	PROTECT(ret_total_states = allocVector(INTSXP, deviceCount));
	PROTECT(ret_threads_per_block = allocVector(INTSXP, deviceCount));
	PROTECT(ret_dev_state_set = allocVector(INTSXP, deviceCount));
	PROTECT(ret_dev_cublas_set = allocVector(INTSXP, deviceCount));
	PROTECT(ret_currentDevice = allocVector(INTSXP, 1));

	for(i=0;i<deviceCount;i++) {
		INTEGER(ret_total_states)[i]=total_states[i];
		INTEGER(ret_threads_per_block)[i]=threads_per_block[i];
		INTEGER(ret_dev_state_set)[i]=dev_state_set[i];
		INTEGER(ret_dev_cublas_set)[i]=dev_cublas_set[i];

	}
	INTEGER(ret_currentDevice)[0]=currentDevice;
	SET_VECTOR_ELT(ret, 0, ret_total_states);
	SET_VECTOR_ELT(ret, 1, ret_threads_per_block);
	SET_VECTOR_ELT(ret, 2, ret_dev_state_set);
	SET_VECTOR_ELT(ret, 3, ret_dev_cublas_set);
	SET_VECTOR_ELT(ret, 4, ret_currentDevice);


	/*GLOBAL int total_states[MAX_DEVICE];
	GLOBAL curandState* dev_states[MAX_DEVICE];
	GLOBAL int threads_per_block[MAX_DEVICE];
	GLOBAL int dev_state_set[MAX_DEVICE];
	GLOBAL int dev_cublas_set[MAX_DEVICE];
	GLOBAL int currentDevice;*/
	UNPROTECT(6);
	return(ret);
}

SEXP get_device()
{

	SEXP ret;
	PROTECT(ret = allocVector(INTSXP, 1));
	INTEGER(ret)[0]=currentDevice;
	UNPROTECT(1);
	return(ret);
}



void free_dev_states(int *silent)
{
	cudaError_t status1;
	if(dev_state_set[currentDevice]==1) {
		if(silent[0]==0)
			Rprintf("Deleting old states on device %d.\n", currentDevice);
		status1=cudaFree((dev_states[currentDevice]));
		if (status1 != cudaSuccess ) {
			error("CUDA memory free error in 'free_(dev_states[currentDevice]).' (%d) \n", (int) status1);
			return;
		}
	}
}

void set_threads_per_block(int *tpb) {
	threads_per_block[currentDevice]=tpb[0];
}

/*
void set_c(double *in_c)
{
	c1=in_c[0];
	c2=in_c[1];
	c3=in_c[2];
}

void get_c(double *in_c)
{
	in_c[0]=c1;
	in_c[1]=c2;
	in_c[2]=c3;
}
void set_(total_states[currentDevice])(int *in_(total_states[currentDevice]))
{
	(total_states[currentDevice])=in_(total_states[currentDevice])[0];
}*/


/*
void check_started()
{
	if(started==0L)
		error("GPU device has not yet been selected. Please use listDevices() and setDevice() to select a divice.")
}*/


/* do some setup*/
__global__ void kernel_setup_curand(curandState *state, int seed, int n)
{
	int id = threadIdx.x + blockIdx.x *  blockDim.x ;
	/* Each thread gets same seed , a different sequence number - no offset */
	if(id<n)
		curand_init(seed, id, 0, &state[id]) ;
}


SEXP setup_curand(SEXP in_total_states, SEXP in_seed, SEXP in_silent, SEXP in_force)
{	//check_started();
	int my_total_states=INTEGER(in_total_states)[0];

	int force = INTEGER(in_force)[0];
	int silent = INTEGER(in_silent)[0];
	int seed=INTEGER(in_seed)[0];
	cudaError_t  cudaStat;
	int doit;
	if(force==1)
		doit=1;
	else if(dev_state_set[currentDevice]==0)
		doit=1;
	else if(total_states[currentDevice]!=my_total_states)
		doit=1;
	else doit=0;

	if(doit==1) {
		if(dev_state_set[currentDevice]==1) {
			if(silent==0)
				Rprintf("Deleting old states on device %d.\n", currentDevice);
			if((dev_states[currentDevice])!=NULL) {
				cudaStat=cudaFree((dev_states[currentDevice]));
				if (cudaStat != cudaSuccess ) {
					error("CUDA memory free error in 'setup_curand.' (%d) \n", (int) cudaStat);
				}
			}
		}
		total_states[currentDevice]=my_total_states;

		if(silent==0)
			Rprintf("Creating new states on device %d.\n", currentDevice);
		/* Allocate space for prng states on device */
		cudaStat = cudaMalloc (( void **)&(dev_states[currentDevice]), (total_states[currentDevice])*sizeof(curandState));
		if (cudaStat != cudaSuccess ) {
			error("Allocation error from 'setup_curand.' (%d)'\n", (int) cudaStat);
		}
		/* Setup prng states */
		int blocksPerGrid = ((total_states[currentDevice]) + (threads_per_block[currentDevice]) - 1) / (threads_per_block[currentDevice]);
		kernel_setup_curand<<<blocksPerGrid, (threads_per_block[currentDevice])>>>((dev_states[currentDevice]), seed, (total_states[currentDevice]));
		cudaStat = cudaDeviceSynchronize();
		if (cudaStat != cudaSuccess ) {
			error("Kernal error from 'setup_curand.' (%d)'\n", (int) cudaStat);
		}

		dev_state_set[currentDevice]=1;
	}
	return in_total_states;
}


void startCublas(int* silent) { // must be called with .C interface
	cublasStatus_t status1;
	if(dev_cublas_set[currentDevice]==0) {
		if(silent[0]==0)
			Rprintf("Starting cublas on device %d.\n", currentDevice);
		status1 = cublasCreate(&(handle[currentDevice]));
		if (status1 != CUBLAS_STATUS_SUCCESS) {
			error("CUBLAS initialization error\n");
		}
		dev_cublas_set[currentDevice]=1;
	}
}

void stopCublas(int* silent) {
	cublasStatus_t status1;
	//check_started();
	if(dev_cublas_set[currentDevice]!=0) {
		if(silent[0]==0)
			Rprintf("Shutting down cublas on device %d", currentDevice);
		status1 =  cublasDestroy((handle[currentDevice]));
		if (status1 != CUBLAS_STATUS_SUCCESS) {
			warning("CUBLAS shutdown error\n");
		}
	}
}

/*
void RlistDevices(int* curdevice, int *memory, int *total, int *silent) {
	int deviceCount = 0;
	int i;
	cudaDeviceProp deviceProp;

	cudaGetDeviceCount(&deviceCount);

	if(deviceCount>20)
		error("to many devices to list.");
	for(i=0;i<deviceCount;i++) {
		cudaGetDeviceProperties(&deviceProp, i);
		memory[i]=deviceProp.totalGlobalMem ;
		if(silent[0]==0) {
			if(current[0]==i)
				Rprintf("%d - \"%s\" (current device)\n", i, deviceProp.name);
			else
				Rprintf("%d - \"%s\"\n", i, deviceProp.name);
			Rprintf("     Total global memory: %d\n", deviceProp.totalGlobalMem );
			Rprintf("     Multiprocessor count: %d\n", deviceProp.multiProcessorCount);
		}
	}

}
*/
void setDevice(int *device, int *silent) {

	cudaError_t status1;
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
#ifdef DEBUG
	Rprintf("%d %d",
			deviceCount,device[0]);
#endif
	if((device[0] < 0) || (device[0] > deviceCount))
		error("The gpu id (%d) number is not valid.",device[0]);
#ifdef DEBUG
	Rprintf("here");
#endif
	status1 = cudaSetDevice(device[0]);
	if (status1 != cudaSuccess) {
		if(status1 == cudaErrorSetOnActiveProcess)
			error("Active process. Can't set device.\n");
		else if(status1 ==  cudaErrorInvalidDevice)
			error("Invalid Device\n");
		else
			error("Unknown errors\n");

	} else {
			currentDevice=device[0];
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, device[0]);
			if(silent[0]==0)
				Rprintf("Now using device %d - \"%s\"\n", device[0], deviceProp.name);
	}
	/*
	GLOBAL __device__ int CUDA_R_Na_int;
	GLOBAL __device__ double CUDA_R_Na_double;
	GLOBAL __device__ float CUDA_R_Na_float;
	R defines the following
	void attribute_hidden InitArithmetic()
	{
	    R_NaInt = INT_MIN;
	    R_NaN = 0.0/R_Zero_Hack;
	    R_NaReal = R_ValueOfNA();
	    R_PosInf = 1.0/R_Zero_Hack;
	    R_NegInf = -1.0/R_Zero_Hack;
	}*/
	float R_NaFloat = (float) R_NaReal;
	cudaMemcpyToSymbol(CUDA_R_Na_int, &R_NaInt, sizeof(int));
	cudaMemcpyToSymbol(CUDA_R_Na_float, &R_NaFloat, sizeof(float));
	cudaMemcpyToSymbol(CUDA_R_Na_double, &R_NaReal, sizeof(double));

}

void deviceReset() {
	cudaError_t cudaStat;
	cudaStat=cudaDeviceReset();
	CUDA_ERROR;

}


void setFlagSpin() {
	cudaError_t cudaStat;
	cudaStat= cudaSetDeviceFlags(cudaDeviceScheduleSpin);
	CUDA_ERROR;
}
void setFlagYield() {
	cudaError_t cudaStat;
	cudaStat= cudaSetDeviceFlags(cudaDeviceScheduleYield);
	CUDA_ERROR;
}
void setFlagBlock() {
	cudaError_t cudaStat;
	cudaStat= cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
	CUDA_ERROR;
}


/*
void d_PrintMatrix(double *d_matrix,int rows, int cols, int startRow, int stopRow) {
	double *matrix = Calloc(rows*cols, double);
	if (matrix == NULL ) {
		Rprintf("d_PrintMatrix: Could not allocate memory.");
	} else {
		cublasGetMatrix(rows, cols, sizeof(double), d_matrix, rows, matrix, rows);
		PrintMatrix(matrix, rows, cols, startRow, stopRow);
		Free(matrix);
	}
}


void PrintMatrix(double matrix[], int rows, int cols, int startRow, int stopRow)
{
	int r,c;
	int row_stop= min(rows,stopRow);
	Rprintf("Matrix is: %d x %d \n", rows, cols);
	for(r=startRow;r<row_stop;r++) {
		Rprintf("[%3d]", r);
		for(c=0; c<cols;c++) {
	//		if( abs(matrix[c*rows + r]) > 100000)
	//			Rprintf("%1.10f ", matrix[c*rows + r]);
	//		else
				Rprintf("  %e  ", matrix[c*rows + r]);
		}
		Rprintf("\n");
	}

}*/
void check_mem(int *freer, int *totr, int *silent) {
	size_t free, total;
	cudaMemGetInfo(&free,&total);
	if(silent[0]==0)
		Rprintf("%d MB free out of %d MB total.\n",free/1048576,total/1048576);
	freer[0]=free;
	totr[0]=total;
	//mem[0]=(int) free;
	//mem[1]=(int) total;
}



SEXP  get_device_info(SEXP property)
{
	int deviceCount = 0;
	int i;
	cudaDeviceProp deviceProp;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
	if (error_id != cudaSuccess)
    {
        error("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
    }

	SEXP ret;


#define LOOK(MPROP,MDPROP) \
		if(strcmp(CHAR(STRING_ELT(property, 0)), #MPROP) == 0) {\
			PROTECT(ret = allocVector(INTSXP, deviceCount));\
			for(i=0;i<deviceCount;i++) {\
				cudaGetDeviceProperties(&deviceProp, i);\
				INTEGER(ret)[i] = deviceProp.MDPROP ;\
			}\
		}

	if(strcmp(CHAR(STRING_ELT(property, 0)), "name") == 0) {
		PROTECT(ret = allocVector(STRSXP, deviceCount));
		for(i=0;i<deviceCount;i++) {
			cudaGetDeviceProperties(&deviceProp, i);
			SET_STRING_ELT(ret, i, mkChar(deviceProp.name));
		}
	} else LOOK(totalGlobalMem,totalGlobalMem)
	else LOOK(sharedMemPerBlock,sharedMemPerBlock)
	else LOOK(regsPerBlock,regsPerBlock)
	else LOOK(warpSize,warpSize)
	else LOOK(memPitch,memPitch)
	else LOOK(maxThreadsPerBlock,maxThreadsPerBlock)
	else LOOK(maxThreadsDim0,maxThreadsDim[0])
	else LOOK(maxThreadsDim1,maxThreadsDim[1])
	else LOOK(maxThreadsDim2,maxThreadsDim[2])
	else LOOK(maxGridSize0,maxGridSize[0])
	else LOOK(maxGridSize1,maxGridSize[1])
	else LOOK(maxGridSize2,maxGridSize[2])
	else LOOK(clockRate,clockRate)
	else LOOK(totalConstMem,totalConstMem)
	else LOOK(major,major)
	else LOOK(minor,minor)
	else LOOK(textureAlignment,textureAlignment)
	else LOOK(deviceOverlap,deviceOverlap)
	else LOOK(multiProcessorCount,multiProcessorCount)
	else LOOK(kernelExecTimeoutEnabled,kernelExecTimeoutEnabled)
	else LOOK(integrated,integrated)
	else LOOK(canMapHostMemory,canMapHostMemory)
	else LOOK(computeMode,computeMode)
	else LOOK(maxTexture1D,maxTexture1D)
	else LOOK(maxTexture2D0,maxTexture2D[0])
	else LOOK(maxTexture2D1,maxTexture2D[1])
	else LOOK(maxTexture3D0,maxTexture3D[0])
	else LOOK(maxTexture3D1,maxTexture3D[1])
	else LOOK(maxTexture3D2,maxTexture3D[2])
	else LOOK(maxTexture1DLayered0,maxTexture1DLayered[0])
	else LOOK(maxTexture1DLayered1,maxTexture1DLayered[1])
	else LOOK(maxTexture2DLayered0,maxTexture2DLayered[0])
	else LOOK(maxTexture2DLayered1,maxTexture2DLayered[1])
	else LOOK(maxTexture2DLayered2,maxTexture2DLayered[2])
	else LOOK(surfaceAlignment,surfaceAlignment)
	else LOOK(concurrentKernels,concurrentKernels)
	else LOOK(ECCEnabled,ECCEnabled)
	else LOOK(pciBusID,pciBusID)
	else LOOK(pciDeviceID,pciDeviceID)
	else LOOK(pciDomainID,pciDomainID)
	else LOOK(tccDriver,tccDriver)
	else LOOK(asyncEngineCount,asyncEngineCount)
	else LOOK(unifiedAddressing,unifiedAddressing)
	else LOOK(memoryClockRate,memoryClockRate)
	else LOOK(memoryBusWidth,memoryBusWidth)
	else LOOK(l2CacheSize,l2CacheSize)
	else LOOK(maxThreadsPerMultiProcessor,maxThreadsPerMultiProcessor)
	else
		error("Property not recognized.");


	UNPROTECT(1);
	return(ret);
}

//void call_gc() {
//	SEXP s, t;
//	PROTECT(t1 = s1 = allocList(2));
//	PROTECT(t2 = s2 = allocList(3));
//
//	SET_TYPEOF(s1, LANGSXP);
//	SET_TYPEOF(s2, LANGSXP);
//
//	SETCAR(t1, install(".Internal")); t1 = CDR(t1);
//	SETCAR(t2, install("gc")); t2 = CDR(t2);
//
//	SETCAR(t, ScalarInteger(digits));
//	SET_TAG(t, install("digits"));
//	eval(s, env);
//	UNPROTECT(1);
//}

/*void
R_init_mylib(DllInfo *info)
{
	Rprintf("Starting cublas...");
	startCublas();
}

void
R_unload_mylib(DllInfo *info)
{
	Rprintf("Stoping cublas...");
	stopCublas();
}
 */

