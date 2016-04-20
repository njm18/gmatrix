
#include <thrust/device_ptr.h>
#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>
#include <Rdefines.h>
#include <R_ext/Rdynload.h>
#include <math.h>
#include <R_ext/PrtUtil.h>
#include <R_ext/Applic.h>
#include <R_ext/Arith.h>
#include <R_ext/Boolean.h>
//#include <cutil_inline.h>
//include "cublas.h"
#include <stdio.h>
#include <cublas_v2.h>
#include <curand_kernel.h>

#if CUDART_VERSION >= 7000
#include <cusolverDn.h>
#endif

#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>


#define IDX2(i ,j ,ld) (((j)*(ld))+(i))



#if CUDART_VERSION < 6500
#define GET_BLOCKS_PER_GRID(n, kern)  \
	int tpb=threads_per_block[currentDevice];\
	int blocksPerGrid = (n + tpb - 1) / (tpb); \
	int operations_per_thread = 1;  \
	if(blocksPerGrid>MAX_BLOCKS) {  \
		blocksPerGrid = MAX_BLOCKS;  \
		int total_threads = blocksPerGrid*tpb; \
		operations_per_thread = (n + total_threads -1) / total_threads; \
	}
#else
#define GET_BLOCKS_PER_GRID(n, kern)  \
    int minGridSize;\
	int tpb;\
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &tpb, kern);\
	int blocksPerGrid = (n + tpb - 1) / (tpb); \
	int operations_per_thread = 1;  \
	if(blocksPerGrid>MAX_BLOCKS) {  \
		blocksPerGrid = MAX_BLOCKS;  \
		int total_threads = blocksPerGrid*tpb; \
		operations_per_thread = (n + total_threads -1) / total_threads; \
	}
#endif

//Rprintf("tpb = %d, operations_per_thread = %d, blocksPerGrid = %d, minGridSize=%d\n", tpb, operations_per_thread, blocksPerGrid, minGridSize);

#define DECERROR0 cudaError_t  cudaStat
#define DECERROR1 cudaError_t  cudaStat, status1
#define DECERROR2 cudaError_t  cudaStat, status1, status2
#define DECERROR3 cudaError_t  cudaStat, status1, status2, status3

//Macros for malloc
#define CUDA_MALLOC(MPTR,MN)  \
		cudaStat = cudaMalloc( (void **)&(MPTR),MN) ;\
		if (cudaStat != cudaSuccess ){\
			R_gc();\
			cudaStat = cudaMalloc( (void **)&(MPTR),MN) ;\
			if (cudaStat != cudaSuccess ){\
				error("CUDA memory allocation error in '%s.' (%s)\n", __func__, cudaGetErrorString(cudaStat));\
			}\
		}

#define CUDA_MALLOC_CLEAN_1(MPTR,MN,MCLEANPTR)  \
		cudaStat = cudaMalloc( (void **)&(MPTR),MN) ;\
		if (cudaStat != cudaSuccess ) {\
			R_gc();\
			cudaStat = cudaMalloc( (void **)&(MPTR),MN) ;\
			if (cudaStat != cudaSuccess ){\
				status1=cudaFree(MCLEANPTR);\
				if (status1 != cudaSuccess) {\
					error("CUDA memory allocation and free error (potential memory leak) in '%s.' (%s)\n", __func__, cudaGetErrorString(cudaStat));\
				}\
				error("CUDA memory allocation error in '%s.' (%s)\n", __func__, cudaGetErrorString(cudaStat));\
			}\
		}

#define CUDA_MALLOC_CLEAN_2(MPTR,MN,MCLEANPTR1,MCLEANPTR2)  \
		cudaStat = cudaMalloc( (void **)&(MPTR),MN ) ;\
		if (cudaStat != cudaSuccess ) {\
			R_gc();\
			cudaStat = cudaMalloc( (void **)&(MPTR),MN) ;\
			if (cudaStat != cudaSuccess ){\
				status1=cudaFree(MCLEANPTR1);\
				status2=cudaFree(MCLEANPTR2);\
				if (status1 != cudaSuccess || status2 != cudaSuccess) {\
					error("CUDA memory allocation error in '%s.' (%s)\n", __func__, cudaGetErrorString(cudaStat));\
				}\
				error("CUDA memory allocation error in '%s.' (%s)\n", __func__, cudaGetErrorString(cudaStat));\
			}\
		}

//Macros for checking errors
#define CUDA_ERROR \
		if (cudaStat != cudaSuccess ) {\
		 error("Error in '%s.' (%s)\n", __func__, cudaGetErrorString(cudaStat));\
		}

//Macros for checking the kernal and cleaning up if there are errors
#define CUDA_CHECK_KERNAL  \
		cudaStat = cudaDeviceSynchronize(); \
		if (cudaStat != cudaSuccess ) {\
			error("Kernal error in '%s.' (%s)\n", __func__, cudaGetErrorString(cudaStat));\
		}

#define CUDA_CHECK_KERNAL_CLEAN_1(MCLEANPTR1)  \
		cudaStat = cudaDeviceSynchronize(); \
		if (cudaStat != cudaSuccess ) {\
			status1=cudaFree(MCLEANPTR1);\
			if (status1 != cudaSuccess ) {\
				error("Kernal error and memory free errors (potential memory leak) in '%s.' (%s)\n", __func__, cudaGetErrorString(cudaStat));\
			}\
		 error("Kernal error in '%s.' (%s)\n", __func__, cudaGetErrorString(cudaStat));\
		}
#define CUDA_CHECK_KERNAL_CLEAN_2(MCLEANPTR1,MCLEANPTR2)  \
		cudaStat = cudaDeviceSynchronize(); \
		if (cudaStat != cudaSuccess ) {\
			status1=cudaFree(MCLEANPTR1);\
			status2=cudaFree(MCLEANPTR2);\
			if (status1 != cudaSuccess || status2 != cudaSuccess) {\
				error("Kernal error and memory free errors (potential memory leak) in '%s.' (%s)\n", __func__, cudaGetErrorString(cudaStat));\
			}\
		 error("Kernal error in '%s.' (%s)\n", __func__, cudaGetErrorString(cudaStat));\
		}

#define CUDA_CHECK_KERNAL_CLEAN_3(MCLEANPTR1,MCLEANPTR2,MCLEANPTR3)  \
		cudaStat = cudaDeviceSynchronize(); \
		if (cudaStat != cudaSuccess ) {\
			status1=cudaFree(MCLEANPTR1);\
			status2=cudaFree(MCLEANPTR2);\
			status3=cudaFree(MCLEANPTR3);\
			if (status1 != cudaSuccess || status2 != cudaSuccess || status3 != cudaSuccess ) {\
				error("Kernal error and memory free errors (potential memory leak) in '%s.' (%s)\n", __func__, cudaGetErrorString(cudaStat));\
			}\
		 error("Kernal error in '%s.' (%s)\n", __func__, cudaGetErrorString(cudaStat));\
		}


//macros for cleaning up
#define CUDA_CLEAN_1(MCLEANPTR1)  \
		status1=cudaFree(MCLEANPTR1);\
		if (status1 != cudaSuccess) {\
			error("Memory free errors (potential memory leak) in '%s.' (%s)\n", __func__, cudaGetErrorString(status1));\
		}

#define CUDA_CLEAN_2(MCLEANPTR1,MCLEANPTR2)  \
		status1=cudaFree(MCLEANPTR1);\
		status2=cudaFree(MCLEANPTR2);\
		if (status1 != cudaSuccess || status2 != cudaSuccess ) {\
			if (status1 != cudaSuccess && status2 == cudaSuccess )\
				error("Memory free errors (potential memory leak) in '%s.' (%s)\n", __func__, cudaGetErrorString(status1));\
			if (status1 == cudaSuccess && status2 != cudaSuccess )\
				error("Memory free errors (potential memory leak) in '%s.' (%s)\n", __func__, cudaGetErrorString(status2));\
			if (status1 != cudaSuccess && status2 != cudaSuccess )\
				error("Memory free errors (potential memory leak) in '%s.' (%s) (%s)\n", __func__, cudaGetErrorString(status1), cudaGetErrorString(status2));\
		}


//memory copy macros (this assumes that DST is on the device and needs to be cleaned up when errors arise)
#define CUDA_MEMCPY_CLEAN(DST, SRC, COUNT,KIND)  \
    cudaStat=cudaMemcpy(DST, SRC, COUNT, KIND) ;\
    if (cudaStat != cudaSuccess) {\
		status1=cudaFree(DST);\
		if (status1 != cudaSuccess  ) {\
			error("Memory copy and memory free errors (potential memory leak) in '%s.' (%s)\n", __func__, cudaGetErrorString(cudaStat));\
		}\
		error("Memory copy error in '%s.' (%s)\n", __func__, cudaGetErrorString(cudaStat));\
    }

#define CUDA_MEMCPY_CLEAN_1(DST, SRC, COUNT,KIND, MCLEANPTR)  \
    cudaStat=cudaMemcpy(DST, SRC, COUNT, KIND) ;\
    if (cudaStat != cudaSuccess) {\
		status1=cudaFree(DST);\
		status2=cudaFree(MCLEANPTR);\
		if (status1 != cudaSuccess || status2 != cudaSuccess ) {\
			error("Memory copy and memory free errors (potential memory leak) in '%s.' (%s)\n", __func__, cudaGetErrorString(cudaStat));\
		}\
		error("Memory copy error in '%s.' (%s)\n", __func__, cudaGetErrorString(cudaStat));\
    }

#define CUDA_MEMCPY_CLEAN_2(DST, SRC, COUNT,KIND, MCLEANPTR1, MCLEANPTR2)  \
    cudaStat=cudaMemcpy(DST, SRC, COUNT, KIND) ;\
    if (cudaStat != cudaSuccess) {\
		status1=cudaFree(DST);\
		status2=cudaFree(MCLEANPTR1);\
		status3=cudaFree(MCLEANPTR2);\
		if (status1 != cudaSuccess || status2 != cudaSuccess || status3 != cudaSuccess ) {\
			error("Memory copy and memory free errors (potential memory leak) in '%s.' (%s)\n", __func__, cudaGetErrorString(cudaStat));\
		}\
		error("Memory copy error in '%s.' (%s)\n", __func__, cudaGetErrorString(cudaStat));\
    }

#define 	PROCESS_TYPE_NO_SIZE\
	int type = INTEGER(in_type)[0];\
	if(type>3)\
		error("Incorrect type passed to '%s.'", __func__);
		
#define 	PROCESS_TYPE_NO_SIZE_SF\
	int type = INTEGER(in_type)[0];\
	if(type>1)\
		error("Incorrect type passed to '%s.'", __func__);

#define 	PROCESS_TYPE\
	int type = INTEGER(in_type)[0];\
	int mysizeof;\
	if(type==0)\
		mysizeof=sizeof(double);\
	else if(type==1)\
		mysizeof=sizeof(float);\
	else if(type==2 || type==3)\
		mysizeof=sizeof(int);\
	else\
		error("Incorrect type passed to '%s.'", __func__);

#define 	PROCESS_TYPE_SF\
	int type = INTEGER(in_type)[0];\
	int mysizeof;\
	if(type==0)\
		mysizeof=sizeof(double);\
	else if(type==1)\
		mysizeof=sizeof(float);\
	else \
		error("Incorrect type passed to '%s.' Type must be 'double' or 'float.'", __func__);

#define PTR_DBL(A) \
		(double *) A->d_vec

#define PTR_FLOAT(A) \
		(float *) A->d_vec

#define PTR_INT(A) \
		(int *) A->d_vec

#define CALL_KERNAL\
		if(type==0) {\
			KERNAL(PTR_DBL, double)\
		} else if(type==1) {\
			KERNAL(PTR_FLOAT, float)\
		} else {\
			KERNAL(PTR_INT, int)\
		}

#define CALL_KERNAL_SF\
		if(type==0) {\
			KERNAL(PTR_DBL, double)\
		} else if(type==1) {\
			KERNAL(PTR_FLOAT, float)\
		}


#define MAX_DEVICE 20
#define MAX_BLOCKS 65000

#ifdef DEFINEGLOBALSHERE
#define GLOBAL
#else
#define GLOBAL extern
#endif

GLOBAL __device__ int CUDA_R_Na_int;
GLOBAL __device__ double CUDA_R_Na_double;
GLOBAL __device__ float CUDA_R_Na_float;
GLOBAL cublasHandle_t handle[MAX_DEVICE];
#if CUDART_VERSION >= 7000
GLOBAL cusolverDnHandle_t  cudshandle[MAX_DEVICE];
#endif

GLOBAL int total_states[MAX_DEVICE] ;
GLOBAL curandState* dev_states[MAX_DEVICE];
GLOBAL int threads_per_block[MAX_DEVICE] ;
GLOBAL int dev_state_set[MAX_DEVICE] ;
GLOBAL int dev_cublas_set[MAX_DEVICE];
GLOBAL int currentDevice ;



union ieee_float {
  unsigned int myint;
  float myfloat;
};
union ieee_double {
  unsigned long mylong;
  double mydouble;
};


#define RNAREAL 0x7ff00000000007A2
#define MYNAFLOAT 0x7F8000FF

//make NA
template <typename T>
__forceinline__ __device__ void MAKE_NA(T *ret) {
	((ieee_double*) ret)->mylong=RNAREAL;
}
template <>
__forceinline__ __device__ void MAKE_NA(float *ret) {
	((ieee_float*) ret)->myint=MYNAFLOAT;
}

template <>
__forceinline__ __device__ void MAKE_NA(int *ret) {
	ret[0]=INT_MIN;
}


//is NA
template <typename T>
__forceinline__ __device__ int IS_NA(T *val) {
	return(((ieee_double*) val)->mylong==RNAREAL);
}
template <>
__forceinline__ __device__ int IS_NA(float *val) {
	return(((ieee_float*) val)->myint==MYNAFLOAT);
}

template <>
__forceinline__ __device__ int IS_NA(int *val) {
	return(INT_MIN==val[0]);
}

//is NAN
template <typename T>
__forceinline__ __device__ int IS_NAN(T val) {
	return(isnan(val));
}
template <>
__forceinline__ __device__ int IS_NAN(float val) {
	return(isnan(val));
}

template <>
__forceinline__ __device__ int IS_NAN(int val) {
	return(INT_MIN==val);
}

//return nan
template <typename T>
__forceinline__ __device__ T RET_NAN(void) {
	return(NAN);
}
template <>
__forceinline__ __device__ float RET_NAN(void) {
	return(NAN);
}

template <>
__forceinline__ __device__ int RET_NAN(void) {
	return(INT_MIN);
}


extern "C" {

struct gpuvec {
	 void *d_vec;
	 int device;
};

struct matrix {
   void *d_vec;
   int rows;
   int cols;
   int ld;
};




#define RNAINT INT_MIN; //works for the moment
#define RNADOUBLE nan(1954); //works for the moment
#define RNAFLOAT nanf(1954); //works for the moment

//general GPU stuff
//void checkAlocationStatus(cublasStatus status);
SEXP get_globals();
void initialize_globals();
SEXP get_device();
void setDevice(int *device, int *silent);
SEXP setup_curand(SEXP in_total_states, SEXP in_seed, SEXP in_silent, SEXP in_force);
void startCublas(int* silent, int *set);
void stopCublas(int* silent) ;
void deviceReset();
void check_mem(int *freer, int *totr, int *silent);
SEXP get_device_info(SEXP property);
void free_dev_states(int *silent);
void set_threads_per_block(int *tpb) ;

//void PrintMatrix(double matrix[], int rows, int cols, int startRow, int stopRow);
//void deviceReset();
//void RlistDevices(int* curdevice, int *memory, int *current, int *total, int *silent);
//SEXP setup_curand(SEXP in_total_states, SEXP in_seed, SEXP in_silent);

//void get_(threads_per_block[currentDevice])(int *in_(threads_per_block[currentDevice]));
//void free_(dev_states[currentDevice])();
//void check_mem();
void setFlagBlock();
void setFlagYield();
void setFlagSpin();

//void set_(total_states[currentDevice])(int *in_(total_states[currentDevice]));
//void get_(total_states[currentDevice])(int *in_(total_states[currentDevice]));
//SEXP setup_curand(SEXP in_(total_states[currentDevice]), SEXP in_seed, SEXP first_time);

//matrix and vector stuff
static void gpu_finalizer(SEXP ext);
SEXP gpu_create(SEXP in_mat, SEXP in_type);
SEXP gpu_register(struct gpuvec *in_mat);
SEXP gpu_get(SEXP ptr, SEXP sn, SEXP in_type);
SEXP gpu_duplicate(SEXP in_vec, SEXP sn, SEXP in_type);
SEXP gpu_rep_m(SEXP in_A,SEXP in_n, SEXP in_N, SEXP in_times_each, SEXP in_type);
SEXP gpu_rep_1(SEXP in_val, SEXP in_N, SEXP in_type);
SEXP gpu_convert(SEXP A_in, SEXP in_N, SEXP in_type, SEXP in_to_type );
SEXP gpu_cpy(SEXP ptr_in, SEXP ptr_out, SEXP sn,SEXP in_type);

//indexing an manipulation
SEXP gpu_numeric_index(SEXP A_in, SEXP n_A_in, SEXP index_in, SEXP n_index_in, SEXP in_type);
SEXP gpu_gmatrix_index_row(SEXP A_in, SEXP n_row_A_in, SEXP n_col_A_in, SEXP index_in, SEXP n_index_in, SEXP in_type);
SEXP gpu_gmatrix_index_col(SEXP A_in, SEXP n_row_A_in, SEXP n_col_A_in, SEXP index_in, SEXP n_index_in, SEXP in_type);
SEXP gpu_gmatrix_index_both(SEXP A_in, SEXP n_row_A_in, SEXP n_col_A_in,
		SEXP index_row_in, SEXP n_index_row_in,SEXP index_col_in, SEXP n_index_col_in, SEXP in_type);
SEXP gpu_naive_transpose(SEXP A_in, SEXP n_row_in, SEXP n_col_in, SEXP in_type);
SEXP gpu_diag_get(SEXP A_in, SEXP n_row_in, SEXP n_col_in, SEXP in_type);
SEXP gpu_diag_set(SEXP A_in, SEXP n_row_in, SEXP n_col_in, SEXP val_in, SEXP n_val_in, SEXP in_type);
SEXP gpu_diag_set_one(SEXP A_in, SEXP n_row_in, SEXP n_col_in, SEXP val_in, SEXP in_type);
SEXP gpu_gmatrix_index_both_set(SEXP A_in, SEXP n_row_A_in, SEXP n_col_A_in, SEXP val_in, SEXP n_val_in,
		SEXP index_row_in, SEXP n_index_row_in,SEXP index_col_in, SEXP n_index_col_in, SEXP in_type);
SEXP gpu_gmatrix_index_col_set(SEXP A_in, SEXP n_row_A_in, SEXP n_col_A_in,  SEXP val_in, SEXP n_val_in, SEXP index_in, SEXP n_index_in, SEXP in_type);
SEXP gpu_gmatrix_index_row_set(SEXP A_in, SEXP n_row_A_in, SEXP n_col_A_in, SEXP val_in, SEXP n_val_in, SEXP index_in, SEXP n_index_in, SEXP in_type);
SEXP gpu_numeric_index_set(SEXP A_in, SEXP n_A_in, SEXP val_in, SEXP n_val_in, SEXP index_in, SEXP n_index_in, SEXP in_type);
SEXP gpu_sum(SEXP A_in, SEXP n_in, SEXP in_type);
SEXP gpu_min(SEXP A_in, SEXP n_in, SEXP in_type);
SEXP gpu_max(SEXP A_in, SEXP n_in, SEXP in_type);
SEXP gpu_sort(SEXP A_in, SEXP n_in, SEXP stable_in, SEXP decreasing_in, SEXP in_type);
SEXP gpu_order(SEXP A_in, SEXP n_in, SEXP stable_in, SEXP decreasing_in, SEXP in_type);
SEXP gpu_which(SEXP A_in, SEXP n_in);
SEXP gpu_seq( SEXP n_in, SEXP init_in, SEXP step_in, SEXP in_type  );
SEXP gpu_if(SEXP H_in, SEXP A_in, SEXP B_in,SEXP snh, SEXP sna, SEXP snb, SEXP in_type);
SEXP gpu_rowLogSums(SEXP in_P, SEXP in_rows, SEXP in_endCol, SEXP in_startCol, SEXP in_type);

//matrix multiplications
SEXP matrix_multiply(SEXP A_in, SEXP B_in, SEXP transa, SEXP transb, SEXP in_type);//ordinary matrix multiplication
SEXP gpu_gmm(SEXP A_in, SEXP B_in, SEXP C_in, SEXP transa, SEXP transb, SEXP accum, SEXP in_type);
SEXP gpu_outer(SEXP A_in, SEXP B_in,SEXP n_A_in, SEXP n_B_in, SEXP op_in, SEXP in_type);
SEXP gpu_kernal_sumby(SEXP A_in, SEXP index1_in,SEXP index2_in,SEXP n_A_in,SEXP n_index_in, SEXP in_type);
SEXP gpu_kronecker(SEXP A_in, SEXP B_in,SEXP n_A_row_in,SEXP n_A_col_in, SEXP n_B_row_in,SEXP n_B_col_in, SEXP in_type);
SEXP gpu_mat_times_diag_vec(SEXP A_in, SEXP B_in, SEXP n_row_in, SEXP n_col_in, SEXP in_type);

SEXP rcusolve_qr(SEXP A_in, SEXP qraux_in);
SEXP rcusolve_modqr_coef(SEXP qr_in, SEXP qraux_in, SEXP B_in);
SEXP rcusolve_svd(SEXP A_in,SEXP  S_in, SEXP U_in,SEXP  VT_in);
SEXP rcusolve_chol(SEXP A_in);

//SEXP rcula_eigen_symm(SEXP A_in, SEXP val_in);
//SEXP rcusolve_dgesv(SEXP A_in, SEXP B_in);
//SEXP check_inverse_condition(SEXP Ain, SEXP Avalsin, SEXP permin, SEXP tolin) ;

//simple binary operations and operations that include in place (IP) procedures
#define BINARYOPDEF(MNAME) \
		SEXP gpu_same_size_##MNAME (SEXP A_in, SEXP B_in, SEXP sn, SEXP in_type);\
		SEXP gpu_scaler_##MNAME (SEXP A_in, SEXP B_in, SEXP sn, SEXP in_type);\
		SEXP gpu_diff_size_##MNAME(SEXP A_in, SEXP B_in, SEXP sna, SEXP snb, SEXP in_type);\
		
#define BINARYOPDEF_IP(MNAME) \
		SEXP gpu_same_size_ip_##MNAME (SEXP A_in, SEXP B_in, SEXP sn, SEXP in_type);\
		SEXP gpu_scaler_ip_##MNAME (SEXP A_in, SEXP B_in, SEXP sn, SEXP in_type);\
		SEXP gpu_diff_size_ip_##MNAME(SEXP A_in, SEXP B_in, SEXP sna, SEXP snb, SEXP in_type);\
		
BINARYOPDEF(add);
BINARYOPDEF(lgspadd);
BINARYOPDEF(mult);
BINARYOPDEF(eq);
BINARYOPDEF(ne);
BINARYOPDEF(gt12);
BINARYOPDEF(gt21);
BINARYOPDEF(lt12);
BINARYOPDEF(lt21);
BINARYOPDEF(gte12);
BINARYOPDEF(gte21);
BINARYOPDEF(lte12);
BINARYOPDEF(lte21);
BINARYOPDEF(sub12);
BINARYOPDEF(sub21);
BINARYOPDEF(div12);
BINARYOPDEF(div21);
BINARYOPDEF(pow12);
BINARYOPDEF(pow21);
BINARYOPDEF(mod12);
BINARYOPDEF(mod21);
BINARYOPDEF(and);
BINARYOPDEF(or);

BINARYOPDEF_IP(add);
BINARYOPDEF_IP(lgspadd);
BINARYOPDEF_IP(mult);
BINARYOPDEF_IP(sub);
BINARYOPDEF_IP(div);
BINARYOPDEF_IP(pow);
BINARYOPDEF_IP(mod);
BINARYOPDEF_IP(and);
BINARYOPDEF_IP(or);

//distributions
SEXP gpu_rnorm(SEXP in_n, SEXP in_mean, SEXP in_sd, SEXP in_n_mean, SEXP in_n_sd, SEXP in_type);
SEXP gpu_pnorm(SEXP in_n, SEXP in_x, SEXP in_mean, SEXP in_sd, SEXP in_n_mean, SEXP in_n_sd,
		SEXP in_log, SEXP in_lower, SEXP in_type);
SEXP gpu_dnorm(SEXP in_n, SEXP in_x, SEXP in_mean, SEXP in_sd, SEXP in_n_mean, SEXP in_n_sd,
		SEXP in_log, SEXP in_type);
SEXP gpu_qnorm(SEXP in_n, SEXP in_x, SEXP in_mean, SEXP in_sd, SEXP in_n_mean, SEXP in_n_sd,
		SEXP in_log, SEXP in_lower, SEXP in_type);

SEXP gpu_rgamma(SEXP in_n, SEXP in_mean, SEXP in_sd, SEXP in_n_mean, SEXP in_n_sd, SEXP in_type);
SEXP gpu_dunif(SEXP in_n, SEXP in_x, SEXP in_min, SEXP in_max, SEXP in_n_min, SEXP in_n_max,
		SEXP in_log, SEXP in_type);

SEXP gpu_runif(SEXP in_n, SEXP in_min, SEXP in_max, SEXP in_n_min, SEXP in_n_max, SEXP in_type);
SEXP gpu_dgamma(SEXP in_n, SEXP in_x, SEXP in_parm1, SEXP in_parm2, SEXP in_n_parm1, SEXP in_n_parm2,
		SEXP in_log, SEXP in_type);

SEXP gpu_rbeta(SEXP in_n, SEXP in_alpha, SEXP in_scale, SEXP in_n_alpha, SEXP in_n_scale, SEXP in_type);
SEXP gpu_dbeta(SEXP in_n, SEXP in_x, SEXP in_parm1, SEXP in_parm2, SEXP in_n_parm1, SEXP in_n_parm2,
		SEXP in_log, SEXP in_type);

SEXP gpu_rbinom(SEXP in_n, SEXP in_alpha, SEXP in_scale, SEXP in_n_alpha, SEXP in_n_scale, SEXP in_type);
SEXP gpu_dbinom(SEXP in_n, SEXP in_x, SEXP in_parm1, SEXP in_parm2, SEXP in_n_parm1, SEXP in_n_parm2,
		SEXP in_log, SEXP in_type);

SEXP gpu_rpois(SEXP in_n, SEXP in_parm1,  SEXP in_n_parm1, SEXP in_type);
SEXP gpu_dpois(SEXP in_n, SEXP in_x, SEXP in_parm1, SEXP in_n_parm1,
		SEXP in_log, SEXP in_type);

SEXP gpu_rsample(SEXP in_P, SEXP in_rows, SEXP in_cols, SEXP in_norm, SEXP in_type);

//elementwise operations
SEXP gpu_one_over(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_sqrt(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_exp(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_expm1(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_log(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_log2(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_log10(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_log1p(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_sin(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_cos(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_tan(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_asin(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_acos(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_atan(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_sinh(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_cosh(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_tanh(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_asinh(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_acosh(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_atanh(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_fabs(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_sign(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_lgamma(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_gamma(SEXP y, SEXP sn, SEXP in_type);

SEXP gpu_ceil( SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_floor(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_round(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_isna( SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_isnan(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_isfinite(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_isinfinite(SEXP y, SEXP sn, SEXP in_type);

SEXP gpu_ip_one_over(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_ip_sqrt(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_ip_exp(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_ip_expm1(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_ip_log(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_ip_log2(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_ip_log10(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_ip_log1p(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_ip_sin(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_ip_cos(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_ip_tan(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_ip_asin(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_ip_acos(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_ip_atan(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_ip_sinh(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_ip_cosh(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_ip_tanh(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_ip_asinh(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_ip_acosh(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_ip_atanh(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_ip_fabs(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_ip_sign(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_ip_lgamma(SEXP y, SEXP sn, SEXP in_type);
SEXP gpu_ip_gamma(SEXP y, SEXP sn, SEXP in_type);



//gfunction
//SEXP gpu_gfunction_call(SEXP args_in, SEXP each_arg_len_in, SEXP fid_in, SEXP varid_in, SEXP outlen_in, SEXP in_type);
SEXP cudaVersion();




//dumb stuff
SEXP asSEXPint(int myint);
struct gpuvec asgpuvec(struct matrix a);
SEXP fromcpuMultmm(SEXP a, SEXP transa, SEXP b, SEXP transb);
}


/*
__global__ void VecAdd(const double* A, const double* B, double* C, int N);
__global__ void myFKernal(const double* bigY, const double* bigX,
		const double* EmpPriorR_x_Sqrt2SigmaInv, double* myF, double x, int N);
__global__ void logKernal(double* y, int N);
__global__ void vec_logspace_add_addKernal(double *a, double *b, double add, int N);
__global__ void vec_logspace_addKernal(double *a, double *b, int N);
*/
