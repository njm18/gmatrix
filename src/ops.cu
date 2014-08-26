
#include "gmatrix.h"






//unary operations
#define ELEMENTWISEOP(MNAME,MCFUN)  \
		template <typename T>\
	    __global__ void kernal_##MNAME (T* x, T* ret, int n, int operations_per_thread) \
		{\
			int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;\
			for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;\
				i < mystop; i+=blockDim.x) {\
				if(i<n) {\
					ret[i] = MCFUN;\
				}\
			}\
		}\
		SEXP gpu_##MNAME (SEXP y, SEXP sn, SEXP in_type)\
		{\
			SEXP ret_final;\
			int n = INTEGER(sn)[0];\
			DECERROR1;\
			PROCESS_TYPE_SF;\
			struct gpuvec *ret = Calloc(1, struct gpuvec);\
			struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(y);\
			CUDA_MALLOC(ret->d_vec, n * mysizeof);\
			GET_BLOCKS_PER_GRID(n);\
			if(type==0)\
				kernal_##MNAME <double><<<blocksPerGrid, (threads_per_block[currentDevice])>>>((double *) A->d_vec,(double *)ret->d_vec, n, operations_per_thread);\
			else if(type==1)\
				kernal_##MNAME <float><<<blocksPerGrid, (threads_per_block[currentDevice])>>>((float *) A->d_vec, (float *) ret->d_vec, n, operations_per_thread);\
			CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);\
			ret_final = gpu_register(ret);\
			return ret_final;\
		}

#define ELEMENTWISEOP_RETURNINT(MNAME,MCFUN)  \
		template <typename T>\
	    __global__ void kernal_##MNAME (T* x, int* ret, int n, int operations_per_thread) \
		{\
			int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;\
			for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;\
				i < mystop; i+=blockDim.x) {\
				if(i<n) {\
					ret[i] = MCFUN;\
				}\
			}\
		}\
		SEXP gpu_##MNAME (SEXP y, SEXP sn, SEXP in_type)\
		{\
			SEXP ret_final;\
			int n = INTEGER(sn)[0];\
			DECERROR1;\
			PROCESS_TYPE_NO_SIZE;\
			struct gpuvec *ret = Calloc(1, struct gpuvec);\
			struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(y);\
			CUDA_MALLOC(ret->d_vec, n * sizeof(int));\
			GET_BLOCKS_PER_GRID(n);\
			if(type==0)\
				kernal_##MNAME <double><<<blocksPerGrid, (threads_per_block[currentDevice])>>>((double *) A->d_vec,(int *)ret->d_vec, n, operations_per_thread);\
			else if(type==1)\
				kernal_##MNAME <float><<<blocksPerGrid, (threads_per_block[currentDevice])>>>((float *) A->d_vec, (int *) ret->d_vec, n, operations_per_thread);\
			else\
				error("'type' must be double or single.");\
			CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);\
			ret_final = gpu_register(ret);\
			return ret_final;\
		}

ELEMENTWISEOP(one_over,1/x[i]);
ELEMENTWISEOP(sqrt,sqrt(x[i]));
ELEMENTWISEOP(exp,exp(x[i]));
ELEMENTWISEOP(expm1,expm1(x[i]));
ELEMENTWISEOP(log,log(x[i]));
ELEMENTWISEOP(log2,log2(x[i]));
ELEMENTWISEOP(log10,log10(x[i]));
ELEMENTWISEOP(log1p,log1p(x[i]));
ELEMENTWISEOP(sin,sin(x[i]));
ELEMENTWISEOP(cos,cos(x[i]));
ELEMENTWISEOP(tan,tan(x[i]));
ELEMENTWISEOP(asin,asin(x[i]));
ELEMENTWISEOP(acos,acos(x[i]));
ELEMENTWISEOP(atan,atan(x[i]));
ELEMENTWISEOP(sinh,sinh(x[i]));
ELEMENTWISEOP(cosh,cosh(x[i]));
ELEMENTWISEOP(tanh,tanh(x[i]));
ELEMENTWISEOP(asinh,asinh(x[i]));
ELEMENTWISEOP(acosh,acosh(x[i]));
ELEMENTWISEOP(atanh,atanh(x[i]));
ELEMENTWISEOP(fabs,fabs(x[i]));
ELEMENTWISEOP(lgamma,lgamma(x[i]));
ELEMENTWISEOP(gamma,tgamma(x[i]));


template <typename T>
__device__ T mysign(T myin) {
	if(myin==0)
		return 0;
	else
		return copysign( 1.0,myin);
}
ELEMENTWISEOP( sign, mysign<T>(x[i]) );

//ELEMENTWISEOP(sign,copysign( 1.0,x[i]));




ELEMENTWISEOP_RETURNINT(ceil,ceil(x[i]));
ELEMENTWISEOP_RETURNINT(floor,floor(x[i]));
ELEMENTWISEOP_RETURNINT(round, rint(x[i]));
ELEMENTWISEOP_RETURNINT(isna, (IS_NA<T>(&(x[i]))) );
ELEMENTWISEOP_RETURNINT(isnan, isnan(x[i]) && !(IS_NA<T>(&(x[i]))) );
ELEMENTWISEOP_RETURNINT(isfinite, isfinite(x[i]) );
ELEMENTWISEOP_RETURNINT(isinfinite, isinf(x[i]) );








//binary operations

#define BINARYOP_SF(MNAME,MCFUN1, MCFUN2, MCFUN3) \
		template <typename T>\
		__global__ void kernal_same_size_##MNAME (T* y, T* x,T* ret, int N, int operations_per_thread)\
		{\
			int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;\
			for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;\
				i < mystop; i+=blockDim.x) {\
				if (i < N) {\
					ret[i] = MCFUN1 ;\
				}\
			}\
		}\
		SEXP gpu_same_size_##MNAME (SEXP A_in, SEXP B_in, SEXP sn, SEXP in_type)\
		{\
			SEXP ret_final;\
			int n = INTEGER(sn)[0];\
			DECERROR1;\
			struct gpuvec *ret = Calloc(1, struct gpuvec);\
			struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);\
			struct gpuvec *B = (struct gpuvec*) R_ExternalPtrAddr(B_in);\
			PROCESS_TYPE_SF;\
			CUDA_MALLOC(ret->d_vec,n * mysizeof) ;\
			GET_BLOCKS_PER_GRID(n);\
			if(type==0)\
				kernal_same_size_##MNAME <double> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((double *) A->d_vec, (double *) B->d_vec,(double *) ret->d_vec, n, operations_per_thread);\
			else if(type==1)\
				kernal_same_size_##MNAME <float> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((float *) A->d_vec, (float *) B->d_vec, (float *) ret->d_vec, n, operations_per_thread);\
			CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);\
			ret_final = gpu_register(ret);\
			return ret_final;\
		}\
		template <typename T>\
		__global__ void kernal_scaler_##MNAME (T* y, T* ret, T c, int N, int operations_per_thread)\
		{\
	int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;\
	for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;\
			i < mystop; i+=blockDim.x) {\
				if (i < N) {\
					ret[i] = MCFUN3 ;\
				}\
			}\
		}\
		SEXP gpu_scaler_##MNAME (SEXP A_in, SEXP B_in, SEXP sn, SEXP in_type)\
		{\
			SEXP ret_final;\
			int n = INTEGER(sn)[0];\
			DECERROR1;\
			struct gpuvec *ret = Calloc(1, struct gpuvec);\
			struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);\
			PROCESS_TYPE_SF;\
			CUDA_MALLOC(ret->d_vec,n * mysizeof) ;\
			GET_BLOCKS_PER_GRID(n);\
			if(type==0) {\
				double B = REAL(B_in)[0];\
				kernal_scaler_##MNAME <double> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((double *) A->d_vec,(double *) ret->d_vec, B, n, operations_per_thread);\
			} else if(type==1) {\
				float B = (float) REAL(B_in)[0];\
				kernal_scaler_##MNAME <float> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((float *) A->d_vec,(float *) ret->d_vec, B, n, operations_per_thread);\
			}\
			CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);\
			ret_final = gpu_register(ret);\
			return ret_final;\
		}\
		template <typename T>\
		__global__ void kernal_diff_size_##MNAME (T* y, T* x, T* ret, int ny, int nx, int operations_per_thread)\
		{\
			int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;\
			for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;\
				i < mystop; i+=blockDim.x) {\
				int j = i % nx;\
				if (i < ny) {\
					ret[i] = MCFUN2 ;\
				}\
			}\
		}\
		SEXP gpu_diff_size_##MNAME(SEXP A_in, SEXP B_in, SEXP sna, SEXP snb, SEXP in_type)\
		{\
			SEXP ret_final;\
			int na = INTEGER(sna)[0];\
			int nb = INTEGER(snb)[0];\
			DECERROR1;\
			struct gpuvec *ret = Calloc(1, struct gpuvec);\
			struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);\
			struct gpuvec *B = (struct gpuvec*) R_ExternalPtrAddr(B_in);\
			PROCESS_TYPE_SF;\
			CUDA_MALLOC(ret->d_vec,na * mysizeof) ;\
			GET_BLOCKS_PER_GRID(na);\
			if(type==0)\
				kernal_diff_size_##MNAME <double> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((double *) A->d_vec, (double *) B->d_vec,(double *) ret->d_vec, na, nb, operations_per_thread);\
			else if(type==1)\
				kernal_diff_size_##MNAME <float> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((float *) A->d_vec, (float *) B->d_vec, (float *) ret->d_vec, na, nb, operations_per_thread);\
			CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);\
			ret_final = gpu_register(ret);\
			return ret_final;\
		}

#define BINARYOP(MNAME,MCFUN1, MCFUN2, MCFUN3) \
		template <typename T>\
		__global__ void kernal_same_size_##MNAME (T* y, T* x,T* ret, int N, int operations_per_thread)\
		{\
			int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;\
			for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;\
				i < mystop; i+=blockDim.x) {\
				if (i < N) {\
					ret[i] = MCFUN1 ;\
				}\
			}\
		}\
		SEXP gpu_same_size_##MNAME (SEXP A_in, SEXP B_in, SEXP sn, SEXP in_type)\
		{\
			SEXP ret_final;\
			int n = INTEGER(sn)[0];\
			DECERROR1;\
			struct gpuvec *ret = Calloc(1, struct gpuvec);\
			struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);\
			struct gpuvec *B = (struct gpuvec*) R_ExternalPtrAddr(B_in);\
			PROCESS_TYPE;\
			CUDA_MALLOC(ret->d_vec,n * mysizeof) ;\
			GET_BLOCKS_PER_GRID(n);\
			if(type==0)\
				kernal_same_size_##MNAME <double> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((double *) A->d_vec, (double *) B->d_vec,(double *) ret->d_vec, n, operations_per_thread);\
			else if(type==1)\
				kernal_same_size_##MNAME <float> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((float *) A->d_vec, (float *) B->d_vec, (float *) ret->d_vec, n, operations_per_thread);\
			else\
				kernal_same_size_##MNAME <int> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((int *) A->d_vec, (int *) B->d_vec, (int *) ret->d_vec, n, operations_per_thread);\
			CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);\
			ret_final = gpu_register(ret);\
			return ret_final;\
		}\
		template <typename T>\
		__global__ void kernal_scaler_##MNAME (T* y, T* ret, T c, int N, int operations_per_thread)\
		{\
			int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;\
			for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;\
				i < mystop; i+=blockDim.x) {\
				if (i < N) {\
					ret[i] = MCFUN3 ;\
				}\
			}\
		}\
		SEXP gpu_scaler_##MNAME (SEXP A_in, SEXP B_in, SEXP sn, SEXP in_type)\
		{\
			SEXP ret_final;\
			int n = INTEGER(sn)[0];\
			DECERROR1;\
			struct gpuvec *ret = Calloc(1, struct gpuvec);\
			struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);\
			PROCESS_TYPE;\
			CUDA_MALLOC(ret->d_vec,n * mysizeof) ;\
			GET_BLOCKS_PER_GRID(n);\
			if(type==0){\
				double B = REAL(B_in)[0];\
				kernal_scaler_##MNAME <double> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((double *) A->d_vec,(double *) ret->d_vec, B, n, operations_per_thread);\
			} else if(type==1) {\
				float B = (float)REAL(B_in)[0];\
				kernal_scaler_##MNAME <float> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((float *) A->d_vec,(float *) ret->d_vec, (float) B, n, operations_per_thread);\
			} else {\
				int B = INTEGER(B_in)[0];\
				kernal_scaler_##MNAME <int> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((int *) A->d_vec,(int *) ret->d_vec, (int) B, n, operations_per_thread);\
			}\
			CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);\
			ret_final = gpu_register(ret);\
			return ret_final;\
		}\
		template <typename T>\
		__global__ void kernal_diff_size_##MNAME (T* y, T* x, T* ret, int ny, int nx, int operations_per_thread)\
		{\
			int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;\
			for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;\
				i < mystop; i+=blockDim.x) {\
				int j = i % nx;\
				if (i < ny) {\
					ret[i] = MCFUN2 ;\
				}\
			}\
		}\
		SEXP gpu_diff_size_##MNAME(SEXP A_in, SEXP B_in, SEXP sna, SEXP snb, SEXP in_type)\
		{\
			SEXP ret_final;\
			int na = INTEGER(sna)[0];\
			int nb = INTEGER(snb)[0];\
			DECERROR1;\
			struct gpuvec *ret = Calloc(1, struct gpuvec);\
			struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);\
			struct gpuvec *B = (struct gpuvec*) R_ExternalPtrAddr(B_in);\
			PROCESS_TYPE;\
			CUDA_MALLOC(ret->d_vec,na * mysizeof) ;\
			GET_BLOCKS_PER_GRID(na);\
			if(type==0)\
				kernal_diff_size_##MNAME <double> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((double *) A->d_vec, (double *) B->d_vec,(double *) ret->d_vec, na, nb, operations_per_thread);\
			else if(type==1)\
				kernal_diff_size_##MNAME <float> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((float *) A->d_vec, (float *) B->d_vec, (float *) ret->d_vec, na, nb, operations_per_thread);\
			else \
				kernal_diff_size_##MNAME <int> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((int *) A->d_vec, (int *) B->d_vec, (int *) ret->d_vec, na, nb, operations_per_thread);\
			CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);\
			ret_final = gpu_register(ret);\
			return ret_final;\
		}

#define BINARYOP_COMPARE(MNAME,MCFUN1, MCFUN2, MCFUN3) \
		template <typename T>\
		__global__ void kernal_same_size_##MNAME (T* y, T* x, int* ret, int N, int operations_per_thread)\
		{\
			int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;\
			for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;\
				i < mystop; i+=blockDim.x) {\
				if (i < N) {\
					ret[i] = MCFUN1 ;\
				}\
			}\
		}\
		SEXP gpu_same_size_##MNAME (SEXP A_in, SEXP B_in, SEXP sn, SEXP in_type)\
		{\
			SEXP ret_final;\
			int n = INTEGER(sn)[0];\
			DECERROR1;\
			struct gpuvec *ret = Calloc(1, struct gpuvec);\
			struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);\
			struct gpuvec *B = (struct gpuvec*) R_ExternalPtrAddr(B_in);\
			PROCESS_TYPE_NO_SIZE;\
			CUDA_MALLOC(ret->d_vec,n * sizeof(int)) ;\
			GET_BLOCKS_PER_GRID(n);\
			if(type==0)\
				kernal_same_size_##MNAME <double> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((double *) A->d_vec, (double *) B->d_vec,(int *) ret->d_vec, n, operations_per_thread);\
			else if(type==1)\
				kernal_same_size_##MNAME <float> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((float *) A->d_vec, (float *) B->d_vec, (int *) ret->d_vec, n, operations_per_thread);\
			else\
				kernal_same_size_##MNAME <int> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((int *) A->d_vec, (int *) B->d_vec, (int *) ret->d_vec, n, operations_per_thread);\
			CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);\
			ret_final = gpu_register(ret);\
			return ret_final;\
		}\
		template <typename T>\
		__global__ void kernal_scaler_##MNAME (T* y, int* ret, T c, int N, int operations_per_thread)\
		{\
	int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;\
	for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;\
			i < mystop; i+=blockDim.x) {\
				if (i < N) {\
					ret[i] = MCFUN3 ;\
				}\
			}\
		}\
		SEXP gpu_scaler_##MNAME (SEXP A_in, SEXP B_in, SEXP sn, SEXP in_type)\
		{\
			SEXP ret_final;\
			int n = INTEGER(sn)[0];\
			DECERROR1;\
			struct gpuvec *ret = Calloc(1, struct gpuvec);\
			struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);\
			PROCESS_TYPE_NO_SIZE;\
			CUDA_MALLOC(ret->d_vec,n * sizeof(int)) ;\
			GET_BLOCKS_PER_GRID(n);\
			if(type==0){ \
				double B = REAL(B_in)[0];\
				kernal_scaler_##MNAME <double> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((double *) A->d_vec,(int *) ret->d_vec, B, n, operations_per_thread);\
			} else if(type==1) {\
				float B = (float) REAL(B_in)[0];\
				kernal_scaler_##MNAME <float> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((float *) A->d_vec,(int *) ret->d_vec, B, n, operations_per_thread);\
			} else {\
				int B = INTEGER(B_in)[0];\
				kernal_scaler_##MNAME <int> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((int *) A->d_vec,(int *) ret->d_vec, (int) B, n, operations_per_thread);\
			}\
			CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);\
			ret_final = gpu_register(ret);\
			return ret_final;\
		}\
		template <typename T>\
		__global__ void kernal_diff_size_##MNAME (T* y, T* x, int* ret, int ny, int nx, int operations_per_thread)\
		{\
			int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;\
			for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;\
				i < mystop; i+=blockDim.x) {\
				int j = i % nx;\
				if (i < ny) {\
					ret[i] = MCFUN2 ;\
				}\
			}\
		}\
		SEXP gpu_diff_size_##MNAME(SEXP A_in, SEXP B_in, SEXP sna, SEXP snb, SEXP in_type)\
		{\
			SEXP ret_final;\
			int na = INTEGER(sna)[0];\
			int nb = INTEGER(snb)[0];\
			DECERROR1;\
			struct gpuvec *ret = Calloc(1, struct gpuvec);\
			struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);\
			struct gpuvec *B = (struct gpuvec*) R_ExternalPtrAddr(B_in);\
			PROCESS_TYPE_NO_SIZE;\
			CUDA_MALLOC(ret->d_vec,na * sizeof(int)) ;\
			GET_BLOCKS_PER_GRID(na);\
			if(type==0)\
				kernal_diff_size_##MNAME <double> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((double *) A->d_vec, (double *) B->d_vec,(int *) ret->d_vec, na, nb, operations_per_thread);\
			else if(type==1)\
				kernal_diff_size_##MNAME <float> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((float *) A->d_vec, (float *) B->d_vec, (int *) ret->d_vec, na, nb, operations_per_thread);\
			else \
				kernal_diff_size_##MNAME <int> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((int *) A->d_vec, (int *) B->d_vec, (int *) ret->d_vec, na, nb, operations_per_thread);\
			CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);\
			ret_final = gpu_register(ret);\
			return ret_final;\
		}


#define BINARYOP_LOGICAL(MNAME,MCFUN1, MCFUN2, MCFUN3) \
		template <typename T>\
		__global__ void kernal_same_size_##MNAME (T* y, T* x, int* ret, int N, int operations_per_thread)\
		{\
			int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;\
			for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;\
				i < mystop; i+=blockDim.x) {\
				if (i < N) {\
					ret[i] = MCFUN1 ;\
				}\
			}\
		}\
		SEXP gpu_same_size_##MNAME (SEXP A_in, SEXP B_in, SEXP sn, SEXP in_type)\
		{\
			SEXP ret_final;\
			int n = INTEGER(sn)[0];\
			DECERROR1;\
			struct gpuvec *ret = Calloc(1, struct gpuvec);\
			struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);\
			struct gpuvec *B = (struct gpuvec*) R_ExternalPtrAddr(B_in);\
			PROCESS_TYPE_NO_SIZE;\
			CUDA_MALLOC(ret->d_vec,n * sizeof(int)) ;\
			GET_BLOCKS_PER_GRID(n);\
			if(type!=3)\
				error("type must be logical for logical operations");\
			kernal_same_size_##MNAME <int> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((int *) A->d_vec, (int *) B->d_vec, (int *) ret->d_vec, n, operations_per_thread);\
			CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);\
			ret_final = gpu_register(ret);\
			return ret_final;\
		}\
		template <typename T>\
		__global__ void kernal_scaler_##MNAME (T* y, int* ret, T c, int N, int operations_per_thread)\
		{\
			int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;\
			for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;\
				i < mystop; i+=blockDim.x) {\
				if (i < N) {\
					ret[i] = MCFUN3 ;\
				}\
			}\
		}\
		SEXP gpu_scaler_##MNAME (SEXP A_in, SEXP B_in, SEXP sn, SEXP in_type)\
		{\
			SEXP ret_final;\
			int n = INTEGER(sn)[0];\
			DECERROR1;\
			struct gpuvec *ret = Calloc(1, struct gpuvec);\
			struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);\
			int B = INTEGER(B_in)[0];\
			PROCESS_TYPE_NO_SIZE;\
			CUDA_MALLOC(ret->d_vec,n * sizeof(int)) ;\
			GET_BLOCKS_PER_GRID(n);\
			if(type!=3)\
				error("type must be logical for logical operations");\
			kernal_scaler_##MNAME <int> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((int *) A->d_vec,(int *) ret->d_vec, B, n, operations_per_thread);\
			CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);\
			ret_final = gpu_register(ret);\
			return ret_final;\
		}\
		template <typename T>\
		__global__ void kernal_diff_size_##MNAME (T* y, T* x, int* ret, int ny, int nx, int operations_per_thread)\
		{\
			int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;\
			for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;\
			i < mystop; i+=blockDim.x) {\
				int j = i % nx;\
				if (i < ny) {\
					ret[i] = MCFUN2 ;\
				}\
			}\
		}\
		SEXP gpu_diff_size_##MNAME(SEXP A_in, SEXP B_in, SEXP sna, SEXP snb, SEXP in_type)\
		{\
			SEXP ret_final;\
			int na = INTEGER(sna)[0];\
			int nb = INTEGER(snb)[0];\
			DECERROR1;\
			struct gpuvec *ret = Calloc(1, struct gpuvec);\
			struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);\
			struct gpuvec *B = (struct gpuvec*) R_ExternalPtrAddr(B_in);\
			PROCESS_TYPE_NO_SIZE;\
			CUDA_MALLOC(ret->d_vec,na * sizeof(int)) ;\
			GET_BLOCKS_PER_GRID(na);\
			if(type!=3)\
				error("type must be logical for logical operations");\
			kernal_diff_size_##MNAME <int> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((int *) A->d_vec, (int *) B->d_vec, (int *) ret->d_vec, na, nb, operations_per_thread);\
			CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);\
			ret_final = gpu_register(ret);\
			return ret_final;\
		}



BINARYOP_SF(pow12, pow(y[i], x[i]), pow(y[i], x[j]), pow(y[i] , c));
BINARYOP_SF(pow21, pow(x[i], y[i]), pow(x[j], y[i]), pow(c, y[i]));

BINARYOP(sub12, y[i] - x[i], y[i] - x[j], y[i] - c);
BINARYOP(sub21, x[i] - y[i], x[j] - y[i], c - y[i]);


BINARYOP(div12, y[i] / x[i], y[i] / x[j], y[i] / c);
BINARYOP(div21, x[i] / y[i], x[j] / y[i], c / y[i]);

BINARYOP_SF(mod12, fmod(y[i] , x[i]), fmod(y[i] , x[j]), fmod(y[i], c));
BINARYOP_SF(mod21, fmod(x[i] , y[i]), fmod(x[j] , y[i]), fmod(c, y[i]));

BINARYOP(mult, y[i] *  x[i], y[i] *  x[j], y[i] *  c);
BINARYOP(add,  y[i] +  x[i], y[i] +  x[j], y[i] +  c);


//double logspace_add (double logx, double logy)
//{
//    return fmax2 (logx, logy) + log1p (exp (-fabs (logx - logy)));
//}
template <typename T>
__device__ T logspaceadd(T logx, T logy) {
	T M = ( ((logx) > (logy)) ? (logx) : (logy) );
	return M + log1p(exp(-fabs(logx-logy)));
}
template <>
__device__ int logspaceadd<int>(int logx, int logy){
	int M = ( ((logx) > (logy)) ? (logx) : (logy) );
	int D = (double)(logx-logy);
	return M + (int)log1p(exp(-fabs((double)D))) ;
}

BINARYOP(lgspadd, logspaceadd(y[i], x[i]), logspaceadd(y[i], x[j]), logspaceadd(y[i], c));


BINARYOP_COMPARE(eq,   y[i] == x[i], y[i] == x[j], y[i] == c);
BINARYOP_COMPARE(ne,   y[i] != x[i], y[i] != x[j], y[i] != c);

BINARYOP_COMPARE(gt12,   y[i] >  x[i], y[i] >  x[j], y[i] >  c);
BINARYOP_COMPARE(gt21,   x[i] >  y[i], x[j] >  y[i], y[i] <  c);

BINARYOP_COMPARE(lt12,   y[i] <  x[i], y[i] <  x[j], y[i] <  c);
BINARYOP_COMPARE(lt21,   x[i] <  y[i], x[j] <  y[i], y[i] >  c);

BINARYOP_COMPARE(gte12,  y[i] >= x[i], y[i] >= x[j], y[i] >= c);
BINARYOP_COMPARE(gte21,  x[i] >= y[i], x[j] >= y[i], y[i] <= c);

BINARYOP_COMPARE(lte12,  y[i] <=  x[i], y[i] <=  x[j], y[i] <=  c);
BINARYOP_COMPARE(lte21,  x[i] <=  y[i], x[j] <=  y[i], y[i] >=  c);

BINARYOP_LOGICAL(and,   y[i] && x[i], y[i] && x[j], y[i] && c);
BINARYOP_LOGICAL(or,   y[i] || x[i], y[i] || x[j], y[i] || c);

/*maybe sometime finish this so that the comparison can be returned as logicals on the cpu
 *
#define compOP(MNAME,MCFUN1, MCFUN2, MCFUN3) \
		__global__ void kernal_same_size_##MNAME (double* y, double* x,int* ret, int N, int operations_per_thread)\
		{\
			int id = blockDim.x * blockIdx.x + threadIdx.x;\
			int mystart = operations_per_thread * id;\
			int mystop = operations_per_thread + mystart;\
			for ( int i = mystart; i < mystop; i++) {\
				if (i < N) {\
					ret[i] = MCFUN1 ;\
				}\
			}\
		}\
		SEXP gpu_same_size_##MNAME (SEXP A_in, SEXP B_in, SEXP sn)\
		{\
			SEXP ret_final;\
			int n = INTEGER(sn)[0];\
			DECERROR1;\
			struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);\
			struct gpuvec *B = (struct gpuvec*) R_ExternalPtrAddr(B_in);\
			int *retgpu;\
			CUDA_MALLOC(retgpu,n * sizeof(int)) ;\
			GET_BLOCKS_PER_GRID(n);\
			kernal_same_size_##MNAME <<<blocksPerGrid, (threads_per_block[currentDevice])>>>(A->d_vec,B->d_vec,retgpu, n, operations_per_thread);\
			CUDA_CHECK_KERNAL_CLEAN_1(retgpu);\
			SEXP ret;\
			PROTECT(ret = allocVector(INTSXP, n));\
			double *h_vec = REAL(ret);\
		    cudaStat=cudaMemcpy(h_vec, retgpu, n * sizeof(int), cudaMemcpyDeviceToHost) ;\
		    if (cudaStat != cudaSuccess)\
		           warning("CUDA memory transfer error in 'gpu_get.'  (%s)\n",  cudaGetErrorString(cudaStat));\
		    UNPROTECT(1);\
			return ret;\
		}\
		__global__ void kernal_scaler_##MNAME (double* y, double* ret, double c, int N, int operations_per_thread)\
		{\
			int id = blockDim.x * blockIdx.x + threadIdx.x;\
			int mystart = operations_per_thread * id;\
			int mystop = operations_per_thread + mystart;\
			for ( int i = mystart; i < mystop; i++) {\
				if (i < N) {\
					ret[i] = MCFUN3 ;\
				}\
			}\
		}\
		SEXP gpu_scaler_##MNAME (SEXP A_in, SEXP B_in, SEXP sn)\
		{\
			SEXP ret_final;\
			int n = INTEGER(sn)[0];\
			DECERROR1;\
			struct gpuvec *ret = Calloc(1, struct gpuvec);\
			struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);\
			double B = REAL(B_in)[0];\
			CUDA_MALLOC(ret->d_vec,n * sizeof(double)) ;\
			GET_BLOCKS_PER_GRID(n);\
			kernal_scaler_##MNAME <<<blocksPerGrid, (threads_per_block[currentDevice])>>>(A->d_vec,ret->d_vec,B, n, operations_per_thread);\
			CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);\
			ret_final = gpu_register(ret);\
			return ret_final;\
		}\
		__global__ void kernal_diff_size_##MNAME (double* y, double* x, double* ret, int ny, int nx, int operations_per_thread)\
		{\
			int id = blockDim.x * blockIdx.x + threadIdx.x;\
			int mystart = operations_per_thread * id;\
			int mystop = operations_per_thread + mystart;\
			for ( int i = mystart; i < mystop; i++) {\
				int j = i % nx;\
				if (i < ny) {\
					ret[i] = MCFUN2 ;\
				}\
			}\
		}\
		SEXP gpu_diff_size_##MNAME(SEXP A_in, SEXP B_in, SEXP sna, SEXP snb)\
		{\
			SEXP ret_final;\
			int na = INTEGER(sna)[0];\
			int nb = INTEGER(snb)[0];\
			DECERROR1;\
			struct gpuvec *ret = Calloc(1, struct gpuvec);\
			struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);\
			struct gpuvec *B = (struct gpuvec*) R_ExternalPtrAddr(B_in);\
			CUDA_MALLOC(ret->d_vec,na * sizeof(double)) ;\
			GET_BLOCKS_PER_GRID(na);\
			kernal_diff_size_##MNAME <<<blocksPerGrid, (threads_per_block[currentDevice])>>>(A->d_vec,B->d_vec,ret->d_vec, na, nb, operations_per_thread);\
			CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);\
			ret_final = gpu_register(ret);\
			return ret_final;\
		}
*/









