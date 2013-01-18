
#include "gmatrix.h"

/*
template <typename T>
__device__ T R_NA3(void) {
	return CUDA_R_Na_float ;
}

template <>
__device__ double R_NA3<double>(void) {
	return CUDA_R_Na_double;
}

template <>
__device__ int R_NA3<int>(void) {
	return CUDA_R_Na_int;
}*/



template <typename T1, typename T2>
__global__ void kernal_convert(T1* y, T2* x, int ny, int operations_per_thread)
{
	int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;
	for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {
		T2 tmpx=x[i];
		T1 tmpy;
		if (i < ny) {
			if(IS_NAN(tmpx)) {
				if(IS_NA<T2>(&(tmpx)))
					MAKE_NA<T1>(&(tmpy));
				else
					tmpy=RET_NAN<T1>();
			} else
				tmpy=  (T1) tmpx;
			y[i]=tmpy;
		}
	}
}



SEXP gpu_convert(SEXP A_in, SEXP in_N, SEXP in_type, SEXP in_to_type )
{
	SEXP ptr;
	struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);
	struct gpuvec *my_gpuvec = Calloc(1, struct gpuvec);

	//int n = length(in_vec);
	int N = INTEGER(in_N)[0];

	DECERROR1;

	//check the from and to types
	int type = INTEGER(in_type)[0];
	if(type > 3)
		error("Incorrect type passed to '%s.'", __func__);

	int to_type = INTEGER(in_to_type)[0];
	int to_mysizeof;
	if(to_type==0)
		to_mysizeof=sizeof(double);
	else if(to_type==1)
		to_mysizeof=sizeof(float);
	else if(to_type==2 || to_type==3)
		to_mysizeof=sizeof(int);
	else
		error("Incorrect type passed to '%s.'", __func__);



//#ifdef DEBUG
//	Rprintf("length = %d\n", n);
//#endif
	CUDA_MALLOC( my_gpuvec->d_vec, N*to_mysizeof );
	GET_BLOCKS_PER_GRID(N);

	if(to_type==0) {
		#define KERNAL(PTR,T)\
		kernal_convert<double, T ><<<blocksPerGrid, (threads_per_block[currentDevice])>>>((double *) (my_gpuvec->d_vec), PTR(A), N, operations_per_thread);
		CALL_KERNAL;
		#undef KERNAL
	} else if(to_type==1) {
		#define KERNAL(PTR,T)\
		kernal_convert<float, T ><<<blocksPerGrid, (threads_per_block[currentDevice])>>>((float *) (my_gpuvec->d_vec), PTR(A), N, operations_per_thread);
		CALL_KERNAL;
		#undef KERNAL
	} else {
		#define KERNAL(PTR,T)\
		kernal_convert<int, T ><<<blocksPerGrid, (threads_per_block[currentDevice])>>>((int *) (my_gpuvec->d_vec), PTR(A), N, operations_per_thread);
		CALL_KERNAL;
		#undef KERNAL
	}
	CUDA_CHECK_KERNAL_CLEAN_1(my_gpuvec->d_vec);


	ptr = gpu_register(my_gpuvec);
	return(ptr);
}
