
#include "gmatrix.h"





static void gpu_finalizer(SEXP ext)
{
	struct gpuvec *gpu_ptr = (struct gpuvec*) R_ExternalPtrAddr(ext);
	cudaError_t  status1;
	if(gpu_ptr==NULL)
		return;

	#ifdef DEBUG
	Rprintf("before free... \n");
	#endif

	//status1=cudaFree(gpu_ptr->d_vec);
	//if (status1 != cudaSuccess ) {
	//	error("CUDA memory free error in 'gpu_finalizer.' (%d) \n", (int) status1);
    //          return;
    //}
	if(gpu_ptr->device!=currentDevice)
		cudaSetDevice(gpu_ptr->device);

	CUDA_CLEAN_1(gpu_ptr->d_vec)

	if(gpu_ptr->device!=currentDevice)
		cudaSetDevice(currentDevice);

	#ifdef DEBUG
    Rprintf("after free... \n");
	#endif

    Free(gpu_ptr);
}


SEXP gpu_register(struct gpuvec *in_vec)
{
	in_vec->device=currentDevice;

    SEXP ext = PROTECT(R_MakeExternalPtr(in_vec, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ext, gpu_finalizer, (Rboolean) 1);
    UNPROTECT(1);
#ifdef DEBUG
    Rprintf("gpu_register: in_vec.d_vec %p \n", in_vec->d_vec);
#endif
    return ext;
}
SEXP gpu_create(SEXP in_vec, SEXP in_type)
{
	SEXP ptr;
	struct gpuvec *my_gpuvec = Calloc(1, struct gpuvec);
	int n = length(in_vec);

	PROCESS_TYPE;
	DECERROR1;
#ifdef DEBUG
Rprintf("length = %d\n", n);
#endif

	CUDA_MALLOC( my_gpuvec->d_vec, n * mysizeof );
	if(type==0) {
		double *hd_vec = REAL(in_vec);
		CUDA_MEMCPY_CLEAN(my_gpuvec->d_vec, hd_vec, n *mysizeof, cudaMemcpyHostToDevice);
	} else if(type==1) {
		error("Cannot directly transfer type 'single.'");
	} else {
		int *hi_vec = INTEGER(in_vec);
		CUDA_MEMCPY_CLEAN(my_gpuvec->d_vec, hi_vec, n *mysizeof, cudaMemcpyHostToDevice);
	}

    ptr = gpu_register(my_gpuvec);
    return(ptr);
}



SEXP gpu_duplicate(SEXP A_in, SEXP sn, SEXP in_type)
{

	int n = INTEGER(sn)[0];
	struct gpuvec *my_gpuvec = Calloc(1, struct gpuvec);
	struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);
	DECERROR1;

	PROCESS_TYPE;

	CUDA_MALLOC( my_gpuvec->d_vec, n * mysizeof );
	CUDA_MEMCPY_CLEAN(my_gpuvec->d_vec, A->d_vec, n * mysizeof, cudaMemcpyDeviceToDevice);

    SEXP ptr = gpu_register(my_gpuvec);
    return(ptr);
}


SEXP gpu_get(SEXP ptr, SEXP sn, SEXP in_type)
{
	DECERROR0;
	struct gpuvec gpu_ptr = ((struct gpuvec*) R_ExternalPtrAddr(ptr))[0];
	int n = INTEGER(sn)[0];
	SEXP ret;


	int type = INTEGER(in_type)[0];\
	int mysizeof;
	if(type==0) {
		mysizeof=sizeof(double);
		PROTECT(ret = allocVector(REALSXP, n));
		double *d_vec = REAL(ret);
		cudaStat=cudaMemcpy(d_vec, (double *) gpu_ptr.d_vec, n * mysizeof, cudaMemcpyDeviceToHost) ; //don't use macro for transfer back to host
		if (cudaStat != cudaSuccess)
			warning("CUDA memory transfer error in 'gpu_get.'  (%s)\n",  cudaGetErrorString(cudaStat));

	} else if(type==1) {
		error("Cannot directly transfer type 'float.' Convert to double first.");
	} else if(type==2){
		mysizeof=sizeof(int);

		PROTECT(ret = allocVector(INTSXP, n));
		int *i_vec = INTEGER(ret);
		cudaStat=cudaMemcpy(i_vec, (int *) gpu_ptr.d_vec, n * mysizeof, cudaMemcpyDeviceToHost) ; //don't use macro for transfer back to host
		if (cudaStat != cudaSuccess)
			warning("CUDA memory transfer error in 'gpu_get.'  (%s)\n",  cudaGetErrorString(cudaStat));

	}  else if(type==3){
		mysizeof=sizeof(int);

		PROTECT(ret = allocVector(LGLSXP, n));
		int *i_vec = INTEGER(ret);
		cudaStat=cudaMemcpy(i_vec, (int *) gpu_ptr.d_vec, n * mysizeof, cudaMemcpyDeviceToHost) ; //don't use macro for transfer back to host
		if (cudaStat != cudaSuccess)
			warning("CUDA memory transfer error in 'gpu_get.'  (%s)\n",  cudaGetErrorString(cudaStat));

	} else
		error("Invalid type in 'gpu_get'.");

#ifdef DEBUG
    Rprintf("length = %d, pointer=%p \n", n * mysizeof, gpu_ptr.d_vec );
#endif

    UNPROTECT(1);
    return ret;
}


struct matrix get_matrix_struct(SEXP A_in) {
	struct matrix A;
	struct gpuvec *gpu_ptr = (struct gpuvec*) R_ExternalPtrAddr(GET_SLOT(A_in, install("ptr")));
	A.d_vec = gpu_ptr->d_vec;
	A.rows = INTEGER(GET_SLOT(A_in, install("nrow")))[0];
	A.cols = INTEGER(GET_SLOT(A_in, install("ncol")))[0];
	A.ld = A.rows;
	return A;
}

SEXP asSEXPint(int myint) { //wraps an integer as a sexp
	SEXP ans;
	PROTECT(ans = allocVector(INTSXP, 1));
	INTEGER(ans)[0] = myint;
	UNPROTECT(1);
	return ans;
}


char* getCublasErrorString(cublasStatus_t stat) {
	static char ret[100];
	char v1[]="Success";
	char v2[]="the library was not initialized";
	char v3[]="the resource allocation failed";
	char v4[]="an invalid numerical value was used as an argument";
	char v5[]="an absent device architectural feature is required";
	char v6[]="an access to GPU memnory space failed";
	char v7[]="the GPU program failed to execute";
	char v8[]="an internal operation failed";
	char v9[]="unknown";
	if(stat==CUBLAS_STATUS_SUCCESS)
		strcpy(ret, v1);
	else if(stat==CUBLAS_STATUS_NOT_INITIALIZED)
		strcpy(ret, v2);
	else if(stat==CUBLAS_STATUS_ALLOC_FAILED)
		strcpy(ret, v3);
	else if(stat==CUBLAS_STATUS_INVALID_VALUE)
		strcpy(ret, v4);
	else if(stat==CUBLAS_STATUS_ARCH_MISMATCH)
		strcpy(ret, v5);
	else if(stat==CUBLAS_STATUS_MAPPING_ERROR)
		strcpy(ret, v6);
	else if(stat==CUBLAS_STATUS_EXECUTION_FAILED)
		strcpy(ret, v7);
	else if(stat==CUBLAS_STATUS_INTERNAL_ERROR)
		strcpy(ret, v8);
	else
		strcpy(ret, v9);
	return(ret);
}


template <typename T>
__global__ void kernal_init_double(T* y, int ny, T setval, int operations_per_thread)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int mystart = operations_per_thread * id;
	int mystop = operations_per_thread + mystart;
	for ( int i = mystart; i < mystop; i++) {
		if (i < ny) {
			y[i] =  setval;
		}
	}

}

SEXP gpu_rep_1(SEXP in_val, SEXP in_N, SEXP in_type)
{
	SEXP ptr;
	struct gpuvec *my_gpuvec = Calloc(1, struct gpuvec);


	//int n = length(in_vec);
	int N = INTEGER(in_N)[0];

	DECERROR1;
	PROCESS_TYPE;
	//problem here in_type may be integer
	double val_double=0;
	float val_float=0;
	int val_int=0;
	if(type==0) {
		val_double= REAL(in_val)[0];
	} else if(type==1) {
		val_double= REAL(in_val)[0];
		val_float= (float) val_double;
	} else
		val_int= INTEGER(in_val)[0];
#ifdef DEBUG
	Rprintf("length = %d\n", n);
#endif
	CUDA_MALLOC( my_gpuvec->d_vec, N*mysizeof );
	GET_BLOCKS_PER_GRID(N);
#define KERNAL(PTR,T)\
	kernal_init_double< T ><<<blocksPerGrid, (threads_per_block[currentDevice])>>>(PTR(my_gpuvec), N, val_##T , operations_per_thread);
	CALL_KERNAL;
#undef KERNAL
	CUDA_CHECK_KERNAL_CLEAN_1(my_gpuvec->d_vec);


	ptr = gpu_register(my_gpuvec);
	return(ptr);
}

template <typename T>
__global__ void kernal_rep(T* y, int n, T* setval, int N, int times_each, int operations_per_thread)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int mystart = operations_per_thread * id;
	int mystop = operations_per_thread + mystart;

	for ( int i = mystart; i < mystop; i++) {
		if (i < N*n) {
		//	printf("i = %d, n = %d, N =%d, imodn=%d,  i / N=%d, time_each=%d \n",i,n,N,i % n,i / N, times_each);
			if(times_each==1) {//times
				y[i] =  setval[i % n];
			} else {
				y[i] =  setval[i / N];
			}
		}
	}
}

SEXP gpu_rep_m(SEXP in_A,SEXP in_n, SEXP in_N, SEXP in_times_each, SEXP in_type)
{
	SEXP ptr;

	struct gpuvec *ret = Calloc(1, struct gpuvec);
	//double *h_vec = REAL(in_vec);
	struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(in_A);

	int n = INTEGER(in_n)[0]; //length(in_A)
	int N = INTEGER(in_N)[0]; //times to replicate
	int times_each = INTEGER(in_times_each)[0];
	PROCESS_TYPE;
	DECERROR1;
//#ifdef DEBUG

//#endif
	int myn=n*N;
	//Rprintf("n = %d, N = %d, times_each = %d, myn=%d\n", n, N, times_each,myn);
	CUDA_MALLOC( ret->d_vec, myn*mysizeof );
	GET_BLOCKS_PER_GRID(myn);

	#define KERNAL(PTR,T)\
		kernal_rep< T ><<<blocksPerGrid, (threads_per_block[currentDevice])>>>(PTR(ret), n, PTR(A), N, times_each, operations_per_thread);
	CALL_KERNAL;
	#undef KERNAL

	CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);
	//error("stop\n");
	//Rprintf("hi\n");
	ptr = gpu_register(ret);
	return(ptr);
}


#define MMPTR_DBL(A) \
		(double *) A.d_vec

#define MMPTR_FLOAT(A) \
		(float *) A.d_vec


#define CALL_CUBLAS\
		if(type==0)\
			CUBLAS(MMPTR_DBL, D)\
		else \
			CUBLAS(MMPTR_FLOAT, S)\


SEXP matrix_multiply(SEXP A_in, SEXP B_in, SEXP transa, SEXP transb, SEXP in_type)
{
	SEXP ret_final;
	struct matrix ret;
	struct matrix A = get_matrix_struct(A_in);
	struct matrix B = get_matrix_struct(B_in);
	int colsOpA=0, rowsOpB=0;
	int TA = LOGICAL_VALUE(transa), TB = LOGICAL_VALUE(transb);
	cublasOperation_t TRANSA , TRANSB ;
	const int stride = 1;
	const double alphaD = 1.0;
	const double betaD = 0.0;
	const float alphaS = 1.0;
	const float betaS = 0.0;
	DECERROR1;
	cublasStatus_t cublasStatus;
	SEXP rownames,colnames, gpu_ptr;
	int type = INTEGER(in_type)[0];\
	if(type>2)
		error("Type must be 'double' or 'float.'");

	if(TA==0) {
		ret.rows=A.rows;
		colsOpA=A.cols;
		PROTECT(rownames= GET_SLOT(A_in, install("rownames")));
		TRANSA=CUBLAS_OP_N;
	}
	else {
		ret.rows=A.cols;
		colsOpA=A.rows;
		PROTECT(rownames= GET_SLOT(A_in, install("colnames")));
		TRANSA = CUBLAS_OP_T;
	}

	if(TB==0) {
		ret.cols=B.cols;
		rowsOpB=B.rows;
		PROTECT(colnames= GET_SLOT(B_in, install("colnames")));
		TRANSB = CUBLAS_OP_N;}
	else {
		ret.cols=B.rows;
		rowsOpB=B.cols;
		PROTECT(colnames= GET_SLOT(B_in, install("rownames")));
		TRANSB = CUBLAS_OP_T;}

	if(rowsOpB!= colsOpA) {
		error("Matrix dimensions do not match for matrix multiplication.\n");
	}
	ret.ld=ret.rows;
#ifdef DEBUG

    Rprintf("sutf: TRANSA = %d, \nTRANSB = %d, \nret.rows = %d, \nret.cols = %d,\ncolsOpA = %d,\n&alpha = NA,\nA.d_vec = %p, A.ld = %d,\nB.d_vec = %p, B.ld = %d,\n&beta = NA,\nret.d_vec = %p, ret.ld = %d\n",
    		TA,
    		TB,
			ret.rows,  //rows of matrix op(A)
			ret.cols,  //cols of matrix op(B)
			colsOpA,  //cols of matrix op(A)
			A.d_vec,A.ld,
			B.d_vec,B.ld,
			ret.d_vec,ret.ld);


#endif
    CUDA_MALLOC(ret.d_vec,ret.cols * ret.rows * sizeof(double));


	if(ret.rows==1) {
		if(ret.cols==1) {
		//	blasStatus_t cublasDdot (cublasHandle_t (handle[currentDevice]), int n,
		//			const double *x, int incx,
		//			const double *y, int incy,
		//			double *result)
		//	REprintf("colsOpA=%d , stride=%d \n",colsOpA,stride );

			//had to jerry-rig this a bit -- everything seemed to break if i changed the mode when the package loads
			cublasStatus = cublasSetPointerMode((handle[currentDevice]), CUBLAS_POINTER_MODE_DEVICE);
			if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
				cudaError_t status1=cudaFree(ret.d_vec);
		    	if (status1 != cudaSuccess ) {
		    		error("CUBLAS pointer mode could not be set and CUDA memory and free errors in 'matrix_multiply'\n");
		    	}
				error("CUBLAS pointer mode could not be set\n");
			}
			#define CUBLAS(PTR,MT) \
			cublasStatus= cublas##MT##dot((handle[currentDevice]), colsOpA,\
					PTR(A),1,\
					PTR(B),1,\
					PTR(ret));
			CALL_CUBLAS;
			#undef CUBLAS

			cublasStatus = cublasSetPointerMode((handle[currentDevice]), CUBLAS_POINTER_MODE_HOST);
			if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
				cudaError_t status1=cudaFree(ret.d_vec);
		    	if (status1 != cudaSuccess ) {
		    		error("CUBLAS pointer mode could not be reset and CUDA memory and free errors in 'matrix_multiply'\n");
		    	}
				error("CUBLAS pointer mode could not be set\n");
			}


		} else {
			if(TRANSB == CUBLAS_OP_T)
				TRANSB=CUBLAS_OP_N;
			else
				TRANSB=CUBLAS_OP_T;

			#define CUBLAS(PTR,MT) \
			cublasStatus=cublas##MT##gemv(\
					(handle[currentDevice]),\
					TRANSB, \
					B.rows,\
					B.cols,\
					&alpha##MT ,\
					PTR(B),B.ld,\
					PTR(A),stride,\
					&beta##MT ,\
					PTR(ret),stride);
			CALL_CUBLAS;
			#undef CUBLAS

		}
	} else {
		if(ret.cols==1) {
			//			cublasDgemv(cublasHandle_t (handle[currentDevice]), cublasOperation_t trans,
			//			int m, int n,
			//			const double *alpha,
			//			const double *A, int lda,
			//			const double *x, int incx,
			//			const double *beta,
			//			double *y, int incy)
			#define CUBLAS(PTR,MT) \
			cublasStatus=cublas##MT##gemv((handle[currentDevice]),\
					TRANSA, \
					A.rows,\
					A.cols,\
					&alpha##MT ,\
					PTR(A),A.ld,\
					PTR(B),stride,\
					&beta##MT ,\
					PTR(ret),stride);
			CALL_CUBLAS;
			#undef CUBLAS

		} else {
			if(colsOpA==1) {
				//kernal_init_double(double* y, int ny, double setval, int operations_per_thread)
				int myn=ret.rows*ret.cols;
				GET_BLOCKS_PER_GRID(myn);
				struct matrix *retptr =&ret;
				//#define val_double 0.0;
				//#define val_float 0.0;
				//#define val_int 0;
				#define KERNAL(PTR,T)\
				 	 kernal_init_double< T ><<<blocksPerGrid, (threads_per_block[currentDevice])>>>(PTR(retptr), myn, 0.0 , operations_per_thread);
				CALL_KERNAL_SF;
				#undef KERNAL

				CUDA_CHECK_KERNAL_CLEAN_1(ret.d_vec);


		//		cublasStatus_t cublasDger (cublasHandle_t (handle[currentDevice]), int m, int n,
		//		  const double *alpha,
		//		  const double *x, int incx,
		//		  const double *y, int incy,
		//		  double *A, int lda)
				#define CUBLAS(PTR,MT) \
				cublasStatus=cublas##MT##ger( (handle[currentDevice]),ret.rows, ret.cols,\
						&alpha##MT ,\
						PTR(A), stride,\
						PTR(B), stride,\
						PTR(ret), ret.ld);
				CALL_CUBLAS;
				#undef CUBLAS

			} else { // finally the general Dgemm
				/*cublasStatus=cublasDgemm((handle[currentDevice]),\
						TRANSA, //transpose\
						TRANSB, //transpose\
						ret.rows,  //rows of matrix op(A)\
						ret.cols,  //cols of matrix op(B)\
						colsOpA,  //cols of matrix op(A)\
						&alpha,\
						PTR(A), A.ld,\
						PTR(B), B.ld,\
						&beta,\
						PTR(ret), ret.ld);*/
				#define CUBLAS(PTR,MT) \
				cublasStatus=cublas##MT##gemm((handle[currentDevice]),\
						TRANSA, \
						TRANSB, \
						ret.rows,  \
						ret.cols, \
						colsOpA,  \
						&alpha##MT ,\
						PTR(A), A.ld,\
						PTR(B), B.ld,\
						&beta##MT ,\
						PTR(ret), ret.ld);
				CALL_CUBLAS;
				#undef CUBLAS
			}
		}
	}


    if (cublasStatus != CUBLAS_STATUS_SUCCESS ) {
    	cudaError_t status1=cudaFree(ret.d_vec);
    	if (status1 != cudaSuccess ) {
    		error("cublas error from 'matrix_multiply.' (%s)'\n Also memory free error (potential memory leak).", getCublasErrorString(cublasStatus));
    	}
    	error("cublas error from 'matrix_multiply.' (%s)'\n", getCublasErrorString(cublasStatus));
    }

	CUDA_CHECK_KERNAL_CLEAN_1(ret.d_vec) ;

    struct gpuvec *my_gpuvec = Calloc(1, struct gpuvec);
    my_gpuvec->d_vec= ret.d_vec;
    PROTECT(gpu_ptr = gpu_register(my_gpuvec));
    PROTECT(ret_final = NEW_OBJECT(MAKE_CLASS("gmatrix")));
    SET_SLOT(ret_final, install("nrow"), PROTECT(asSEXPint(ret.rows)));
    SET_SLOT(ret_final, install("ncol"), PROTECT(asSEXPint(ret.cols)));
    SET_SLOT(ret_final, install("rownames"), rownames);
    SET_SLOT(ret_final, install("colnames"), colnames);
    SET_SLOT(ret_final, install("ptr"), gpu_ptr);
    SET_SLOT(ret_final, install("type"), PROTECT(asSEXPint(type)));
    SET_SLOT(ret_final, install("device"), PROTECT(asSEXPint(currentDevice)));
    UNPROTECT(8L);


    return ret_final;
}



/***********************************************************/
template <typename T>
__global__ void kernal_outer(T* x, T* y, T* ret, int n_x,  int op, int N, int operations_per_thread)
{
	int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;
	for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {
		if (i < N) {
			int col = i / n_x;
			int row = i-n_x*col;
			if(op==1)
				ret[i] = x[row] * y[col];
			else if(op==2)
				ret[i] = x[row] + y[col];
			else if(op==3)
				ret[i] =x[row] - y[col];
			else if(op==4)
				ret[i] =y[row] - x[col];
			else if(op==5)
				ret[i] =x[row] / y[col];
			else if(op==6)
				ret[i] =y[row] / x[col];
			else if(op==7)
				ret[i] = pow(x[row] , y[col]) ;
			else if(op==8)
				ret[i] = pow(y[row] , x[col]) ;

		}
	}
}
SEXP gpu_outer(SEXP A_in, SEXP B_in,SEXP n_A_in, SEXP n_B_in, SEXP op_in, SEXP in_type)
{
	SEXP ret_final;
	int n_A = INTEGER(n_A_in)[0];
	int n_B = INTEGER(n_B_in)[0];
	int n = n_A*n_B;
	int op = INTEGER(op_in)[0];

	DECERROR1;
    struct gpuvec *ret = Calloc(1, struct gpuvec);
	struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);
	struct gpuvec *B = (struct gpuvec*) R_ExternalPtrAddr(B_in);

	//Rprintf("n=%d, nA = %d, n_b=%d", n,n_A,n_B);
	//allocate
	PROCESS_TYPE_SF; //only works for float or double (power is the problem)
	CUDA_MALLOC(ret->d_vec,n * mysizeof) ;

	GET_BLOCKS_PER_GRID(n);
//(double* x, double* y, double* ret, int n_x,  int op, int N, int operations_per_thread)
	#define KERNAL(PTR, T) \
		kernal_outer< T ><<<blocksPerGrid, (threads_per_block[currentDevice])>>>( PTR(A), PTR(B),PTR(ret),\
			n_A, op, n, operations_per_thread);
	CALL_KERNAL_SF;
	#undef KERNAL


	 CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec) ;


    ret_final = gpu_register(ret);
    return ret_final;
}



/***********************************************************/

template <typename T>
__global__ void kernal_kronecker(T* x, T* y, T* ret, int n_row_x,int n_col_x,  int n_row_y, int n_col_y, int N, int operations_per_thread)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int mystart = operations_per_thread * id;
	int mystop = operations_per_thread + mystart;
	int n_row_ret=n_row_x*n_row_y;
	//int n_col_ret=n_col_x*n_col_y;

	for ( int i = mystart; i < mystop; i++) {
		if (i < N) {
			int col = i / n_row_ret;
			int row = i-n_row_ret*col;
			int col_x = col / n_col_y;
			int row_x = row / n_row_y;
			int col_y = col - col_x * n_col_y;
			int row_y = row  - row_x * n_row_y;
			ret[i]=x[IDX2(row_x ,col_x ,n_row_x)]*y[IDX2(row_y ,col_y ,n_row_y)];
		}
	}
}

/*
__global__ void kernal_kronecker(double* x, double* y, double* ret, int n_row_x,int n_col_x,  int n_row_y, int n_col_y, int N, int operations_per_thread)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int mystart = operations_per_thread * id;
	int mystop = operations_per_thread + mystart;
	int n_row_ret=n_row_x*n_row_y;
	//int n_col_ret=n_col_x*n_col_y;

	for ( int i = mystart; i < mystop; i++) {
		if (i < N) {
			int col = i / n_row_ret;
			int row = i-n_row_ret*col;
			int col_x = col / n_col_y;
			int row_x = row / n_row_y;
			int col_y = col - col_x * n_col_y;
			int row_y = row  - row_x * n_row_y;
			ret[i]=x[IDX2(row_x ,col_x ,n_row_x)]*y[IDX2(row_y ,col_y ,n_row_y)];
		}
	}
}*/
SEXP gpu_kronecker(SEXP A_in, SEXP B_in,SEXP n_A_row_in,SEXP n_A_col_in, SEXP n_B_row_in,SEXP n_B_col_in, SEXP in_type)
{
	SEXP ret_final;
	int n_A_row = INTEGER(n_A_row_in)[0];
	int n_A_col = INTEGER(n_A_col_in)[0];
	int n_B_row = INTEGER(n_B_row_in)[0];
	int n_B_col = INTEGER(n_B_col_in)[0];

	int n = n_A_row*n_B_row*n_A_col*n_B_col;

	DECERROR1;
    struct gpuvec *ret = Calloc(1, struct gpuvec);
	struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);
	struct gpuvec *B = (struct gpuvec*) R_ExternalPtrAddr(B_in);

	//Rprintf("n=%d, nA = %d, n_b=%d", n,n_A,n_B);
	//allocate
	PROCESS_TYPE;
	CUDA_MALLOC(ret->d_vec,n * mysizeof) ;


	GET_BLOCKS_PER_GRID(n);
//kernal_kronecker(double* x, double* y, double* ret, int n_row_x, int n_col_x,  int n_row_y, int n_col_y, int N, int operations_per_thread)
	#define KERNAL(PTR,T)\
	kernal_kronecker< T ><<<blocksPerGrid, (threads_per_block[currentDevice])>>>(PTR(A),PTR(B),PTR(ret),\
			n_A_row, n_A_col,n_B_row,n_B_col, n, operations_per_thread);
	CALL_KERNAL;
	#undef KERNAL

	 CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec) ;


    ret_final = gpu_register(ret);
    return ret_final;
}



/***********************************************************/


template <typename T>
__global__ void kernal_sumby(T* x, T* ret, int n_x, int n_ret, int* k_start_vec,  int* k_stop_vec, int operations_per_thread)
{
	double tmp_sum;
	int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;
	for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {
		if (i < n_ret) {
			int k_start= k_start_vec[i]-1;
			int k_stop = k_stop_vec[i];
			tmp_sum=0;
			//printf("i=%d, k_start = %d, k_stop = %d , \n", i, k_start, k_stop);
			for(int k =k_start; k<k_stop; k++)
				if(0<=k && k<n_x)
					tmp_sum+=x[k];
			ret[i]=tmp_sum;
		}
	}
}

SEXP gpu_kernal_sumby(SEXP A_in, SEXP index1_in,SEXP index2_in,SEXP n_A_in,SEXP n_index_in, SEXP in_type)
{
	SEXP ret_final;
	int n_A = INTEGER(n_A_in)[0];
	int n_index = INTEGER(n_index_in)[0];


	DECERROR3;
    struct gpuvec *ret = Calloc(1, struct gpuvec);
	struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);
	int *index1 = INTEGER(index1_in);
	int *index2 = INTEGER(index2_in);
	int *d_index1;
	int *d_index2;


	//allocate index
	PROCESS_TYPE;
	CUDA_MALLOC(d_index1, n_index * sizeof(int)) ;
	CUDA_MEMCPY_CLEAN(d_index1, index1, n_index * sizeof(int), cudaMemcpyHostToDevice);


	CUDA_MALLOC_CLEAN_1(d_index2, n_index * sizeof(int),d_index1);
	CUDA_MEMCPY_CLEAN_1(d_index2, index2, n_index * sizeof(int), cudaMemcpyHostToDevice, d_index1);

	CUDA_MALLOC_CLEAN_2(ret->d_vec,  n_index*mysizeof, d_index1, d_index2);

	GET_BLOCKS_PER_GRID(n_index);
//kernal_sumby(double* x, double* ret, int n_x, int n_ret, int* k_start_vec,  int* k_stop_vec, int operations_per_thread)
	#define KERNAL(PTR,T)\
	kernal_sumby< T ><<<blocksPerGrid, (threads_per_block[currentDevice])>>> (PTR(A),PTR(ret),\
			n_A, n_index, d_index1, d_index2, operations_per_thread);
	CALL_KERNAL;
	#undef KERNAL

	CUDA_CHECK_KERNAL_CLEAN_3(ret->d_vec,d_index1,d_index2) ;



	status1=cudaFree(d_index1);
	status2=cudaFree(d_index2);
	if (status1 != cudaSuccess || status2 != cudaSuccess) {
		error("CUDA memory free error in 'gpu_gmatrix_index_both'\n");
	}
    ret_final = gpu_register(ret);
    return ret_final;
}



/*********************************************************************/

template <typename T>
__global__ void kernal_mat_times_diag_vec(T* x, T *y, T* ret, int n_row_x, int n, int operations_per_thread)
{
	int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;
	for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {
		if (i < n) {
			int col = i / n_row_x;
			ret[i] = x[i] * y[col] ;
		}
	}

}
SEXP gpu_mat_times_diag_vec(SEXP A_in, SEXP B_in, SEXP n_row_in, SEXP n_col_in, SEXP in_type)
{
	SEXP ret_final;
	int n_row = INTEGER(n_row_in)[0];
	int n_col = INTEGER(n_col_in)[0];
	int n=n_row*n_col;
	DECERROR1;
    struct gpuvec *ret = Calloc(1, struct gpuvec);
	struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);
	struct gpuvec *B = (struct gpuvec*) R_ExternalPtrAddr(B_in);

	//allocate
	PROCESS_TYPE;
	CUDA_MALLOC(ret->d_vec,n * mysizeof) ;

	GET_BLOCKS_PER_GRID(n);

	//kernal_mat_times_diag_vec(double* x, double* y, double* ret, int n_row_x, int n, int operations_per_thread)
	#define KERNAL(PTR,T)\
	kernal_mat_times_diag_vec< T ><<<blocksPerGrid, (threads_per_block[currentDevice])>>>(PTR(A),PTR(B),\
			PTR(ret),n_row, n, operations_per_thread);
	CALL_KERNAL;
	#undef KERNAL

	CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec) ;

    ret_final = gpu_register(ret);
    return ret_final;
}


/*********************************************************************/


SEXP gpu_sum(SEXP A_in, SEXP n_in, SEXP in_type)
{
	SEXP ret_final;
	int n = INTEGER(n_in)[0];

	//cublasStatus_t cublasStatus;
	struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);
	int type = INTEGER(in_type)[0];\

	if(type==0) {
		PROTECT(ret_final = allocVector(REALSXP, 1));
		double* ret = REAL(ret_final);
		thrust::device_ptr<double> dev_ptr( (double *) A->d_vec);
		ret[0]= thrust::reduce(dev_ptr, dev_ptr + n, (double) 0, thrust::plus<double>());
	} else if(type==1){
		PROTECT(ret_final = allocVector(REALSXP, 1));
		double* ret = REAL(ret_final);
		float rettmp;
		thrust::device_ptr<float> dev_ptr((float *)A->d_vec);
		rettmp=thrust::reduce(dev_ptr, dev_ptr + n, (float) 0, thrust::plus<float>());
		ret[0]= (double) rettmp;
	} else {
		PROTECT(ret_final = allocVector(REALSXP, 1));
		double* ret = REAL(ret_final);
		int rettmp;
		thrust::device_ptr<int> dev_ptr((int *) A->d_vec);
		rettmp=thrust::reduce(dev_ptr, dev_ptr + n, (int) 0, thrust::plus<int>());
		ret[0]= (int) rettmp;
	}

	UNPROTECT(1);

	return ret_final;
}



SEXP gpu_min(SEXP A_in, SEXP n_in, SEXP in_type)
{
	SEXP ret_final;
	int n = INTEGER(n_in)[0];

	//cublasStatus_t cublasStatus;
	struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);
	int type = INTEGER(in_type)[0];\

	if(type==0) {
		PROTECT(ret_final = allocVector(REALSXP, 1));
		double* ret = REAL(ret_final);
		thrust::device_ptr<double> dev_ptr( (double *) A->d_vec);
		ret[0]= thrust::reduce(dev_ptr, dev_ptr + n, HUGE_VAL, thrust::minimum<double>());
	} else if(type==1){
		PROTECT(ret_final = allocVector(REALSXP, 1));
		double* ret = REAL(ret_final);
		float rettmp;
		thrust::device_ptr<float> dev_ptr((float *)A->d_vec);
		rettmp=thrust::reduce(dev_ptr, dev_ptr + n,  HUGE_VALF, thrust::minimum<float>());
		ret[0]= (double) rettmp;
	} else {
		PROTECT(ret_final = allocVector(REALSXP, 1));
		double* ret = REAL(ret_final);
		int rettmp;
		thrust::device_ptr<int> dev_ptr((int *) A->d_vec);
		rettmp=thrust::reduce(dev_ptr, dev_ptr + n, INT_MAX, thrust::minimum<int>());
		ret[0]= (int) rettmp;
	}

	UNPROTECT(1);

	return ret_final;
}


SEXP gpu_max(SEXP A_in, SEXP n_in, SEXP in_type)
{
	SEXP ret_final;
	int n = INTEGER(n_in)[0];

	//cublasStatus_t cublasStatus;
	struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);
	int type = INTEGER(in_type)[0];\

	if(type==0) {
		PROTECT(ret_final = allocVector(REALSXP, 1));
		double* ret = REAL(ret_final);
		thrust::device_ptr<double> dev_ptr( (double *) A->d_vec);
		ret[0]= thrust::reduce(dev_ptr, dev_ptr + n,  -HUGE_VAL, thrust::maximum<double>());
	} else if(type==1){
		PROTECT(ret_final = allocVector(REALSXP, 1));
		double* ret = REAL(ret_final);
		float rettmp;
		thrust::device_ptr<float> dev_ptr((float *)A->d_vec);
		rettmp=thrust::reduce(dev_ptr, dev_ptr + n,  -HUGE_VALF, thrust::maximum<float>());
		ret[0]= (double) rettmp;
	} else {
		PROTECT(ret_final = allocVector(REALSXP, 1));
		double* ret = REAL(ret_final);
		int rettmp;
		thrust::device_ptr<int> dev_ptr((int *) A->d_vec);
		rettmp=thrust::reduce(dev_ptr, dev_ptr + n, INT_MIN, thrust::maximum<int>());
		ret[0]= (int) rettmp;
	}

	UNPROTECT(1);

	return ret_final;
}



#define MYSORT(T)\
		thrust::device_ptr< T > dev_ptr(( T *)A->d_vec);\
		if(stable==0) {\
			if(decreasing==0)\
				thrust::sort(dev_ptr, dev_ptr + n);\
			else\
				thrust::sort(dev_ptr, dev_ptr + n,  thrust::greater< T >());\
		} else {\
			if(decreasing==0)\
				thrust::stable_sort(dev_ptr, dev_ptr + n);\
			else\
				thrust::stable_sort(dev_ptr, dev_ptr + n,  thrust::greater< T >());\
		}


SEXP gpu_sort(SEXP A_in, SEXP n_in, SEXP stable_in, SEXP decreasing_in, SEXP in_type)
{

	int n = INTEGER(n_in)[0];
	int stable=INTEGER(n_in)[0];
	struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);
	int type = INTEGER(stable_in)[0];
	int decreasing = INTEGER(decreasing_in)[0];

	if(type==0) {
		MYSORT(double);
	} else if(type==1){
		MYSORT(float);
	} else {
		MYSORT(int);
	}
	return asSEXPint(1L);
}

//order
#define MYORDER(T)\
		thrust::device_ptr<T> dev_ptr((T *)A->d_vec);\
		if(stable==0) {\
			if(decreasing==0)\
				thrust::sort_by_key(dev_ptr, dev_ptr + n, ret_ptr);\
			else\
				thrust::sort_by_key(dev_ptr, dev_ptr + n, ret_ptr, thrust::greater<T>());\
		} else {\
			if(decreasing==0)\
				thrust::stable_sort_by_key(dev_ptr, dev_ptr + n, ret_ptr);\
			else\
				thrust::stable_sort_by_key(dev_ptr, dev_ptr + n, ret_ptr, thrust::greater<T>());\
		}


SEXP gpu_order(SEXP A_in, SEXP n_in, SEXP stable_in, SEXP decreasing_in, SEXP in_type)
{

	int n = INTEGER(n_in)[0];
	int stable=INTEGER(stable_in)[0];
	struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);
	int type = INTEGER(in_type)[0];
	int decreasing = INTEGER(decreasing_in)[0];
	struct gpuvec *ret = Calloc(1, struct gpuvec);
	DECERROR0;
	CUDA_MALLOC(ret->d_vec, n * sizeof(int));
	thrust::device_ptr<int> ret_ptr((int *)ret->d_vec);
	thrust::sequence(ret_ptr, ret_ptr+n, 1, 1);

	if(type==0) {
		MYORDER(double);
	} else if(type==1){
		MYORDER(float);
	} else {
		MYORDER(int);
	}

	SEXP ret_final;
    PROTECT(ret_final = NEW_OBJECT(MAKE_CLASS("gvector")));
    SET_SLOT(ret_final, install("length"), PROTECT(asSEXPint(n)));
    SET_SLOT(ret_final, install("ptr"),   PROTECT(gpu_register(ret)));
    SET_SLOT(ret_final, install("type"), PROTECT(asSEXPint(2L)));
    SET_SLOT(ret_final, install("device"), PROTECT(asSEXPint(currentDevice)));
    UNPROTECT(5L);

	return ret_final;
}

//which

struct is_true
{
  __host__ __device__
  bool operator()(const int &x)
  {
    return x != 0;
  }
};


SEXP gpu_which(SEXP A_in, SEXP n_in)
{

	int n = INTEGER(n_in)[0];
	int stable=INTEGER(n_in)[0];
	struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);
	struct gpuvec *ret = Calloc(1, struct gpuvec);

	DECERROR0;


	thrust::device_ptr<int> sten((int *)A->d_vec);
	int tot=thrust::reduce(sten, sten + n, (int) 0, thrust::plus<int>());

	thrust::counting_iterator<int> index(1); // create implicit range [0, 1, 2, ...)counting_iterator<int> indices_end(vec.size());
//	counting_iterator<int> indices_end(n);


	CUDA_MALLOC(ret->d_vec, tot* sizeof(int));
	thrust::device_ptr<int> ret_ptr((int *)ret->d_vec);

	thrust::copy_if(index, index + n, sten, ret_ptr, is_true());
	//thrust::copy_if(make_zip_iterator(make_tuple(dev_ptr, indices_begin)),
	//		make_zip_iterator(make_tuple(dev_ptr + n, indices_end)),
	//		make_zip_iterator(make_tuple(junk, ret_ptr)), is_true());
    //struct gpuvec *my_gpuvec = Calloc(1, struct gpuvec);
    //my_gpuvec->d_vec= ret.d_vec;
	//SEXP type= PROTECT(asSEXPint(4L));
	SEXP ret_final;
    PROTECT(ret_final = NEW_OBJECT(MAKE_CLASS("gvector")));
    SET_SLOT(ret_final, install("length"), PROTECT(asSEXPint(tot)));
    SET_SLOT(ret_final, install("ptr"),   PROTECT(gpu_register(ret)));
    SET_SLOT(ret_final, install("type"), PROTECT(asSEXPint(2L)));
    SET_SLOT(ret_final, install("device"), PROTECT(asSEXPint(currentDevice)));
    UNPROTECT(5L);
	return 	ret_final;
}

#define MYSEQ(T)\
		thrust::device_ptr<T> ret_ptr((T *)ret->d_vec);\
		thrust::sequence(ret_ptr, ret_ptr + n, init, step);\

SEXP gpu_seq( SEXP n_in, SEXP init_in, SEXP step_in, SEXP in_type  )
{

	int n = INTEGER(n_in)[0];
//	int type = INTEGER(stable_in)[0];
	struct gpuvec *ret = Calloc(1, struct gpuvec);
	PROCESS_TYPE;
	DECERROR0;
	CUDA_MALLOC(ret->d_vec, n * mysizeof) ;

	if(type==0) {
		double init = REAL(init_in)[0];
		double step = REAL(step_in)[0];
		MYSEQ(double);
	} else if(type==1){
		float init = (float) REAL(init_in)[0];
		float step = (float) REAL(step_in)[0];
		MYSEQ(float);
	} else if(type==2){
		int init = INTEGER(init_in)[0];
		int step = INTEGER(step_in)[0];
		MYSEQ(int);
	} else
		error("Incorrect type.");
	SEXP ret_final;
    PROTECT(ret_final = NEW_OBJECT(MAKE_CLASS("gvector")));
    SET_SLOT(ret_final, install("length"), PROTECT(asSEXPint(n)));
    SET_SLOT(ret_final, install("ptr"),   PROTECT(gpu_register(ret)));
    SET_SLOT(ret_final, install("type"), PROTECT(asSEXPint(type)));
    SET_SLOT(ret_final, install("device"), PROTECT(asSEXPint(currentDevice)));
    UNPROTECT(5L);

	return 	ret_final;
}

template <typename T>
	__global__ void kernal_if(int *h, T* y, T* x, T* ret, int nh, int ny, int nx, int operations_per_thread)
	{
		int id = blockDim.x * blockIdx.x + threadIdx.x;
		int mystart = operations_per_thread * id;
		int mystop = operations_per_thread + mystart;
		for ( int i = mystart; i < mystop; i++) {
			if (i < nh) {
				ret[i] = h[i]  ? y[i % ny] : x[i % nx] ;
			}
		}
	}

SEXP gpu_if(SEXP H_in, SEXP A_in, SEXP B_in,SEXP snh, SEXP sna, SEXP snb, SEXP in_type)
{
	SEXP ret_final;
	int nh = INTEGER(snh)[0];
	int na = INTEGER(sna)[0];
	int nb = INTEGER(snb)[0];
	DECERROR1;
	struct gpuvec *ret = Calloc(1, struct gpuvec);
	struct gpuvec *H = (struct gpuvec*) R_ExternalPtrAddr(H_in);
	struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);
	struct gpuvec *B = (struct gpuvec*) R_ExternalPtrAddr(B_in);
	PROCESS_TYPE;
	CUDA_MALLOC(ret->d_vec,nh * mysizeof);
	GET_BLOCKS_PER_GRID(na);

	if(type==0)
		kernal_if<double> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((int *) H->d_vec, (double *) A->d_vec, (double *) B->d_vec,(double *) ret->d_vec, nh, na, nb, operations_per_thread);
	else if(type==1)
		kernal_if<float> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((int *) H->d_vec,(float *) A->d_vec, (float *) B->d_vec, (float *) ret->d_vec,nh, na, nb, operations_per_thread);
	else
		kernal_if<int> <<<blocksPerGrid, (threads_per_block[currentDevice])>>>((int *) H->d_vec,(int *) A->d_vec, (int *) B->d_vec, (int *) ret->d_vec,nh, na, nb, operations_per_thread);

	CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);
	ret_final = gpu_register(ret);
	return ret_final;
}

//rowLogSums
template <typename T>
__global__ void kernal_rowLogSums(T* P, T* ret, int rows, int cols, int operations_per_thread)
{
	int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;
	int j;
	T M, s;
	for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {
		if (i < rows) {
			M=log(0.0);
			for(j=0;j<cols;j++) {
				if(P[j*rows+i]>M)
					M=P[j*rows+i];
			}

			//calculate the cumulative sum without overflow
			s=exp(P[i]-M);
			for(j=1;j<cols;j++) {
				s=s + exp(P[j*rows+i]-M);
			}
			ret[i]=log(s)+M;
		}
	}

}
SEXP gpu_rowLogSums(SEXP in_P, SEXP in_rows, SEXP in_cols, SEXP in_type)
{

	struct gpuvec *ret = Calloc(1, struct gpuvec);
	struct gpuvec *P = (struct gpuvec*) R_ExternalPtrAddr(in_P);
	int rows = INTEGER(in_rows)[0];
	int cols = INTEGER(in_cols)[0];

	DECERROR1;
	//allocate
	PROCESS_TYPE;
	CUDA_MALLOC(ret->d_vec,rows * mysizeof) ;

	GET_BLOCKS_PER_GRID(rows);

	//kernal_mat_times_diag_vec(double* x, double* y, double* ret, int n_row_x, int n, int operations_per_thread)
	#define KERNAL(PTR,T)\
	kernal_rowLogSums< T ><<<blocksPerGrid, (threads_per_block[currentDevice])>>>(PTR(P),PTR(ret),\
			rows, cols, operations_per_thread);
	CALL_KERNAL_SF;
	#undef KERNAL

	CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec) ;

    SEXP ret_final = gpu_register(ret);
    return ret_final;
}

/*
SEXP gpu_max_pos(SEXP A_in, SEXP n_in)
{
	SEXP ret_final;
	int n = INTEGER(n_in)[0];


//	cublasStatus_t cublasStatus;


	struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);
	PROTECT(ret_final = allocVector(INTSXP, 1));
	int* ret = INTEGER(ret_final);
	thrust::device_ptr<double> dev_ptr(A->d_vec);
	ret[0]= thrust::reduce(dev_ptr, dev_ptr + n, (double) 0, thrust::maximum<double>());
	UNPROTECT(1);

	return ret_final;
}

SEXP gpu_min_pos(SEXP A_in, SEXP n_in)
{
	SEXP ret_final;
	int n = INTEGER(n_in)[0];


	//cublasStatus_t cublasStatus;


	struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);
	PROTECT(ret_final = allocVector(INTSXP, 1));
	int* ret = INTEGER(ret_final);
	thrust::device_ptr<double> dev_ptr(A->d_vec);
	ret[0]= thrust::reduce(dev_ptr, dev_ptr + n, (double) 0,thrust::minimum<double>());
	UNPROTECT(1);
	UNPROTECT(1);

	return ret_final;
}
*/
