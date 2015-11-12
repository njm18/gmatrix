
#include "gmatrix.h"



struct gptr {
	 void *d_vec;
};


struct gvec {
	 void *d_vec;
	 int length;
	 int type;
	 int device;
};

struct gmat {
   void *d_vec;
   int nrow;
   int ncol;
   int type;
   int device;
};



struct gmat get_gmat_struct(SEXP A_in) {
	struct gmat A;
	struct gptr *gpu_ptr = (struct gptr*) R_ExternalPtrAddr(GET_SLOT(A_in, install("ptr")));
	A.d_vec = gpu_ptr->d_vec;
	A.nrow = INTEGER(GET_SLOT(A_in, install("nrow")))[0];
	A.ncol = INTEGER(GET_SLOT(A_in, install("ncol")))[0];
	A.type = INTEGER(GET_SLOT(A_in, install("type")))[0];
	A.device = INTEGER(GET_SLOT(A_in, install("device")))[0];
	return A;
}

struct gvec get_gvec_struct(SEXP A_in) {
	struct gvec A;
	struct gptr *gpu_ptr = (struct gptr*) R_ExternalPtrAddr(GET_SLOT(A_in, install("ptr")));
	A.d_vec = gpu_ptr->d_vec;
	A.length = INTEGER(GET_SLOT(A_in, install("length")))[0];
	A.type = INTEGER(GET_SLOT(A_in, install("type")))[0];
	A.device = INTEGER(GET_SLOT(A_in, install("device")))[0];
	return A;
}


void check_error(cusolverStatus_t s) {
	if(s!=CUSOLVER_STATUS_SUCCESS) {
		if(s==CUSOLVER_STATUS_NOT_INITIALIZED)
			error("CUSOLVER not initialized.");
		else if(s==CUSOLVER_STATUS_ALLOC_FAILED)
			error("CUSOLVER allocation failed.");
		else if(s==CUSOLVER_STATUS_INVALID_VALUE)
			error("CUSOLVER invalid value.");
		else if(s==CUSOLVER_STATUS_ARCH_MISMATCH)
			error("CUSOLVER architectural mismatch.");
		else if(s==CUSOLVER_STATUS_EXECUTION_FAILED)
			error("CUSOLVER execution failed.");
		else if(s==CUSOLVER_STATUS_INTERNAL_ERROR)
			error("CUSOLVER internal error.");
		else if(s==CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED)
			error("CUSOLVER matrix type not supported.");
		else
			error("CUSOLVER unknown error.");
	}
}

#define SET_UP_DEVINFO\
	int *devInfo = NULL;\
	int devInfo_h = 0;\
	CUDA_MALLOC(devInfo,sizeof(int));\
	CUDA_MEMCPY_CLEAN(devInfo, &devInfo_h, sizeof(int),cudaMemcpyHostToDevice);


#define CHECK_DEVINFO_CLEAN_2(MCLEANPTR1, MCLEANPTR2)  \
		status1 = cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);\
		if(devInfo_h !=0L || status1 != cudaSuccess) {\
			if(status1 != cudaSuccess)\
			    Rprintf("Transfer not successful.");\
			status1=cudaFree(MCLEANPTR1);\
			if (status1 != cudaSuccess ) {\
				error("devInfo_h error and memory free errors (potential memory leak) in '%s.' (%s)\n", __func__, cudaGetErrorString(status1));\
			}\
			status1=cudaFree(MCLEANPTR2);\
			if (status1 != cudaSuccess ) {\
				error("devInfo_h error and memory free errors (potential memory leak) in '%s.' (%s)\n", __func__, cudaGetErrorString(status1));\
			}\
			error("Bad devInfo = %d .\n", devInfo_h);\
		}
		
#define CHECK_DEVINFO_CLEAN_1(MCLEANPTR1)  \
		status1 = cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);\
		if(devInfo_h !=0L || status1 != cudaSuccess) {\
		    if(status1 != cudaSuccess)\
			    Rprintf("Transfer not successful.");\
			status1=cudaFree(MCLEANPTR1);\
			if (status1 != cudaSuccess ) {\
				error("devInfo_h error and memory free errors (potential memory leak) in '%s.' (%s)\n", __func__, cudaGetErrorString(status1));\
			}\
			error("Bad devInfo = %d.\n", devInfo_h);\
		}
		
//note first check clean is for workspace size and needs 
#define CHECK_CLEAN_WORK(MCLEANPTR1)  \
		if(s!=CUSOLVER_STATUS_SUCCESS) {\
			status1=cudaFree(MCLEANPTR1);\
			if (status1 != cudaSuccess ) {\
				error("CUSOLVER error and memory free errors (potential memory leak) in '%s.' (%s)\n", __func__, cudaGetErrorString(status1));\
			}\
			check_error(s);\
		}\
		CHECK_DEVINFO_CLEAN_1(MCLEANPTR1)



#define CHECK_CUSOLVE_CLEAN_2(MCLEANPTR1, MCLEANPTR2)  \
		CUDA_CHECK_KERNAL_CLEAN_2(MCLEANPTR1, MCLEANPTR2);\
		if(s!=CUSOLVER_STATUS_SUCCESS) {\
			status1=cudaFree(MCLEANPTR1);\
			if (status1 != cudaSuccess ) {\
				error("CUSOLVER error and memory free errors (potential memory leak) in '%s.' (%s)\n", __func__, cudaGetErrorString(status1));\
			}\
			status1=cudaFree(MCLEANPTR2);\
			if (status1 != cudaSuccess ) {\
				error("CUSOLVER error and memory free errors (potential memory leak) in '%s.' (%s)\n", __func__, cudaGetErrorString(status1));\
			}\
			check_error(s);\
		}\
		CHECK_DEVINFO_CLEAN_2(MCLEANPTR1, MCLEANPTR2)
		
#define CHECK_CUBLAS_CLEAN_2(MCLEANPTR1, MCLEANPTR2)  \
		CUDA_CHECK_KERNAL_CLEAN_2(MCLEANPTR1, MCLEANPTR2);\
		if(cublas_status != CUBLAS_STATUS_SUCCESS ) {\
			status1=cudaFree(MCLEANPTR1);\
			if (status1 != cudaSuccess ) {\
				error("CUBLAS error and memory free errors (potential memory leak) in '%s.' (%s)\n", __func__, cudaGetErrorString(status1));\
			}\
			status1=cudaFree(MCLEANPTR2);\
			if (status1 != cudaSuccess ) {\
				error("CUBLAS error and memory free errors (potential memory leak) in '%s.' (%s)\n", __func__, cudaGetErrorString(status1));\
			}\
			error("CUBLAS error.");\
		}
		
	

SEXP rcusolve_qr(SEXP A_in, SEXP qraux_in)
{
	#if CUDART_VERSION < 7000
	error("Please upgrade to a newer version of CUDA to perform this action.");
	#else
    int m, n;
 	struct gmat A = get_gmat_struct(A_in);
	struct gptr *qraux = (struct gptr*) R_ExternalPtrAddr(qraux_in);
	
	int  lwork = 0;
	void *workspace = NULL;

	
	cusolverStatus_t s = CUSOLVER_STATUS_SUCCESS;
    cudaError_t status1 = cudaSuccess, status2 = cudaSuccess, cudaStat=cudaSuccess;


	if (A.type >1L)
		error("'a' must be a of type 'double' or 'single'");

    m = A.nrow;//Adims[0];
    n = A.ncol;//Adims[1];
	
	SET_UP_DEVINFO;
	
    if(A.type==0L) {
		//Work space creation
		s = cusolverDnDgeqrf_bufferSize(
			cudshandle[currentDevice], m, n, (double *) A.d_vec, m,
			&lwork);
		CHECK_CLEAN_WORK(devInfo);
		CUDA_MALLOC_CLEAN_1(workspace,sizeof(double)*lwork, devInfo);
		//Run QR Decomposition
		//cusolverDnDgeqrf(cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *TAU, double *Workspace, int Lwork, int *devInfo );
    	s = cusolverDnDgeqrf(cudshandle[currentDevice], m, n,
			(double *) A.d_vec, m,
			(double *) qraux->d_vec,
			(double *) workspace, lwork, devInfo );
		CHECK_CUSOLVE_CLEAN_2(workspace,devInfo);

    } else {
		//Work space creation
		s = cusolverDnSgeqrf_bufferSize(
			cudshandle[currentDevice], m, n, (float *) A.d_vec, m,
			&lwork);
		CHECK_CLEAN_WORK(devInfo);
		CUDA_MALLOC_CLEAN_1(workspace,sizeof(float)*lwork, devInfo);
		//Run QR Decomposition
		//cusolverDnDgeqrf(cusolverDnHandle_t handle, int m, int n, float *A, int lda, float *TAU, float *Workspace, int Lwork, int *devInfo );
    	s = cusolverDnSgeqrf(cudshandle[currentDevice], m, n,
			(float *) A.d_vec, m,
			(float *) qraux->d_vec,
			(float *) workspace, lwork, devInfo );
		CHECK_CUSOLVE_CLEAN_2(workspace,devInfo);

	}

	CUDA_CLEAN_2(workspace,devInfo);
	#endif
    return asSEXPint(1L);
}



SEXP rcusolve_modqr_coef(SEXP qr_in, SEXP qraux_in, SEXP B_in) {
	#if CUDART_VERSION < 7000
	error("Please upgrade to a newer version of CUDA to perform this action.");
	#else
    int m, n, nrhs, k;

	struct gmat qr = get_gmat_struct(qr_in);
	struct gmat B = get_gmat_struct(B_in);
	struct gvec qraux = get_gvec_struct(qraux_in);

	int  lwork = 0;

	void *workspace = NULL;
	const double oneD = 1;
	const float oneS = 1;
	cusolverStatus_t s = CUSOLVER_STATUS_SUCCESS;
    cudaError_t status1 = cudaSuccess, status2 = cudaSuccess, cudaStat=cudaSuccess;
	cublasStatus_t cublas_status;

	
	m = qr.nrow;//Adims[0];
    n = qr.ncol;//Adims[1];
    k = qraux.length;
	if(n>m)
		error("Underdetermined systems not implemented");
    if(B.nrow != m)
    	error("right-hand side should have %d not %d rows", m,B.nrow );
    nrhs = B.ncol;
	SET_UP_DEVINFO;
    if(qr.type==0L) {
		//Work space creation
		s = cusolverDnDgeqrf_bufferSize(
			cudshandle[currentDevice], m, n, (double *) qr.d_vec, m,
			&lwork);
		CHECK_CLEAN_WORK(devInfo);
		CUDA_MALLOC_CLEAN_1(workspace,sizeof(double)*lwork, devInfo);

		//cusolver_status= cudsDormqr( cudenseH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, nrhs, m, d_A, lda, d_tau, d_B, ldb, d_work, lwork, devInfo);
    	s=cusolverDnDormqr(cudshandle[currentDevice], CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, nrhs, k,
        		(double *) qr.d_vec, m,
				(double *)qraux.d_vec,
				(double *) B.d_vec, m,
				(double *) workspace, 
				lwork, devInfo );
		CHECK_CUSOLVE_CLEAN_2(workspace,devInfo);
		cublas_status = cublasDtrsm( (handle[currentDevice]), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
				n, nrhs, &oneD, (double *) qr.d_vec, m, (double *) B.d_vec, m);				
		CHECK_CUBLAS_CLEAN_2(workspace,devInfo);

    } else {
		//Work space creation
		s = cusolverDnSgeqrf_bufferSize(
			cudshandle[currentDevice], m, n, (float *) qr.d_vec, m,
			&lwork);
		CHECK_CLEAN_WORK(devInfo);
		CUDA_MALLOC_CLEAN_1(workspace,sizeof(float)*lwork, devInfo);

		//cusolver_status= cudsDormqr( cudenseH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, nrhs, m, d_A, lda, d_tau, d_B, ldb, d_work, lwork, devInfo);
    	s=cusolverDnSormqr(cudshandle[currentDevice], CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, nrhs, k,
        		(float *) qr.d_vec, m,
				(float *)qraux.d_vec,
				(float *) B.d_vec, m,
				(float *) workspace, 
				lwork, devInfo );
		CHECK_CUSOLVE_CLEAN_2(workspace,devInfo);
		cublas_status = cublasStrsm( (handle[currentDevice]), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
				n, nrhs, &oneS, (float *) qr.d_vec, m, (float *) B.d_vec, m);				
		CHECK_CUBLAS_CLEAN_2(workspace,devInfo);

	}
	CUDA_CLEAN_2(workspace,devInfo);
	#endif
    return asSEXPint(1L);
}


template <typename T>
__global__ void kernal_lower_0(T* ret, int n_row, int n_col, int operations_per_thread)
{
	int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;
	for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {
		int col = i / n_row;
		int row = i-n_row*col;
		if (row < n_row && col<n_col && row>col) {
				ret[IDX2(row, col ,n_col)] = 0 ;
		}
	}

}
void lower_0(gmat *ret, int type)
{

	int n_row = ret->nrow;
	int n_col =  ret->ncol;
	int n = n_row*n_col;
	DECERROR1;
	
	//allocate
	#define KERNAL(PTR,T)\
		GET_BLOCKS_PER_GRID(n,kernal_lower_0< T >);\
		kernal_lower_0< T ><<<blocksPerGrid, (tpb)>>>(PTR(ret),n_row,n_col, operations_per_thread);
	CALL_KERNAL;
	#undef KERNAL
	CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);
}
SEXP rcusolve_chol(SEXP A_in)
{
	#if CUDART_VERSION < 7000
	error("Please upgrade to a newer version of CUDA to perform this action.");
	#else
    int m, n;
 	struct gmat A = get_gmat_struct(A_in);

	

	int  lwork = 0;
	void *workspace = NULL;


	
	cusolverStatus_t s = CUSOLVER_STATUS_SUCCESS;
    cudaError_t status1 = cudaSuccess, status2 = cudaSuccess, cudaStat=cudaSuccess;


	if (A.type >1L)
		error("'a' must be a of type 'double' or 'single'");

    m = A.nrow;//Adims[0];
    n = A.ncol;//Adims[1];
    if(m!=n)
		error("Rows and Cols must be the same.");
	
	SET_UP_DEVINFO;
    if(A.type==0L) {
		//Work space creation
		s = cusolverDnDpotrf_bufferSize(cudshandle[currentDevice], CUBLAS_FILL_MODE_UPPER, m, (double *) A.d_vec, m,  &lwork);
		CHECK_CLEAN_WORK(devInfo);
		CUDA_MALLOC_CLEAN_1(workspace,sizeof(double)*lwork, devInfo);
		
		//Run QR Decomposition
		//cusolverStatus_t cusolverDnDpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float *A, int lda, float *Workspace, int Lwork, int *devInfo ); 
    	s = cusolverDnDpotrf (cudshandle[currentDevice], CUBLAS_FILL_MODE_UPPER, m,
			(double *) A.d_vec, m,
			(double *) workspace, lwork, devInfo);
		CHECK_CUSOLVE_CLEAN_2(workspace,devInfo);
		lower_0(&A, A.type);
		
    } else {
		s = cusolverDnSpotrf_bufferSize(cudshandle[currentDevice], CUBLAS_FILL_MODE_UPPER, m, (float *) A.d_vec, m,  &lwork);
		CHECK_CLEAN_WORK(devInfo);
		CUDA_MALLOC_CLEAN_1(workspace,sizeof(float)*lwork, devInfo);
		
		//Run QR Decomposition
		//cusolverStatus_t cusolverDnSpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float *A, int lda, float *Workspace, int Lwork, int *devInfo ); 
    	s = cusolverDnSpotrf (cudshandle[currentDevice], CUBLAS_FILL_MODE_UPPER, m,
			(float *) A.d_vec, m,
			(float *) workspace, lwork, devInfo);
		CHECK_CUSOLVE_CLEAN_2(workspace,devInfo);
		lower_0(&A, A.type);
	}

	CUDA_CLEAN_2(workspace,devInfo);
	#endif
    return asSEXPint(1L);
}

SEXP rcusolve_svd(SEXP A_in, SEXP  S_in, SEXP  U_in, SEXP  VT_in)
{
	#if CUDART_VERSION < 7000
	error("Please upgrade to a newer version of CUDA to perform this action.");
	#else
    int m, n;
 	struct gmat A = get_gmat_struct(A_in);
	struct gptr *S = (struct gptr*) R_ExternalPtrAddr(S_in);
	struct gmat U = get_gmat_struct(U_in);
	struct gmat VT = get_gmat_struct(VT_in);	
	
	const char Achar = 'A';
	int  lwork = 0;
	void *workspace = NULL;
	void *hostworkspace = NULL;

	
	cusolverStatus_t s = CUSOLVER_STATUS_SUCCESS;
    cudaError_t status1 = cudaSuccess, status2 = cudaSuccess, cudaStat=cudaSuccess;


	if (A.type >1L)
		error("'a' must be a of type 'double' or 'single'");

    m = A.nrow;//Adims[0];
    n = A.ncol;//Adims[1];

	SET_UP_DEVINFO;
    if(A.type==0L) {
		//Work space creation
		s = cusolverDnDgesvd_bufferSize(cudshandle[currentDevice], m, n, &lwork);
		CHECK_CLEAN_WORK(devInfo);
		CUDA_MALLOC_CLEAN_1(workspace,sizeof(double)*lwork, devInfo);
		hostworkspace = (void *) R_alloc(lwork, sizeof(double));
		
		//Run QR Decomposition
		//usolverDnDgesvd (cusolverDnHandle_t handle, char jobu, char jobvt, int m, int n, double *A, int lda, double *S, double *U, int ldu, double *VT, int ldvt, double *Work, int Lwork, double *rwork, int *devInfo);

    	s = cusolverDnDgesvd (cudshandle[currentDevice], Achar, Achar, m, n,
			(double *) A.d_vec, m,
			(double *) S->d_vec,
			(double *) U.d_vec, m,
			(double *) VT.d_vec, n,
			(double *) workspace, lwork, (double *) hostworkspace, devInfo);
			// double *Work, int Lwork, double *rwork, int *devInfo)
		CHECK_CUSOLVE_CLEAN_2(workspace,devInfo);

    } else {
		//Work space creation
		s = cusolverDnDgesvd_bufferSize(cudshandle[currentDevice], m, n, &lwork);
		CHECK_CLEAN_WORK(devInfo);
		CUDA_MALLOC_CLEAN_1(workspace,sizeof(float)*lwork, devInfo);
		hostworkspace = (void *) R_alloc(lwork, sizeof(float));
		
		//Run QR Decomposition
		//usolverDnDgesvd (cusolverDnHandle_t handle, char jobu, char jobvt, int m, int n, float *A, int lda, float *S, float *U, int ldu, float *VT, int ldvt, float *Work, int Lwork, float *rwork, int *devInfo);

    	s = cusolverDnSgesvd (cudshandle[currentDevice], Achar, Achar, m, n,
			(float *) A.d_vec, m,
			(float *) S->d_vec,
			(float *) U.d_vec, m,
			(float *) VT.d_vec, n,
			(float *) workspace, lwork, (float *) hostworkspace, devInfo);
			// float *Work, int Lwork, float *rwork, int *devInfo)
		CHECK_CUSOLVE_CLEAN_2(workspace,devInfo);
	}

	CUDA_CLEAN_2(workspace,devInfo);
	#endif
    return asSEXPint(1L);
}



/*
SEXP rcusolve_dgesv(SEXP A_in, SEXP B_in)
{
    int n;
    culaStatus s;

	struct gmat A = get_gmat_struct(A_in);
	struct gmat B = get_gmat_struct(B_in);


    if (A.type>1L)
    	error(("'a' must be of type 'single' or 'double.'"));
    if (B.type>1L)
    	error(("'b' must be of type 'single' or 'double.'"));

    n = A.nrow;
    if(n == 0) error(("'a' is 0-diml"));
    int p = B.ncol;
    if(p == 0) error(("no right-hand side in 'b'"));
    if(A.ncol != n)
    	error(("'a' (%d x %d) must be square"), n, A.ncol);
    if(B.nrow != n)
    	error(("'b' (%d x %d) must be compatible with 'a' (%d x %d)"), B.nrow, p, n, n);
    SEXP ret;
    PROTECT(ret = allocVector(INTSXP, n));
    int *ipiv = INTEGER(ret);
    for(int tmp=0;tmp<n;tmp++)
    	ipiv[tmp]=1;
    if(A.type==0L)
    	s = culaDgesv(n, p, (double *)A.d_vec, n, ipiv, (double *)B.d_vec, n);
    else
    	s = culaSgesv(n, p, (float *) A.d_vec, n, ipiv, (float *)B.d_vec, n);

    check_error(s);
    UNPROTECT(1);
    return ret;
	
}


SEXP check_inverse_condition(SEXP Ain, SEXP Avalsin, SEXP permin, SEXP tolin) {
	double tol = REAL(tolin)[0];
	double *A = REAL(Ain);
	double *Avals= REAL(Avalsin);
	int *ipiv = INTEGER(permin);
	double rcond;
	int info;

	int *Adims = INTEGER(coerceVector(getAttrib(Ain, R_DimSymbol), INTSXP));
    int n = Adims[0];
	double anorm = F77_CALL(dlange)("1", &n, &n, A, &n, (double*) NULL);
	double *work = (double *) R_alloc(4*n, sizeof(double));
	F77_CALL(dgecon)("1", &n, Avals, &n, &anorm, &rcond, work, ipiv, &info);
	if (rcond < tol)
		error(("system is computationally singular: reciprocal condition number = %g"),
				rcond);
	return(asSEXPreal(rcond));

}

SEXP rcula_eigen_symm(SEXP A_in, SEXP val_in)
{
	culaStatus s;
	struct gmat A = get_gmat_struct(A_in);
	struct gvec val = get_gvec_struct(val_in);

	if(A.nrow != val.length)
		error("dim mismatch");

	if(A.type==0L)
		s=culaDeviceDsyev('V', 'U', A.ncol, (double *) A.d_vec , A.nrow, (double *) val.d_vec );
	else
		s=culaDeviceSsyev('V', 'U', A.ncol, (float *) A.d_vec , A.nrow, (float *) val.d_vec );


	check_error(s);

	return asSEXPint(1L);
}
*/

