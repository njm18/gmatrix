
#include "gmatrix.h"

/*
template <typename T>
__device__ bool myisfinite(T x) {
	return isfinite(x) ;
}


template <>
__device__ bool myisfinite<int>(int x) {
	return CUDA_R_Na_int==x;
}

template <typename T>
__device__ T R_NA1(void) {
	return CUDA_R_Na_float ;
}

template <>
__device__ double R_NA1<double>(void) {
	return CUDA_R_Na_double;
}

template <>
__device__ int R_NA1<int>(void) {
	return CUDA_R_Na_int;
}

*/

template <typename T>
__global__ void kernal_numeric_index(T* y, int n_y, T* ret, int n_ret, int* index,
		int operations_per_thread)
{

	int j;
	int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;
	for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {
		__syncthreads();
		if (i < n_ret) {
			j=index[i];
			//printf("id = %d, i=%d, j = %d \n",id, i, j);
			if( j==INT_MIN || j>n_y || j<1 )
				MAKE_NA<T>(&(ret[i]));
			else {
				j=j-1;
				ret[i] = y[j];
			}
		}
	}


}


SEXP gpu_numeric_index(SEXP A_in, SEXP n_A_in, SEXP index_in, SEXP n_index_in, SEXP in_type)
{
	SEXP ret_final;

	struct gpuvec *d_index = (struct gpuvec*) R_ExternalPtrAddr(index_in);

	int n_ret = INTEGER(n_index_in)[0];
	int n_A = INTEGER(n_A_in)[0];
	DECERROR1;
    struct gpuvec *ret = Calloc(1, struct gpuvec);
	struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);
	//double myna=R_NaReal;//ok so passing in R_NaReal can't be optimal but it works


	PROCESS_TYPE;
	//allocate  and copy
	//CUDA_MALLOC( d_index, n_ret*sizeof(int) );
	//CUDA_MEMCPY_CLEAN(d_index, index, n_ret * sizeof(int), cudaMemcpyHostToDevice);
	CUDA_MALLOC(ret->d_vec, n_ret * mysizeof);


	GET_BLOCKS_PER_GRID(n_ret);
	//kernal_mult_scaler<<<blocksPerGrid, (threads_per_block[currentDevice])>>>(A->d_vec,PTR(ret),1, n, operations_per_thread);
#define KERNAL(PTR,T)\
	kernal_numeric_index< T ><<<blocksPerGrid, (threads_per_block[currentDevice])>>>(PTR(A), n_A,\
			PTR(ret),	n_ret, (int *) (d_index->d_vec), operations_per_thread);
	CALL_KERNAL;
	#undef KERNAL
	CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);





    ret_final = gpu_register(ret);
    return ret_final;
}


template <typename T>
__global__ void kernal_gmatrix_index_row(
		T* y, int n_row_y, int n_col_y,
		T* ret, int n_row_ret, int* index,
		int operations_per_thread)
{

	int row_j;
	int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;
	for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {
		__syncthreads();
		int col = i / n_row_ret;
		int row = i-n_row_ret*col;
		if (row < n_row_ret && col < n_col_y  ) { //cols of ret and y are the same
			row_j=index[row];// ok this could be made more efficeint b/c it only needs to be looked up once for each row
			//printf("id = %d, i=%d, row_j = %d, row = %d, col = %d, pos = %d, n_row_ret = %d, n_col_y=%d\n",
			//		id, i, row_j, row, col, IDX2(row_j ,col ,n_row_y), n_row_ret,n_col_y);
			if(row_j==INT_MIN || row_j>n_row_y || row_j<1)
				MAKE_NA<T>(&(ret[i]));
			else {
				row_j=row_j-1;
				ret[i] = y[IDX2(row_j ,col ,n_row_y)];

			}
		}
	}

}


SEXP gpu_gmatrix_index_row(SEXP A_in, SEXP n_row_A_in, SEXP n_col_A_in, SEXP index_in, SEXP n_index_in, SEXP in_type)
{
	SEXP ret_final;
	struct gpuvec *d_index = (struct gpuvec*) R_ExternalPtrAddr(index_in);
	int n_row_ret = INTEGER(n_index_in)[0];
	int n_row_A = INTEGER(n_row_A_in)[0];
	int n_col_A = INTEGER(n_col_A_in)[0];
	DECERROR1;
	struct gpuvec *ret = Calloc(1, struct gpuvec);
	struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);
	//double myna=R_NaReal;//ok so passing in R_NaReal can't be optimal but it works
	PROCESS_TYPE;

	//allocate and copy
	//CUDA_MALLOC(d_index, n_row_ret* sizeof(int));
	//CUDA_MEMCPY_CLEAN(d_index, index, n_row_ret* sizeof(int), cudaMemcpyHostToDevice);
	CUDA_MALLOC(ret->d_vec , n_row_ret*n_col_A * mysizeof) ;

	GET_BLOCKS_PER_GRID(n_row_ret*n_col_A);
#define KERNAL(PTR,T)\
	kernal_gmatrix_index_row< T ><<<blocksPerGrid, (threads_per_block[currentDevice])>>>(PTR(A), n_row_A, n_col_A,\
			PTR(ret),	n_row_ret, (int *) (d_index->d_vec), operations_per_thread);

			CALL_KERNAL;
	#undef KERNAL
	CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);


	ret_final = gpu_register(ret);
	return ret_final;
}


template <typename T>
__global__ void kernal_gmatrix_index_col(
		T* y, int n_row_y, int n_col_y,
		T* ret, int n_col_ret, int* index,
		int operations_per_thread)
{

	int col_j;
	int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;
	for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {
		__syncthreads();
		int col = i / n_row_y;
		int row = i-n_row_y*col;
		if (row < n_row_y && col < n_col_ret  ) { //cols of ret and y are the same
			col_j=index[col];// ok this could be made more efficeint b/c it only needs to be looked up once for each row
			//printf("id = %d, i=%d, col_j = %d, row = %d, col = %d, pos = %d, n_col_ret = %d, n_col_y=%d\n",
			//		id, i, col_j, row, col, IDX2(row ,col_j ,n_row_y), n_col_ret,n_col_y);
			if(col_j==INT_MIN || col_j>n_col_y || col_j<1)
				MAKE_NA<T>(&(ret[i]));
			else {
				col_j=col_j-1;
				ret[i] = y[IDX2(row ,col_j ,n_row_y)];

			}
		}
	}

}


SEXP gpu_gmatrix_index_col(SEXP A_in, SEXP n_row_A_in, SEXP n_col_A_in, SEXP index_in, SEXP n_index_in, SEXP in_type)
{
	SEXP ret_final;
	struct gpuvec *d_index = (struct gpuvec*) R_ExternalPtrAddr(index_in);

	int n_col_ret = INTEGER(n_index_in)[0];
	int n_row_A = INTEGER(n_row_A_in)[0];
	int n_col_A = INTEGER(n_col_A_in)[0];
	DECERROR1;
	struct gpuvec *ret = Calloc(1, struct gpuvec);
	struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);
	//double myna=R_NaReal;//ok so passing in R_NaReal can't be optimal but it works
	PROCESS_TYPE;

	//CUDA_MALLOC(d_index, n_col_ret* sizeof(int));
	//CUDA_MEMCPY_CLEAN(d_index, index, n_col_ret* sizeof(int), cudaMemcpyHostToDevice);
	CUDA_MALLOC(ret->d_vec , n_col_ret*n_row_A  * mysizeof) ;

	GET_BLOCKS_PER_GRID(n_col_ret*n_row_A);
#define KERNAL(PTR,T)\
	kernal_gmatrix_index_col< T ><<<blocksPerGrid, (threads_per_block[currentDevice])>>>(PTR(A), n_row_A, n_col_A,\
			PTR(ret),	n_col_ret, (int *) (d_index->d_vec), operations_per_thread);
		CALL_KERNAL;
	#undef KERNAL
	CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);



	ret_final = gpu_register(ret);
	return ret_final;
}


template <typename T>
__global__ void kernal_gmatrix_index_both(
		T* y, int n_row_y, int n_col_y,
		T* ret, int n_row_ret, int n_col_ret,
		int* index_row, int* index_col,
		int operations_per_thread)
{

	int col_j, row_j;
	int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;
	for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {
		__syncthreads();
		int col = i / n_row_ret;
		int row = i-n_row_ret*col;
		if (row < n_row_ret && col < n_col_ret  ) { //cols of ret and y are the same
			col_j=index_col[col];// ok this could be made more efficeint b/c it only needs to be looked up once for each row
			row_j=index_row[row];
			//printf("id = %d, i=%d, col_j = %d, row_j = %d, row = %d, col = %d, pos = %d, n_col_ret = %d, n_row_ret=%d\n",
			//		id, i, col_j, row_j, row, col, IDX2(row_j ,col_j ,n_row_y), n_col_ret,n_row_ret);
			if(col_j==INT_MIN || col_j>n_col_y || col_j<1 || row_j==INT_MIN || row_j>n_row_y || row_j<1 )
				MAKE_NA<T>(&(ret[i]));
			else {
				col_j=col_j-1;
				row_j=row_j-1;
				ret[i] = y[IDX2(row_j ,col_j ,n_row_y)];

			}
		}
	}

}





SEXP gpu_gmatrix_index_both(SEXP A_in, SEXP n_row_A_in, SEXP n_col_A_in,
		SEXP index_row_in, SEXP n_index_row_in,SEXP index_col_in, SEXP n_index_col_in, SEXP in_type)
{
	SEXP ret_final;
	struct gpuvec *d_index_row = (struct gpuvec*) R_ExternalPtrAddr(index_row_in);
	struct gpuvec *d_index_col = (struct gpuvec*) R_ExternalPtrAddr(index_col_in);


	int n_row_ret = INTEGER(n_index_row_in)[0];
	int n_col_ret = INTEGER(n_index_col_in)[0];
	int n_row_A = INTEGER(n_row_A_in)[0];
	int n_col_A = INTEGER(n_col_A_in)[0];
	DECERROR1;
	struct gpuvec *ret = Calloc(1, struct gpuvec);
	struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);
	//double myna=R_NaReal;//ok so passing in R_NaReal can't be optimal but it works
	PROCESS_TYPE;
	//allocate index
	//CUDA_MALLOC(d_index_row,   n_row_ret*sizeof(int));
	//CUDA_MEMCPY_CLEAN(d_index_row, index_row, n_row_ret*sizeof(int), cudaMemcpyHostToDevice);

	//CUDA_MALLOC_CLEAN_1(d_index_col,   n_col_ret*sizeof(int), d_index_row);
	//CUDA_MEMCPY_CLEAN_1(d_index_col, index_col, n_col_ret*sizeof(int), cudaMemcpyHostToDevice, d_index_row);

	CUDA_MALLOC(ret->d_vec , n_col_ret*n_row_ret*mysizeof) ;

	GET_BLOCKS_PER_GRID(n_col_ret*n_row_ret);
#define KERNAL(PTR,T)\
	kernal_gmatrix_index_both< T ><<<blocksPerGrid, (threads_per_block[currentDevice])>>>(PTR(A), n_row_A, n_col_A,\
			PTR(ret),	n_row_ret,n_col_ret, (int *) (d_index_row->d_vec), (int *) (d_index_col->d_vec), operations_per_thread);
	CALL_KERNAL;
	#undef KERNAL
	CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);



	ret_final = gpu_register(ret);
	return ret_final;
}


// Set index

/*
#define CREATE_ERROR_STATE  \
		__device__ int d_error_state=0;\
		int h_error_state;
#define CHECK_ERROR_STATE  \
		cudaMemcpyFromSymbol(&h_error_state, "d_error_state", sizeof(int), 0, cudaMemcpyDeviceToHost);
		if(h_error_state==1); \
			error("The index cannot contain an 'NA' or be outside the boundaries.");
*/
template <typename T>
__global__ void kernal_numeric_index_set(T* y, int n_y, T* val, int n_val,int n_replace, int* index,
		 int operations_per_thread)
{

	int j;

	int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;
	for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {
		__syncthreads();
		if (i < n_replace) {
			j=index[i];
			//printf("id = %d, i=%d, j = %d \n",id, i, j);

			if(!(j==INT_MIN || j>n_y || j<1)) {
				j=j-1;
				y[j]=val[i % n_val];
			}
		}
	}


}


SEXP gpu_numeric_index_set(SEXP A_in, SEXP n_A_in, SEXP val_in, SEXP n_val_in, SEXP index_in, SEXP n_index_in, SEXP in_type)
{

	struct gpuvec *d_index = (struct gpuvec*) R_ExternalPtrAddr(index_in);
	int n_index = INTEGER(n_index_in)[0];
	int n_A = INTEGER(n_A_in)[0];
	int n_val = INTEGER(n_val_in)[0];
	DECERROR0;
	struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);
	struct gpuvec *val = (struct gpuvec*) R_ExternalPtrAddr(val_in);
	//double myna=R_NaReal;//ok so passing in R_NaReal can't be optimal but it works

	if(n_index%n_val!=0)
		error("Size of value and index do not match.");

	PROCESS_TYPE_NO_SIZE;
	//allocate index
	//CUDA_MALLOC(d_index, n_val * sizeof(int));
	//CUDA_MEMCPY_CLEAN(d_index, index, n_val* sizeof(int), cudaMemcpyHostToDevice);

	GET_BLOCKS_PER_GRID(n_val);
	//kernal_mult_scaler< T ><<<blocksPerGrid, (threads_per_block[currentDevice])>>>(PTR(A),PTR(val),1, n, operations_per_thread);
#define KERNAL(PTR,T)\
	kernal_numeric_index_set< T ><<<blocksPerGrid, (threads_per_block[currentDevice])>>>(PTR(A), n_A,\
			PTR(val),	n_val, n_index,  (int *) (d_index->d_vec), operations_per_thread);
	CALL_KERNAL;
	#undef KERNAL
	//CUDA_CHECK_KERNAL_CLEAN_1(d_index);
	CUDA_CHECK_KERNAL;

    return n_val_in;
}






template <typename T>
__global__ void kernal_gmatrix_index_row_set(
		T* y, int n_row_y, int n_col_y,
		T* val, int n_val, int n_row_replace, int* index,
		 int operations_per_thread)
{

	int row_j;
	int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;
	for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {
		__syncthreads();
		int col = i / n_row_replace;
		int row = i-n_row_replace*col;
		if (row < n_row_replace && col < n_col_y  ) { //cols of val and y are the same
			row_j=index[row];// ok this could be made more efficeint b/c it only needs to be looked up once for each row
			//printf("id = %d, i=%d, row_j = %d, row = %d, col = %d, pos = %d, n_row_replace = %d, n_col_y=%d\n",
			//		id, i, row_j, row, col, IDX2(row_j ,col ,n_row_y), n_row_replace,n_col_y);
			if(!(row_j==INT_MIN || row_j>n_row_y || row_j<1)) {
				row_j=row_j-1;
				y[IDX2(row_j ,col ,n_row_y)]= val[i% n_val] ;

			}
		}
	}

}


SEXP gpu_gmatrix_index_row_set(SEXP A_in, SEXP n_row_A_in, SEXP n_col_A_in, SEXP val_in, SEXP n_val_in, SEXP index_in, SEXP n_index_in, SEXP in_type)
{

	struct gpuvec *d_index = (struct gpuvec*) R_ExternalPtrAddr(index_in);
	int n_row_replace = INTEGER(n_index_in)[0];
	int n_row_A = INTEGER(n_row_A_in)[0];
	int n_col_A = INTEGER(n_col_A_in)[0];
	int n_val = INTEGER(n_val_in)[0];
	DECERROR0;
	struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);
	struct gpuvec *val = (struct gpuvec*) R_ExternalPtrAddr(val_in);
	//double myna=R_NaReal;//ok so passing in R_NaReal can't be optimal but it works

	if((n_row_replace*n_col_A)%n_val!=0)
		error("Size of value and index do not match.");
	PROCESS_TYPE_NO_SIZE;
	//allocate index
	//CUDA_MALLOC(d_index, n_row_replace * sizeof(int));
	//CUDA_MEMCPY_CLEAN(d_index, index, n_row_replace* sizeof(int), cudaMemcpyHostToDevice);

	GET_BLOCKS_PER_GRID(n_row_replace*n_col_A);
#define KERNAL(PTR,T)\
	kernal_gmatrix_index_row_set< T ><<<blocksPerGrid, (threads_per_block[currentDevice])>>>(PTR(A), n_row_A, n_col_A,\
			PTR(val),n_val,	n_row_replace,  (int *) (d_index->d_vec), operations_per_thread);
	CALL_KERNAL;
	#undef KERNAL
	CUDA_CHECK_KERNAL;

	//CUDA_CLEAN_1(d_index);

	return n_val_in;
}



template <typename T>
__global__ void kernal_gmatrix_index_col_set(
		T* y, int n_row_y, int n_col_y,
		T* val, int n_val, int n_col_replace, int* index,
		 int operations_per_thread)
{

	int col_j;
	int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;
	for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {
		__syncthreads();
		int col = i / n_row_y;
		int row = i-n_row_y*col;
		if (row < n_row_y && col < n_col_replace  ) { //cols of val and y are the same
			col_j=index[col];// ok this could be made more efficeint b/c it only needs to be looked up once for each row
			//printf("id = %d, i=%d, col_j = %d, row = %d, col = %d, pos = %d, n_col_replace = %d, n_col_y=%d\n",
			//		id, i, col_j, row, col, IDX2(row ,col_j ,n_row_y), n_col_replace,n_col_y);
			if(!(col_j==INT_MIN || col_j>n_col_y || col_j<1)) {
				col_j=col_j-1;
				y[IDX2(row ,col_j ,n_row_y)]=val[i % n_val] ;

			}
		}
	}

}


SEXP gpu_gmatrix_index_col_set(SEXP A_in, SEXP n_row_A_in, SEXP n_col_A_in,  SEXP val_in, SEXP n_val_in, SEXP index_in, SEXP n_index_in, SEXP in_type)
{

	struct gpuvec *d_index = (struct gpuvec*) R_ExternalPtrAddr(index_in);

	int n_col_replace = INTEGER(n_index_in)[0];
	int n_row_A = INTEGER(n_row_A_in)[0];
	int n_col_A = INTEGER(n_col_A_in)[0];
	int n_val = INTEGER(n_val_in)[0];

	DECERROR0;
	struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);
	struct gpuvec *val = (struct gpuvec*) R_ExternalPtrAddr(val_in);
	//double myna=R_NaReal;//ok so passing in R_NaReal can't be optimal but it works

	if( (n_col_replace*n_row_A) %n_val !=0)
		error("Size of value and index do not match.");

	PROCESS_TYPE_NO_SIZE;
	//allocate index
	//CUDA_MALLOC(d_index, n_col_replace * sizeof(int));
	//CUDA_MEMCPY_CLEAN(d_index, index, n_col_replace* sizeof(int), cudaMemcpyHostToDevice);


	GET_BLOCKS_PER_GRID(n_col_replace*n_row_A);
#define KERNAL(PTR,T)\
	kernal_gmatrix_index_col_set< T ><<<blocksPerGrid, (threads_per_block[currentDevice])>>>(PTR(A), n_row_A, n_col_A,\
			PTR(val),n_val, n_col_replace,  (int *) (d_index->d_vec), operations_per_thread);
	CALL_KERNAL;
	#undef KERNAL
	CUDA_CHECK_KERNAL;

	//CUDA_CLEAN_1(d_index);

	return n_val_in;
}



template <typename T>
__global__ void kernal_gmatrix_index_both_set(
		T* y, int n_row_y, int n_col_y,
		T* val, int n_val, int n_row_replace, int n_col_replace,
		int* index_row, int* index_col,
		 int operations_per_thread)
{

	int col_j, row_j;
	int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;
	for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {
		__syncthreads();
		int col = i / n_row_replace;
		int row = i-n_row_replace*col;
		if (row < n_row_replace && col < n_col_replace  ) { //cols of val and y are the same
			col_j=index_col[col];// ok this could be made more efficeint b/c it only needs to be looked up once for each row
			row_j=index_row[row];
			//printf("id = %d, i=%d, col_j = %d, row_j = %d, row = %d, col = %d, pos = %d, n_col_replace = %d, n_row_replace=%d\n",
			//		id, i, col_j, row_j, row, col, IDX2(row_j ,col_j ,n_row_y), n_col_replace,n_row_replace);
			if(!(col_j==INT_MIN || col_j>n_col_y || col_j<1 || row_j==INT_MIN || row_j>n_row_y || row_j<1 )) {
				col_j=col_j-1;
				row_j=row_j-1;
				y[IDX2(row_j ,col_j ,n_row_y)]=val[i % n_val];

			}
		}
	}

}





SEXP gpu_gmatrix_index_both_set(SEXP A_in, SEXP n_row_A_in, SEXP n_col_A_in, SEXP val_in, SEXP n_val_in,
		SEXP index_row_in, SEXP n_index_row_in,SEXP index_col_in, SEXP n_index_col_in, SEXP in_type)
{

	struct gpuvec *d_index_row = (struct gpuvec*) R_ExternalPtrAddr(index_row_in);
	struct gpuvec *d_index_col = (struct gpuvec*) R_ExternalPtrAddr(index_col_in);
	int n_row_replace = INTEGER(n_index_row_in)[0];
	int n_col_replace = INTEGER(n_index_col_in)[0];
	int n_row_A = INTEGER(n_row_A_in)[0];
	int n_col_A = INTEGER(n_col_A_in)[0];
	int n_val = INTEGER(n_val_in)[0];

	DECERROR0;
	struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);
	struct gpuvec *val = (struct gpuvec*) R_ExternalPtrAddr(val_in);
	//double myna=R_NaReal;//ok so passing in R_NaReal can't be optimal but it works

	if(((n_row_replace*n_col_replace) % n_val) != 0)
		error("Size of value and index do not match.");
	PROCESS_TYPE_NO_SIZE;
	//allocate index
	//CUDA_MALLOC(d_index_row,   n_row_replace*sizeof(int));
	//CUDA_MEMCPY_CLEAN(d_index_row, index_row, n_row_replace*sizeof(int), cudaMemcpyHostToDevice);

	//CUDA_MALLOC_CLEAN_1(d_index_col,   n_col_replace*sizeof(int), d_index_row);
	//CUDA_MEMCPY_CLEAN_1(d_index_col, index_col, n_col_replace*sizeof(int), cudaMemcpyHostToDevice, d_index_row);


	GET_BLOCKS_PER_GRID(n_col_replace*n_row_replace);
#define KERNAL(PTR,T)\
	kernal_gmatrix_index_both_set< T ><<<blocksPerGrid, (threads_per_block[currentDevice])>>>(PTR(A), n_row_A, n_col_A,\
			PTR(val),	n_val, n_row_replace,n_col_replace,  (int *) (d_index_row->d_vec),  (int *) (d_index_col->d_vec), operations_per_thread);
	CALL_KERNAL;
	#undef KERNAL
	CUDA_CHECK_KERNAL;

	//CUDA_CLEAN_2(d_index_row,d_index_col);

	return n_val_in;
}


//other


template <typename T>
__global__ void kernal_naive_transpose(T* y, T* ret, int n_row, int n_col, int operations_per_thread)
{

	int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;
	for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {
		int col = i / n_row;
		int row = i-n_row*col;
		if (row < n_row && col<n_col) {
			//printf("n_row = %d, n_col = %d, row = %d ,col = %d, pos = %d, i=%d\n",
			//		n_row,n_col,row,col, IDX2(col, row ,n_col), i);
			ret[IDX2(col, row ,n_col)] = y[i] ;
		}
	}

}
SEXP gpu_naive_transpose(SEXP A_in, SEXP n_row_in, SEXP n_col_in, SEXP in_type)
{
	SEXP ret_final;
	int n_row = INTEGER(n_row_in)[0];
	int n_col = INTEGER(n_col_in)[0];
	int n = n_row*n_col;
	DECERROR1;
    struct gpuvec *ret = Calloc(1, struct gpuvec);
	struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);
	PROCESS_TYPE;
	//allocate
	CUDA_MALLOC(ret->d_vec,   n *mysizeof);

	GET_BLOCKS_PER_GRID(n);
#define KERNAL(PTR,T)\
	kernal_naive_transpose< T ><<<blocksPerGrid, (threads_per_block[currentDevice])>>>(PTR(A),PTR(ret),n_row,n_col, operations_per_thread);
	CALL_KERNAL;
	#undef KERNAL
	CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);

    ret_final = gpu_register(ret);
    return ret_final;
}


template <typename T>
__global__ void kernal_diag_get(T* y, T* ret, int n_row, int n,int operations_per_thread)
{

	int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;
	for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {
		if (i < n) {
			ret[i] = y[IDX2(i, i ,n_row)] ;
		}
	}

}
SEXP gpu_diag_get(SEXP A_in, SEXP n_row_in, SEXP n_col_in, SEXP in_type)
{
	SEXP ret_final;
	int n_row = INTEGER(n_row_in)[0];
	int n_col = INTEGER(n_col_in)[0];
	int n = min(n_row,n_col);
	DECERROR1;
    struct gpuvec *ret = Calloc(1, struct gpuvec);
	struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);

	PROCESS_TYPE;
	//allocate
	CUDA_MALLOC(ret->d_vec,   n *mysizeof);
	//cudaStat = cudaMalloc ((void**)&(PTR(ret)) , n * sizeof(double)) ;

	GET_BLOCKS_PER_GRID(n);
#define KERNAL(PTR,T)\
	kernal_diag_get< T ><<<blocksPerGrid, (threads_per_block[currentDevice])>>>(PTR(A),PTR(ret),n_row,n, operations_per_thread);
	CALL_KERNAL;
	#undef KERNAL
	CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);

    ret_final = gpu_register(ret);
    return ret_final;
}

template <typename T>
__global__ void kernal_diag_set(T* y, T* val, int n_row, int n,int operations_per_thread)
{

	int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;
	for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {

		if (i < n) {
			//printf("%d,  %d, %f \n ",IDX2(i, i ,n_row), i, val[i]);
			y[IDX2(i, i ,n_row)] =val[i] ;
		}
	}

}
SEXP gpu_diag_set(SEXP A_in, SEXP n_row_in, SEXP n_col_in, SEXP val_in, SEXP n_val_in, SEXP in_type)
{

	int n_row = INTEGER(n_row_in)[0];
	int n_col = INTEGER(n_col_in)[0];
	int n_val = INTEGER(n_val_in)[0];
	int n = min(n_row,n_col);
	if(n_val != n)
		error("replacement diagonal has wrong length");
	DECERROR0;
	struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);
	struct gpuvec *val = (struct gpuvec*) R_ExternalPtrAddr(val_in);
	PROCESS_TYPE_NO_SIZE;

	GET_BLOCKS_PER_GRID(n);
#define KERNAL(PTR,T)\
	kernal_diag_set< T ><<<blocksPerGrid, (threads_per_block[currentDevice])>>>(PTR(A),PTR(val),n_row, n, operations_per_thread);
	CALL_KERNAL;
	#undef KERNAL
	CUDA_CHECK_KERNAL;


    return A_in;
}



template <typename T>
__global__ void kernal_diag_set_one(T* y, T val, int n_row, int n,int operations_per_thread)
{

	int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;
	for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {

		if (i < n) {
			//printf("%d,  %d, %f \n ",IDX2(i, i ,n_row), i, val[i]);
			y[IDX2(i, i ,n_row)] =val ;
		}
	}

}
SEXP gpu_diag_set_one(SEXP A_in, SEXP n_row_in, SEXP n_col_in, SEXP val_in, SEXP in_type)
{

	int n_row = INTEGER(n_row_in)[0];
	int n_col = INTEGER(n_col_in)[0];
	int n = min(n_row,n_col);

	DECERROR0;
	struct gpuvec *A = (struct gpuvec*) R_ExternalPtrAddr(A_in);
	//double val = REAL(val_in)[0];
	PROCESS_TYPE_NO_SIZE;
	double val_double=0;
	float val_float=0;
	int val_int=0;
	if(type==0) {
		val_double= REAL(val_in)[0];
	} else if(type==1) {
		val_double= REAL(val_in)[0];
		val_float= (float) val_double;
	} else
		val_int= INTEGER(val_in)[0];




	GET_BLOCKS_PER_GRID(n);
#define KERNAL(PTR,T)\
	kernal_diag_set_one< T ><<<blocksPerGrid, (threads_per_block[currentDevice])>>>(PTR(A),val_##T ,n_row, n, operations_per_thread);
	CALL_KERNAL;
	#undef KERNAL
	CUDA_CHECK_KERNAL;

    return A_in;
}



/*
SEXP fromcpuMultmm(SEXP a, SEXP transa, SEXP b, SEXP transb)
{
	cudaError_t  cudaStat;
	SEXP c;
	int TA = LOGICAL_VALUE(transa), TB = LOGICAL_VALUE(transb),
			*DIMA = INTEGER(GET_DIM(a)), *DIMB = INTEGER(GET_DIM(b)),
			M = DIMA[TA], N = DIMB[!TB], K = DIMA[!TA],
			LDA = DIMA[0], LDB = DIMB[0], LDC = M;
	cublasOperation_t TRANSA = (TA==0 ? CUBLAS_OP_N : CUBLAS_OP_T), TRANSB = (TB==0 ? CUBLAS_OP_N : CUBLAS_OP_T);

	double *A = REAL(PROTECT(AS_NUMERIC(a))),
			*B = REAL(PROTECT(AS_NUMERIC(b))),
			*d_A, *d_B, *d_C;

	if(DIMB[TB] != K) error("non-conformable matrices");

	c = PROTECT(allocMatrix(REALSXP, M, N));
	cudaStat = cudaMalloc ((void**)&d_A, M * K * sizeof(double)) ;
	if (cudaStat != cudaSuccess )
		error("CUDA memory allocation error in 'matrix_multiply'. (%d)\n", (int) cudaStat);
	cudaStat = cudaMalloc ((void**)&d_B, K * N * sizeof(double)) ;
	if (cudaStat != cudaSuccess )
		error("CUDA memory allocation error in 'matrix_multiply'. (%d)\n", (int) cudaStat);
	cudaStat = cudaMalloc ((void**)&d_C, M * N * sizeof(double)) ;
	if (cudaStat != cudaSuccess )
		error("CUDA memory allocation error in 'matrix_multiply'. (%d)\n", (int) cudaStat);


	cudaStat=cudaMemcpy(d_A, A, M * K * sizeof(double), cudaMemcpyHostToDevice) ;
	if (cudaStat != cudaSuccess) {
		cudaError_t status1=cudaFree(d_A);
		if (status1 != cudaSuccess ) {
			error("CUDA memory copy and free errors in 'matrix_create'\n");
		}
		error("CUDA memory copy error in 'matrix_create.'  (%d)'\n", (int) cudaStat);
	}

	cudaStat=cudaMemcpy(d_B, B, K * N * sizeof(double), cudaMemcpyHostToDevice) ;
	if (cudaStat != cudaSuccess) {
		cudaError_t status1=cudaFree(d_A);
		if (status1 != cudaSuccess ) {
			error("CUDA memory copy and free errors in 'matrix_create'\n");
		}
		error("CUDA memory copy error in 'matrix_create.'  (%d)'\n", (int) cudaStat);
	}
    double alpha=1;
    double beta=0;

	cublasDgemm((handle[currentDevice]), TRANSA, TRANSB, M, N, K, &alpha, d_A, LDA, d_B, LDB, &beta, d_C, LDC);


    cudaStat=cudaMemcpy(REAL(c), d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost) ;
    if (cudaStat != cudaSuccess)
           warning("CUDA memory transfer error in 'matrix_create.' (%d)\n", (int) cudaStat);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

	UNPROTECT(3);

	return c;
}



SEXP matrix_create(SEXP in_mat)
{
	int nrows = INTEGER(getAttrib(in_mat,R_DimSymbol))[0];
	int ncols = INTEGER(getAttrib(in_mat,R_DimSymbol))[1];
	SEXP rownames,colnames, ptr;
	PROTECT(rownames= GET_VECTOR_ELT(getAttrib(in_mat, R_DimNamesSymbol),0));
	PROTECT(colnames = GET_VECTOR_ELT(getAttrib(in_mat, R_DimNamesSymbol),1));

    struct gpuvec *my_gpuvec = Calloc(1, struct gpuvec);
    double *h_vec = REAL(in_mat);
    cudaError_t  cudaStat;



    Rprintf("nrows = %d, ncol = %d \n", nrows, ncols);
    cudaStat = cudaMalloc( (void **)&(my_gpuvec->d_vec), nrows * ncols * sizeof(double)) ;
    if (cudaStat != cudaSuccess )
           error("CUDA memory allocation error in 'matrix_create.' (%d)\n",(int) cudaStat);

    cudaStat=cudaMemcpy(my_gpuvec->d_vec, h_vec, nrows * ncols * sizeof(double), cudaMemcpyHostToDevice) ;
    if (cudaStat != cudaSuccess)
           error("CUDA memory copy error in 'matrix_create.  (%d)'\n", (int) cudaStat);

    PROTECT(ptr = gpu_register(my_matrix));

    SEXP ret_final = PROTECT(NEW_OBJECT(MAKE_CLASS("gpumatrix")));
    SET_SLOT(ret_final, install("nrow"), nrows);
    SET_SLOT(ret_final, install("ncol"), ncols);
    SET_SLOT(ret_final, install("rownames"), rownames);
    SET_SLOT(ret_final, install("colnames"), colnames);
    SET_SLOT(ret_final, install("ptr"), ptr);


    UNPROTECT(4);
    return(ret_final);
}

SEXP gpu_get_matrix(SEXP A_in)
{
	cudaError_t  cudaStat;
	struct matrix A = get_matrix_struct(A_in);
	SEXP ret, dim;
	PROTECT(ret = allocVector(REALSXP, A.rows * A.cols));
	PROTECT(dim = allocVector(INTSXP, 2));
	INTEGER(dim)[0] = A.rows; INTEGER(dim)[1] = A.cols;
	setAttrib(ret, R_DimSymbol, dim);

	double *h_vec = REAL(ret);
    Rprintf("length = %d", A.rows * A.cols * sizeof(double));
    cudaStat=cudaMemcpy(h_vec, A.d_vec, A.rows * A.cols * sizeof(double), cudaMemcpyDeviceToHost) ;
    if (cudaStat != cudaSuccess)
           warning("CUDA memory transfer error in 'matrix_create.' (%d)\n", (int) cudaStat);
    UNPROTECT(2);
    return ret;
}

    CUBLAS_STATUS_SUCCESS         =0,
    CUBLAS_STATUS_NOT_INITIALIZED =1,
    CUBLAS_STATUS_ALLOC_FAILED    =3,
    CUBLAS_STATUS_INVALID_VALUE   =7,
    CUBLAS_STATUS_ARCH_MISMATCH   =8,
    CUBLAS_STATUS_MAPPING_ERROR   =11,
    CUBLAS_STATUS_EXECUTION_FAILED=13,
    CUBLAS_STATUS_INTERNAL_ERROR  =14


*/
