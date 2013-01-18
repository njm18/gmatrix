
#include "gmatrix.h"



//#define DEBUG

template <typename T>
__device__ T cuda_runi(curandState *state) {
	return curand_uniform(state) ;
}
template <>
__device__ double cuda_runi<double>(curandState *state) {
	return curand_uniform_double(state) ;
}

template <typename T>
__device__ T cuda_rnrm(curandState *state) {
	return curand_normal(state) ;
}
template <>
__device__ double cuda_rnrm<double>(curandState *state) {
	return curand_normal_double(state);
}

/*
template <typename T>
__device__ T cuda_rnrm(curandState *state) {
	T ret;
	if(T==double)
		ret = curand_normal_double(state);
	else
		ret = curand_normal(state);
	return(ret)
}*/



#define UNI cuda_runi<T>(&localState)
#define NRM  cuda_rnrm<T>(&localState)
#define UNI2 cuda_runi<T>(state)
#define NRM2 cuda_rnrm<T>(state)

//sims per thread must be even and two or greater
#define SETUP_RAND\
	int simulations_per_thread = ((total_states[currentDevice]) + n - 1) / (total_states[currentDevice]);\
	int total_threads = (n +simulations_per_thread) / simulations_per_thread;\
	int blocksPerGrid = (total_threads + (threads_per_block[currentDevice]) - 1) / (threads_per_block[currentDevice]);


/******************************
 *          normal
 ******************************/
template <typename T>
__global__ void kernel_rnorm(curandState* state, int sims_per_thread,
		T* param1, T* param2, int n1, int n2, int n, T* ret)
{
	int id =blockDim.x * blockIdx.x  + threadIdx.x;
	curandState localState = state[id];

	int mystop = blockDim.x * (blockIdx.x+1) * sims_per_thread;
	for ( int i = (blockDim.x*blockIdx.x*sims_per_thread +threadIdx.x);i<mystop;i+=blockDim.x) {
		/* Generate pseudorandom  numbers*/
		if(i<n)
			ret[i]=NRM*param2[i % n2] + param1[i % n1];
	}

	/* Copy state back to global memory */
	state[id] = localState;
}



SEXP gpu_rnorm(SEXP in_n, SEXP in_mean, SEXP in_sd, SEXP in_n_mean, SEXP in_n_sd, SEXP in_type)
{

    struct gpuvec *ret = Calloc(1, struct gpuvec);
	struct gpuvec *mean = (struct gpuvec*) R_ExternalPtrAddr(in_mean);
	struct gpuvec *sd = (struct gpuvec*) R_ExternalPtrAddr(in_sd);
	int n_mean = INTEGER(in_n_mean)[0];
	int n_sd = INTEGER(in_n_sd)[0];
	int n=INTEGER(in_n)[0];
	DECERROR1;
	PROCESS_TYPE_SF;
	//allocate
	CUDA_MALLOC(ret->d_vec,n * mysizeof);

	SETUP_RAND;

	#define KERNAL(PTR,T)\
	kernel_rnorm< T><<<blocksPerGrid, (threads_per_block[currentDevice])>>>((dev_states[currentDevice]), simulations_per_thread,\
			PTR(mean),PTR(sd), n_mean, n_sd, n, PTR(ret));
	CALL_KERNAL_SF;
	#undef KERNAL

	CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);

    SEXP ret_final = gpu_register(ret);
    return ret_final;

}




#define MY_Z (x[i] - param1[i % n1])/sd


template <typename T>
__global__ void kernel_dnorm(T* x, T* param1, T* param2, int n1, int n2, int n,
		T* ret, int log1, int operations_per_thread)
{

	T z;
	int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;
	for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {

		if(i < n) {
			T sd=param2[i % n2];
			if(sd<0)
				ret[i]=NAN;
			else if(log1==0) {
				z= MY_Z;
				ret[i]=exp(-(M_LN_SQRT_2PI  + log(sd) + z*z/2) );
			} else {
				z= MY_Z;
				ret[i]=-(M_LN_SQRT_2PI  + log(sd) + z*z/2);
			}
		}
	}
}

SEXP gpu_dnorm(SEXP in_n, SEXP in_x, SEXP in_mean, SEXP in_sd, SEXP in_n_mean, SEXP in_n_sd,
		SEXP in_log, SEXP in_type)
{

    struct gpuvec *ret = Calloc(1, struct gpuvec);
	struct gpuvec *mean = (struct gpuvec*) R_ExternalPtrAddr(in_mean);
	struct gpuvec *sd = (struct gpuvec*) R_ExternalPtrAddr(in_sd);
	struct gpuvec *x = (struct gpuvec*) R_ExternalPtrAddr(in_x);
	int n_mean = INTEGER(in_n_mean)[0];
	int n_sd = INTEGER(in_n_sd)[0];
	int n=INTEGER(in_n)[0];
	DECERROR1;
	int log = LOGICAL(in_log)[0];
	PROCESS_TYPE;
	//allocate
	CUDA_MALLOC(ret->d_vec,n * mysizeof);

	//sims per thread must be even and two or greater
	GET_BLOCKS_PER_GRID(n);
	#define KERNAL(PTR,T)\
	kernel_dnorm< T><<<blocksPerGrid, (threads_per_block[currentDevice])>>>(PTR(x),\
			PTR(mean),PTR(sd), n_mean, n_sd, n, PTR(ret), log, operations_per_thread);

	CALL_KERNAL_SF;
	#undef KERNAL
	CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);

    SEXP ret_final = gpu_register(ret);
    return ret_final;
}

template <typename T>
__global__ void kernel_pnorm(T* x, T* param1, T* param2, int n1, int n2, int n,
		T* ret, int lg, int low, int operations_per_thread)
{

	T tmp;
	int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;
	for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {
		if(i < n){
			T sd=param2[i % n2];
			if(sd<0)
				ret[i]=NAN;
			else{
				if(low!=0)
					tmp=.5*(1+erf(MY_Z*M_SQRT1_2));
				else
					tmp=.5*(1+erf(-MY_Z*M_SQRT1_2));
				if(lg==0)
					ret[i]=tmp;
				else
					ret[i]=log(tmp);
			}
		}
	}

}




SEXP gpu_pnorm(SEXP in_n, SEXP in_x, SEXP in_mean, SEXP in_sd, SEXP in_n_mean, SEXP in_n_sd,
		SEXP in_log, SEXP in_lower, SEXP in_type)
{

    struct gpuvec *ret = Calloc(1, struct gpuvec);
	struct gpuvec *mean = (struct gpuvec*) R_ExternalPtrAddr(in_mean);
	struct gpuvec *sd = (struct gpuvec*) R_ExternalPtrAddr(in_sd);
	struct gpuvec *x = (struct gpuvec*) R_ExternalPtrAddr(in_x);
	int n_mean = INTEGER(in_n_mean)[0];
	int n_sd = INTEGER(in_n_sd)[0];
	int n=INTEGER(in_n)[0];
	DECERROR1;
	int log = LOGICAL(in_log)[0];
	int low =LOGICAL(in_lower)[0];

	//allocate
	PROCESS_TYPE;
	CUDA_MALLOC(ret->d_vec,n *mysizeof);

	//sims per thread must be even and two or greater
	GET_BLOCKS_PER_GRID(n);
	#define KERNAL(PTR,T)\
	kernel_pnorm< T><<<blocksPerGrid, (threads_per_block[currentDevice])>>>(PTR(x),\
			PTR(mean),PTR(sd), n_mean, n_sd, n, PTR(ret), log, low, operations_per_thread);
	CALL_KERNAL_SF;
	#undef KERNAL
	CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);
    SEXP ret_final = gpu_register(ret);
    return ret_final;
}



template <typename T>
__global__ void kernel_qnorm(T* x, T* param1, T* param2, int n1, int n2, int n,
		T* ret, int lg, int low, int operations_per_thread)
{

	T tmp;
	int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;
	for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {
		if(i < n){
			if(lg==0)
				tmp=x[i];
			else
				tmp=exp(x[i]);
			tmp=M_SQRT2*erfinv(2*tmp-1);

			if(low==0)
				tmp=-1*tmp;
			T sd=param2[i % n2];
			if(sd>=0)
				ret[i]=tmp*sd + param1[i % n1];
			else
				ret[i]=NAN;

		}
	}
}




SEXP gpu_qnorm(SEXP in_n, SEXP in_x, SEXP in_mean, SEXP in_sd, SEXP in_n_mean, SEXP in_n_sd,
		SEXP in_log, SEXP in_lower, SEXP in_type)
{

    struct gpuvec *ret = Calloc(1, struct gpuvec);
	struct gpuvec *mean = (struct gpuvec*) R_ExternalPtrAddr(in_mean);
	struct gpuvec *sd = (struct gpuvec*) R_ExternalPtrAddr(in_sd);
	struct gpuvec *x = (struct gpuvec*) R_ExternalPtrAddr(in_x);
	int n_mean = INTEGER(in_n_mean)[0];
	int n_sd = INTEGER(in_n_sd)[0];
	int n=INTEGER(in_n)[0];
	DECERROR1;
	int log = LOGICAL(in_log)[0];
	int low = LOGICAL(in_lower)[0];

	//allocate
		PROCESS_TYPE;
	CUDA_MALLOC(ret->d_vec,n *mysizeof);

	//sims per thread must be even and two or greater
	GET_BLOCKS_PER_GRID(n);
	#define KERNAL(PTR,T)\
	kernel_qnorm< T><<<blocksPerGrid, (threads_per_block[currentDevice])>>>(PTR(x),\
			PTR(mean),PTR(sd), n_mean, n_sd, n, PTR(ret), log, low, operations_per_thread);
	CALL_KERNAL_SF;
	#undef KERNAL
	CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);
    SEXP ret_final = gpu_register(ret);
    return ret_final;
}

/******************************
 *          uniform
 ******************************/
template <typename T>
__global__ void kernel_dunif(T* x, T* param1, T* param2, int n1, int n2, int n,
		T* ret, int lg, int operations_per_thread)
{
	int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;
	for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {
		if(i < n) {
			T mymin=param1[i % n1];
			T mymax=param2[i % n2];
			if(mymin>=mymax){
				ret[i]=NAN;
			} else if(lg==0) {
				if(mymin <= x[i] && x[i] <= mymax )
					ret[i]=1/(mymax - mymin);
				else
					ret[i]=0;
			} else {
				if(mymin <= x[i] && x[i] <= mymax )
					ret[i]=log(1/(mymax - mymin));
				else
					ret[i]=log(0.0);
			}
		}
	}
}




SEXP gpu_dunif(SEXP in_n, SEXP in_x, SEXP in_min, SEXP in_max, SEXP in_n_min, SEXP in_n_max,
		SEXP in_log, SEXP in_type)
{

    struct gpuvec *ret = Calloc(1, struct gpuvec);
	struct gpuvec *min = (struct gpuvec*) R_ExternalPtrAddr(in_min);
	struct gpuvec *max = (struct gpuvec*) R_ExternalPtrAddr(in_max);
	struct gpuvec *x = (struct gpuvec*) R_ExternalPtrAddr(in_x);
	int n_min = INTEGER(in_n_min)[0];
	int n_max = INTEGER(in_n_max)[0];
	int n=INTEGER(in_n)[0];
	int log = LOGICAL(in_log)[0];

	DECERROR1;

	//allocate
		PROCESS_TYPE;
	CUDA_MALLOC(ret->d_vec,n *mysizeof);

	//sims per thread must be even and two or greater
	GET_BLOCKS_PER_GRID(n);
	#define KERNAL(PTR,T)\
	kernel_dunif< T><<<blocksPerGrid, (threads_per_block[currentDevice])>>>(PTR(x),\
			PTR(min),PTR(max), n_min, n_max, n, PTR(ret), log, operations_per_thread);
	CALL_KERNAL_SF;
	#undef KERNAL
	CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);
    SEXP ret_final = gpu_register(ret);
    return ret_final;

}

template <typename T>
__global__ void kernel_runif(curandState* state, int sims_per_thread,
		T* param1, T* param2, int n1, int n2, int n, T* ret)
{
	int id =blockDim.x * blockIdx.x  + threadIdx.x;
	curandState localState = state[id];

	int mystop = blockDim.x * (blockIdx.x+1) * sims_per_thread;
	for ( int i = blockDim.x * blockIdx.x * sims_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {
		T mymin=param1[i % n1];
		//printf("i = %d, p1 = %d, p2 = %d\n", i,param1[i % n1], param2[i % n2] );
		if(i<n)
			ret[i]= mymin + UNI*(param2[i % n2]-mymin);
	}
	/* Copy state back to global memory */
	state[id] = localState;
}


SEXP gpu_runif(SEXP in_n, SEXP in_min, SEXP in_max, SEXP in_n_min, SEXP in_n_max, SEXP in_type)
{

    struct gpuvec *ret = Calloc(1, struct gpuvec);
	struct gpuvec *min = (struct gpuvec*) R_ExternalPtrAddr(in_min);
	struct gpuvec *max = (struct gpuvec*) R_ExternalPtrAddr(in_max);
	int n_min = INTEGER(in_n_min)[0];
	int n_max = INTEGER(in_n_max)[0];
	int n=INTEGER(in_n)[0];
	DECERROR1;

	//allocate
		PROCESS_TYPE;
	CUDA_MALLOC(ret->d_vec,n *mysizeof);


	SETUP_RAND;

#ifdef DEBUG
	Rprintf("simulations_per_thread=%d, blocksPerGrid=%d, n=%d, n_mean=%d, n_sd=%d",
			simulations_per_thread,blocksPerGrid,n,n_min,n_max);
#endif

	#define KERNAL(PTR,T)\
	kernel_runif< T><<<blocksPerGrid, (threads_per_block[currentDevice])>>>((dev_states[currentDevice]), simulations_per_thread,\
			PTR(min),PTR(max), n_min, n_max, n, PTR(ret));

	CALL_KERNAL_SF;
	#undef KERNAL
	CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);
    SEXP ret_final = gpu_register(ret);
    return ret_final;

}

/******************************
 *          gamma
 ******************************/


/* use MARSAGLIA and TSANG (2000)
 * A Simple Method for Generating Gamma Variables
 * Algorithm from paper:
 * (1) Setup: d=a-1/3, c=1/sqrt(9d).
 * (2) Generate: v=(1+c*x)^3, with x standard normal.
 * (3) if v>0 and log(UNI) < 0.5*x^2+d-d*v+d*log(v) return d*v.
 * (4) go back to step 2.
 */
template <typename T>
__device__ T rgamma(curandState *state, T shape, T scale) {
	T U, x, d, c, v, alpha,less_than_one_correction; //gamma generation stuff
	__syncthreads();
	less_than_one_correction=1.0;
	//T one=1.0;
	alpha=shape;
	if(alpha<1) {
		less_than_one_correction=
				pow(UNI2, 1/alpha);
		alpha=alpha+1.0;
		if(alpha<1){ //make sure that the initial alpha was greater than 0
			alpha=10;
			less_than_one_correction=NAN;
		}
	}
	__syncthreads();


	d=alpha-1.0/3.0;
	c=1/sqrt(9.0*d);
	for(;;) {
		do {
			// Generate normals
			x=NRM2;
			v=1.0+c*x;
			v=v*v*v;
		} while(v<=0.0);
		__syncthreads();
		U= (0.5*x*x) + d*(1. - v + log(v));//reuse U to save register space
		if(log(UNI2) < U) {
			//if(i<n) {
			U=d*v*scale*less_than_one_correction; //reuse U to save register space
			if(U<0) { //make sure that no negative values were passed in.
				U=NAN;//return(NAN);//ret[0]=NAN;
			} else {
				//return(U);//ret[0]=U;}
				//}
				break;
			}
		}
	}
	return(U);
}

template <typename T>
__global__ void kernel_rgamma(curandState* state, int sims_per_thread,
		T* param1, T* param2, int n1, int n2, int n, T* ret)
{
	int id =blockDim.x * blockIdx.x  + threadIdx.x;
	curandState localState = state[id];

	int mystop = blockDim.x * (blockIdx.x+1) * sims_per_thread;
	for ( int i = blockDim.x * blockIdx.x * sims_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {
		if(i<n) {
			ret[i]=rgamma<T>(&localState, param1[i % n1], param2[i % n2]);
		}
	}
	/* Copy state back to global memory */
	state[id] = localState;
}



SEXP gpu_rgamma(SEXP in_n, SEXP in_alpha, SEXP in_scale, SEXP in_n_alpha, SEXP in_n_scale, SEXP in_type)
{

    struct gpuvec *ret = Calloc(1, struct gpuvec);
	struct gpuvec *alpha = (struct gpuvec*) R_ExternalPtrAddr(in_alpha);
	struct gpuvec *scale = (struct gpuvec*) R_ExternalPtrAddr(in_scale);
	int n_alpha = INTEGER(in_n_alpha)[0];
	int n_scale = INTEGER(in_n_scale)[0];
	int n=INTEGER(in_n)[0];
	DECERROR1;

	//allocate
		PROCESS_TYPE;
	CUDA_MALLOC(ret->d_vec,n *mysizeof);

	 SETUP_RAND;
#ifdef DEBUG
	Rprintf("simulations_per_thread=%d, blocksPerGrid=%d, n=%d, n_mean=%d, n_sd=%d",
			blocksPerGrid,n,n_alpha,n_scale);
#endif
	#define KERNAL(PTR,T)\
	kernel_rgamma< T><<<blocksPerGrid, (threads_per_block[currentDevice])>>>((dev_states[currentDevice]), simulations_per_thread,\
			PTR(alpha),PTR(scale), n_alpha, n_scale, n, PTR(ret));
	CALL_KERNAL_SF;
	#undef KERNAL
	CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);
    SEXP ret_final = gpu_register(ret);
    return ret_final;

}

template <typename T>
__global__ void kernel_dgamma(T* x, T* param1, T* param2, int n1, int n2, int n,
		T* ret, int lg, int operations_per_thread)
{
	int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;
	for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {
		if(i < n) {
			T shape=param1[i % n1];
			T scale=param2[i % n2];
			T myx=x[i];
			if(scale<0 || shape<0)
				ret[i]=NAN;
			else if(lg==0) {
				if( myx >= 0 )
					ret[i]=exp( -shape*log(scale) - lgamma(shape) +  (shape-1)*log(myx) - myx/scale );
				else
					ret[i]=0;
			} else {
				if( myx >= 0 )
					ret[i]= -shape*log(scale) - lgamma(shape) +  (shape-1)*log(myx) - myx/scale ;
				else
					ret[i]=log(0.0);
			}
		}
	}
}



SEXP gpu_dgamma(SEXP in_n, SEXP in_x, SEXP in_parm1, SEXP in_parm2, SEXP in_n_parm1, SEXP in_n_parm2,
		SEXP in_log, SEXP in_type)
{

    struct gpuvec *ret = Calloc(1, struct gpuvec);
	struct gpuvec *parm1 = (struct gpuvec*) R_ExternalPtrAddr(in_parm1);
	struct gpuvec *parm2 = (struct gpuvec*) R_ExternalPtrAddr(in_parm2);
	struct gpuvec *x = (struct gpuvec*) R_ExternalPtrAddr(in_x);
	int n_parm1 = INTEGER(in_n_parm1)[0];
	int n_parm2 = INTEGER(in_n_parm2)[0];
	int n=INTEGER(in_n)[0];
	int log = LOGICAL(in_log)[0];
	DECERROR1;

	//allocate

	PROCESS_TYPE;
	CUDA_MALLOC(ret->d_vec,n *mysizeof);
	//sims per thread must be even and two or greater
	GET_BLOCKS_PER_GRID(n);
	#define KERNAL(PTR,T)\
	kernel_dgamma< T><<<blocksPerGrid, (threads_per_block[currentDevice])>>>(PTR(x),\
			PTR(parm1),PTR(parm2), n_parm1, n_parm2, n, PTR(ret), log, operations_per_thread);
	CALL_KERNAL_SF;
	#undef KERNAL
	CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);
	SEXP ret_final = gpu_register(ret);
	return ret_final;

}
/******************************
 *          beta
 ******************************/
template <typename T>
__device__ T rbeta(curandState *state, T alpha, T beta)
{
	T z1=rgamma<T>(state,alpha, 1);
	T z2=rgamma<T>(state,beta, 1);
	return (z1/(z1+z2));
}

template <typename T>
__global__ void kernel_rbeta(curandState* state, int sims_per_thread,
		T* param1, T* param2, int n1, int n2, int n, T* ret)
{
	int id =blockDim.x * blockIdx.x  + threadIdx.x;
	curandState localState = state[id];

	int mystop = blockDim.x * (blockIdx.x+1) * sims_per_thread;
	for ( int i = blockDim.x * blockIdx.x * sims_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {
		if(i<n) {
			ret[i]=rbeta<T>(&localState, param1[i % n1], param2[i % n2]);
		}
	}
	/* Copy state back to global memory */
	state[id] = localState;
}



SEXP gpu_rbeta(SEXP in_n, SEXP in_alpha, SEXP in_scale, SEXP in_n_alpha, SEXP in_n_scale, SEXP in_type)
{

    struct gpuvec *ret = Calloc(1, struct gpuvec);
	struct gpuvec *alpha = (struct gpuvec*) R_ExternalPtrAddr(in_alpha);
	struct gpuvec *scale = (struct gpuvec*) R_ExternalPtrAddr(in_scale);
	int n_alpha = INTEGER(in_n_scale)[0];
	int n_scale = INTEGER(in_n_scale)[0];
	int n=INTEGER(in_n)[0];
	DECERROR1;

	//allocate
		PROCESS_TYPE;
	CUDA_MALLOC(ret->d_vec,n *mysizeof);

	 SETUP_RAND
#ifdef DEBUG
	Rprintf("simulations_per_thread=%d, blocksPerGrid=%d, n=%d, n_mean=%d, n_sd=%d",
			blocksPerGrid,n,n_alpha,n_scale);
#endif
	#define KERNAL(PTR,T)\
	kernel_rbeta< T><<<blocksPerGrid, (threads_per_block[currentDevice])>>>((dev_states[currentDevice]), simulations_per_thread,\
			PTR(alpha),PTR(scale), n_alpha, n_scale, n, PTR(ret));

	CALL_KERNAL_SF;
	#undef KERNAL
	CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);


    SEXP ret_final = gpu_register(ret);
    return ret_final;

}

template <typename T>
__global__ void kernel_dbeta(T* x, T* param1, T* param2, int n1, int n2, int n,
		T* ret, int lg, int operations_per_thread)
{

	T tmp;
	int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;
	for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {

		T alpha=param1[i % n1];
		T beta=param2[i % n2];
		T myx=x[i];
		if(alpha<=0 || beta<=0)
			tmp=NAN;
		else if(lg==0) {
			if(myx==0) {
				if(alpha<1)
					tmp = -log(0.0);
				else if(alpha==1)
					tmp = exp( lgamma(alpha+ beta) -lgamma(alpha) - lgamma(beta));
				else
					tmp = 0;

			} else if(myx==1) {
				if(beta<1)
					tmp = -log(0.0);
				else if(beta==1)
					tmp = exp( lgamma(alpha+ beta) -lgamma(alpha) - lgamma(beta));
				else
					tmp = 0;
			} else if( myx > 0 && myx<1 )
				tmp=exp( lgamma(alpha+ beta) -lgamma(alpha) - lgamma(beta) + (alpha-1)*log(myx) + (beta-1)*log(1-myx) );
			else
				tmp=0;
		} else {
			if(myx==0) {
				if(alpha<1)
					tmp = -log(0.0);
				else if(alpha==1)
					tmp = ( lgamma(alpha+ beta) -lgamma(alpha) - lgamma(beta));
				else
					tmp = log(0.0);

			} else if(myx==1) {
				if(beta<1)
					tmp = -log(0.0);
				else if(beta==1)
					tmp = ( lgamma(alpha+ beta) -lgamma(alpha) - lgamma(beta));
				else
					tmp = log(0.0);
			} else if( myx > 0 && myx<1 )
				tmp=lgamma(alpha+ beta) -lgamma(alpha) - lgamma(beta) + (alpha-1)*log(myx) + (beta-1)*log(1-myx);
			else
				tmp=log(0.0);
		}
		__syncthreads();
		if(i < n) {
			ret[i]=tmp;
		}
	}
}




SEXP gpu_dbeta(SEXP in_n, SEXP in_x, SEXP in_parm1, SEXP in_parm2, SEXP in_n_parm1, SEXP in_n_parm2,
		SEXP in_log, SEXP in_type)
{

    struct gpuvec *ret = Calloc(1, struct gpuvec);
	struct gpuvec *parm1 = (struct gpuvec*) R_ExternalPtrAddr(in_parm1);
	struct gpuvec *parm2 = (struct gpuvec*) R_ExternalPtrAddr(in_parm2);
	struct gpuvec *x = (struct gpuvec*) R_ExternalPtrAddr(in_x);
	int n_parm1 = INTEGER(in_n_parm1)[0];
	int n_parm2 = INTEGER(in_n_parm2)[0];
	int n=INTEGER(in_n)[0];
	int log = LOGICAL(in_log)[0];
	DECERROR1;

	//allocate
		PROCESS_TYPE;
	CUDA_MALLOC(ret->d_vec,n *mysizeof);

	//sims per thread must be even and two or greater
	GET_BLOCKS_PER_GRID(n);
	#define KERNAL(PTR,T)\
	kernel_dbeta< T><<<blocksPerGrid, (threads_per_block[currentDevice])>>>(PTR(x),\
						PTR(parm1),PTR(parm2), n_parm1, n_parm2, n, PTR(ret), log, operations_per_thread);
	CALL_KERNAL_SF;
	#undef KERNAL
	CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);
    SEXP ret_final = gpu_register(ret);
    return ret_final;

}



/******************************
 *          binomial
 ******************************/



template <typename T>
__device__ T lchoose( T n,  T m)
{
	T ret;
	if(m > n) {
		ret=(log(0.0));
	}  else if(n<0 ||m == n || m == 0 ) {
		ret=(0.0);
	}  else {
		ret=(lgamma(n+1) -lgamma(m+1)-lgamma(n-m + 1));
	}
	return ret;
}

/* modified from the GSL library which uses the Knuth method
 */
template <typename T>
__device__ int
 rbinom(curandState *state, int n,  T p)
{
  int i, a, b, k = 0;

  while (n > 10)
    {
      T X;
      a = 1 + (n / 2);
      b = 1 + n - a;

      X = rbeta<T>(state, (T) a, (T) b);

      if (X >= p)
        {
          n = a - 1;
          p /= X;
        }
      else
        {
          k += a;
          n = b - 1;
          p = (p - X) / (1 - X);
        }
    }

  for (i = 0; i < n; i++)
    {
      T u = UNI2;
      if (u < p)
        k++;
    }
 // printf("p = %f, n = %d, k = %d \n", p, n, k);
  return k;
}

template <typename T>
__global__ void kernel_dbinom(T* x, T* param1, T* param2, int n1, int n2, int n,
		T* ret, int lg, int operations_per_thread)
{
	int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;
	for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {
		if(i < n) {
			T N=param1[i % n1];
			T p=param2[i % n2];
			T myx=x[i];
			if(N<=0 || p<0 || p>1)
				ret[i]=NAN;
			else if(lg==0) {
				if(myx>=0 && myx<=N && floor(myx)==myx )
					ret[i]=exp( lchoose<T>( N, myx) + myx*log(p)+ (N-myx)*log(1-p) );
				else
					ret[i]=0;
			} else {
				if(myx>=0 && myx<=N && floor(myx)==myx)
					ret[i]=lchoose<T>( N, myx) + myx*log(p)+ (N-myx)*log(1-p);
				else
					ret[i]=log(0.0);
			}
		}
	}
}



SEXP gpu_dbinom(SEXP in_n, SEXP in_x, SEXP in_parm1, SEXP in_parm2, SEXP in_n_parm1, SEXP in_n_parm2,
		SEXP in_log, SEXP in_type)
{

    struct gpuvec *ret = Calloc(1, struct gpuvec);
	struct gpuvec *parm1 = (struct gpuvec*) R_ExternalPtrAddr(in_parm1);
	struct gpuvec *parm2 = (struct gpuvec*) R_ExternalPtrAddr(in_parm2);
	struct gpuvec *x = (struct gpuvec*) R_ExternalPtrAddr(in_x);
	int n_parm1 = INTEGER(in_n_parm1)[0];
	int n_parm2 = INTEGER(in_n_parm2)[0];
	int n=INTEGER(in_n)[0];
	int log = LOGICAL(in_log)[0];
	DECERROR1;

	//allocate
	PROCESS_TYPE;
	CUDA_MALLOC(ret->d_vec,n *mysizeof);

	//sims per thread must be even and two or greater
	GET_BLOCKS_PER_GRID(n);
	#define KERNAL(PTR,T)\
	kernel_dbinom< T><<<blocksPerGrid, (threads_per_block[currentDevice])>>>(PTR(x),\
						PTR(parm1),PTR(parm2), n_parm1, n_parm2, n, PTR(ret), log, operations_per_thread);
	CALL_KERNAL_SF;
	#undef KERNAL
	CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);
    SEXP ret_final = gpu_register(ret);
    return ret_final;

}

template <typename T>
__global__ void kernel_rbinom(curandState* state, int sims_per_thread,
		T* param1, T* param2, int n1, int n2, int n, int* ret)
{
	int id =blockDim.x * blockIdx.x  + threadIdx.x;
	curandState localState = state[id];

	int mystop = blockDim.x * (blockIdx.x+1) * sims_per_thread;
	for ( int i = blockDim.x * blockIdx.x * sims_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {
		if(i<n) {
			int N = (int) param1[i % n1];
			T p = param2[i % n2];
			//printf("%d %d %f \n", p, (int) param1[i % n1], param2[i % n2]);
			if(N<0 || p<0 || p>1)
				ret[i]=INT_MIN;
			else
				ret[i]= rbinom<T>(&localState, (int) param1[i % n1], param2[i % n2]);
		}
	}
	/* Copy state back to global memory */
	state[id] = localState;
}



SEXP gpu_rbinom(SEXP in_n, SEXP in_parm1, SEXP in_parm2, SEXP in_n_parm1, SEXP in_n_parm2, SEXP in_type)
{

    struct gpuvec *ret = Calloc(1, struct gpuvec);
	struct gpuvec *parm1 = (struct gpuvec*) R_ExternalPtrAddr(in_parm1);
	struct gpuvec *parm2 = (struct gpuvec*) R_ExternalPtrAddr(in_parm2);
	int n_parm1 = INTEGER(in_n_parm1)[0];
	int n_parm2 = INTEGER(in_n_parm2)[0];
	int n=INTEGER(in_n)[0];
	DECERROR1;

	//allocate
	PROCESS_TYPE_NO_SIZE;
	CUDA_MALLOC(ret->d_vec,n *sizeof(int));

	SETUP_RAND;
#ifdef DEBUG
	Rprintf("simulations_per_thread=%d, blocksPerGrid=%d, n=%d, n_parm1=%d, n_parm2=%d",
			simulations_per_thread,blocksPerGrid,n,n_parm1,n_parm2);
#endif
		#define KERNAL(PTR,T)\
	kernel_rbinom< T><<<blocksPerGrid, (threads_per_block[currentDevice])>>>((dev_states[currentDevice]), simulations_per_thread,\
			PTR(parm1),PTR(parm2), n_parm1, n_parm2, n, (int *) ret->d_vec);
	CALL_KERNAL_SF;
	#undef KERNAL
	CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);
    SEXP ret_final = gpu_register(ret);
    return ret_final;

}








/******************************
 *          poisson
 ******************************/



/* modified from the GSL library which uses the Knuth method
 */
template <typename T>
__device__ int
rpois(curandState *state,  T mu)
{
        T emu;
        T prod = 1.0;
        int k = 0;

        while (mu > 10.0)
        {
                int m = mu * (7.0 / 8.0);

                T X = rgamma<T>(state, (T) m, 1.0);//gsl_ran_gamma_int (r, m);

                if (X >= mu) {
                        return k + rbinom<T>(state, m -1 , mu/X);//gsl_ran_binomial (r, mu / X, m - 1);
                } else {
                        k += m;
                        mu -= X;
                }
        }

        /* This following method works well when mu is small */
        __syncthreads();//little scared of this but...
        emu = exp (-mu);

        do
        {
                prod *=  UNI2; //gsl_rng_uniform (r);
                k++;
        }
        while (prod > emu);

        return k - 1;

}




template <typename T>
__global__ void kernel_dpois(T* x, T* param1,  int n1, int n,
                T* ret, int lg, int operations_per_thread)
{
	int mystop = blockDim.x * (blockIdx.x+1) * operations_per_thread;
	for ( int i = blockDim.x * blockIdx.x * operations_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {
                if(i < n) {
                        T mu=param1[i % n1];
                        T myx=x[i];
                        if( mu<0)
                                ret[i]=NAN;
                        else if(lg==0) {
                                if(myx>=0 && floor(myx)==myx)
                                        ret[i]=exp(-mu - lgamma(myx+1) + myx*log(mu));
                                else
                                        ret[i]=0;
                        } else {
                                if(myx>=0 && floor(myx)==myx)
                                        ret[i]=-mu - lgamma(myx+1) + myx*log(mu);
                                else
                                        ret[i]=log(0.0);
                        }
                }
        }
}


SEXP gpu_dpois(SEXP in_n, SEXP in_x, SEXP in_parm1, SEXP in_n_parm1,
		SEXP in_log, SEXP in_type)
{

	struct gpuvec *ret = Calloc(1, struct gpuvec);
	struct gpuvec *parm1 = (struct gpuvec*) R_ExternalPtrAddr(in_parm1);
	struct gpuvec *x = (struct gpuvec*) R_ExternalPtrAddr(in_x);
	int n_parm1 = INTEGER(in_n_parm1)[0];
	int n=INTEGER(in_n)[0];
	int log = LOGICAL(in_log)[0];
	DECERROR1;

	//allocate
		PROCESS_TYPE;
	CUDA_MALLOC(ret->d_vec,n *mysizeof);

	//sims per thread must be even and two or greater
	GET_BLOCKS_PER_GRID(n);
	#define KERNAL(PTR,T)\
	kernel_dpois< T><<<blocksPerGrid, (threads_per_block[currentDevice])>>>(PTR(x),\
			PTR(parm1), n_parm1,  n, PTR(ret), log, operations_per_thread);
	cudaStat = cudaDeviceSynchronize();
	CALL_KERNAL_SF;
	#undef KERNAL
	CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);
	SEXP ret_final = gpu_register(ret);
	return ret_final;

}


template <typename T>
__global__ void kernel_rpois(curandState* state, int sims_per_thread,
		T* param1,  int n1,  int n, int* ret)
{
	int id =blockDim.x * blockIdx.x  + threadIdx.x;
	curandState localState = state[id];

	int mystop = blockDim.x * (blockIdx.x+1) * sims_per_thread;
	for ( int i = blockDim.x * blockIdx.x * sims_per_thread  + threadIdx.x;
			i < mystop; i+=blockDim.x) {
		if(i<n) {

			T mu= param1[i % n1];
			if(mu<0)
				ret[i]=INT_MIN;
			else
				ret[i]= rpois<T>(&localState, param1[i % n1]);
		}
	}
	/* Copy state back to global memory */
	state[id] = localState;
}



SEXP gpu_rpois(SEXP in_n, SEXP in_parm1,  SEXP in_n_parm1, SEXP in_type)
{

	struct gpuvec *ret = Calloc(1, struct gpuvec);
	struct gpuvec *parm1 = (struct gpuvec*) R_ExternalPtrAddr(in_parm1);
	int n_parm1 = INTEGER(in_n_parm1)[0];
	int n=INTEGER(in_n)[0];
	DECERROR1;

	//allocate
	PROCESS_TYPE_NO_SIZE;
	CUDA_MALLOC(ret->d_vec,n *sizeof(int));


	SETUP_RAND;
#ifdef DEBUG
	Rprintf("simulations_per_thread=%d, blocksPerGrid=%d, n=%d, n_parm1=%d, n_parm2=%d",
			simulations_per_thread,blocksPerGrid,n,n_parm1,n_parm2);
#endif
	#define KERNAL(PTR,T)\
	kernel_rpois< T><<<blocksPerGrid, (threads_per_block[currentDevice])>>>((dev_states[currentDevice]), simulations_per_thread,\
			PTR(parm1), n_parm1, n, (int *) ret->d_vec);
	CALL_KERNAL_SF;
	#undef KERNAL
	CUDA_CHECK_KERNAL_CLEAN_1(ret->d_vec);
	SEXP ret_final = gpu_register(ret);
	return ret_final;

}
