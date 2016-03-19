# This a a modified version of the 'configure.ac' from
# the 'magma' R package by Brian Smith which is licensed by
# under GNU GPL 3 or later.

AC_INIT([gmatrix], 0.3.0)
AC_CONFIG_AUX_DIR([tools])

#################### GCC Compiler ####################

AC_PROG_CC(gcc)

#################### CUDA Toolkit ####################
: ${CUDA_TMP=`which nvcc`}
AC_ARG_WITH([cuda-home],
	[AC_HELP_STRING([--with-cuda-home=DIR],
		[full path to CUDA installation @<:@which nvcc@:>@])],
	[CUDA_HOME="$withval"], [CUDA_HOME=${CUDA_TMP%%nvcc}..])
AC_MSG_RESULT([setting CUDA home directory to ${CUDA_HOME}])

NVCC=${CUDA_HOME}/bin/nvcc
AC_CHECK_FILE([${NVCC}],,
	[AC_MSG_ERROR([nvcc not found!])])

AC_ARG_WITH([arch],
	[AC_HELP_STRING([--with-arch=SM],
		[compute capability of the target device @<:@sm_20@:>@])],
	[ARCH="$withval"], [ARCH=sm_20])
AC_MSG_RESULT([setting compute capability of the target device to ${ARCH}])


NCFLAGS="-arch=${ARCH} -O2 -DADD_"
CUDA_INC=${CUDA_HOME}/include
AC_MSG_CHECKING([for CUDA headers])
AC_CHECK_FILE([${CUDA_INC}/cublas_v2.h],
	[NCFLAGS="${NCFLAGS} -I${CUDA_INC}"],
	[AC_MSG_ERROR([CUDA headers not found!])]
)

AC_MSG_CHECKING([whether this is a 64-bit version of CUDA])
AC_CHECK_FILE([${CUDA_HOME}/lib64/libcublas.so], [check_bit=64-bit])
if test "${check_bit}" == "64-bit"; then
	LDFLAGS="-shared -L${CUDA_HOME}/lib64"
else
	LDFLAGS="-shared -L${CUDA_HOME}/lib"
fi

AC_MSG_CHECKING([whether cusolver is used])
AC_CHECK_FILE([${CUDA_INC}/cusolverDn.h], [check_cusolve=T])
if test "${check_cusolve}" == "T"; then
	LIBS="-lcudart -lcublas -lcusolver"
else
	LIBS="-lcudart -lcublas"
fi




#################### R Software ####################

: ${R_HOME=`R RHOME`}
AC_ARG_WITH([r-home],
	[AC_HELP_STRING([--with-r-home=DIR],
		[full path to R installation @<:@R RHOME@:>@])],
	[R_HOME=$withval]
)
AC_MSG_RESULT([setting R home directory to ${R_HOME}])
AC_CHECK_PROG([check_R], [R], [yes],, [${R_HOME}/bin])
if test -z "${check_R}"; then
	AC_MSG_ERROR([R not found!])
fi 

R_INCLUDE_DIR=$(R --slave --no-save -e "cat(R.home('include'))")
AC_ARG_WITH([r-include],
	[AC_HELP_STRING([--with-r-include=DIR],
		[full path to R include files @<:@R --slave --no-save -e "cat(R.home('include'))"@:>@])],
	[R_INCLUDE_DIR=$withval]
)
AC_MSG_RESULT([setting R header files to ${R_INCLUDE_DIR}])

R_CPICFLAGS=`"${R_HOME}/bin/R" CMD config CPICFLAGS`
NCFLAGS="${NCFLAGS} -Xcompiler \"-O2 -DADD_ ${R_CPICFLAGS} -I${R_INCLUDE_DIR}\""
NCFLAGS="${NCFLAGS//[$'\t\r\n ']}"
		
AC_CANONICAL_HOST
# Check for which host we are on and setup a few things
case $host_os in
  darwin* )
    TMPLDFLAGS=`R CMD config --ldflags`
    LDFLAGS="${LDFLAGS} -Xlinker \"${TMPLDFLAGS}\""
        ;;
esac

AC_SUBST([NVCC])
AC_SUBST([NCFLAGS])
AC_SUBST([LDFLAGS])
AC_SUBST([LIBS])

AC_CONFIG_FILES([src/Makefile])
AC_OUTPUT