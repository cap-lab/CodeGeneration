/*
 * uem_common.h
 *
 *  Created on: 2017. 8. 5.
 *      Author: jej
 */

#ifndef SRC_COMMON_INCLUDE_UEM_COMMON_H_
#define SRC_COMMON_INCLUDE_UEM_COMMON_H_

#ifdef __cplusplus
extern "C"
{
#endif

#ifndef IN
    #define IN
#endif

#ifndef OUT
    #define OUT
#endif

#ifndef TRUE
#   define TRUE (1)
#endif

#ifndef FALSE
#   define FALSE (0)
#endif

#undef NULL

#ifndef NULL
#   define NULL (void *) (0)
#endif

#ifndef MIN
    #define MIN(a, b) ((a) < (b) ? (a): (b))
#endif

#ifndef MAX
    #define MAX(a, b) ((a) > (b) ? (a): (b))
#endif

#define INVALID_TASK_ID (-1)
#define INVALID_SCHEDULE_ID (-1)
#define INVALID_MODE_ID (-1)
#define INVALID_CHANNEL_ID (-1)
#define INVALID_CHUNK_INDEX (-1)
#define INVALID_TIMER_SLOT_ID (-1)
#define INVALID_TIME_VALUE (0)
#define VARIABLE_SAMPLE_RATE (-1)
#define MAPPING_NOT_SPECIFIED (-1)
#define CHUNK_NUM_NOT_INITIALIZED (-1)
#define INVALID_ARRAY_INDEX (-1)

#ifdef ARDUINO
#define UEM_RESULT_CATEGORY_INFO  0x1000
#define UEM_RESULT_CATEGORY_ERROR 0x2000
#else
#define UEM_RESULT_CATEGORY_INFO  0x10000000
#define UEM_RESULT_CATEGORY_ERROR 0x20000000
#endif

#define UEM_MODULE_KERNEL 0x1000

typedef enum _EUemResult {
    ERR_UEM_NOERROR            = 0,

	ERR_UEM_INFORMATION        = UEM_RESULT_CATEGORY_INFO,
	// Insert information here
	ERR_UEM_USER_CANCELED,
	ERR_UEM_FOUND_DATA,
	ERR_UEM_ALREADY_DONE,
	ERR_UEM_NO_CHANGE,
	ERR_UEM_TRUNCATED,
	ERR_UEM_SUSPEND,
	ERR_UEM_SKIP_THIS,
	ERR_UEM_NOT_REACHED_YET,
	ERR_UEM_IN_PROGRESS,

	ERR_UEM_ERROR            = UEM_RESULT_CATEGORY_ERROR,

	// Insert error at the end of error enumeration
	ERR_UEM_UNKNOWN,
	ERR_UEM_INVALID_PARAM,
	ERR_UEM_INVALID_HANDLE,
	ERR_UEM_OUT_OF_MEMORY,
	ERR_UEM_NO_DATA,
	ERR_UEM_INTERNAL_FAIL,
	ERR_UEM_MUTEX_ERROR,
	ERR_UEM_STATIC_HANDLE,
	ERR_UEM_TIME_EXPIRED,
	ERR_UEM_INTERRUPT,

	ERR_UEM_DATA_DUPLICATED,
	ERR_UEM_ILLEGAL_CONTROL,
	ERR_UEM_ILLEGAL_DATA,
	ERR_UEM_NOT_SUPPORTED_YET,
	ERR_UEM_NOT_FOUND,
	ERR_UEM_CONVERSION_ERROR,
	ERR_UEM_CUDA_INVALID_VALUE,				//cudaErrorInvalidValue
	ERR_UEM_CUDA_MEMORY_ALLOCATION,			//cudaErrorMemoryAllocation
	ERR_UEM_CUDA_INVALID_DEVICE_POINTER,	//cudaErrorInvalidDevicePointer
	ERR_UEM_CUDA_INITIALIZATION,			//cudaErrorInitializationError

	ERR_UEM_CUDA_INVALID_MEMCPY_DIRECTION,	//cudaErrorInvalidMemcpyDirection
	ERR_UEM_UNAVAILABLE_DATA,
	ERR_UEM_NOT_SUPPORTED,
	ERR_UEM_INVALID_SOCKET,
	ERR_UEM_INVALID_SERIAL,
	ERR_UEM_SOCKET_ERROR,
	ERR_UEM_CONNECT_ERROR,
	ERR_UEM_LISTEN_ERROR,
	ERR_UEM_SELECT_ERROR,
	ERR_UEM_NET_TIMEOUT,

	ERR_UEM_ACCEPT_ERROR,
	ERR_UEM_NET_SEND_ERROR,
	ERR_UEM_NET_RECEIVE_ERROR,
	ERR_UEM_BIND_ERROR,
	ERR_UEM_READ_BLOCK,
	ERR_UEM_WRITE_BLOCK,

} uem_result;


typedef int uem_size;

#ifdef ARDUINO
	typedef unsigned long uem_time;
	typedef signed char uem_bool;
#else
	typedef long long uem_time;
	typedef int uem_bool;
#endif

typedef enum _EUemModuleId {
	// UEM Common module
	ID_UEM_THREAD             = 0,
	ID_UEM_THREAD_MUTEX       = 1,
	ID_UEM_THREAD_EVENT       = 2,
	ID_UEM_DYNAMIC_LINKED_LIST= 3,
	ID_UEM_STACK			  = 4,
	ID_UEM_SOCKET			  = 5,
	ID_UEM_SERIAL			  = 6,
	ID_UEM_FIXED_SIZE_QUEUE = 7,

	// UEM Kernel module
	ID_UEM_KERNEL_MODULE = UEM_MODULE_KERNEL,
	ID_UEM_CPU_TASK_MANAGER,
	ID_UEM_CPU_COMPOSITE_TASK_MANAGER,
	ID_UEM_CPU_GENERAL_TASK_MANAGER,
	ID_UEM_SERIAL_COMMUNICATION_MANAGER,


} EUemModuleId;

#define ARGUMENT_CHECK

#define ARRAYLEN(array)	 		(sizeof(array)/sizeof(array[0]))

#define IS_VALID_HANDLE(handle, id) (handle != NULL && (*((int *)(handle))) == id)


#define MAX_TASK_NAME_LEN (100)

#define EXIT_FLAG_READ 	(0x1)
#define EXIT_FLAG_WRITE (0x2)

#ifdef __cplusplus
}
#endif

//#define _DEBUG
#define DEBUG_PRINT


#ifdef DEBUG_PRINT
	#if defined(HAVE_PRINTF)
		#include <stdio.h>

		#define UEM_DEBUG_PRINT(fmt,args...) printf(fmt, ## args )
	#elif defined(ARDUINO)
		#include <UCPrint.h>

		#define UEM_DEBUG_PRINT(fmt,args...) UCPrint_format(fmt, ## args )
	#else
		#define UEM_DEBUG_PRINT(fmt,args...)
	#endif
#else
	#define UEM_DEBUG_PRINT(fmt,args...)
#endif

#ifdef _DEBUG
	#if defined(HAVE_PRINTF)
		#include <unistd.h>
		#include <stdio.h>
		#include <errno.h>

		#define ERRIFGOTO(res, label) if(((res) & ERR_UEM_ERROR)!=ERR_UEM_NOERROR) {fprintf(stderr, "error! %08x (%s:%d)\n", res, __FILE__,__LINE__);goto label;}
		#define ERRASSIGNGOTO(res, err, label) {res=err; fprintf(stderr, "error! %08x (%s:%d)\n", res, __FILE__,__LINE__); goto label;}
		#define IFVARERRASSIGNGOTO(var, val, res, err, label) if((var)==(val)) {res=err;fprintf(stderr, "error! %08x (%s:%d)\n", res, __FILE__,__LINE__);goto label;}
		#define ERRMEMGOTO(var, res, label) if((var)==NULL) {res=ERR_UEM_OUT_OF_MEMORY;fprintf(stderr, "memory error! %08x (%s:%d)\n", res, __FILE__,__LINE__);goto label;}
	#elif defined(ARDUINO)
		#include <UCPrint.h>

		#define ERRIFGOTO(res, label) if(((res) & ERR_UEM_ERROR)!=ERR_UEM_NOERROR) {UCPrint_format("error! %08x (%s:%d)\n", res, __FILE__,__LINE__);goto label;}
		#define ERRASSIGNGOTO(res, err, label) {res=err; UCPrint_format("error! %08x (%s:%d)\n", res, __FILE__,__LINE__); goto label;}
		#define IFVARERRASSIGNGOTO(var, val, res, err, label) if((var)==(val)) {res=err;UCPrint_format("error! %08x (%s:%d)\n", res, __FILE__,__LINE__);goto label;}
		#define ERRMEMGOTO(var, res, label) if((var)==NULL) {res=ERR_UEM_OUT_OF_MEMORY;UCPrint_format("memory error! %08x (%s:%d)\n", res, __FILE__,__LINE__);goto label;}
	#else
		#define ERRIFGOTO(res, label) if(((res) & ERR_UEM_ERROR)!=ERR_UEM_NOERROR) {goto label;}
		#define ERRASSIGNGOTO(res, err, label) {res=err; goto label;}
		#define IFVARERRASSIGNGOTO(var, val, res, err, label) if((var)==(val)) {res=err;goto label;}
		#define ERRMEMGOTO(var, res, label) if((var)==NULL) {res=ERR_UEM_OUT_OF_MEMORY;goto label;}
	#endif
#else

#define ERRIFGOTO(res, label) if(((res) & ERR_UEM_ERROR)!=ERR_UEM_NOERROR) {goto label;}
#define ERRASSIGNGOTO(res, err, label) {res=err; goto label;}
#define IFVARERRASSIGNGOTO(var, val, res, err, label) if((var)==(val)) {res=err;goto label;}
#define ERRMEMGOTO(var, res, label) if((var)==NULL) {res=ERR_UEM_OUT_OF_MEMORY;goto label;}
#endif

#define UEMASSIGNGOTO(res, err, label) {res=err; goto label;}




#endif /* SRC_COMMON_INCLUDE_UEM_COMMON_H_ */
