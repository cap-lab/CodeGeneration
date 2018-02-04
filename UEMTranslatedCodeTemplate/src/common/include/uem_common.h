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


#define UEM_RESULT_CATEGORY_INFO  0x10000000
#define UEM_RESULT_CATEGORY_ERROR 0x20000000

#define UEM_MODULE_KERNEL 0x10000000

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


} uem_result;

typedef int uem_bool;
typedef int uem_size;


typedef enum _EUemModuleId {
	// UEM Common module
	ID_UEM_THREAD             = 0,
	ID_UEM_THREAD_MUTEX       = 1,
	ID_UEM_THREAD_EVENT       = 2,
	ID_UEM_DYNAMIC_LINKED_LIST= 3,
	ID_UEM_STACK			  = 4,

	// UEM Kernel module
	ID_UEM_KERNEL_MODULE = UEM_MODULE_KERNEL,
	ID_UEM_CPU_TASK_MANAGER,
	ID_UEM_CPU_COMPOSITE_TASK_MANAGER,
	ID_UEM_CPU_GENERAL_TASK_MANAGER,


} EUemModuleId;

#define ARGUMENT_CHECK

#define ARRAYLEN(array)	 		(sizeof(array)/sizeof(array[0]))

#define IS_VALID_HANDLE(handle, id) (handle != NULL && (*((int *)(handle))) == id)


#define EXIT_FLAG_READ 	(0x1)
#define EXIT_FLAG_WRITE (0x2)


#define _DEBUG

#ifdef _DEBUG

#include <unistd.h>
#include <stdio.h>
#include <errno.h>

#define ERRIFGOTO(res, label) if(((res) & ERR_UEM_ERROR)!=ERR_UEM_NOERROR) {fprintf(stderr, "error! %08x (%s:%d)\n", res, __FILE__,__LINE__);goto label;}
#define ERRASSIGNGOTO(res, err, label) {res=err; fprintf(stderr, "error! %08x (%s:%d)\n", res, __FILE__,__LINE__); goto label;}
#define IFVARERRASSIGNGOTO(var, val, res, err, label) if((var)==(val)) {res=err;fprintf(stderr, "error! %08x (%s:%d)\n", res, __FILE__,__LINE__);goto label;}
#else
#define ERRIFGOTO(res, label) if(((res) & ERR_UEM_ERROR)!=ERR_UEM_NOERROR) {goto label;}
#define ERRASSIGNGOTO(res, err, label) {res=err; goto label;}
#define IFVARERRASSIGNGOTO(var, val, res, err, label) if((var)==(val)) {res=err;goto label;}

#endif

#define UEMASSIGNGOTO(res, err, label) {res=err; goto label;}
#define ERRMEMGOTO(var, res, label) if((var)==NULL) {res=ERR_UEM_OUT_OF_MEMORY;goto label;}


#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_INCLUDE_UEM_COMMON_H_ */
