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

#ifndef NULL
#   define NULL (void *) (0)
#endif


#define UEM_RESULT_CATEGORY_INFO  0x10000000
#define UEM_RESULT_CATEGORY_ERROR 0x20000000

typedef enum _EUemResult {
    ERR_UEM_NOERROR            = 0,

	ERR_UEM_INFORMATION        = UEM_RESULT_CATEGORY_INFO,
	// Insert information here
	ERR_UEM_USER_CANCELED,
	ERR_UEM_FOUND_DATA,

	ERR_UEM_ERROR            = UEM_RESULT_CATEGORY_ERROR,

	// Insert error at the end of error enumeration
	ERR_UEM_UNKNOWN,
	ERR_UEM_INVALID_PARAM,
	ERR_UEM_INVALID_HANDLE,
	ERR_UEM_OUT_OF_MEMORY,
	ERR_UEM_NO_DATA,

} uem_result;

typedef int uem_bool;

typedef enum _EUemModuleId {
	ID_UEM_THREAD             = 0,
	ID_UEM_THREAD_MUTEX       = 1,
	ID_UEM_THREAD_EVENT       = 2,
	ID_UEM_DYNAMIC_LINKED_LIST= 3,
} EUemModuleId;

#define ARRAYLEN(array)	 		(sizeof(array)/sizeof(array[0]))

#define IS_VALID_HANDLE(handle, id) (handle != NULL && (*((int *)(handle))) == id)

#define ERRIFGOTO(res, label) if(((res) & ERR_UEM_ERROR)!=ERR_UEM_NOERROR) {goto label;}
#define ERRASSIGNGOTO(res, err, label) {res=err; goto label;}
#define UEMASSIGNGOTO(res, err, label) {res=err; goto label;}
#define IFVARERRASSIGNGOTO(var, val, res, err, label) if((var)==(val)) {res=err;goto label;}
#define ERRMEMGOTO(var, res, label) if((var)==NULL) {res=ERR_UEM_OUT_OF_MEMORY;goto label;}

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_INCLUDE_UEM_COMMON_H_ */
