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

#ifndef TRUE
	#define TRUE 1
#endif

#ifndef FALSE
	#define FALSE 0
#endif

#ifndef IN
	#define IN
#endif

#ifndef OUT
	#define OUT
#endif

#define UEM_RESULT_CATEGORY_INFO  0x10000000
#define UEM_RESULT_CATEGORY_ERROR 0x20000000

typedef enum _EUemResult {
    ERR_UEM_NOERROR            = 0,

	ERR_UEM_INFORMATION        = UEM_RESULT_CATEGORY_INFO,
	// Insert information here

	ERR_UEM_ERROR            = UEM_RESULT_CATEGORY_ERROR,
	// Insert error here
	ERR_UEM_UNKNOWN,


} uem_result;

typedef int uem_bool;

typedef enum _EUemModuleId {
	ID_UEM_THREAD             = 0,
	ID_UEM_THREAD_MUTEX       = 1,
	ID_UEM_THREAD_EVENT       = 2,
} EUemModuleId;

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_INCLUDE_UEM_COMMON_H_ */
