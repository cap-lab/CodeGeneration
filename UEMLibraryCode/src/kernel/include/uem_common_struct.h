/*
 * uem_common_data.h
 *
 *  Created on: 2018. 9. 10.
 *      Author: chjej202
 */

#ifndef SRC_KERNEL_INCLUDE_UEM_COMMON_STRUCT_H_
#define SRC_KERNEL_INCLUDE_UEM_COMMON_STRUCT_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _STask STask;
typedef struct _SChannel SChannel;

typedef struct _STimer {
	int nSlotId;
	uem_bool bAlarmChecked;
	uem_time tTimeInMilliSec;
	uem_time tAlarmTime;
} STimer;

typedef struct _STaskFunctions {
	FnUemTaskInit fnInit;
	FnUemTaskGo fnGo;
	FnUemTaskWrapup fnWrapup;
} STaskFunctions;

typedef struct _STaskGraph {
	ETaskGraphType enType;
	STask *astTasks;
	int nNumOfTasks;
	STask *pstParentTask;
} STaskGraph;

typedef union _UParamValue {
	int nParam;
	double dbParam;
} UParamValue;

typedef struct _STaskParameter {
	int nParamId;
	EParameterType enType;
	const char *pszParamName;
	UParamValue uParamValue;
} STaskParameter;


#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UEM_COMMON_STRUCT_H_ */
