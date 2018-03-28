/*
 * UFControl.h
 *
 *  Created on: 2017. 8. 11.
 *      Author: jej
 */

#ifndef SRC_API_INCLUDE_UFCONTROL_H_
#define SRC_API_INCLUDE_UFCONTROL_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_result UFControl_RunTask (IN char *pszTaskName);
uem_result UFControl_StopTask (IN char *pszTaskName, IN uem_bool bDelayedStop);
uem_result UFControl_SuspendTask (IN char *pszTaskName);
uem_result UFControl_ResumeTask (IN char *pszTaskName);
uem_result UFControl_CallTask (IN char *pszTaskName);

#ifdef __cplusplus
}
#endif

#endif /* SRC_API_INCLUDE_UFCONTROL_H_ */
