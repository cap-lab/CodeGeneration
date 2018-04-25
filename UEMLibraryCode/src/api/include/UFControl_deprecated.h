/*
 * UFControl_deprecated.h
 *
 *  Created on: 2017. 8. 13.
 *      Author: jej
 */

#ifndef SRC_API_INCLUDE_UFCONTROL_DEPRECATED_H_
#define SRC_API_INCLUDE_UFCONTROL_DEPRECATED_H_

#ifdef __cplusplus
extern "C"
{
#endif

void SYS_REQ_END_TASK(int nCallerTaskId, char *pszTaskName);
void SYS_REQ_RUN_TASK(int nCallerTaskId, char *pszTaskName);
void SYS_REQ_STOP_TASK(int nCallerTaskId, char *pszTaskName);
void SYS_REQ_SUSPEND_TASK(int nCallerTaskId, char *pszTaskName);
void SYS_REQ_RESUME_TASK(int nCallerTaskId, char *pszTaskName);
void SYS_REQ_CALL_TASK(int nCallerTaskId, char *pszTaskName);

#ifdef __cplusplus
}
#endif

#endif /* SRC_API_INCLUDE_UFCONTROL_DEPRECATED_H_ */
