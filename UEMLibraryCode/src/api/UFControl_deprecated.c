/*
 * UFControl_deprecated.c
 *
 *  Created on: 2017. 8. 24.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UFControl.h>


void SYS_REQ_END_TASK(int nCallerTaskId, char *pszTaskName)
{
	UFControl_StopTask(nCallerTaskId, pszTaskName, TRUE);
}


void SYS_REQ_RUN_TASK(int nCallerTaskId, char *pszTaskName)
{
	UFControl_RunTask(nCallerTaskId, pszTaskName);
}


void SYS_REQ_STOP_TASK(int nCallerTaskId, char *pszTaskName)
{
	UFControl_StopTask(nCallerTaskId, pszTaskName, FALSE);
}


void SYS_REQ_CALL_TASK(int nCallerTaskId, char *pszTaskName)
{
	UFControl_CallTask(nCallerTaskId, pszTaskName);
}


#ifndef API_LITE
void SYS_REQ_SUSPEND_TASK(int nCallerTaskId, char *pszTaskName)
{
	UFControl_SuspendTask(nCallerTaskId, pszTaskName);
}


void SYS_REQ_RESUME_TASK(int nCallerTaskId, char *pszTaskName)
{
	UFControl_ResumeTask(nCallerTaskId, pszTaskName);
}
#endif
