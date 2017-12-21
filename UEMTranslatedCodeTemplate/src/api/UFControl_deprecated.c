/*
 * UFControl_deprecated.c
 *
 *  Created on: 2017. 8. 24.
 *      Author: jej
 */

#include <uem_common.h>

#include <UFControl.h>


void SYS_REQ_END_TASK(char *pszTaskName)
{
	UFControl_StopTask(pszTaskName, TRUE);
}


void SYS_REQ_RUN_TASK(char *pszTaskName)
{
	UFControl_RunTask(pszTaskName);
}


void SYS_REQ_STOP_TASK(char *pszTaskName)
{
	UFControl_StopTask(pszTaskName, FALSE);
}


void SYS_REQ_SUSPEND_TASK(char *pszTaskName)
{
	UFControl_SuspendTask(pszTaskName);
}


void SYS_REQ_RESUME_TASK(char *pszTaskName)
{
	UFControl_ResumeTask(pszTaskName);
}


void SYS_REQ_CALL_TASK(char *pszTaskName)
{
	UFControl_CallTask(pszTaskName);
}


