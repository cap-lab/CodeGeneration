/*
 * UKCPUTaskCommon.c
 *
 *  Created on: 2018. 2. 5.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCTime.h>

#include <uem_data.h>

#include <UKTime.h>
#include <UKCPUTaskCommon.h>


#define MIN_SLEEP_DURATION (10)
#define MAX_SLEEP_DURATION (100)

uem_result UKCPUTaskCommon_CheckTaskState(ECPUTaskState enOldState, ECPUTaskState enNewState)
{
	uem_result result = ERR_UEM_UNKNOWN;

	if(enNewState == enOldState)
	{
		UEMASSIGNGOTO(result, ERR_UEM_ALREADY_DONE, _EXIT);
	}

	switch(enOldState)
	{
	case TASK_STATE_RUNNING:
		// do nothing
		break;
	case TASK_STATE_SUSPEND:
		if(enNewState == TASK_STATE_STOPPING)
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
		}
		break;
	case TASK_STATE_STOP:
		if(enNewState != TASK_STATE_RUNNING && enNewState != TASK_STATE_STOPPING)
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
		}
		break;
	case TASK_STATE_STOPPING:
		if(enNewState != TASK_STATE_STOP)
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
		}
		break;
	default:
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
		break;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



uem_result UKCPUTaskCommon_HandleTimeDrivenTask(STask *pstCurrentTask, FnUemTaskGo fnGo, IN OUT long long *pllNextTime,
										IN OUT int *pnRunCount, IN OUT int *pnNextMaxRunCount)
{
	uem_result result = ERR_UEM_UNKNOWN;
	long long llCurTime = 0;
	long long llNextTime = 0;
	int nMaxRunCount = 0;
	int nRunCount = 0;

	llNextTime = *pllNextTime;
	nRunCount = *pnRunCount;
	nMaxRunCount = *pnNextMaxRunCount;

	result = UCTime_GetCurTickInMilliSeconds(&llCurTime);
	ERRIFGOTO(result, _EXIT);
	if(llCurTime <= llNextTime) // time is not passed
	{
		if(nRunCount < nMaxRunCount) // run count is available
		{
			nRunCount++;
			fnGo(pstCurrentTask->nTaskId);
		}
		else // all run count is used and time is not passed yet
		{
			// if period is long, sleep by time passing
			if(llNextTime - llCurTime > MAX_SLEEP_DURATION)
			{
				UCTime_Sleep(MAX_SLEEP_DURATION);
			}
			else if(llNextTime - llCurTime > MIN_SLEEP_DURATION) // left time is more than SLEEP_DURATION ms
			{
				UCTime_Sleep(MIN_SLEEP_DURATION);
			}
			else
			{
				// otherwise, busy wait
			}
		}
	}
	else // time is passed, reset
	{
		result = UKTime_GetNextTimeByPeriod(llNextTime, pstCurrentTask->nPeriod, pstCurrentTask->enPeriodMetric,
										&llNextTime, &nMaxRunCount);
		ERRIFGOTO(result, _EXIT);
		nRunCount = 0;
	}

	*pllNextTime = llNextTime;
	*pnRunCount = nRunCount;
	*pnNextMaxRunCount = nMaxRunCount;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



