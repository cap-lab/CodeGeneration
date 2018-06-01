/*
 * UKTimer.c
 *
 *  Created on: 2018. 5. 1.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <uem_data.h>

#include <UCBasic.h>
#include <UCTime.h>

#include <UKTask.h>

#define TIMER_UNIT_SEC "S"
#define TIMER_UNIT_MILLSEC "MS"
#define TIMER_UNIT_MICROSEC "US"


#define TIMER_METRIC_GAP (1000)


static uem_result convertTimeToMilliSec(IN int nTimeValue, IN char *pszTimeUnit, OUT int *pnMilliSec)
{
	uem_result result = ERR_UEM_UNKNOWN;

	// compare also the last NULL string, so size is sizeof(TIMER_UNIT_XXX)
	if(UC_memcmp(pszTimeUnit, TIMER_UNIT_SEC, sizeof(TIMER_UNIT_SEC)) == 0)
	{
		*pnMilliSec = nTimeValue * TIMER_METRIC_GAP;
	}
	else if(UC_memcmp(pszTimeUnit, TIMER_UNIT_MILLSEC, sizeof(TIMER_UNIT_MILLSEC)) == 0)
	{
		*pnMilliSec = nTimeValue;
	}
	else if(UC_memcmp(pszTimeUnit, TIMER_UNIT_MICROSEC, sizeof(TIMER_UNIT_MICROSEC)) == 0)
	{
		// Microsecond is not supported because its resolution is too small to wait.
		if(nTimeValue < TIMER_METRIC_GAP)
		{
			*pnMilliSec = 1;
		}
		else
		{
			*pnMilliSec = nTimeValue / TIMER_METRIC_GAP;
		}
	}
	else
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}


	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

static uem_result findEmptyTimerSlot(STimer *pstTimer, int *pnTimerIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	long long llCurTime = 0;
	int nTimerIndex = INVALID_TIMER_SLOT_ID;

	for(nLoop = 0 ; nLoop < g_nTimerSlotNum; nLoop++)
	{
		if(pstTimer[nLoop].nTimeInMilliSec == INVALID_TIME_VALUE)
		{
			nTimerIndex = nLoop;
			break;
		}
	}

	if(nTimerIndex == INVALID_TIMER_SLOT_ID)
	{
		for(nLoop = 0 ; nLoop < g_nTimerSlotNum; nLoop++)
		{
			if(pstTimer[nLoop].bAlarmChecked == TRUE)
			{
				break;
			}
		}
	}

	if(nTimerIndex == INVALID_TIMER_SLOT_ID)
	{
		for(nLoop = 0 ; nLoop < g_nTimerSlotNum; nLoop++)
		{
			if(pstTimer[nLoop].llAlarmTime < llCurTime)
			{
				break;
			}
		}
	}

	if(nTimerIndex == INVALID_TIMER_SLOT_ID)
	{
		ERRASSIGNGOTO(result, ERR_UEM_UNAVAILABLE_DATA, _EXIT);
	}

	*pnTimerIndex = nTimerIndex;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}



uem_result UKTimer_SetAlarm (IN int nCallerTaskId, IN int nTimeValue, IN char *pszTimeUnit, OUT int *pnTimerId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstCallerTask = NULL;
	int nMilliSec = 0;
	long long llCurTime = 0;
	int nTimerIndex = INVALID_TIMER_SLOT_ID;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pszTimeUnit, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pnTimerId, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if(nTimeValue < 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	result = UKTask_GetTaskFromTaskId(nCallerTaskId, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	if(pstCallerTask->enType != TASK_TYPE_CONTROL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	result = convertTimeToMilliSec(nTimeValue, pszTimeUnit, &nMilliSec);
	ERRIFGOTO(result, _EXIT);

	result = UCTime_GetCurTickInMilliSeconds(&llCurTime);
	ERRIFGOTO(result, _EXIT);

	result = findEmptyTimerSlot(pstCallerTask->astTimer, &nTimerIndex);
	ERRIFGOTO(result, _EXIT);

	pstCallerTask->astTimer[nTimerIndex].bAlarmChecked = FALSE;
	pstCallerTask->astTimer[nTimerIndex].nTimeInMilliSec = nMilliSec;
	pstCallerTask->astTimer[nTimerIndex].llAlarmTime = llCurTime + nMilliSec;

	*pnTimerId = pstCallerTask->astTimer[nTimerIndex].nSlotId;

	result = ERR_UEM_NOERROR;
_EXIT:;
	return result;
}


static uem_result getTimerSlotIndex(STimer *pstTimer, int nTimerId, int *pnTimerIndex)
{
	uem_result result = ERR_UEM_UNKNOWN;
	int nLoop = 0;
	int nTimerIndex = INVALID_TIMER_SLOT_ID;

	if(pstTimer[nTimerId].nSlotId == nTimerId)
	{
		nTimerIndex = nTimerId;
	}
	else
	{
		for(nLoop = 0 ; nLoop < g_nTimerSlotNum ; nLoop++)
		{
			if(pstTimer[nLoop].nSlotId == nTimerId)
			{
				nTimerIndex = nLoop;
				break;
			}
		}

		if(nLoop == g_nTimerSlotNum)
		{
			ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_DATA, _EXIT);
		}
	}

	*pnTimerIndex = nTimerIndex;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTimer_GetAlarmed (IN int nCallerTaskId, IN int nTimerId, OUT uem_bool *pbTimerPassed)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstCallerTask = NULL;
	int nTimerIndex = INVALID_TIMER_SLOT_ID;
	long long llCurTime = 0;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(pbTimerPassed, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if(nTimerId < 0 || nTimerId >= g_nTimerSlotNum)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	result = UKTask_GetTaskFromTaskId(nCallerTaskId, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	if(pstCallerTask->enType != TASK_TYPE_CONTROL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	result = getTimerSlotIndex(pstCallerTask->astTimer, nTimerId, &nTimerIndex);
	ERRIFGOTO(result, _EXIT);

	result = UCTime_GetCurTickInMilliSeconds(&llCurTime);
	ERRIFGOTO(result, _EXIT);

	if(pstCallerTask->astTimer[nTimerIndex].nTimeInMilliSec == INVALID_TIME_VALUE)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}

	if(pstCallerTask->astTimer[nTimerIndex].llAlarmTime < llCurTime)
	{
		pstCallerTask->astTimer[nTimerIndex].bAlarmChecked = TRUE;
		*pbTimerPassed = TRUE;
	}
	else
	{
		*pbTimerPassed = FALSE;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}


uem_result UKTimer_Reset (IN int nCallerTaskId, IN int nTimerId)
{
	uem_result result = ERR_UEM_UNKNOWN;
	STask *pstCallerTask = NULL;
	int nTimerIndex = INVALID_TIMER_SLOT_ID;
#ifdef ARGUMENT_CHECK
	if(nTimerId < 0 || nTimerId >= g_nTimerSlotNum)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	result = UKTask_GetTaskFromTaskId(nCallerTaskId, &pstCallerTask);
	ERRIFGOTO(result, _EXIT);

	if(pstCallerTask->enType != TASK_TYPE_CONTROL)
	{
		ERRASSIGNGOTO(result, ERR_UEM_ILLEGAL_CONTROL, _EXIT);
	}

	result = getTimerSlotIndex(pstCallerTask->astTimer, nTimerId, &nTimerIndex);
	ERRIFGOTO(result, _EXIT);

	pstCallerTask->astTimer[nTimerIndex].nTimeInMilliSec = INVALID_TIME_VALUE;
	pstCallerTask->astTimer[nTimerIndex].bAlarmChecked = FALSE;
	pstCallerTask->astTimer[nTimerIndex].llAlarmTime = 0;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

