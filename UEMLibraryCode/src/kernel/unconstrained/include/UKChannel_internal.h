/*
 * UKChannel_internal.h
 *
 *  Created on: 2018. 8. 28.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKCHANNEL_INTERNAL_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKCHANNEL_INTERNAL_H_

#include <uem_common.h>

#include <uem_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_bool UKChannel_IsTaskSourceTask(int nTaskId);
uem_bool UKChannel_IsPortRateAvailableTask(int nTaskId, char *pszModeName);
uem_result UKChannel_SetExit();
uem_result UKChannel_SetExitByTaskId(int nTaskId);
uem_result UKChannel_ClearExitByTaskId(int nTaskId);
uem_result UKChannel_ClearChannelInSubgraph(int nParentTaskId);
uem_result UKChannel_FillInitialDataBySourceTaskId(int nTaskId);
uem_result UKChannel_GetChunkNumAndLen(SPort *pstPort, OUT int *pnChunkNum, OUT int *pnChunkLen);
uem_result UKChannel_PopLoopTaskBroadcastingDataFromQueueByTaskId(int nLoopTaskId, int nTaskId, int nNumOfDataToPop);

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKCHANNEL_INTERNAL_H_ */
