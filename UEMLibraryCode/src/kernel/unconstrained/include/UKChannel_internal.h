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

/**
 * @brief
 *
 * This function
 *
 * @param nTaskId
 *
 * @return
 */
uem_bool UKChannel_IsTaskSourceTask(int nTaskId);

/**
 * @brief
 *
 * This function
 *
 * @param nTaskId
 * @param pszModeName
 *
 * @return
 */
uem_bool UKChannel_IsPortRateAvailableTask(int nTaskId, char *pszModeName);

/**
 * @brief
 *
 * This function
 *
 * @return
 */
uem_result UKChannel_SetExit();

/**
 * @brief
 *
 * This function
 *
 * @param nTaskId
 *
 * @return
 */
uem_result UKChannel_SetExitByTaskId(int nTaskId);

/**
 * @brief
 *
 * This function
 *
 * @param nTaskId
 *
 * @return
 */
uem_result UKChannel_ClearExitByTaskId(int nTaskId);

/**
 * @brief
 *
 * This function
 *
 * @param nParentTaskId
 *
 * @return
 */
uem_result UKChannel_ClearChannelInSubgraph(int nParentTaskId);

/**
 * @brief
 *
 * This function
 *
 * @param nTaskId
 *
 * @return
 */
uem_result UKChannel_FillInitialDataBySourceTaskId(int nTaskId);

/**
 * @brief
 *
 * This function
 *
 * @param pstPort
 * @param[out] pnChunkNum
 * @param[out] pnChunkLen
 *
 * @return
 */
uem_result UKChannel_GetChunkNumAndLen(SPort *pstPort, OUT int *pnChunkNum, OUT int *pnChunkLen);

/**
 * @brief
 *
 * This function
 *
 * @param nLoopTaskId
 * @param nTaskId
 * @param nNumOfDataToPop
 *
 * @return
 */
uem_result UKChannel_PopLoopTaskBroadcastingDataFromQueueByTaskId(int nLoopTaskId, int nTaskId, int nNumOfDataToPop);

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKCHANNEL_INTERNAL_H_ */
