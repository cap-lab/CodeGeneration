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
 * @brief Check the task is a source task.
 *
 * This function checks the task is a source task which has no input channels in the task graph.
 *
 * @param nTaskId target task ID.
 *
 * @return If the task is a source task, this function returns TRUE. Otherwise, it returns FALSE.
 */
uem_bool UKChannel_IsTaskSourceTask(int nTaskId);

/**
 * @brief Check input ports are available for the current mode in MTM task.
 *
 * This function checks all the input ports for a specific mode @a pszModeName in MTM task, \n
 * and there is a port with sample rate is greater than 0. \n
 * This function is used for checking the task can be runnable on mode @a pszModeName.
 *
 * @param nTaskId target task ID.
 * @param pszModeName mode name.
 *
 * @return If the task's input ports are available, this function returns TRUE. Otherwise, it returns FALSE.
 */
uem_bool UKChannel_IsPortRateAvailableTask(int nTaskId, char *pszModeName);

/**
 * @brief Set exit flags on all channels.
 *
 * This function sets exit flags on all channels.
 *
 * @warning This function is not used currently.
 *
 * @return This function returns ERR_UEM_NOERROR.
 */
uem_result UKChannel_SetExit();

/**
 * @brief Set exit flags on channels connected with a specific task.
 *
 * This function sets exit flags on channels connected with a specific task with ID @a nTaskId.
 *
 * @param nTaskId target task ID.
 *
 * @return This function returns ERR_UEM_NOERROR.
 */
uem_result UKChannel_SetExitByTaskId(int nTaskId);

/**
 * @brief Clear exit flags on channels connected with a specific task.
 *
 * This function clears exit flags on channels connected with a specific task.
 *
 * @param nTaskId target task ID.
 *
 * @return This function returns ERR_UEM_NOERROR.
 */
uem_result UKChannel_ClearExitByTaskId(int nTaskId);

/**
 * @brief Clear all channel information located inside of a specific task graph.
 *
 * This function clears all channel information located inside of a specific task graph with ID @a nParentTaskId.
 * It will also clear all the channels located in the subgraphs of @a nParentTaskId.
 *
 * @param nParentTaskId a task ID of parent task graph.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_ILLEGAL_CONTROL and corresponding results from SChannelAPI's fnClear(). \n
 *         @ref ERR_UEM_ILLEGAL_CONTROL can be occurred when the fnClear() function is not defined.
 */
uem_result UKChannel_ClearChannelInSubgraph(int nParentTaskId);

/**
 * @brief Fill initial data from a source task.
 *
 * This function fills initial data from a source task. A source task is a task which generates an output data to channel.
 *
 * @param nTaskId source task ID.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - corresponding results from SChannelAPI's fnFillInitialData().
 */
uem_result UKChannel_FillInitialDataBySourceTaskId(int nTaskId);

/**
 * @brief Calculate chunk number and chunk length of the port.
 *
 * This function calculates chunk number and chunk length from port information. \n
 * In most case, chunk number is 1, and chunk length is the size of sample data size (sample rate * sample size). \n
 * For Loop task with distributing port, chunk number is multiplied by loop count, and chunk length is divided by loop count.
 *
 * @param pstPort a target port.
 * @param[out] pnChunkNum number of chunk.
 * @param[out] pnChunkLen length of chunk.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKChannel_GetChunkNumAndLen(SPort *pstPort, OUT int *pnChunkNum, OUT int *pnChunkLen);

/**
 * @brief Pop data from queue when the task is loop task and port is broadcasting port.
 *
 * This function pops out @a nNumOfDataToPop of data from queue \n
 * if the task is located inside a loop task (or task itself is a loop task) and the port is broadcasting port.
 * Data pop is occurred under the loop task of @a nLoopTaskId.
 *
 * @param nLoopTaskId ID of target loop task.
 * @param nTaskId target task id to pop data.
 * @param nNumOfDataToPop number of items to pop.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - corresponding results from SChannelAPI's fnGetNumOfAvailableData() and fnReadFromQueue().
 */
uem_result UKChannel_PopLoopTaskBroadcastingDataFromQueueByTaskId(int nLoopTaskId, int nTaskId, int nNumOfDataToPop);

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKCHANNEL_INTERNAL_H_ */
