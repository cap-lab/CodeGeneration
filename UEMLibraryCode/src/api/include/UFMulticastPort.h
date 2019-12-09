/*
 * UFMulticastPort.h
 *
 *  Created on: 2019. 6. 20.
 *      Author: wecracy
 */

#ifndef SRC_API_INCLUDE_UFMULTICASTPORT_H_
#define SRC_API_INCLUDE_UFMULTICASTPORT_H_


#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Return group ID and port ID corresponding to the task and port name.
 *
 * This function is called by the TASK_INIT function where the first argument should be TASK_ID macro. TASK_ID means the ID of the current task.
 *
 *
 * @param nTaskId id of task.
 * @param szPortName port name.
 * @param[out] pnMulticastGroupId returned group id.
 * @param[out] pnMulticastPortId returned port id.
 *
 * @return
 *
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_NOT_FOUND is returned if there is no matched port. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid szPortName, pnMulticastGroupId, pnMulticastPortId arguments.
 */
uem_result UFMulticastPort_Initialize(IN int nTaskId, IN const char *szPortName, OUT int *pnMulticastGroupId, OUT int *pnMulticastPortId);

/**
 * @brief Receive data on a port.
 *
 * @param nMulticastGroupId ID of group to receive data.
 * @param nMulticastPortId ID of port to receive data.
 * @param pBuffer buffer to receive data.
 * @param nDataToRead buffer size.
 * @param[out] pnDataRead size of received data.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid nDataToRead, pBuffer, pnDataRead arguments. \n
 * @ref ERR_UEM_NOT_FOUND is returned if there is no matched port.
 */
uem_result UFMulticastPort_ReadFromBuffer (IN int nMulticastGroupId, IN int nMulticastPortId, IN OUT unsigned char *pBuffer, IN int nDataToRead, OUT int *pnDataRead);

/**
 * @brief Send data to a specific chunk index on a port whose type is buffer.
 *
 * @param nMulticastGroupId ID of group to send data.
 * @param nMulticastPortId ID of port to send data.
 * @param pBuffer buffer to send data.
 * @param nDataToWrite buffer size.
 * @param[out] pnDataWritten sent data size.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid nDataToWrite, pBuffer, pnDataWritten arguments. \n
 * @ref ERR_UEM_NOT_FOUND is returned if there is no matched port.
 */
uem_result UFMulticastPort_WriteToBuffer (IN int nMulticastGroupId, IN int nMulticastPortId, IN unsigned char *pBuffer, IN int nDataToWrite, OUT int *pnDataWritten);

/**
 * @brief Get buffer size of the multicast group.
 *
 * @param nMulticastGroupId ID of group to get the buffer size.
 * @param[out] pnMulticastGroupSize group buffer size.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid pnMulticastGroupSize argument. \n
 * @ref ERR_UEM_NOT_FOUND is returned if there is no matched group.
 */
uem_result UFMulticastPort_GetMulticastSize(IN int nMulticastGroupId, OUT int *pnMulticastGroupSize);

#ifdef __cplusplus
}
#endif

#endif /* SRC_API_INCLUDE_UFMULTICASTPORT_H_ */
