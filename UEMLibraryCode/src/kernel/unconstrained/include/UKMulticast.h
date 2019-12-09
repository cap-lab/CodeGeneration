/*
 * UKMulticast.h
 *
 *  Created on: 2019. 6. 20.
 *      Author: wecracy
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKMULTICAST_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKMULTICAST_H_

#include <uem_common.h>
#include <uem_multicast_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Initialize a multicast.
 *
 * This function performs multicast initialization. \n
 * This function at first initialize socket APIs if exists, \n
 * then initialize the groups' communication if the initialization API exists, \n
 * and initialize the ports' communication if the initialization API exists.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error.
 */
uem_result UKMulticast_Initialize();

/**
 * @brief Get multicast group id by task and port name.
 *
 * This function returns multicast group id by given task name and port name.
 *
 * @param nTaskId task id.
 * @param szPortName port name.
 * @param[out] pnMulticastGroupId multicast group id.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_NOT_FOUND is returned if there is no matched port with given task name and port name. \n
 * @ref ERR_UEM_INVALID_PARAM is returned if there is error in making new string.
 */
uem_result UKMulticast_GetMulticastGroupIdByTaskAndPortName(IN int nTaskId, IN const char *szPortName, OUT int *pnMulticastGroupId);

/**
 * @brief Get multicast port id by task and port name.
 *
 * This function returns multicast port id by given task name and port name.
 *
 * @param nTaskId task id.
 * @param szPortName port name.
 * @param[out] pnMulticastPortId multicast port id.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_NOT_FOUND is returned if there is no matched port with given task name and port name. \n
 * @ref ERR_UEM_INVALID_PARAM is returned if there is error in making new string.
 */
uem_result UKMulticast_GetMulticastPortIdByTaskAndPortName(IN int nTaskId, IN const char *szPortName, OUT int *pnMulticastPortId);

/**
 * @brief Get multicast group maximum buffer size by multicast group id.
 *
 * This function returns multicast group maximum buffer size by given multicast group id.
 *
 * @param nMulticastGroupId multicast group id.
 * @param[out] pnMulticastGroupSize multicast group maximum buffer size.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_NOT_FOUND is returned if there is no matched group with given multicast group id.
 */
uem_result UKMulticast_GetMulticastGroupSize(IN int nMulticastGroupId, OUT int *pnMulticastGroupSize);

/**
 * @brief Get communication structure.
 *
 * This function returns the communication structure which include the API structure and the socket.
 *
 * @param astCommunicationList candidate communication list.
 * @param nCommunicationTypeNum number of candidate communication.
 * @param enCommunicationType target communication type.
 * @param[out] pstCommunication target communication structure.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_NOT_FOUND is returned if there is no matched communication with the given communication type. \n
 * @ref ERR_UEM_INVALID_PARAM is returned if astCommunicationList or pstCommunication is null.
 */
uem_result UKMulticast_GetCommunication(IN SMulticastCommunication *astCommunicationList, IN int nCommunicationTypeNum, IN EMulticastCommunicationType enCommunicationType, OUT SMulticastCommunication **pstCommunication);

/**
 * @brief Send data by the multicast communication methods
 *
 * This function sends data by the multicast communication methods. \n
 * This function calls the writeToBuffer function specified for each multicastAPI allocated for each multicast communication. \n
 * When a multicast port is connected to multiple communication, it calls all corresponding multicastAPIs.
 *
 * @param nMulticastGroupId ID of multicast group to send data.
 * @param nMulticastPortId ID of multicast port to send data.
 * @param pBuffer buffer to send data.
 * @param nDataToWrite buffer size.
 * @param[out] pnDataWritten sent data size.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid @a pBuffer. \n
 * @ref ERR_UEM_NOT_FOUND is returned if there is no matched multicast port with the given multicast group id and port id. \n
 * error could be propagated from fnWriteToBuffer function of multicastAPI.
 */
uem_result UKMulticast_WriteToBuffer(IN int nMulticastGroupId, IN int nMulticastPortId, IN unsigned char *pBuffer, IN int nDataToWrite, OUT int *pnDataWritten);

/**
 * @brief receive data from multicast
 *
 * This function receives data by the multicast communication methods. \n
 * This function calls the ReadFromBuffer function specified for each multicastAPI allocated for each multicast communication. \n
 * When a multicast port is connected to multiple communication, it calls all corresponding multicastAPIs.
 *
 * @param nMulticastGroupId ID of multicast group to send data.
 * @param nMulticastPortId ID of multicast port to send data.
 * @param pBuffer buffer to receive data.
 * @param nDataToRead data length to read.
 * @param[out] pnDataRead received data size.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM for invalid @a pBuffer. \n
 * @ref ERR_UEM_NOT_FOUND is returned if there is no matched multicast port with the given multicast group id and port id. \n
 * error could be propagated from fnReadFromBuffer function of multicastAPI.
 */
uem_result UKMulticast_ReadFromBuffer(IN int nMulticastGroupId, IN int nMulticastPortId, IN OUT unsigned char *pBuffer, IN int nDataToRead, OUT int *pnDataRead);

/**
 * @brief Perform multicast finalizing.
 * This function performs multicast finalizing jobs.
 * This function at first finalize the ports' communication if exists, \n
 * then finalize the groups' communication if the finalization API exists, \n
 * and finalize the socket APIs if the finalization API exists.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error.
 */
uem_result UKMulticast_Finalize();


#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKMULTICAST_H_ */
