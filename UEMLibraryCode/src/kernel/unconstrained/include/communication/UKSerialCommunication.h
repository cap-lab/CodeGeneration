/*
 * UKSerialCommunication.h
 *
 *  Created on: 2019. 5. 23.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSERIALCOMMUNICATION_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSERIALCOMMUNICATION_H_


#include <uem_common.h>

#include <UKVirtualCommunication.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief
 *
 * This function
 *
 * @param phSocket
 * @param pSocketInfo
 *
 * @return
 */
uem_result UKSerialCommunication_Create(OUT HVirtualSocket *phSocket, void *pSocketInfo);

/**
 * @brief
 *
 * This function
 *
 * @param phSocket
 *
 * @return
 */
uem_result UKSerialCommunication_Destroy(IN OUT HVirtualSocket *phSocket);

/**
 * @brief
 *
 * This function
 *
 * @param hSocket
 * @param nTimeout
 *
 * @return
 */
uem_result UKSerialCommunication_Connect(HVirtualSocket hSocket, int nTimeout);

/**
 * @brief
 *
 * This function
 *
 * @param hSocket
 *
 * @return
 */
uem_result UKSerialCommunication_Disconnect(HVirtualSocket hSocket);

/**
 * @brief
 *
 * This function
 *
 * @param hSocket
 *
 * @return
 */
uem_result UKSerialCommunication_Listen(HVirtualSocket hSocket);

/**
 * @brief
 *
 * This function
 *
 * @param hSocket
 * @param nTimeout
 * @param hAcceptedSocket
 *
 * @return
 */
uem_result UKSerialCommunication_Accept(HVirtualSocket hSocket, int nTimeout, IN OUT HVirtualSocket hAcceptedSocket);

/**
 * @brief
 *
 * This function
 *
 * @param hSocket
 * @param nTimeout
 * @param pData
 * @param nDataLen
 * @param [out] pnSentSize
 *
 * @return
 */
uem_result UKSerialCommunication_Send(HVirtualSocket hSocket, IN int nTimeout, IN char *pData, IN int nDataLen, OUT int *pnSentSize);

/**
 * @brief
 *
 * This function
 *
 * @param hSocket
 * @param nTimeout
 * @param pBuffer
 * @param nBufferLen
 * @param[out] pnReceivedSize
 *
 * @return
 */
uem_result UKSerialCommunication_Receive(HVirtualSocket hSocket, IN int nTimeout, IN OUT char *pBuffer, IN int nBufferLen, OUT int *pnReceivedSize);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSERIALCOMMUNICATION_H_ */
