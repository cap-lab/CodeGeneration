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
 * @brief Create a serial communication socket.
 *
 * This function creates a serial communication socket. \n
 * This is an implementation function of UKVirtualCommunication of fnCreate().
 *
 * @param[out] phSocket a socket handle to be created.
 * @param pSocketInfo serial communication options.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_OUT_OF_MEMORY.
 */
uem_result UKSerialCommunication_Create(OUT HVirtualSocket *phSocket, void *pSocketInfo);

/**
 * @brief Destroy a serial communication socket.
 *
 * This function destroys a serial communication socket. \n
 * This is an implementation function of UKVirtualCommunication of fnDestroy().
 *
 * @param[in,out] phSocket a socket handle to be destroyed.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INVALID_HANDLE.
 */
uem_result UKSerialCommunication_Destroy(IN OUT HVirtualSocket *phSocket);

/**
 * @brief Connect through a serial communication socket.
 *
 * This function opens a serial communication as a master role. \n
 * This is an implementation function of UKVirtualCommunication of fnConnect().
 *
 * @param hSocket a socket handle.
 * @param nTimeout (not used).
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_SERIAL.
 *         @ref ERR_UEM_INVALID_SERIAL can be occurred when the serial port cannot be opened.
 */
uem_result UKSerialCommunication_Connect(HVirtualSocket hSocket, int nTimeout);

/**
 * @brief Disconnect through a serial communication socket.
 *
 * This function closes a serial communication as a master role. \n
 * This is an implementation function of UKVirtualCommunication of fnDisconnect().
 *
 * @param hSocket a socket handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *          Errors to be returned - @ref ERR_UEM_INVALID_HANDLE.
 */
uem_result UKSerialCommunication_Disconnect(HVirtualSocket hSocket);

/**
 * @brief Listen through a serial communication socket.
 *
 * This function opens a serial communication as a slave role. \n
 * This is an implementation function of UKVirtualCommunication of fnListen().
 *
 * @param hSocket a socket handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 */
uem_result UKSerialCommunication_Listen(HVirtualSocket hSocket);

/**
 * @brief Not used.
 *
 * This function is not used because serial communication does not have accept behavior. \n
 * This is an implementation function of UKVirtualCommunication of fnAccept().
 *
 * @param hSocket a socket handle.
 * @param nTimeout (not used).
 * @param hAcceptedSocket (not used).
 *
 * @return @ref ERR_UEM_SKIP_THIS is always returned. \n
 */
uem_result UKSerialCommunication_Accept(HVirtualSocket hSocket, int nTimeout, IN OUT HVirtualSocket hAcceptedSocket);

/**
 * @brief Send data through a serial communication socket.
 *
 * This function sends data through a serial communication socket. \n
 * This is an implementation function of UKVirtualCommunication of fnSend().
 *
 * @param hSocket a socket handle.
 * @param nTimeout timeout value waiting for ready to send.
 * @param pData data to send.
 * @param nDataLen size of data.
 * @param [out] pnSentSize amount of data sent.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 */
uem_result UKSerialCommunication_Send(HVirtualSocket hSocket, IN int nTimeout, IN char *pData, IN int nDataLen, OUT int *pnSentSize);

/**
 * @brief Receive data from a serial communication socket.
 *
 * This function receives data from a serial communication socket. \n
 * This is an implementation function of UKVirtualCommunication of fnReceive().
 *
 * @param hSocket a socket handle.
 * @param nTimeout timeout value waiting for new data to read.
 * @param[in,out] pBuffer pBuffer buffer to store received data.
 * @param nBufferLen size of the buffer.
 * @param[out] pnReceivedSize pnReceivedSize amount of size received.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 */
uem_result UKSerialCommunication_Receive(HVirtualSocket hSocket, IN int nTimeout, IN OUT char *pBuffer, IN int nBufferLen, OUT int *pnReceivedSize);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSERIALCOMMUNICATION_H_ */
