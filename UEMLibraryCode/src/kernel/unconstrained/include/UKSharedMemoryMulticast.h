/*
 * UKSharedMemoryMulticast.h
 *
 *  Created on: 2019. 6. 20.
 *      Author: wecracy
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKSHAREDMEMORYMULTICAST_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKSHAREDMEMORYMULTICAST_H_

#include <uem_common.h>

#include <uem_data.h>

#include <UKMulticast.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SSharedMemoryMulticast {
	void *pData;
	int nDataLen;
	HThreadMutex hMutex; // Multicast global mutex
} SSharedMemoryMulticast;

/**
 * @brief Initialize multicast shared memory
 *
 * This function initializes shared memory.
 *
 * @param pstMulticastGroup target group for initializing.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         It propagates the errors of \n
 *         @ref UKMulticast_GetCommunication \n
 *         @ref UKMulticastMemory_Initialize.
 */
uem_result UKSharedMemoryMulticastGroup_Initialize(SMulticastGroup *pstMulticastGroup);

/**
 * @brief receive data from shared memory, multicast
 *
 * This function receives data from the shared memory, multicast.
 *
 * @param pstMulticastPort receiver of data.
 * @param pBuffer buffer to receive data.
 * @param nDataToRead data length to read.
 * @param[out] pnDataRead received data size.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         It propagates the errors of \n
 *         @ref UKMulticast_GetCommunication \n
 *         @ref UKMulticastMemory_ReadFromBuffer.
 */
uem_result UKSharedMemoryMulticast_ReadFromBuffer(SMulticastPort *pstMulticastPort, IN OUT unsigned char *pBuffer, IN int nDataToRead, OUT int *pnDataRead);

/**
 * @brief Send data to shared memory, multicast
 *
 * This function sends data to the shared memory, multicast.
 *
 * @param pstMulticastPort sender.
 * @param pData buffer to send data.
 * @param nDataToWrite buffer size.
 * @param[out] pnDataWritten sent data size.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         It propagates the errors of \n
 *         @ref UKMulticast_GetCommunication \n
 *         @ref UKMulticastMemory_WriteToBuffer.
 */
uem_result UKSharedMemoryMulticast_WriteToBuffer(SMulticastPort *pstMulticastPort, IN unsigned char *pBuffer, IN int nDataToWrite, OUT int *pnDataWritten);

/**
 * @brief Finalizing multicast shared memory
 *
 * This function initializes shared memory.
 *
 * @param pstMulticastGroup target group for Finalizing.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         It propagates the errors of \n
 *         @ref UKMulticast_GetCommunication \n
 *         @ref UKMulticastMemory_Finalize.
 */
uem_result UKSharedMemoryMulticastGroup_Finalize(SMulticastGroup *pstMulticastGroup);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKSHAREDMEMORYMULTICAST_H_ */
