/*
 * UKMulticastMemory.h
 *
 *  Created on: 2019. 6. 20.
 *      Author: wecracy
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKMULTICASTMEMORY_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKMULTICASTMEMORY_H_

#include <uem_common.h>

#include <uem_data.h>

#include <UKSharedMemoryMulticast.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Initialize multicast shared memory.
 *
 * This function initializes shared memory.
 *
 * @param pstMulticastGroup target group for initializing.
 * @param pstSharedMemoryMulticast shared memory socket for target group.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         @ref ERR_UEM_INVALID_PARAM is returned if pstSharedMemoryMulticast is null. \n
 *         error could be propagated from @ref UCThreadMutex_Create.
 */
uem_result UKMulticastMemory_Initialize(SMulticastGroup *pstMulticastGroup, SSharedMemoryMulticast *pstSharedMemoryMulticast);

/**
 * @brief receive data from shared memory, multicast
 *
 * This function receives data by the shared memory, multicast.
 *
 * @param pstMulticastPort receiver of data.
 * @param pstSharedMemoryMulticast shared memory socket for target group.
 * @param pBuffer buffer to receive data.
 * @param nDataToRead data length to read.
 * @param[out] pnDataRead received data size.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         @ref ERR_UEM_INVALID_PARAM is returned if pstSharedMemoryMulticast is null. \n
 *         It propagates the errors of \n
 *         @ref UCThreadMutex_Lock \n
 *         fnCopyFromMemory, which is @ref UKGPUSystem_CopyHostToDeviceMemory (for port located in GPU) or @ref UKHostSystem_CopyFromMemory (for port located in CPU).
 */
uem_result UKMulticastMemory_ReadFromBuffer(SMulticastPort *pstMulticastPort, SSharedMemoryMulticast *pstSharedMemoryMulticast, IN OUT unsigned char *pBuffer, IN int nDataToRead, OUT int *pnDataRead);

/**
 * @brief Send data to shared memory, multicast
 *
 * This function sends data to the shared memory, multicast.
 *
 * @param pstMulticastPort sender.
 * @param pstSharedMemoryMulticast shared memory socket for target group.
 * @param pBuffer buffer to send data.
 * @param nDataToWrite buffer size.
 * @param[out] pnDataWritten sent data size.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         @ref ERR_UEM_INVALID_PARAM is returned if pstSharedMemoryMulticast is null. \n
 *         It propagates the errors of \n
 *         @ref UCThreadMutex_Lock \n
 *         fnCopyToMemory, which is @ref UKGPUSystem_CopyDeviceToHostMemory (for port located in GPU) or @ref UKHostSystem_CopyToMemory (for port located in CPU).
 */
uem_result UKMulticastMemory_WriteToBuffer(SMulticastPort *pstMulticastPort, SSharedMemoryMulticast *pstSharedMemoryMulticast, IN unsigned char *pBuffer, IN int nDataToWrite, OUT int *pnDataWritten);

/**
 * @brief Finalize multicast shared memory.
 *
 * This function finalizes shared memory.
 *
 * @param pstMulticastGroup target group for finalizing.
 * @param pstSharedMemoryMulticast shared memory socket for target group.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         @ref ERR_UEM_INVALID_PARAM is returned if pstSharedMemoryMulticast is null. \n
 *         error could be propagated from @ref UCThreadMutex_Destroy.
 */
uem_result UKMulticastMemory_Finalize(SMulticastGroup *pstMulticastGroup, SSharedMemoryMulticast *pstSharedMemoryMulticast);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKMULTICASTMEMORY_H_ */
