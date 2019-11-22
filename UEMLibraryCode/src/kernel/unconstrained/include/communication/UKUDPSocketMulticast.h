/*
 * UKUDPSocketMulticast.h
 *
 *  Created on: 2018. 6. 20.
 *      Author: wecracy
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKUDPSOCKETMULTICAST_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKUDPSOCKETMULTICAST_H_

#include <uem_common.h>

#include <UCDynamicSocket.h>

#include <uem_data.h>
#include <uem_udp_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SUDPMulticastReceiver SUDPMulticastReceiver;

typedef struct _SUDPMulticast{
	SUDPInfo stUDPInfo;
	int *anReceivers;
	int nReceiverNum;
	int *anSenders;
	int nSenderNum;
	SUDPMulticastReceiver *pstUDPMulticastReceiver;
}SUDPMulticast;

typedef struct _SUDPMulticastReceiver{
	SUDPMulticast *pstUDPMulticast;
	SUDPSocket stReceiverSocket;
	HThread hManagementThread;
	uem_bool bExit;
}SUDPMulticastReceiver;

typedef struct _SUDPMulticastSender{
	SUDPMulticast *pstUDPMulticast;
	SUDPSocket stSenderSocket;
}SUDPMulticastSender;

/**
 * @brief Initialize Multicast UDP Receivers.
 *
 * This function initializes Multicast UDP Receivers. \n
 * This function loads a server and accept clients from different devices.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         It propagates the errors of \n
 *         @ref UKUDPSocketMulticast_AllocBuffer \n
 *         @ref UKUDPSocketMulticast_SocketInitialize \n
 *         @ref UCDynamicSocket_Bind \n
 *         @ref UCThread_Create.
 */
uem_result UKUDPSocketMulticastAPI_Initialize();

/**
 * @brief Initialize a Multicast UDP sender.
 *
 * This function initializes a Multicast UDP sender.
 *
 * @param pstMulticastPort sender
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         It propagates the errors of \n
 *         @ref UKMulticast_GetCommunication \n
 *         @ref UKUDPSocketMulticast_AllocBuffer \n
 *         @ref UKUDPSocketMulticast_SocketInitialize.
 */
uem_result UKUDPSocketMulticastPort_Initialize(IN SMulticastPort *pstMulticastPort);

/**
 * @brief Send data to UDP, multicast
 *
 * This function sends data to the UDP, multicast.
 *
 * @param pstMulticastPort sender.
 * @param pData buffer to send data.
 * @param nDataToWrite buffer size.
 * @param[out] pnDataWritten sent data size.
 *
 * @return
 * @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_CONVERSION_ERROR is returned when fail to convert the header data to little endian. \n
 * It propagates the errors of \n
 * @ref UKMulticast_GetCommunication \n
 * @ref UCDynamicSocket_Sendto \n
 * @ref fnCopyToMemory, which is UKGPUSystem_CopyDeviceToHostMemory (for port located in GPU) or UKHostSystem_CopyToMemory (for port located in CPU).
 */
uem_result UKUDPSocketMulticast_WriteToBuffer(IN SMulticastPort *pstMulticastPort, IN unsigned char *pData, IN int nDataToWrite, OUT int *pnDataWritten);

/**
 * @brief Finalize a Multicast UDP sender.
 *
 * This function finalize a Multicast UDP sender. \n
 *
 * @param pstMulticastPort sender
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error.
 *         error could be propagated from UKMulticast_GetCommunication.
 */
uem_result UKUDPSocketMulticastPort_Finalize(IN SMulticastPort *pstMulticastPort);

/**
 * @brief Finalize Multicast UDP Receivers.
 *
 * This function finalize Multicast UDP Receivers.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error.
 *         error could be propagated from UCThread_Destroy.
 */
uem_result UKUDPSocketMulticastAPI_Finalize();

extern SUDPMulticast g_astMulticastUDPList[];
extern int g_nMulticastUDPNum;

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKUDPSOCKETMULTICAST_H_ */
