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
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_OUT_OF_MEMORY, and \n
 *         errors corresponding to @ref SVirtualCommunicationAPI fnCreate() and fnListen().
 */
uem_result UKUDPSocketMulticastAPI_Initialize();

/**
 * @brief Initialize a Multicast UDP sender.
 *
 * This function initializes a Multicast UDP sender. \n
 * This function loads a server and accept clients from different devices.
 *
 * @param pstMulticastPort sender of the UDP
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_OUT_OF_MEMORY, and \n
 *         errors corresponding to @ref SVirtualCommunicationAPI fnCreate() and fnListen().
 */
uem_result UKUDPSocketMulticastPort_Initialize(IN SMulticastPort *pstMulticastPort);
uem_result UKUDPSocketMulticast_WriteToBuffer(IN SMulticastPort *pstMulticastPort, IN unsigned char *pData, IN int nDataToWrite, OUT int *pnDataWritten);
uem_result UKUDPSocketMulticastPort_Finalize(IN SMulticastPort *pstMulticastPort);
uem_result UKUDPSocketMulticastAPI_Finalize();

extern SUDPMulticast g_astMulticastUDPList[];
extern int g_nMulticastUDPNum;

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKUDPSOCKETMULTICAST_H_ */
