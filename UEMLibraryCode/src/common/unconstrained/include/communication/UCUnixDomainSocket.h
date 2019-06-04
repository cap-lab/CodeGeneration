/*
 * UCUnixDomainSocket.h
 *
 *  Created on: 2018. 10. 3.
 *      Author: chjej202
 */

#ifndef SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_UDS_UCUNIXDOMAINSOCKET_H_
#define SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_UDS_UCUNIXDOMAINSOCKET_H_

#include <uem_common.h>

#include <UCDynamicSocket.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief
 *
 * This function
 *
 * @param hSocket
 *
 * @return
 */
uem_result UCUnixDomainSocket_Bind(HSocket hSocket);

/**
 * @brief
 *
 * This function
 *
 * @param hServerSocket
 * @param hClientSocket
 *
 * @return
 */
uem_result UCUnixDomainSocket_Accept(HSocket hServerSocket, HSocket hClientSocket);

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
uem_result UCUnixDomainSocket_Connect(HSocket hSocket, IN int nTimeout);

/**
 * @brief
 *
 * This function
 *
 * @param hSocket
 *
 * @return
 */
uem_result UCUnixDomainSocket_Destroy(HSocket hSocket);

#ifdef __cplusplus
}
#endif


#endif /* SRC_COMMON_UNCONSTRAINED_INCLUDE_COMMUNICATION_UDS_UCUNIXDOMAINSOCKET_H_ */
