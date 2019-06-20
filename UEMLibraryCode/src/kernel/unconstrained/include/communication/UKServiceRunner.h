/*
 * UKServiceRunner.h
 *
 *  Created on: 2019. 5. 27.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSERVICERUNNER_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSERVICERUNNER_H_

#include <uem_common.h>

#include <UKVirtualCommunication.h>

#include <uem_remote_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Start aggregated service.
 *
 * This function starts aggregated service which multiple channels share single connection to communicate with another device. \n
 * This function is used when the connection role is a slave.
 *
 * @param pstServiceInfo structure of aggregate service to start.
 * @param pSocketInfo socket information going to be used by aggregate service.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         @ref ERR_UEM_SKIP_THIS is returned when this service does not have any corresponding channels. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_OUT_OF_MEMORY, and \n
 *         errors corresponding to @ref SVirtualCommunicationAPI fnCreate().
 */
uem_result UKServiceRunner_StartAggregatedService(SAggregateServiceInfo *pstServiceInfo, void *pSocketInfo);

/**
 * @brief Stop aggregated service.
 *
 * This function stops aggregated service.
 *
 * @param pstServiceInfo structure of aggregate service to stop.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKServiceRunner_StopAggregatedService(SAggregateServiceInfo *pstServiceInfo);

/**
 * @brief Start aggregated client service.
 *
 * This function starts aggregated client service which multiple channels share single connection to communicate with another device. \n
 * This function is used when the connection role is a master.
 *
 * @param pstServiceInfo structure of aggregate client service to start.
 * @param pSocketInfo socket information going to be used by aggregate service.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         @ref ERR_UEM_SKIP_THIS is returned when this service does not have any corresponding channels. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_OUT_OF_MEMORY, and \n
 *         errors corresponding to @ref SVirtualCommunicationAPI fnCreate().
 */
uem_result UKServiceRunner_StartAggregatedClientService(SAggregateServiceInfo *pstServiceInfo, void *pSocketInfo);

/**
 * @brief Stop aggregated client service.
 *
 * This function stops aggregated client service.
 *
 * @param pstServiceInfo structure of aggregate client service to stop.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM.
 */
uem_result UKServiceRunner_StopAggregatedClientService(SAggregateServiceInfo *pstServiceInfo);

/**
 * @brief Start individual service.
 *
 * This function starts individual service which each channel has their own connection. \n
 * This service is needed when the device role is a server. \n
 * Individual service accepts clients and pass the socket information to each corresponding channels. \n
 *
 * @param pstServiceInfo structure of individual service to start.
 * @param pSocketInfo socket information going to be used by individual service.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_OUT_OF_MEMORY, and \n
 *         errors corresponding to @ref SVirtualCommunicationAPI fnCreate() and fnListen().
 */
uem_result UKServiceRunner_StartIndividualService(SIndividualServiceInfo *pstServiceInfo, void *pSocketInfo);

/**
 * @brief Stop individual service.
 *
 * This function stops individual service.
 *
 * @param pstServiceInfo structure of individual service to stop.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INTERNAL_FAIL, and \n
 *         errors corresponding to @ref SVirtualCommunicationAPI fnDestroy(). \n
 *         @ref ERR_UEM_INTERNAL_FAIL can be occurred when threads created by this module is not terminated properly.
 */
uem_result UKServiceRunner_StopIndividualService(SIndividualServiceInfo *pstServiceInfo);

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSERVICERUNNER_H_ */
