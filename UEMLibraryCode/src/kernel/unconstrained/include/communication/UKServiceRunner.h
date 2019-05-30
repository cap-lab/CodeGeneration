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


uem_result UKServiceRunner_StartAggregatedService(SAggregateServiceInfo *pstServiceInfo, void *pSocketInfo);
uem_result UKServiceRunner_StopAggregatedService(SAggregateServiceInfo *pstServiceInfo);
uem_result UKServiceRunner_StartAggregatedClientService(SAggregateServiceInfo *pstServiceInfo, void *pSocketInfo);
uem_result UKServiceRunner_StopAggregatedClientService(SAggregateServiceInfo *pstServiceInfo);

uem_result UKServiceRunner_StartIndividualService(SIndividualServiceInfo *pstServiceInfo, void *pSocketInfo);
uem_result UKServiceRunner_StopIndividualService(SIndividualServiceInfo *pstServiceInfo);


#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_COMMUNICATION_UKSERVICERUNNER_H_ */
