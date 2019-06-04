/*
 * UKSerialModule.h
 *
 *  Created on: 2018. 10. 25.
 *      Author: jej
 */

#ifndef SRC_KERNEL_CONSTRAINED_INCLUDE_COMMUNICATION_UKSERIALMODULE_H_
#define SRC_KERNEL_CONSTRAINED_INCLUDE_COMMUNICATION_UKSERIALMODULE_H_

#include <uem_common.h>

#include <uem_channel_data.h>
#include <uem_bluetooth_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief
 *
 * This function
 *
 * @return
 */
uem_result UKSerialModule_Initialize();

/**
 * @brief
 *
 * This function
 *
 * @return
 */
uem_result UKSerialModule_Finalize();

/**
 * @brief
 *
 * This function
 *
 * @param pstSerialInfo
 * @param pstChannel
 *
 * @return
 */
uem_result UKSerialModule_SetChannel(SSerialInfo *pstSerialInfo, SChannel *pstChannel);

/**
 * @brief
 *
 * This function
 *
 * @return
 */
uem_result UKSerialModule_Run();


#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_CONSTRAINED_INCLUDE_COMMUNICATION_UKSERIALMODULE_H_ */
