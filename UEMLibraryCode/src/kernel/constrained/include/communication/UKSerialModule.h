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
 * @brief Initialize serial communication module.
 *
 * This function initializes serial communication module. \n
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_ILLEGAL_DATA if received message type is @ref MESSAGE_TYPE_HANDSHAKE.
 */
uem_result UKSerialModule_Initialize();

/**
 * @brief (not used) Finalize serial communication module.
 *
 * (not used) This function finalizes serial communication module. \n
 * Currently do nothing.
 *
 * @return This function always returns @ref ERR_UEM_NOERROR.
 */
uem_result UKSerialModule_Finalize();

/**
 * @brief Set channel to SerialInfo channelList.
 *
 * This function sets channel to SerialInfo channelList. \n
 *
 * @param pstSerialInfo a single SerialInfo structure.
 * @param pstChannel a single channel structure.
 *
 * @return  @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_ILLEGAL_DATA if SerialInfo's SetChannelAccessNum is bigger than or equal to MaxChannelAccessNum.
 */
uem_result UKSerialModule_SetChannel(SSerialInfo *pstSerialInfo, SChannel *pstChannel);

/**
 * @brief Run Serial Communication.
 *
 * This function performs serial communication on serial channels. \n
 * Depending on received data message type, this function handle received data/received request and \n
 * make Requesting message.
 *
 * @return  @ref ERR_UEM_NOERROR is returned if there is no error. \n
 * @ref ERR_UEM_INVALID_PARAM if received message type @ref MESSAGE_TYPE_AVAILABLE_INDEX, @ref MESSAGE_TYPE_NONE, or @ref MESSAGE_TYPE_HANDSHAKE, or NULL. \n
 */
uem_result UKSerialModule_Run();


#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_CONSTRAINED_INCLUDE_COMMUNICATION_UKSERIALMODULE_H_ */
