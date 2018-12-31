/*
 * uem_lite_protocol_data.h
 *
 *  Created on: 2018. 10. 25.
 *      Author: jej
 */

#ifndef SRC_KERNEL_INCLUDE_UEM_LITE_PROTOCOL_DATA_H_
#define SRC_KERNEL_INCLUDE_UEM_LITE_PROTOCOL_DATA_H_

#include <uem_common.h>

#include <uem_protocol_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define HANDSHAKE_DEVICE_KEY_LSB_INDEX (0)
#define HANDSHAKE_DEVICE_KEY_MSB_INDEX (1)

#define READ_QUEUE_CHANNEL_ID_INDEX (0)
#define READ_QUEUE_SIZE_TO_READ_INDEX (1)

#define READ_BUFFER_CHANNEL_ID_INDEX (0)
#define READ_BUFFER_SIZE_TO_READ_INDEX (1)

#define AVAILABLE_DATA_CHANNEL_ID_INDEX (0)

#define RESULT_CHANNEL_ID_INDEX (0)
#define RESULT_REQUEST_PACKET_INDEX (1)
#define RESULT_ERROR_CODE_INDEX (2)
#define RESULT_BODY_SIZE_INDEX (3)
#define RESULT_RETURN_VALUE_INDEX (3)


#define MAX_MESSAGE_PARAMETER (4)

#define HANDSHAKE_TIMEOUT (3)
#define SEND_TIMEOUT (3)
#define RECEIVE_TIMEOUT (3)

#define PRE_HEADER_LENGTH (1)
#define MESSAGE_PACKET_SIZE (1)
#define MESSAGE_PARAMETER_SIZE (2)
#define HANDSHAKE_RETRY_COUNT (3)

#define MAX_HEADER_LENGTH (MESSAGE_PACKET_SIZE + (MESSAGE_PARAMETER_SIZE * MAX_MESSAGE_PARAMETER))
#define MIN_HEADER_LENGTH MESSAGE_PACKET_SIZE


#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_INCLUDE_UEM_LITE_PROTOCOL_DATA_H_ */
