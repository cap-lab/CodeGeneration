/*
 * uem_protocol_data.h
 *
 *  Created on: 2018. 10. 5.
 *      Author: chjej202
 */

#ifndef SRC_KERNEL_INCLUDE_UEM_PROTOCOL_DATA_H_
#define SRC_KERNEL_INCLUDE_UEM_PROTOCOL_DATA_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum _EMessageType {
	MESSAGE_TYPE_HANDSHAKE = 0,
	MESSAGE_TYPE_READ_QUEUE = 1,
	MESSAGE_TYPE_READ_BUFFER = 2,
	MESSAGE_TYPE_AVAILABLE_INDEX = 3,
	MESSAGE_TYPE_AVAILABLE_DATA = 4,
	MESSAGE_TYPE_RESULT = 5,

	MESSAGE_TYPE_NONE = -1,
} EMessageType;

typedef enum _EProtocolError {
	ERR_UEMPROTOCOL_NOERROR = 0,
	ERR_UEMPROTOCOL_ERROR = -1,
	ERR_UEMPROTOCOL_INTERNAL = -2,
} EProtocolError;

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_INCLUDE_UEM_PROTOCOL_DATA_H_ */
