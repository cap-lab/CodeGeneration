/*
 * UKSerialCommunicationManager.c
 *
 *  Created on: 2018. 10. 6.
 *      Author: chjej202
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UKUEMLiteProtocol.h>

#include <UKSerialCommunicationManager.h>

typedef struct _SSerialCommunicationManager {
	HUEMLiteProtocol hSendProtocol;
	HUEMLiteProtocol hReceiveProtocol;
} SSerialCommunicationManager;

