/*
 * uem_multicast_data.h
 *
 *  Created on: 2019. 6. 20.
 *      Author: wecracy
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_UEM_MULTICAST_DATA_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_UEM_MULTICAST_DATA_H_

#include <uem_common.h>

#include <UCThreadMutex.h>
#include <UCThreadEvent.h>
#include <UCThread.h>

#include <uem_enum.h>

#include <uem_memory_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SMulticastPort SMulticastPort;
typedef struct _SMulticastGroup SMulticastGroup;
typedef struct _SMulticastAPI SMulticastAPI;

typedef struct _SMulticastCommunication{
	EMulticastCommunicationType enCommunicationType;
	SMulticastAPI *pstMulticastAPI;
	void *pstSocket;
}SMulticastCommunication;

typedef struct _SMulticastPort {
	int nTaskId;
	int nMulticastPortId;
	const char *pszPortName;
	EPortDirection enDirection;
	SGenericMemoryAccess *pstMemoryAccessAPI; // in GPU or CPU?
	SMulticastGroup *pstMulticastGroup;
	SMulticastCommunication *astCommunicationList;
	int nCommunicationTypeNum;
} SMulticastPort;

typedef struct _SMulticastGroup {
	int nMulticastGroupId;
	const char *pszGroupName;
	int nBufSize;
	SMulticastPort *astInputPort;
	int nInputPortNum;
	SMulticastPort *astOutputPort;
	int nOutputPortNum;
	SMulticastCommunication *astCommunicationList;
	int nCommunicationTypeNum;
} SMulticastGroup;

typedef uem_result (*FnMulticastAPIInitialize)();
typedef uem_result (*FnMulticastAPIFinalize)();
typedef uem_result (*FnMulticastGroupInitialize)(SMulticastGroup *pstMulticastGroup);
typedef uem_result (*FnMulticastGroupFinalize)(SMulticastGroup *pstMulticastGroup);
typedef uem_result (*FnMulticastPortInitialize)(SMulticastPort *pstMulticastPort);
typedef uem_result (*FnMulticastPortFinalize)(SMulticastPort *pstMulticastPort);
typedef uem_result (*FnMulticastReadFromBuffer)(SMulticastPort *pstMulticastPort, IN OUT unsigned char *pBuffer, IN int nDataToRead, OUT int *pnDataRead);
typedef uem_result (*FnMulticastWriteToBuffer)(SMulticastPort *pstMulticastPort, IN unsigned char *pBuffer, IN int nDataToWrite, OUT int *pnDataWritten);

typedef struct _SMulticastAPI {
	FnMulticastAPIInitialize fnAPIInitialize;
	FnMulticastAPIFinalize fnAPIFinalize;
	FnMulticastGroupInitialize fnGroupInitialize;
	FnMulticastGroupFinalize fnGroupFinalize;
	FnMulticastPortInitialize fnPortInitialize;
	FnMulticastPortFinalize fnPortFinalize;
	FnMulticastReadFromBuffer fnReadFromBuffer;
	FnMulticastWriteToBuffer fnWriteToBuffer;
} SMulticastAPI;

extern SMulticastGroup g_astMulticastGroups[];
extern int g_nMulticastGroupNum;

extern SMulticastAPI *g_astMulticastAPIList[];
extern int g_nMulticastAPINum;

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UEM_MULTICAST_DATA_H_ */
