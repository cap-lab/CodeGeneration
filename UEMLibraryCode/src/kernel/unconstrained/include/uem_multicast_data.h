/*
 * uem_multicast_data.h
 *
 *  Created on: 2019. 3. 30.
 *      Author: wecracy
 */

#ifndef SRC_KERNEL_INCLUDE_UEM_MULTICAST_DATA_H_
#define SRC_KERNEL_INCLUDE_UEM_MULTICAST_DATA_H_

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

typedef struct _SSharedMemoryMulticast {
	void *pBuffer;
	void *pDataStart;
	void *pDataEnd;
	int nDataLen;
	int nReadReferenceCount;
	int nWriteReferenceCount;
	HThreadMutex hMutex; // Multicast global mutex
	SGenericMemoryAccess *pstMemoryAccessAPI;
} SSharedMemoryMulticast;

typedef struct _SMulticastPort SMulticastPort;

typedef struct _SMulticastPort {
	int nTaskId;
	int nMulticastPortId;
	const char *pszPortName;
    EPortDirection eDirection;
	SGenericMemoryAccess *pstMemoryAccessAPI; // in GPU or CPU?
	SMulticastGroup *pMulticastGroup;
	void *pMulticastSendGateList; // only for Output Port
} SMulticastPort;

typedef struct _SMulticastCommunicationInfo {
	EMulticastCommunicationType eCommunicationType;
	void *pAdditionalCommunicationInfo;
}SMulticastCommunicationInfo;

typedef struct _SMulticastGroup {
	int nMulticastGroupId;
	const char *pszGroupName;
	int nBufSize;
	SMulticastPort *pstInputPort;
	int nInputPortNum;
	SMulticastCommunicationInfo **pstInputCommunicationInfo;
    int nInputCommunicationTypeNum;
	SMulticastPort *pstOutputPort;
	int nOutputPortNum;
	SMulticastCommunicationInfo **pstOutputCommunicationInfo;
    int nOutputCommunicationTypeNum;
	SSharedMemoryMulticast *pMulticastStruct;
	void **pMulticastRecvGateList;
} SMulticastGroup;

typedef uem_result (*FnMulticastInitialize)(SMulticastGroup *pstMulticastGroup);
typedef uem_result (*FnMulticastReadFromBuffer)(SMulticastGroup *pstMulticastGroup, IN OUT unsigned char *pBuffer, IN int nDataToRead, IN int nChunkIndex, OUT int *pnDataRead);
typedef uem_result (*FnMulticastWriteToBuffer)(SMulticastGroup *pstMulticastGroup, IN unsigned char *pBuffer, IN int nDataToWrite, IN int nChunkIndex, OUT int *pnDataWritten);
typedef uem_result (*FnMulticastClear)(SMulticastGroup *pstMulticastGroup);
typedef uem_result (*FnMulticastFinalize)(SMulticastGroup *pstMulticastGroup);
typedef uem_result (*FnMulticastAPIInitialize)();
typedef uem_result (*FnMulticastAPIFinalize)();

typedef struct _SMulticastAPI {
	FnMulticastInitialize fnInitialize;
	FnMulticastReadFromBuffer fnReadFromBuffer;
	FnMulticastWriteToBuffer fnWriteToBuffer;
	FnMulticastClear fnClear;
	FnMulticastFinalize fnFinalize;
	FnMulticastAPIInitialize fnAPIInitialize;
	FnMulticastAPIFinalize fnAPIFinalize;
} SMulticastAPI;

extern SMulticastGroup g_astMulticastGroups[];
extern int g_nMulticastGroupNum;

extern SMulticastAPI *g_astMulticastAPIList[];
extern int g_nMulticastAPINum;

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UEM_MULTICAST_DATA_H_ */
