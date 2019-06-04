/*
 * UCSerialPort.h
 *
 *  Created on: 2019. 02. 18, modified from UCDynamicSocket.h & UCBluetoothSocket.h
 *      Author: dowhan1128
 */

#ifndef SRC_COMMON_INCLUDE_UCSERIAL_H_
#define SRC_COMMON_INCLUDE_UCSERIAL_H_

#include <uem_common.h>

#include <UCString.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SSerialPortInfo
{
    // Serial file path
    //for example,      /dev/ttyUSB0 for SERIAL_MASTER(hostPC),
	//     				/dev/ttyAMA0 for SERIAL_SLAVE(raspi),
    char *pszSerialPortPath;
}SSerialPortInfo;

typedef struct _SUCSerialPort
{
	EUemModuleId enID;
    int nSerialfd;
    uem_string_struct stSerialPortPath;
    char *pszSerialPath;
} SUCSerialPort;

typedef struct _SUCSerialPort *HSerialPort;

/**
 * @brief
 *
 * This function
 *
 * @param pstSerialPortInfo
 * @param[out] phSerialPort
 *
 * @return
 */
uem_result UCSerialPort_Create(IN SSerialPortInfo *pstSerialPortInfo, OUT HSerialPort *phSerialPort);

/**
 * @brief
 *
 * This function
 *
 * @param phSerialPort
 *
 * @return
 */
uem_result UCSerialPort_Destroy(IN OUT HSerialPort *phSerialPort);

/**
 * @brief
 *
 * This function
 *
 * @param hClientSerial
 *
 * @return
 */
uem_result UCSerialPort_Open(HSerialPort hClientSerial);

/**
 * @brief
 *
 * This function
 *
 * @param hClientSerial
 *
 * @return
 */
uem_result UCSerialPort_Close(HSerialPort hClientSerial);

/**
 * @brief
 *
 * This function
 *
 * @param hSerialPort
 * @param nTimeout
 * @param pData
 * @param nDataLen
 * @param[out] pnSentSize
 *
 * @return
 */
uem_result UCSerialPort_Send(HSerialPort hSerialPort, IN int nTimeout, IN char *pData, IN int nDataLen, OUT int *pnSentSize);

/**
 * @brief
 *
 * This function
 *
 * @param hSerialPort
 * @param nTimeout
 * @param pBuffer
 * @param nBufferLen
 * @param[out] pnReceivedSize
 *
 * @return
 */
uem_result UCSerialPort_Receive(HSerialPort hSerialPort, IN int nTimeout, IN OUT char *pBuffer, IN int nBufferLen, OUT int *pnReceivedSize);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_INCLUDE_UCSERIAL_H_ */
