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
 * @brief Create a serial communication port handle.
 *
 * This function creates a serial communication port handle. @ref SSerailPortInfo is needed to create new serial port.
 *
 * @param pstSerialPortInfo serial communication settings
 * @param[out] phSerialPort a serial port handle to be created.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_OUT_OF_MEMORY.
 */
uem_result UCSerialPort_Create(IN SSerialPortInfo *pstSerialPortInfo, OUT HSerialPort *phSerialPort);

/**
 * @brief destroy a serial communication port handle.
 *
 * This function destroys a serial communication port handle.
 *
 * @param[in,out] phSerialPort a serial port handle to be destroyed.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_PARAM, @ref ERR_UEM_INVALID_HANDLE.
 */
uem_result UCSerialPort_Destroy(IN OUT HSerialPort *phSerialPort);

/**
 * @brief Open a serial port.
 *
 * This function opens a serial port which can be communicated with other devices through wire or USB.
 *
 * @param hSerialPort a serial port handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_SERIAL.
 */
uem_result UCSerialPort_Open(HSerialPort hSerialPort);

/**
 * @brief Close a serial port.
 *
 * This function closes a serial port to finish communication.
 *
 * @param hSerialPort a serial port handle.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE.
 */
uem_result UCSerialPort_Close(HSerialPort hSerialPort);


/**
 * @brief Send data through serial port.
 *
 * This function sends data through serial port.
 *
 * @param hSerialPort a serial port handle.
 * @param nTimeout timeout value waiting for ready to send.
 * @param pData data to send.
 * @param nDataLen size of data.
 * @param[out] pnSentSize amount of data transmitted.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, \n
 *         @ref ERR_UEM_NET_SEND_ERROR, @ref ERR_UEM_SELECT_ERROR, @ref ERR_UEM_NET_TIMEOUT. \n
 *         @ref ERR_UEM_NET_SEND_ERROR can be occurred when the write operation is failed. \n
 *         @ref ERR_UEM_SELECT_ERROR can be occurred when the select operation is failed. \n
 *         @ref ERR_UEM_NET_TIMEOUT can be occurred if the write is not available until timeout.
 */
uem_result UCSerialPort_Send(HSerialPort hSerialPort, IN int nTimeout, IN char *pData, IN int nDataLen, OUT int *pnSentSize);

/**
 * @brief Receive data from serial port.
 *
 * This function receives data from serial port.
 *
 * @param hSerialPort a serial port handle
 * @param nTimeout timeout value waiting for new data to read.
 * @param[in,out] pBuffer buffer to store received data.
 * @param nBufferLen size of the buffer.
 * @param[out] pnReceivedSize amount of size received.
 *
 * @return @ref ERR_UEM_NOERROR is returned if there is no error. \n
 *         Errors to be returned - @ref ERR_UEM_INVALID_HANDLE, @ref ERR_UEM_INVALID_PARAM, \n
 *         @ref ERR_UEM_NET_RECEIVE_ERROR, @ref ERR_UEM_SELECT_ERROR, @ref ERR_UEM_NET_TIMEOUT. \n
 *         @ref ERR_UEM_NET_RECEIVE_ERROR can be occurred when the read operation is failed. \n
 *         @ref ERR_UEM_SELECT_ERROR can be occurred when the select operation is failed. \n
 *         @ref ERR_UEM_NET_TIMEOUT can be occurred if the write is not available until timeout.
 */
uem_result UCSerialPort_Receive(HSerialPort hSerialPort, IN int nTimeout, IN OUT char *pBuffer, IN int nBufferLen, OUT int *pnReceivedSize);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_INCLUDE_UCSERIAL_H_ */
