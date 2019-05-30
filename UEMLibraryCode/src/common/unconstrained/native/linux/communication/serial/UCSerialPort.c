/*
 * UCSerialPort.c, modified from UCDynamicSocket.c & UCBluetoothSocket.c
 *
 *  Created on: 2019. 02. 18.
 *      Author: dowhan1128
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

// this code is not run on Windows because I didn't call any WSAStartup or WSACleanup.
// ifdefs are used for removing compile errors on mingw32 build

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/types.h>
#include <termios.h>

#include <UCBasic.h>
#include <UCAlloc.h>
#include <UCString.h>
#include <UCSerialPort.h>

#define SERIAL_FD_NOT_SET (-1)
#define error_message(format, args...) fprintf(stderr, format, args)

#ifndef DEFAULT_BAUD_RATE
	#define DEFAULT_BAUD_RATE (38400) //should be same with Arduino DATA_SERIAL_DEFAULT_BAUD_RATE when communicating with Arduino
#endif


static uem_result setSerialPath(SUCSerialPort *pstSerialPort, char *pszSerialPath)
{
    uem_result result = ERR_UEM_UNKNOWN;
    uem_string_struct stInputPath;

    result = UCString_New(&stInputPath, pszSerialPath, UEMSTRING_CONST);
    ERRIFGOTO(result, _EXIT);

    if(UCString_Length(&stInputPath) > 0)
    {
        pstSerialPort->pszSerialPath = (char *) UCAlloc_calloc(UCString_Length(&stInputPath)+1, sizeof(char));
        ERRMEMGOTO(pstSerialPort->pszSerialPath, result, _EXIT);

        result = UCString_New(&(pstSerialPort->stSerialPortPath), pstSerialPort->pszSerialPath, (UCString_Length(&stInputPath)+1) * sizeof(char));
        ERRIFGOTO(result, _EXIT);

        result = UCString_Set(&(pstSerialPort->stSerialPortPath), &stInputPath);
        ERRIFGOTO(result, _EXIT);
    }

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

uem_result UCSerialPort_Create(IN SSerialPortInfo *pstSerialPortInfo, OUT HSerialPort *phSerialPort)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSerialPort *pstSerialPort = NULL;
#ifdef ARGUMENT_CHECK
    IFVARERRASSIGNGOTO(pstSerialPortInfo, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
    IFVARERRASSIGNGOTO(phSerialPort, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif

    pstSerialPort = (SUCSerialPort *) UCAlloc_malloc(sizeof(SUCSerialPort));
    ERRMEMGOTO(pstSerialPort, result, _EXIT);

    pstSerialPort->enID = ID_UEM_SERIAL;
    pstSerialPort->nSerialfd = SERIAL_FD_NOT_SET; //not yet open port.
    pstSerialPort->pszSerialPath = NULL;

    if(pstSerialPortInfo->pszSerialPortPath != NULL) // socket path is used
    {
        result = setSerialPath(pstSerialPort, pstSerialPortInfo->pszSerialPortPath);
        ERRIFGOTO(result, _EXIT);
    }

    *phSerialPort = (HSerialPort) pstSerialPort;

    result = ERR_UEM_NOERROR;
_EXIT:
    if(result != ERR_UEM_NOERROR && pstSerialPort != NULL)
    {
        UCSerialPort_Destroy((HSerialPort *)&pstSerialPort);
    }
    return result;
}

uem_result UCSerialPort_Destroy(IN OUT HSerialPort *phSerialPort)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSerialPort *pstSerialPort = NULL;
#ifdef ARGUMENT_CHECK
    IFVARERRASSIGNGOTO(phSerialPort, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

    if(IS_VALID_HANDLE(*phSerialPort, ID_UEM_SERIAL) == FALSE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
    }
#endif
    pstSerialPort = (SUCSerialPort *) *phSerialPort;


    if(pstSerialPort->nSerialfd != SERIAL_FD_NOT_SET)
    {
		UCSerialPort_Close(pstSerialPort);
    }

    SAFEMEMFREE(pstSerialPort->pszSerialPath);
    SAFEMEMFREE(pstSerialPort);

    *phSerialPort = NULL;

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

//timeout check.
//for example, passing pstReadSet arguments makes it wait until read or timeout.
static uem_result selectTimeout(int nSerialfd, fd_set *pstReadSet, fd_set *pstWriteSet, fd_set *pstExceptSet, int nTimeout)
{
    uem_result result = ERR_UEM_UNKNOWN;
    struct timeval stTimeVal;
    int nRet = 0;

    if(pstReadSet != NULL)
    {
        FD_ZERO(pstReadSet);
        FD_SET(nSerialfd, pstReadSet);
    }

    if(pstWriteSet != NULL)
    {
        FD_ZERO(pstWriteSet);
        FD_SET(nSerialfd, pstWriteSet);
    }

    if(pstExceptSet != NULL)
   {
       FD_ZERO(pstExceptSet);
       FD_SET(nSerialfd, pstExceptSet);
   }

    stTimeVal.tv_sec = nTimeout;
    stTimeVal.tv_usec = 0;

    nRet = select(nSerialfd+1, pstReadSet, pstWriteSet, pstExceptSet, &stTimeVal);
    if(nRet < 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_SELECT_ERROR, _EXIT);
    }
    else if(nRet == 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_NET_TIMEOUT, _EXIT);
    }
    else
    {
        result = ERR_UEM_NOERROR;
    }
_EXIT:
    return result;
}


//convert conventional baud_rate form to appropriate form
static speed_t setBaudRate(int baud_rate_int){
	switch(baud_rate_int){
			case 0       : return B0      ;                /* hang up */
			case 50      : return B50     ;
			case 75      : return B75     ;
			case 110     : return B110    ;
			case 134     : return B134    ;
			case 150     : return B150    ;
			case 200     : return B200    ;
			case 300     : return B300    ;
			case 600     : return B600    ;
			case 1200    : return B1200   ;
			case 1800    : return B1800   ;
			case 2400    : return B2400   ;
			case 4800    : return B4800   ;
			case 9600    : return B9600   ;
			case 19200   : return B19200  ;
			case 38400   : return B38400  ;
			case 57600   : return B57600  ;
			case 115200  : return B115200 ;
			case 230400  : return B230400 ;
			case 460800  : return B460800 ;
			case 500000  : return B500000 ;
			case 576000  : return B576000 ;
			case 921600  : return B921600 ;
			case 1000000 : return B1000000;
			case 1152000 : return B1152000;
			case 1500000 : return B1500000;
			case 2000000 : return B2000000;
			case 2500000 : return B2500000;
			case 3000000 : return B3000000;
			case 3500000 : return B3500000;
			case 4000000 : return B4000000;
		default : return -1; //error case
	}
}

static int open_port(char str[])
{
	 //int fd = open(str, O_RDWR | O_NOCTTY | O_NONBLOCK); // ?? NDELAY or NONBLOCK?
	    int fd = open(str, O_RDWR | O_NOCTTY | O_SYNC); // ?? NDELAY or NONBLOCK?

	  if (fd == -1)
	  {
	        error_message("open_port: Unable to open %s\n",str);
	  }
	  else
	        fcntl(fd, F_SETFL, 0);

	  struct termios options;
	  memset (&options, 0, sizeof options);
	  tcgetattr(fd, &options); //this gets the current options set for the port
	  if (tcgetattr (fd, &options) != 0)
	  {
	      error_message ("error %d from tcgetattr", errno);
	      return -1;
	  }

	  // setting the options


	  cfsetispeed(&options, setBaudRate(DEFAULT_BAUD_RATE)); //input baudrate
	  cfsetospeed(&options, setBaudRate(DEFAULT_BAUD_RATE)); // output baudrate
	  options.c_cflag = (options.c_cflag & ~CSIZE) | CS8;     // 8-bit chars
	  options.c_iflag &= ~IGNBRK;         // disable break processing
	  options.c_lflag = 0;                // no signaling chars, no echo,
	  // no canonical processing
	  options.c_oflag = 0;                // no remapping, no delays
	  options.c_cc[VMIN]  = 1;            // read doesn't block
	  options.c_cc[VTIME] = 0;            // 0.5 seconds read timeout

	  options.c_iflag &= ~(IXON | IXOFF | IXANY); // shut off xon/xoff ctrl

	  options.c_cflag |= (CLOCAL | CREAD);// ignore modem controls,
	  // enable reading
	  options.c_cflag &= ~(PARENB | PARODD);      // shut off parity
	  options.c_cflag |= 0;
	  options.c_cflag &= ~CSTOPB;
	  options.c_cflag &= ~CRTSCTS;

	  if (tcsetattr (fd, TCSANOW, &options) != 0)
	  {
	      error_message ("error %d from tcsetattr", errno);
	      return -1;
	  }

	  return (fd);

}

uem_result UCSerialPort_Open(HSerialPort hSerialPort)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSerialPort *pstSerialPort = NULL;

#ifdef ARGUMENT_CHECK
    if(IS_VALID_HANDLE(hSerialPort, ID_UEM_SERIAL) == FALSE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
    }
#endif
    pstSerialPort = (SUCSerialPort *) hSerialPort;

    char *portname = pstSerialPort->pszSerialPath;
	int nSerialfd = open_port(portname);
	if (nSerialfd < 0)
	{
		error_message ("error %d opening %s: %s", errno, portname, strerror (errno));
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_SERIAL, _EXIT);
	}
	hSerialPort->nSerialfd = nSerialfd;

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

uem_result UCSerialPort_Close(HSerialPort hSerialPort)
{
    uem_result result = ERR_UEM_NOERROR;

    if(hSerialPort->nSerialfd != SERIAL_FD_NOT_SET)
    {
    	close(hSerialPort->nSerialfd);
    	hSerialPort->nSerialfd = SERIAL_FD_NOT_SET;
    }

    return result;
}

uem_result UCSerialPort_Send(HSerialPort hSerialPort, IN int nTimeout, IN char *pData, IN int nDataLen, OUT int *pnSentSize)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSerialPort *pstSerialPort = NULL;
    fd_set stWriteSet;
    int nDataSent = 0;
#ifdef ARGUMENT_CHECK
    IFVARERRASSIGNGOTO(pData, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

    if(IS_VALID_HANDLE(hSerialPort, ID_UEM_SERIAL) == FALSE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
    }

    if(nTimeout <= 0 || nDataLen <= 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
    }
#endif
    pstSerialPort = (SUCSerialPort *) hSerialPort;

    result = selectTimeout(pstSerialPort->nSerialfd, NULL, &stWriteSet, NULL, nTimeout);
    ERRIFGOTO(result, _EXIT);

    nDataSent = write(pstSerialPort->nSerialfd, pData, nDataLen);
	usleep ((nDataLen + 25) * 100);             // sleep enough to transmit. //TODO : check whether this sleep code is needed or not.

    if(nDataSent < 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_NET_SEND_ERROR, _EXIT);
    }

    if(pnSentSize != NULL)
    {
        *pnSentSize = nDataSent;
    }

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

uem_result UCSerialPort_Receive(HSerialPort hSerialPort, IN int nTimeout, IN OUT char *pBuffer, IN int nBufferLen, OUT int *pnReceivedSize)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSerialPort *pstSerialPort = NULL;
    fd_set stReadSet;
    int nDataReceived = 0;
#ifdef ARGUMENT_CHECK
    IFVARERRASSIGNGOTO(pBuffer, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

    if(IS_VALID_HANDLE(hSerialPort, ID_UEM_SERIAL) == FALSE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_HANDLE, _EXIT);
    }

    if(nTimeout <= 0 || nBufferLen <= 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
    }
#endif
    pstSerialPort = (SUCSerialPort *) hSerialPort;

    result = selectTimeout(pstSerialPort->nSerialfd, &stReadSet, NULL, NULL, nTimeout);
    ERRIFGOTO(result, _EXIT);

    nDataReceived = read(hSerialPort->nSerialfd, pBuffer, nBufferLen); // read up to nBufferLen characters

    if(nDataReceived <= 0)
    {
        ERRASSIGNGOTO(result, ERR_UEM_NET_RECEIVE_ERROR, _EXIT);
    }

    if(pnReceivedSize != NULL)
    {
        *pnReceivedSize = nDataReceived;
    }

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}
