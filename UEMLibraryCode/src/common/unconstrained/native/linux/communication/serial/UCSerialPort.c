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
	#define DEFAULT_BAUD_RATE (9600)
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

uem_result UCSerialPort_Create(IN SSerialPortInfo *pstSerialPortInfo, IN uem_bool bIsServer, OUT HSerialPort *phSerialPort)
{
    uem_result result = ERR_UEM_UNKNOWN;
    SUCSerialPort *pstSerialPort = NULL;
#ifdef ARGUMENT_CHECK
    IFVARERRASSIGNGOTO(pstSerialPortInfo, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
    IFVARERRASSIGNGOTO(phSerialPort, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

    if(bIsServer != TRUE && bIsServer != FALSE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
    }
    if(bIsServer == TRUE && pstSerialPortInfo->pszSerialPortPath == NULL)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
    }
#endif

    pstSerialPort = (SUCSerialPort *) UCAlloc_malloc(sizeof(SUCSerialPort));
    ERRMEMGOTO(pstSerialPort, result, _EXIT);

    pstSerialPort->enID = ID_UEM_SERIAL;
    pstSerialPort->bIsServer = bIsServer;
    pstSerialPort->portfd = SERIAL_FD_NOT_SET; //not yet open port.
    pstSerialPort->nSerialfd = SERIAL_FD_NOT_SET;
    pstSerialPort->pszSerialPath = pstSerialPortInfo->pszSerialPortPath;

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


    if(pstSerialPort->portfd != SERIAL_FD_NOT_SET)
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
static uem_result selectTimeout(int portfd, fd_set *pstReadSet, fd_set *pstWriteSet, fd_set *pstExceptSet, int nTimeout)
{
    uem_result result = ERR_UEM_UNKNOWN;
    struct timeval stTimeVal;
    int nRet = 0;

    if(pstReadSet != NULL)
    {
        FD_ZERO(pstReadSet);
        FD_SET(portfd, pstReadSet);
    }

    if(pstWriteSet != NULL)
    {
        FD_ZERO(pstWriteSet);
        FD_SET(portfd, pstWriteSet);
    }

    if(pstExceptSet != NULL)
   {
       FD_ZERO(pstExceptSet);
       FD_SET(portfd, pstExceptSet);
   }

    stTimeVal.tv_sec = nTimeout;
    stTimeVal.tv_usec = 0;

    nRet = select(portfd+1, pstReadSet, pstWriteSet, pstExceptSet, &stTimeVal);
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
		case 0       : return 0000000;                /* hang up */
		case 50      : return 0000001;
		case 75      : return 0000002;
		case 110     : return 0000003;
		case 134     : return 0000004;
		case 150     : return 0000005;
		case 200     : return 0000006;
		case 300     : return 0000007;
		case 600     : return 0000010;
		case 1200    : return 0000011;
		case 1800    : return 0000012;
		case 2400    : return 0000013;
		case 4800    : return 0000014;
		case 9600    : return 0000015;
		case 19200   : return 0000016;
		case 38400   : return 0000017;
		case 57600   : return 0010001;
		case 115200  : return 0010002;
		case 230400  : return 0010003;
		case 460800  : return 0010004;
		case 500000  : return 0010005;
		case 576000  : return 0010006;
		case 921600  : return 0010007;
		case 1000000 : return 0010010;
		case 1152000 : return 0010011;
		case 1500000 : return 0010012;
		case 2000000 : return 0010013;
		case 2500000 : return 0010014;
		case 3000000 : return 0010015;
		case 3500000 : return 0010016;
		case 4000000 : return 0010017;
		default : return -1; //error case
	}
}

static int open_port(char str[])
{
    int fd = open(str, O_RDWR | O_NOCTTY | O_SYNC); // O_SYNC? NDELAY or NONBLOCK?
    //NOCTTY : this program doesn't want to be the "controlling terminal" for that port.

    if (fd == -1)
    {
        perror("open_port: Unable to open port. ");
        return fd;
    }
    else
        fcntl(fd, F_SETFL, 0);

    struct termios options;
    tcgetattr(fd, &options); //this gets the current options set for the port

    // setting the options
    cfsetispeed(&options, setBaudRate(DEFAULT_BAUD_RATE)); //input baudrate
    cfsetospeed(&options, setBaudRate(DEFAULT_BAUD_RATE)); // output baudrate
    options.c_cflag |= (CLOCAL | CREAD); // ?? enable receicer and set local mode
    //options.c_cflag &= ~CSIZE; /* mask the character size bits */
    options.c_cflag |= CS8;    /* select 8 data bits */
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); // choosing raw input
    options.c_iflag &= ~INPCK; // disable parity check
    options.c_iflag &= ~(IXON | IXOFF | IXANY); // disable software flow control
    options.c_oflag |= OPOST; // ?? choosing processed output
    options.c_cc[VMIN] = 1; // Wait until x bytes read (blocks!)
    options.c_cc[VTIME] = 0; // Wait x * 0.1s for input (unblocks!)

    // settings for no parity bit
    options.c_cflag &= ~PARENB;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;

    tcsetattr(fd, TCSANOW, &options); //set the new options ... TCSANOW specifies all option changes to occur immediately

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
#ifdef ARGUMENT_CHECK
    if(pstSerialPort->bIsServer == TRUE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_SERIAL, _EXIT);
    }
#endif

    char *portname = pstSerialPort->pszSerialPath;
	int portfd = open_port(portname);
	if (portfd < 0)
	{
			error_message ("error %d opening %s: %s", errno, portname, strerror (errno));
	}
	hSerialPort->portfd = portfd;

    result = ERR_UEM_NOERROR;
_EXIT:
    return result;
}

uem_result UCSerialPort_Close(HSerialPort hSerialPort)
{
    uem_result result = ERR_UEM_NOERROR;
	close(hSerialPort->portfd);
	hSerialPort->portfd = SERIAL_FD_NOT_SET;
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
#ifdef ARGUMENT_CHECK
    if(pstSerialPort->bIsServer == TRUE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_SERIAL, _EXIT);
    }
#endif
    result = selectTimeout(pstSerialPort->portfd, NULL, &stWriteSet, NULL, nTimeout);
    ERRIFGOTO(result, _EXIT);

    nDataSent = write(pstSerialPort->portfd, pData, nDataLen);
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
#ifdef ARGUMENT_CHECK
    if(pstSerialPort->bIsServer == TRUE)
    {
        ERRASSIGNGOTO(result, ERR_UEM_INVALID_SERIAL, _EXIT);
    }
#endif
    result = selectTimeout(pstSerialPort->portfd, &stReadSet, NULL, NULL, nTimeout);
    ERRIFGOTO(result, _EXIT);

    //에러 발생 지점
    nDataReceived = read(hSerialPort->portfd, pBuffer, nBufferLen); // read up to nBufferLen characters

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
