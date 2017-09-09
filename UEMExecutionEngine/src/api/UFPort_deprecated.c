/*
 * UFPort_deprecated.c
 *
 *  Created on: 2017. 8. 24.
 *      Author: jej
 */


int PORT_INITIALIZE(int nTaskId, const char *pszPortName)
{
	int nChannelId;

	return nChannelId;
}

int MQ_RECEIVE(int nChannelId, unsigned char *pBuffer, int nLen)
{
	int nDataRead;

	return nDataRead;
}


int MQ_SEND(int nChannelId, unsigned char *pBuffer, int nLen)
{
	int nDataWritten;

	return nDataWritten;
}


int MQ_AVAILABLE(int nChannelId)
{
	int nAvailableDataLen;

	return nAvailableDataLen;
}


int AC_RECEIVE(int nChannelId, unsigned char *pBuffer, int nLen, int nIndex)
{
	int nDataRead;

	return nDataRead;
}


int AC_SEND(int nChannelId, unsigned char *pBuffer, int nLen, int nIndex)
{
	int nDataWritten;

	return nDataWritten;
}


int AC_CHECK(int nChannelId)
{
	int nAvailableChunkIndex;

	return nAvailableChunkIndex;
}


int BUF_RECEIVE(int nChannelId, unsigned char *pBuffer, int nLen)
{
	int nDataRead;

	return nDataRead;
}


int BUF_SEND(int nChannelId, unsigned char *pBuffer, int nLen)
{
	int nDataWritten;

	return nDataWritten;
}

