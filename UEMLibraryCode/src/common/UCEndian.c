/*
 * UCEndian.c
 *
 *  Created on: 2018. 10. 6.
 *      Author: chjej202
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <UCEndian.h>

uem_bool UCEndian_SystemIntToLittleEndianChar(int nValue, char *pBuffer, int nBufferLen)
{
	// TODO: need to fix it with config.h about sizeof(int) == sizeof(short)
    if(nBufferLen < sizeof(int) || sizeof(int) == sizeof(short))
    {
        return FALSE;
    }
#ifdef WORDS_BIGENDIAN
    pBuffer[0] = nValue & 0xFF;
    pBuffer[1] = (nValue >> 8) & 0xFF;
    pBuffer[2] = (nValue >> 16) & 0xFF;
    pBuffer[3] = (nValue >> 24) & 0xFF;
#else
    int *pnDst;
    pnDst = (int *) pBuffer;
    *pnDst = nValue;
#endif
    return TRUE;
}

uem_bool UCEndian_SystemShortToLittleEndianChar(short sValue, char *pBuffer, int nBufferLen)
{
    if(nBufferLen < sizeof(short))
    {
        return FALSE;
    }
#ifdef WORDS_BIGENDIAN
    pBuffer[0] = sValue & 0xFF;
    pBuffer[1] = (sValue >> 8) & 0xFF;
#else
    short *psDst;
    psDst = (short *) pBuffer;
    *psDst = sValue;
#endif

    return TRUE;
}


uem_bool UCEndian_LittleEndianCharToSystemInt(char *pBuffer, int nBufferLen, int *pnValue)
{
	// TODO: need to fix it with config.h about sizeof(int) == sizeof(short)
    if(nBufferLen < sizeof(int) || sizeof(int) == sizeof(short))
    {
        return FALSE;
    }
#ifdef WORDS_BIGENDIAN
    *pnValue =  pBuffer[0];
    *pnValue = *pnValue | ((int) pBuffer[1] << 8);
    *pnValue = *pnValue | ((int) pBuffer[2] << 16);
    *pnValue = *pnValue | ((int) pBuffer[3] << 24);
#else
    int *pnDst;
    pnDst = (int *) pBuffer;
    *pnValue = *pnDst;
#endif
    return TRUE;
}

uem_bool UCEndian_LittleEndianCharToSystemShort(char *pBuffer, int nBufferLen, short *psValue)
{
    if(nBufferLen < sizeof(short))
    {
        return FALSE;
    }
#ifdef WORDS_BIGENDIAN
    *psValue =  pBuffer[0];
    *psValue = *psValue | ((short) pBuffer[1] << 8);
#else
    short *psDst;
    psDst = (short *) pBuffer;
    *psValue = *psDst;
#endif
    return TRUE;
}
