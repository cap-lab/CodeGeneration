/*
 * UCString.c
 *
 *  Created on: 2017. 12. 24.
 *      Author: jej
 */

#include <uem_common.h>
#include <UCBasic.h>

#include <UCString.h>

#define UEMSTRING_CHARTYPE char

uem_result UCString_New(uem_string strToSet, char *pBuffer, int nBufLen)
{
	uem_result result = ERR_UEM_UNKNOWN;
	uem_string_struct *pstStr = NULL;
	int nLoop = 0;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(strToSet, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(pBuffer, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);

	if(nBufLen <= 0)
	{
		ERRASSIGNGOTO(result, ERR_UEM_INVALID_PARAM, _EXIT);
	}
#endif
	pstStr = strToSet;

    pstStr->nBufferLen = nBufLen;
    pstStr->nStringLen = 0;
    pstStr->pszStr = pBuffer;

    for(nLoop = 0 ; nLoop < nBufLen ; nLoop++)
    {
    	if(pBuffer[nLoop] == '\0')
		{
			break;
		}
    }

    pstStr->nStringLen = nLoop;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UCString_SetLow(uem_string strDst, const char *pszSrc, int nSrcBufLen)
{
    uem_result result = ERR_UEM_UNKNOWN;
    int nSrcRealLen = 0;
    int nLoop = 0;
    uem_string_struct *pstStrDst = NULL;
#ifdef ARGUMENT_CHECK
    IFVARERRASSIGNGOTO(strDst, NULL, result, ERR_UEM_INVALID_HANDLE, _EXIT);
#endif
    pstStrDst = strDst;

    if(nSrcBufLen <= 0 || pszSrc == NULL)
    {
        // length is 0
        if(pstStrDst->pszStr != NULL)
        {
            pstStrDst->pszStr[0] = '\0';
            pstStrDst->nStringLen = 0;
        }
        else
        {
            pstStrDst->nStringLen = 0;
        }
        result = ERR_UEM_NOERROR;
    }
    else
    {
        for(nLoop = 0; nLoop < nSrcBufLen ; nLoop++)
        {
            if(pszSrc[nLoop] == '\0')
            {
                break;
            }
        }

        nSrcRealLen = nLoop;

        // truncate the string
        if(nSrcRealLen >= pstStrDst->nBufferLen)
        {
        	nSrcRealLen = pstStrDst->nBufferLen;
        	result = ERR_UEM_TRUNCATED;
        }
        else // there is enough space to copy a source string
        {
        	result = ERR_UEM_NOERROR;
        }

        // copy string
        UC_memcpy(pstStrDst->pszStr, pszSrc, nSrcRealLen*sizeof(UEMSTRING_CHARTYPE));
        pstStrDst->pszStr[nSrcRealLen] = '\0';
        pstStrDst->nStringLen = nSrcRealLen;
    }

_EXIT:
    return result;
}

uem_bool UCString_IsEqual(uem_string strCompare1, uem_string strCompare2)
{
    uem_bool bEqual = FALSE;
    uem_string_struct *pstCompare1 = NULL;
    uem_string_struct *pstCompare2 = NULL;

    IFVARERRASSIGNGOTO(strCompare1, NULL, bEqual, FALSE, _EXIT);
    IFVARERRASSIGNGOTO(strCompare2, NULL, bEqual, FALSE, _EXIT);

    pstCompare1 = (uem_string_struct *) strCompare1;
    pstCompare2 = (uem_string_struct *) strCompare2;

    if(pstCompare1->nStringLen != pstCompare2->nStringLen)
    {
        UEMASSIGNGOTO(bEqual, FALSE, _EXIT);
    }

    if(UC_memcmp(pstCompare1->pszStr, pstCompare2->pszStr, pstCompare1->nStringLen*sizeof(UEMSTRING_CHARTYPE)) == 0)
    {
        bEqual = TRUE;
    }
    else
    {
        bEqual = FALSE;
    }
_EXIT:
    return bEqual;
}


