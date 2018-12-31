/*
 * UCBasic.c
 *
 *  Created on: 2017. 8. 11.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>

#include <uem_common.h>

#include <UCBasic.h>

void *UC_memcpy(void *pDest, const void *pSrc, int nSize)
{
	void *pMemory = NULL;

	pMemory = memcpy(pDest, pSrc, nSize);

	return pMemory;
}


int UC_memcmp(void *pCompare1, void *pCompare2, int nSize)
{
	int result;

	result = memcmp(pCompare1, pCompare2, nSize);

	return result;
}


void *UC_memset(void *pDest, int nContents, int nSize)
{
	void *pMemory = NULL;

	pMemory = memset(pDest, nContents, nSize);

	return pMemory;
}


