/*
 * UCBasic.c
 *
 *  Created on: 2017. 8. 11.
 *      Author: jej
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdlib.h>
#include <string.h>

#include <uem_common.h>

#include <UCBasic.h>

void *UC_malloc(int nSize)
{
	void *pMemory = NULL;

	pMemory = malloc(nSize);

	return pMemory;
}


void *UC_calloc(int nNumOfElements, int nSize)
{
	void *pMemory = NULL;

	pMemory = calloc(nNumOfElements, nSize);

	return pMemory;
}


void *UC_realloc(void *pMem, int nSize)
{
	void *pMemory = NULL;

	pMemory = realloc(pMem, nSize);

	return pMemory;
}

void UC_free(void *pMem)
{
	free(pMem);
}


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


