/*
 * UCAlloc.c
 *
 *  Created on: 2018. 9. 3.
 *      Author: chjej202
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

void *UCAlloc_malloc(int nSize)
{
	void *pMemory = NULL;

	pMemory = malloc(nSize);

	return pMemory;
}


void *UCAlloc_calloc(int nNumOfElements, int nSize)
{
	void *pMemory = NULL;

	pMemory = calloc(nNumOfElements, nSize);

	return pMemory;
}


void *UCAlloc_realloc(void *pMem, int nSize)
{
	void *pMemory = NULL;

	pMemory = realloc(pMem, nSize);

	return pMemory;
}

void UCAlloc_free(void *pMem)
{
	free(pMem);
}
