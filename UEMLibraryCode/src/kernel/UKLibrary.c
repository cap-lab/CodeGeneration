/*
 * UKLibrary.c
 *
 *  Created on: 2018. 2. 14.
 *      Author: jej
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <uem_data.h>

void UKLibrary_Initialize()
{
	int nLoop = 0;
	for(nLoop = 0 ; nLoop < g_nLibraryInfoNum ; nLoop++)
	{
		g_stLibraryInfo[nLoop].fnInit();
	}
}


void UKLibrary_Finalize()
{
	int nLoop = 0;
	for(nLoop = 0 ; nLoop < g_nLibraryInfoNum ; nLoop++)
	{
		g_stLibraryInfo[nLoop].fnWrapup();
	}
}
