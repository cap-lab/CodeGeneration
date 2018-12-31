/*
 * UFLoop.h
 *
 *  Created on: 2018. 8. 30.
 *      Author: DG-SHIN
 */

#ifndef SRC_API_INCLUDE_UFLOOP_H_
#define SRC_API_INCLUDE_UFLOOP_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_result UFLoop_GetIteration(IN int nCallerTaskId, IN int nTaskThreadId, OUT int *pnTaskIteration);
uem_result UFLoop_StopNextIteration(IN int nCallerTaskId);

#ifdef __cplusplus
}
#endif


#endif /* SRC_API_INCLUDE_UFLOOP_H_ */
