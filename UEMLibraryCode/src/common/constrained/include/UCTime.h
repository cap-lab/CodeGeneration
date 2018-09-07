/*
 * UCTime.h
 *
 *  Created on: 2018. 9. 6.
 *      Author: chjej202
 */

#ifndef SRC_COMMON_CONSTRAINED_INCLUDE_UCTIME_H_
#define SRC_COMMON_CONSTRAINED_INCLUDE_UCTIME_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_result UCTime_GetCurTickInMilliSeconds(unsigned long *pulTime);

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_CONSTRAINED_INCLUDE_UCTIME_H_ */
