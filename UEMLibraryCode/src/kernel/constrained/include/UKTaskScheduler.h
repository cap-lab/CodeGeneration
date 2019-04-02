/*
 * UKTaskScheduler.h
 *
 *  Created on: 2018. 9. 6.
 *      Author: chjej202
 */

#ifndef SRC_KERNEL_CONSTRAINED_INCLUDE_UKTASKSCHEDULER_H_
#define SRC_KERNEL_CONSTRAINED_INCLUDE_UKTASKSCHEDULER_H_

#include <uem_common.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_result UKTaskScheduler_Init();
uem_result UKTaskScheduler_Run();

#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_CONSTRAINED_INCLUDE_UKTASKSCHEDULER_H_ */
