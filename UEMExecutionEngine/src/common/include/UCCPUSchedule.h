/*
 * UCCPUSchedule.h
 *
 *  Created on: 2017. 9. 27.
 *      Author: jej
 */

#ifndef SRC_COMMON_INCLUDE_UCCPUSCHEDULE_H_
#define SRC_COMMON_INCLUDE_UCCPUSCHEDULE_H_

#include "uem_common.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _SCPUSchedule SCPUSchedule;


uem_result UCCPUSchedule_Init(SCPUSchedule *pstCPU);
uem_result UCCPUSchedule_SetCPU(SCPUSchedule *pstCPU);
uem_result UCCPUSchedule_ClearCPU(SCPUSchedule *pstCPU);


#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_INCLUDE_UCCPUSCHEDULE_H_ */
