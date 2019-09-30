/*
 * UKModeTransitionModelController.h
 *
 *  Created on: 2019. 9. 27.
 *      Author: jej
 */

#ifndef SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKMODETRANSITIONMODELCONTROLLER_H_
#define SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKMODETRANSITIONMODELCONTROLLER_H_

#include <uem_common.h>

#include <uem_data.h>

#ifdef __cplusplus
extern "C"
{
#endif

uem_result UKModeTransitionMachineController_HandleModelComposite(STaskGraph *pstGraph, void *pCurrentTaskHandle, void *pCurrentThreadHandle);
uem_result UKModeTransitionMachineController_GetTaskIterationIndex(STask *pstMTMTask, int nCurrentIteration, int OUT *pnIndex);
uem_result UKModeTransitionMachineController_Clear(STaskGraph *pstTaskGraph);
uem_result UKModeTransitionMachineController_ChangeTaskThreadState(STaskGraph *pstGraph, void *pCurrentTaskHandle, void *pCurrentThreadHandle, ECPUTaskState enTargetState, OUT ECPUTaskState *penState);

#ifdef __cplusplus
}
#endif


#endif /* SRC_KERNEL_UNCONSTRAINED_INCLUDE_UKMODETRANSITIONMODELCONTROLLER_H_ */
