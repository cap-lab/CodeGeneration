/*
 * UFSystem_deprecated.h
 *
 *  Created on: 2017. 8. 13.
 *      Author: jej
 */

#ifndef SRC_API_INCLUDE_UFSYSTEM_DEPRECATED_H_
#define SRC_API_INCLUDE_UFSYSTEM_DEPRECATED_H_

#ifdef __cplusplus
extern "C"
{
#endif

#ifndef API_LITE
/**
 * @brief (Deprecated) Kill running program.
 *
 * @param nCallerTaskId id of caller task(currently not used).
 *
 * @return
 *
 */
void SYS_REQ_KILL(int nCallerTaskId);

/**
 * @brief (Deprecated) Stop execution of each task, then set whole program to be terminated.
 *
 * @param nCallerTaskId id of caller task(currently not used).
 *
 * @return
 */
void SYS_REQ_STOP(int nCallerTaskId);
#endif

#ifdef __cplusplus
}
#endif

#endif /* SRC_API_INCLUDE_UFSYSTEM_DEPRECATED_H_ */
