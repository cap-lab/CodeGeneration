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
void SYS_REQ_KILL(int nCallerTaskId);
void SYS_REQ_STOP(int nCallerTaskId);
#endif

#ifdef __cplusplus
}
#endif

#endif /* SRC_API_INCLUDE_UFSYSTEM_DEPRECATED_H_ */
