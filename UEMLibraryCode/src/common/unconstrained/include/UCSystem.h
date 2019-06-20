/*
 * UCSystem.h
 *
 *  Created on: 2018. 1. 10.
 *      Author: jej
 */

#ifndef SRC_COMMON_INCLUDE_UCSYSTEM_H_
#define SRC_COMMON_INCLUDE_UCSYSTEM_H_

#include "uem_common.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief terminate a program.
 *
 * This function forcedly terminates the current program. This is a wrapper function exit();
 *
 */
void UCSystem_Exit();

#ifdef __cplusplus
}
#endif

#endif /* SRC_COMMON_INCLUDE_UCSYSTEM_H_ */
