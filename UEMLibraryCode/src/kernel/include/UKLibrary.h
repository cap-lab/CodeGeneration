/*
 * UKLibrary.h
 *
 *  Created on: 2018. 2. 14.
 *      Author: jej
 */

#ifndef SRC_KERNEL_INCLUDE_UKLIBRARY_H_
#define SRC_KERNEL_INCLUDE_UKLIBRARY_H_

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Initialize library task.
 *
 * This function initializes library task by calling tasks' init function.
 *
 */
void UKLibrary_Initialize();

/**
 * @brief Finalize library task.
 *
 * This function finalizes library task by calling tasks' wrapup function.
 *
 */
void UKLibrary_Finalize();


#ifdef __cplusplus
}
#endif

#endif /* SRC_KERNEL_INCLUDE_UKLIBRARY_H_ */
