/*
 * Copyright 2008 Sony Corporation of America
 *
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this Library and associated documentation files (the
 * "Library"), to deal in the Library without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Library, and to
 * permit persons to whom the Library is furnished to do so, subject to
 * the following conditions:
 *
 *  The above copyright notice and this permission notice shall be
 *  included in all copies or substantial portions of the Library.
 *
 *  If you modify the Library, you may copy and distribute your modified
 *  version of the Library in object code or as an executable provided
 *  that you also do one of the following:
 *
 *   Accompany the modified version of the Library with the complete
 *   corresponding machine-readable source code for the modified version
 *   of the Library; or,
 *
 *   Accompany the modified version of the Library with a written offer
 *   for a complete machine-readable copy of the corresponding source
 *   code of the modified version of the Library.
 *
 *
 * THE LIBRARY IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * LIBRARY OR THE USE OR OTHER DEALINGS IN THE LIBRARY.
 */

#ifndef MARS_TASK_SEMAPHORE_H
#define MARS_TASK_SEMAPHORE_H

/**
 * \file
 * \ingroup group_mars_task_semaphore
 * \brief <b>[host]</b> MARS Task Semaphore API
 */

#include <stdint.h>
#include <mars/task_semaphore_types.h>

struct mars_context;

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * \ingroup group_mars_task_semaphore
 * \brief <b>[host]</b> Creates a task semaphore.
 *
 * This function will allocate an instance of the task semaphore.
 * The semaphore allows for tasks to wait until a semaphore can be obtained.
 * The semaphore should be used in pairs with calls to acquire and release.
 *
 * <b>Key Parameters</b>:
 * \n \n
 * \e count
 * - Specify the total number of entities that can have access to the semaphore
 * simultaneously.
 * - Must not be greater than \ref MARS_TASK_SEMAPHORE_WAIT_MAX.
 *
 * \param[in] mars		- pointer to MARS context
 * \param[out] semaphore_ea	- ea of semaphore instance
 * \param[in] count		- initial number of task accesses allowed
 * \return
 *	MARS_SUCCESS		- successfully created semaphore
 * \n	MARS_ERROR_NULL		- null pointer is specified
 * \n	MARS_ERROR_PARAMS	- count exceeds allowed limit
 */
int mars_task_semaphore_create(struct mars_context *mars,
			       uint64_t *semaphore_ea,
			       int32_t count);

/**
 * \ingroup group_mars_task_semaphore
 * \brief <b>[host]</b> Destroys a task semaphore.
 *
 * This function will free any resources allocated during creation of the task
 * semaphore.
 *
 * \param[in] semaphore_ea	- ea of semaphore instance
 * \return
 *	MARS_SUCCESS		- successfully destroyed semaphore
 * \n	MARS_ERROR_NULL		- null pointer specified
 * \n	MARS_ERROR_ALIGN	- ea not properly aligned
 */
int mars_task_semaphore_destroy(uint64_t semaphore_ea);

#if defined(__cplusplus)
}
#endif

#endif
