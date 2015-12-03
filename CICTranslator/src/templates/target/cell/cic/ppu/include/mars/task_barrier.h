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

#ifndef MARS_TASK_BARRIER_H
#define MARS_TASK_BARRIER_H

/**
 * \file
 * \ingroup group_mars_task_barrier
 * \brief <b>[host]</b> MARS Task Barrier API
 */

#include <stdint.h>
#include <mars/task_barrier_types.h>

struct mars_context;

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * \ingroup group_mars_task_barrier
 * \brief <b>[host]</b> Creates a task barrier.
 *
 * This function will allocate an instance of the task barrier.
 * The barrier allows for tasks within the barrier group to wait until all
 * tasks arrive at some synchronization point and notify the barrier.
 * All tasks included in the barrier group should call the notify and wait in
 * pairs.
 *
 * <b>Key Parameters</b>:
 * \n \n
 * \e total
 * - Specify total number of tasks that will be associated with this barrier.
 * - Total must be a value between 1 and \ref MARS_TASK_BARRIER_WAIT_MAX.
 *
 * \param[in] mars		- pointer to MARS context
 * \param[out] barrier_ea	- ea of barrier instance
 * \param[in] total		- number of notifies before barrier released
 * \return
 *	MARS_SUCCESS		- successfully created barrier
 * \n	MARS_ERROR_NULL		- null pointer specified
 * \n	MARS_ERROR_PARAMS	- total is 0 or exceeds allowed limit
 */
int mars_task_barrier_create(struct mars_context *mars,
			     uint64_t *barrier_ea,
			     uint32_t total);

/**
 * \ingroup group_mars_task_barrier
 * \brief <b>[host]</b> Destroys a task barrier.
 *
 * This function will free any resources allocated during creation of the task
 * barrier.
 *
 * \param[in] barrier_ea	- ea of barrier instance
 * \return
 *	MARS_SUCCESS		- successfully destroyed barrier
 * \n	MARS_ERROR_NULL		- null pointer specified
 * \n	MARS_ERROR_ALIGN	- ea not properly aligned
 */
int mars_task_barrier_destroy(uint64_t barrier_ea);

#if defined(__cplusplus)
}
#endif

#endif
