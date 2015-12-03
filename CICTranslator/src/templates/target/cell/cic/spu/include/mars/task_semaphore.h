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
 * \brief <b>[MPU]</b> MARS Task Semaphore API
 */

#include <stdint.h>
#include <mars/task_semaphore_types.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * \ingroup group_mars_task_semaphore
 * \brief <b>[MPU]</b> Acquires a task semaphore.
 * <b>(Task Switch Call)</b>
 *
 * \note The <b>[MPU]</b> call may result in a task switch and put this
 * task into the waiting state. Understand all the limitations before calling
 * a <b>Task Switch Call</b> (<b>See</b> \ref sec_7_5).
 *
 * This function will attempt to acquire the semaphore.
 * If the total number of current accesses of the semaphore is greater than or
 * equal to the total allowed specified at semaphore creation, the caller task
 * will enter a waiting state until some other tasks release the semaphore and
 * becomes available for acquiring.
 *
 * \param[in] semaphore_ea	- ea of initialized semaphore instance
 * \return
 *	MARS_SUCCESS		- successfully acquired semaphore
 * \n	MARS_ERROR_NULL		- ea is 0
 * \n	MARS_ERROR_ALIGN	- ea not aligned properly
 * \n	MARS_ERROR_LIMIT	- maximum number of tasks already waiting
 * \n	MARS_ERROR_FORMAT	- no context save area specified
 */
int mars_task_semaphore_acquire(uint64_t semaphore_ea);

/**
 * \ingroup group_mars_task_semaphore
 * \brief <b>[MPU]</b> Releases a task semaphore.
 *
 * This function will release a previously acquired semaphore.
 * If their are other tasks currently waiting to acquire a semaphore, calling
 * this function will resume a waiting task to allow it to acquire the
 * semaphore.
 *
 * \param[in] semaphore_ea	- ea of initialized semaphore instance
 * \return
 *	MARS_SUCCESS		- successfully released semaphore
 * \n	MARS_ERROR_NULL		- ea is 0
 * \n	MARS_ERROR_ALIGN	- ea not aligned properly
 */
int mars_task_semaphore_release(uint64_t semaphore_ea);

#if defined(__cplusplus)
}
#endif

#endif
