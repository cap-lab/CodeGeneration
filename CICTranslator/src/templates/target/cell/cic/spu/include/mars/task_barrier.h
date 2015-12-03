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
 * \brief <b>[MPU]</b> MARS Task Barrier API
 */

#include <stdint.h>
#include <mars/task_barrier_types.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * \ingroup group_mars_task_barrier
 * \brief <b>[MPU]</b> Notifies a task barrier.
 * <b>(Task Switch Call)</b>
 *
 * \note The <b>[MPU]</b> call may result in a task switch and put this
 * task into the waiting state. Understand all the limitations before calling
 * a <b>Task Switch Call</b> (<b>See</b> \ref sec_7_5).
 *
 * This function notifies the barrier that the caller task has reached the
 * synchronization point. Once a task reaches the synchronization point and
 * notifies the barrier, it may handle other processing before calling the
 * required \ref mars_task_barrier_wait.
 *
 * If all tasks from the previous barrier cycle have not reached the
 * synchronization point and notified the barrier yet, the caller task will
 * enter a waiting state until the previous barrier cycle is released.
 *
 * \param[in] barrier_ea	- ea of initialized barrier instance
 * \return
 *	MARS_SUCCESS		- successfully notified barrier
 * \n	MARS_ERROR_NULL		- ea is 0
 * \n	MARS_ERROR_ALIGN	- ea not properly aligned
 * \n	MARS_ERROR_LIMIT	- maximum number of tasks already waiting
 * \n	MARS_ERROR_FORMAT	- no context save area specified
 */
int mars_task_barrier_notify(uint64_t barrier_ea);

/**
 * \ingroup group_mars_task_barrier
 * \brief <b>[MPU]</b> Notifies a task barrier.
 *
 * This function notifies the barrier that the caller task has reached the
 * synchronization point. Once a task reaches the synchronization point and
 * notifies the barrier, it may handle other processing before calling the
 * required \ref mars_task_barrier_wait.
 *
 * If all tasks from the previous barrier cycle have not reached the
 * synchronization point and notified the barrier yet, this function will
 * return immediately with \ref MARS_ERROR_BUSY.
 *
 * \param[in] barrier_ea	- ea of initialized barrier instance
 * \return
 *	MARS_SUCCESS		- successfully notified barrier
 * \n	MARS_ERROR_NULL		- ea is 0
 * \n	MARS_ERROR_ALIGN	- ea not properly aligned
 * \n	MARS_ERROR_BUSY		- previous waits have not completed
 */
int mars_task_barrier_try_notify(uint64_t barrier_ea);

/**
 * \ingroup group_mars_task_barrier
 * \brief <b>[MPU]</b> Waits on a task barrier.
 * <b>(Task Switch Call)</b>
 *
 * \note The <b>[MPU]</b> call may result in a task switch and put this
 * task into the waiting state. Understand all the limitations before calling
 * a <b>Task Switch Call</b> (<b>See</b> \ref sec_7_5).
 *
 * This function should be called after the caller task has previously called
 * \ref mars_task_barrier_notify. This function should be called when the task
 * that reached the synchronization point is ready to wait for others to arrive
 * at the synchronization point.
 *
 * If not all tasks associated with the barrier have reached the synchronization
 * point and notified the barrier at the time of this call, the caller task will
 * enter a waiting state until all notifications are received from the other
 * tasks and the barrier is released.
 *
 * \param[in] barrier_ea	- ea of initialized barrier instance
 * \return
 *	MARS_SUCCESS		- barrier is released
 * \n	MARS_ERROR_NULL		- ea is 0
 * \n	MARS_ERROR_ALIGN	- ea not properly aligned
 * \n	MARS_ERROR_LIMIT	- maximum number of tasks already waiting
 * \n	MARS_ERROR_FORMAT	- no context save area specified
 */
int mars_task_barrier_wait(uint64_t barrier_ea);

/**
 * \ingroup group_mars_task_barrier
 * \brief <b>[MPU]</b> Waits on a task barrier.
 *
 * This function should be called after the caller task has previously called
 * \ref mars_task_barrier_notify. This function should be called when the task
 * that reached the synchronization point is ready to wait for others to arrive
 * at the synchronization point.
 *
 * If not all tasks associated with the barrier have reached the synchronization
 * point and notified the barrier at the time of this call, this function will
 * return immediately with \ref MARS_ERROR_BUSY.
 *
 * \param[in] barrier_ea	- ea of initialized barrier instance
 * \return
 *	MARS_SUCCESS		- barrier is released
 * \n	MARS_ERROR_NULL		- ea is 0
 * \n	MARS_ERROR_ALIGN	- ea not properly aligned
 * \n	MARS_ERROR_BUSY		- not all notifications arrived yet
 */
int mars_task_barrier_try_wait(uint64_t barrier_ea);

#if defined(__cplusplus)
}
#endif

#endif
