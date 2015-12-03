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

#ifndef MARS_TASK_EVENT_FLAG_H
#define MARS_TASK_EVENT_FLAG_H

/**
 * \file
 * \ingroup group_mars_task_event_flag
 * \brief <b>[host]</b> MARS Task Event Flag API
 */

#include <stdint.h>
#include <mars/task_event_flag_types.h>

struct mars_context;

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * \ingroup group_mars_task_event_flag
 * \brief <b>[host]</b> Creates a task event flag.
 *
 * This function will allocate an instance of the task event flag.
 * The event flag allows for tasks to wait on specific events.
 * The event flag should be used in pairs with calls to set an event flag and
 * wait for an event flag.
 *
 * <b>Key Parameters</b>:
 * \n \n
 * \e direction
 * - Specify the communication direction of the event flag.
 * - Must be one of \ref MARS_TASK_EVENT_FLAG_HOST_TO_MPU,
 * \ref MARS_TASK_EVENT_FLAG_MPU_TO_HOST, \ref MARS_TASK_EVENT_FLAG_MPU_TO_MPU.
 *
 * \e clear_mode
 * - Specify when the event flag is cleared.
 * - Specify \ref MARS_TASK_EVENT_FLAG_CLEAR_AUTO to have event flag bits
 * cleared automatically when a task waiting on the event flag bits has been
 * notified of the event.
 * - Specify \ref MARS_TASK_EVENT_FLAG_CLEAR_MANUAL to have the event flag bits
 * remain set until the user manually clears them through
 * \ref mars_task_event_flag_clear.
 *
 * \param[in] mars		- pointer to MARS context
 * \param[out] event_flag_ea	- ea of event flag instance
 * \param[in] direction		- direction of the event flag
 * \param[in] clear_mode	- behavior of how the event flag is cleared
 * \return
 *	MARS_SUCCESS		- successfully created event flag
 * \n	MARS_ERROR_NULL		- null pointer specified
 * \n	MARS_ERROR_PARAMS	- invalid direction or clear mode specified
 */
int mars_task_event_flag_create(struct mars_context *mars,
				uint64_t *event_flag_ea,
				uint8_t direction,
				uint8_t clear_mode);

/**
 * \ingroup group_mars_task_event_flag
 * \brief <b>[host]</b> Destroys a task event flag.
 *
 * This function will free any resources allocated during creation of the task
 * event flag.
 *
 * \param[in] event_flag_ea	- ea of event flag instance
 * \return
 *	MARS_SUCCESS		- successfully destroyed event flag
 * \n	MARS_ERROR_NULL		- null pointer specified
 * \n	MARS_ERROR_ALIGN	- ea not properly aligned
 */
int mars_task_event_flag_destroy(uint64_t event_flag_ea);

/**
 * \ingroup group_mars_task_event_flag
 */
int mars_task_event_flag_clear(uint64_t event_flag_ea, uint32_t bits);

/**
 * \ingroup group_mars_task_event_flag
 */
int mars_task_event_flag_set(uint64_t event_flag_ea, uint32_t bits);

/**
 * \ingroup group_mars_task_event_flag
 */
int mars_task_event_flag_wait(uint64_t event_flag_ea,
			      uint32_t mask, uint8_t mask_mode,
			      uint32_t *bits);

/**
 * \ingroup group_mars_task_event_flag
 */
int mars_task_event_flag_try_wait(uint64_t event_flag_ea,
				  uint32_t mask, uint8_t mask_mode,
				  uint32_t *bits);

#if defined(__cplusplus)
}
#endif

#endif
