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

#ifndef MARS_WORKLOAD_QUEUE_H
#define MARS_WORKLOAD_QUEUE_H

/**
 * \file
 * \ingroup group_mars_workload_queue
 * \brief <b>[host]</b> MARS Workload Queue API
 */

#include <mars/workload_types.h>

struct mars_context;

#if defined(__cplusplus)
extern "C" {
#endif

/* These functions should only be used internally by the MARS context */
int mars_workload_queue_create(struct mars_context *mars);
int mars_workload_queue_destroy(struct mars_context *mars);
int mars_workload_queue_exit(struct mars_context *mars);

/**
 * \ingroup group_mars_workload_queue
 * \brief <b>[host]</b> Begins adding workload to workload queue.
 *
 * This function will begin the process to add a workload to the workload queue.
 * This only initiates the add operation.
 * This function must be completed with a matching call to
 * \ref mars_workload_queue_add_end to guarantee the completion of the add
 * operation.
 *
 * If workload_ea is not NULL, the ea of the workload will be returned.
 *
 * This call will lock the workload queue until the matching call to
 * \ref mars_workload_queue_add_end is made.
 * The user should make any necessary updates to the returned workload context
 * in between this begin call and the end call.
 *
 * \param[in] mars		- address of pointer to MARS context
 * \param[out] id		- pointer to return workload id
 * \param[out] workload_ea	- address of pointer to workload context ea
 * \return
 *	MARS_SUCCESS		- workload adding started
 * \n	MARS_ERROR_NULL		- null pointer specified
 * \n	MARS_ERROR_PARAMS	- invalid mars context specified
 * \n	MARS_ERROR_LIMIT	- workload queue is full
 */
int mars_workload_queue_add_begin(struct mars_context *mars,
				  uint16_t *id,
				  uint64_t *workload_ea);

// by iloy 2009.03.02
#ifndef CIC
#define CIC (1)
#endif
#if defined(CIC) && (CIC==1)
int mars_workload_queue_add_begin_with_affinity(struct mars_context *mars,
				  uint16_t *id,
                  uint16_t affinity,
				  uint64_t *workload_ea);
#endif

/**
 * \ingroup group_mars_workload_queue
 * \brief <b>[host]</b> Ends adding of specified workload.
 *
 * This function will complete a add operation previously initiated with
 * \ref mars_workload_queue_add_begin.
 * This function must be called in pair for each call to
 * \ref mars_workload_queue_add_begin to guarantee the completion of the
 * initiated add operation.
 *
 * \param[in] mars		- address of pointer to MARS context
 * \param[in] id		- id of workload to end add
 * \return
 *	MARS_SUCCESS		- workload adding complete
 * \n	MARS_ERROR_NULL		- null pointer specified
 * \n	MARS_ERROR_PARAMS	- invalid mars context or workload id specified
 * \n	MARS_ERROR_STATE	- workload adding not started
 */
int mars_workload_queue_add_end(struct mars_context *mars,
				uint16_t id);

/**
 * \ingroup group_mars_workload_queue
 * \brief <b>[host]</b> Cancels adding of specified workload.
 *
 * This function will cancel an add operation previously initiated with
 * \ref mars_workload_queue_add_begin.
 * If scheduling is canceled, \ref mars_workload_queue_add_end should not
 * be called.
 *
 * \param[in] mars		- address of pointer to MARS context
 * \param[in] id		- id of workload to cancel add
 * \return
 *	MARS_SUCCESS		- workload adding canceled
 * \n	MARS_ERROR_NULL		- null pointer specified
 * \n	MARS_ERROR_PARAMS	- invalid mars context or workload id specified
 * \n	MARS_ERROR_STATE	- workload adding not started
 */
int mars_workload_queue_add_cancel(struct mars_context *mars,
				   uint16_t id);

/**
 * \ingroup group_mars_workload_queue
 * \brief <b>[host]</b> Begins removing workload from workload queue.
 *
 * This function will begin the process to remove a workload from the workload
 * queue.
 * This only initiates the remove operation.
 * This function must be completed with a matching call to
 * \ref mars_workload_queue_remove_end to guarantee the completion of the remove
 * operation.
 *
 * If workload_ea is not NULL, the ea of the workload will be returned.
 *
 * This call will lock the workload queue until the matching call to
 * \ref mars_workload_queue_remove_end is made.
 * The user should make any necessary updates to the returned workload context
 * in between this begin call and the end call.
 *
 * \param[in] mars		- address of pointer to MARS context
 * \param[in] id		- id of workload to begin remove
 * \param[out] workload_ea	- address of pointer to workload context ea
 * \return
 *	MARS_SUCCESS		- workload removing started
 * \n	MARS_ERROR_NULL		- null pointer specified
 * \n	MARS_ERROR_PARAMS	- invalid mars context specified
 * \n	MARS_ERROR_STATE	- specified workload not added or finished
 */
int mars_workload_queue_remove_begin(struct mars_context *mars,
				     uint16_t id,
				     uint64_t *workload_ea);

/**
 * \ingroup group_mars_workload_queue
 * \brief <b>[host]</b> Ends removing of specified workload.
 *
 * This function will complete a remove operation previously initiated with
 * \ref mars_workload_queue_remove_begin.
 * This function must be called in pair for each call to
 * \ref mars_workload_queue_remove_begin to guarantee the completion of the
 * initiated remove operation.
 *
 * \param[in] mars		- address of pointer to MARS context
 * \param[in] id		- id of workload
 * \return
 *	MARS_SUCCESS		- workload removing complete
 * \n	MARS_ERROR_PARAMS	- invalid mars context or workload id specified
 * \n	MARS_ERROR_STATE	- workload removing not started
 */
int mars_workload_queue_remove_end(struct mars_context *mars,
				   uint16_t id);

/**
 * \ingroup group_mars_workload_queue
 * \brief <b>[host]</b> Cancels removing of specified workload.
 *
 * This function will cancel a remove operation previously initiated with
 * \ref mars_workload_queue_remove_begin.
 * If removing is canceled, \ref mars_workload_queue_remove_end should not
 * be called.
 *
 * \param[in] mars		- address of pointer to MARS context
 * \param[in] id		- id of workload
 * \return
 *	MARS_SUCCESS		- workload removing canceled
 * \n	MARS_ERROR_PARAMS	- invalid mars context or workload id specified
 * \n	MARS_ERROR_STATE	- workload removing not started
 */
int mars_workload_queue_remove_cancel(struct mars_context *mars,
				      uint16_t id);

/**
 * \ingroup group_mars_workload_queue
 * \brief <b>[host]</b> Begins scheduling of specified workload.
 *
 * This function will begin scheduling the workload specified.
 * This only initiates the scheduling of the workload.
 * This function must be completed with a matching call to
 * \ref mars_workload_queue_schedule_end to guarantee the completion of the
 * scheduling.
 *
 * If workload_ea is not NULL, the ea of the workload will be returned.
 *
 * This call will lock the workload queue until the matching call to
 * \ref mars_workload_queue_schedule_end is made.
 * The user should make any necessary updates to the returned workload context
 * in between this begin call and the end call.
 *
 * \param[in] mars		- address of pointer to MARS context
 * \param[in] id		- id of workload
 * \param[in] priority		- scheduling priority of workload
 * \param[out] workload_ea	- address of pointer to workload context ea
 * \return
 *	MARS_SUCCESS		- workload scheduling started
 * \n	MARS_ERROR_NULL		- null pointer specified
 * \n	MARS_ERROR_PARAMS	- invalid mars context or workload id specified
 * \n	MARS_ERROR_STATE	- specified workload not added or finished
 */
int mars_workload_queue_schedule_begin(struct mars_context *mars,
				       uint16_t id, uint8_t priority,
				       uint64_t *workload_ea);

/**
 * \ingroup group_mars_workload_queue
 * \brief <b>[host]</b> Ends scheduling of specified workload.
 *
 * This function will complete a schedule operation previously initiated with
 * \ref mars_workload_queue_schedule_begin.
 * This function must be called in pair for each call to
 * \ref mars_workload_queue_schedule_begin to guarantee the completion of the
 * initiated schedule operation.
 *
 * \param[in] mars		- address of pointer to MARS context
 * \param[in] id		- id of workload
 * \return
 *	MARS_SUCCESS		- workload scheduling complete
 * \n	MARS_ERROR_NULL		- null pointer specified
 * \n	MARS_ERROR_PARAMS	- invalid mars context or workload id specified
 * \n	MARS_ERROR_STATE	- workload scheduling not started
 */
int mars_workload_queue_schedule_end(struct mars_context *mars,
				     uint16_t id);

/**
 * \ingroup group_mars_workload_queue
 * \brief <b>[host]</b> Cancels scheduling of specified workload.
 *
 * This function will cancel a schedule operation previously initiated with
 * \ref mars_workload_queue_schedule_begin.
 * If scheduling is canceled, \ref mars_workload_queue_schedule_end should not
 * be called.
 *
 * \param[in] mars		- address of pointer to MARS context
 * \param[in] id		- id of workload
 * \return
 *	MARS_SUCCESS		- workload scheduling canceled
 * \n	MARS_ERROR_NULL		- null pointer specified
 * \n	MARS_ERROR_PARAMS	- invalid mars context or workload id specified
 * \n	MARS_ERROR_STATE	- workload scheduling not started
 */
int mars_workload_queue_schedule_cancel(struct mars_context *mars,
					uint16_t id);

/**
 * \ingroup group_mars_workload_queue
 * \brief <b>[host]</b> Waits for specified workload to finish.
 *
 * This function will block and wait until the specified workload finishes.
 *
 * If workload_ea is not NULL, the ea of the workload will be returned.
 *
 * \param[in] mars		- address of pointer to MARS context
 * \param[in] id		- id of workload
 * \param[out] workload_ea	- address of pointer to workload context ea
 * \return
 *	MARS_SUCCESS		- workload is finished
 * \n	MARS_ERROR_NULL		- null pointer specified
 * \n	MARS_ERROR_PARAMS	- invalid mars context or workload id specified
 * \n	MARS_ERROR_STATE	- invalid workload specified
 */
int mars_workload_queue_wait(struct mars_context *mars,
			     uint16_t id,
			     uint64_t *workload_ea);

/**
 * \ingroup group_mars_workload_queue
 * \brief <b>[host]</b> Waits for specified workload to finish.
 *
 * This function will check whether the workload specified is finished or not
 * and return immediately without blocking.
 *
 * If workload_ea is not NULL, the ea of the workload will be returned.
 *
 * \param[in] mars		- address of pointer to MARS context
 * \param[in] id		- id of workload
 * \param[out] workload_ea	- address of pointer to workload context ea
 * \return
 *	MARS_SUCCESS		- workload is finished
 * \n	MARS_ERROR_NULL		- null pointer specified
 * \n	MARS_ERROR_PARAMS	- invalid mars context or workload id specified
 * \n	MARS_ERROR_STATE	- invalid workload specified
 * \n	MARS_ERROR_BUSY		- workload has not yet finished
 */
int mars_workload_queue_try_wait(struct mars_context *mars,
				 uint16_t id,
				 uint64_t *workload_ea);

/**
 * \ingroup group_mars_workload_queue
 * \brief <b>[host]</b> Sends signal to specified workload.
 *
 * This function will send a signal to the specified workload.
 *
 * \param[in] mars		- address of pointer to MARS context
 * \param[in] id		- id of workload
 * \return
 *	MARS_SUCCESS		- signalled workload
 * \n	MARS_ERROR_NULL		- null pointer specified
 * \n	MARS_ERROR_PARAMS	- invalid mars context or workload id specified
 * \n	MARS_ERROR_STATE	- invalid workload specified
 */
int mars_workload_queue_signal_send(struct mars_context *mars,
				    uint16_t id);

#if defined(__cplusplus)
}
#endif

#endif
