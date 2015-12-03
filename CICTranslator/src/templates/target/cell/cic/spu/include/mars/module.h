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

#ifndef MARS_MODULE_H
#define MARS_MODULE_H

/**
 * \file
 * \ingroup group_mars_workload_module
 * \brief <b>[MPU]</b> MARS Workload Module API
 */

#include <stdint.h>

#include <mars/mutex_types.h>
#include <mars/workload_types.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * \ingroup group_mars_workload_module
 * \brief <b>[MPU]</b> Entry point for workload module.
 *
 * This function is the main entry point for the workload module. All workload
 * modules will need to have a definition of this function. This function is
 * called from the MARS kernel when a workload context that specifies this
 * workload module is scheduled for execution.
 *
 * \note Returning from this function is equivalent to calling
 * \ref mars_module_workload_finish.
 */
void mars_module_main(void);

/**
 * \ingroup group_mars_workload_module
 * \brief <b>[MPU]</b> Gets tick counter value.
 *
 * \note Counter's frequency depends on runtime environment.
 *
 * \return
 *	uint32_t		- 32-bit tick counter value
 */
uint32_t mars_module_get_ticks(void);

/**
 * \ingroup group_mars_workload_module
 * \brief <b>[MPU]</b> Gets ea of MARS context.
 *
 * \return
 *	uint64_t		- ea of MARS context
 */
uint64_t mars_module_get_mars_context_ea(void);

/**
 * \ingroup group_mars_workload_module
 * \brief <b>[MPU]</b> Gets id of kernel that the module is being executed on.
 *
 * \return
 *	uint16_t		- id of MARS kernel
 */
uint16_t mars_module_get_kernel_id(void);

/**
 * \ingroup group_mars_workload_module
 * \brief <b>[MPU]</b> Gets id of current workload context.
 *
 * \return
 *	uint16_t		- id of workload
 */
uint16_t mars_module_get_workload_id(void);

/**
 * \ingroup group_mars_workload_module
 * \brief <b>[MPU]</b> Gets pointer to current workload context.
 *
 * \return
 *	struct mars_workload_context *	- pointer to current workload context
 */
struct mars_workload_context *mars_module_get_workload(void);

/**
 * \ingroup group_mars_workload_module
 * \brief <b>[MPU]</b> Gets pointer to workload context specified by id.
 *
 * \param[in] id		- id of workload
 * \return
 *	struct mars_workload_context *	- pointer to specified workload context
 */
struct mars_workload_context *mars_module_get_workload_by_id(uint16_t id);

/**
 * \brief MARS workload module query types
 *
 * These are the query types you can pass into \ref mars_module_workload_query
 */
enum {
	MARS_QUERY_IS_CACHED = 0,	/**< query if workload is cached */
	MARS_QUERY_IS_INITIALIZED,	/**< query if workload is initialized */
	MARS_QUERY_IS_READY,		/**< query if workload is ready */
	MARS_QUERY_IS_WAITING,		/**< query if workload is waiting */
	MARS_QUERY_IS_RUNNING,		/**< query if workload is running */
	MARS_QUERY_IS_FINISHED,		/**< query if workload is finished */
	MARS_QUERY_IS_SIGNAL_SET,	/**< query if workload signal is set */
};

/**
 * \ingroup group_mars_workload_module
 * \brief <b>[MPU]</b> Returns whether or not specified query is satisfied.
 *
 * \param[in] id		- id of workload
 * \param[in] query		- query type
 * \return
 *	int			- non-zero if query satisfied
 */
int mars_module_workload_query(uint16_t id, int query);

/**
 * \ingroup group_mars_workload_module
 * \brief <b>[MPU]</b> Sets calling workload to wait for completion of specified workload.
 *
 * \note This function only sets the id of workload to wait for completion.
 * The caller should also \ref mars_module_workload_wait immediately after this
 * call so the calling workload yields execution and enters the waiting state.
 *
 * \param[in] id		- id of workload
 * \return
 *	MARS_SUCCESS		- id of workload to wait for set
 * \n	MARS_ERROR_PARAMS	- invalid workload id specified
 */
int mars_module_workload_wait_set(uint16_t id);

/**
 * \ingroup group_mars_workload_module
 * \brief <b>[MPU]</b> Sets calling workload to not wait for completion of any workloads.
 *
 * \return
 *	MARS_SUCCESS		- id of workload to wait for reset
 * \n	MARS_ERROR_PARAMS	- invalid workload id specified
 */
int mars_module_workload_wait_reset(void);

/**
 * \ingroup group_mars_workload_module
 * \brief <b>[MPU]</b> Sets signal for specified workload.
 *
 * \param[in] id		- id of workload
 * \return
 *	MARS_SUCCESS		- signal set
 * \n	MARS_ERROR_PARAMS	- invalid workload id specified
 */
int mars_module_workload_signal_set(uint16_t id);

/**
 * \ingroup group_mars_workload_module
 * \brief <b>[MPU]</b> Resets signal for specified workload.
 *
 * \return
 *	MARS_SUCCESS		- signal reset
 * \n	MARS_ERROR_PARAMS	- invalid workload id specified
 */
int mars_module_workload_signal_reset(void);

/**
 * \ingroup group_mars_workload_module
 * \brief <b>[MPU]</b> Begins scheduling of specified workload.
 *
 * This function will begin scheduling the workload specified.
 * This only initiates the scheduling of the workload.
 * This function must be completed with a matching call to
 * \ref mars_module_workload_schedule_end to guarantee the completion of the
 * scheduling.
 *
 * This call will lock the workload queue until the matching call to
 * \ref mars_module_workload_schedule_end is made.
 * The user should make any necessary updates to the returned workload context
 * in between this begin call and the end call.
 *
 * \param[in] id		- id of workload
 * \param[in] priority		- scheduling priority of workload
 * \param[out] workload		- address of pointer to workload context
 * \return
 *	MARS_SUCCESS		- workload scheduling started
 * \n	MARS_ERROR_PARAMS	- invalid workload id specified
 * \n	MARS_ERROR_STATE	- specified workload not added or finished
 */
int mars_module_workload_schedule_begin(uint16_t id, uint8_t priority,
				struct mars_workload_context **workload);

/**
 * \ingroup group_mars_workload_module
 * \brief <b>[MPU]</b> Ends scheduling of specified workload.
 *
 * This function will complete a schedule operation previously initiated with
 * \ref mars_module_workload_schedule_begin.
 * This function must be called in pair for each call to
 * \ref mars_module_workload_schedule_begin to guarantee the completion of the
 * initiated schedule operation.
 *
 * \return
 *	MARS_SUCCESS		- workload scheduling complete
 * \n	MARS_ERROR_PARAMS	- invalid workload id specified
 * \n	MARS_ERROR_STATE	- workload scheduling not started
 */
int mars_module_workload_schedule_end(uint16_t id);

/**
 * \ingroup group_mars_workload_module
 * \brief <b>[MPU]</b> Cancels scheduling of specified workload.
 *
 * This function will cancel a schedule operation previously initiated with
 * \ref mars_module_workload_schedule_begin.
 * If scheduling is canceled, \ref mars_module_workload_schedule_end should not
 * be called.
 *
 * \return
 *	MARS_SUCCESS		- workload scheduling canceled
 * \n	MARS_ERROR_PARAMS	- invalid workload id specified
 * \n	MARS_ERROR_STATE	- workload scheduling not started
 */
int mars_module_workload_schedule_cancel(uint16_t id);

/**
 * \ingroup group_mars_workload_module
 * \brief <b>[MPU]</b> Returns execution to kernel with workload in wait state.
 *
 * This function will yield execution of the calling workload module and return
 * execution back to the kernel. The workload currently being processed will be
 * put into a waiting state.
 *
 * \note This function will exit the workload module and is not re-entrant.
 */
void mars_module_workload_wait(void);

/**
 * \ingroup group_mars_workload_module
 * \brief <b>[MPU]</b> Returns execution to kernel with workload in ready state.
 *
 * This function will yield execution of the calling workload module and return
 * execution back to the kernel. The workload currently being processed will be
 * put into a ready state.
 *
 * \note This function will exit the workload module and is not re-entrant.
 */
void mars_module_workload_yield(void);

/**
 * \ingroup group_mars_workload_module
 * \brief <b>[MPU]</b> Returns execution to kernel with workload in finished state.
 *
 * This function will yield execution of the calling workload module and return
 * execution back to the kernel. The workload currently being processed will be
 * put into a finished state.
 *
 * \note This function will exit the workload module and is not re-entrant.
 */
void mars_module_workload_finish(void);

/**
 * \ingroup group_mars_workload_module
 * \brief <b>[MPU]</b> Notify host a particular 32-bit area is modified.
 *
 * \param[in] watch_point_ea	- ea of modified area
 *
 * \return
 *	MARS_SUCCESS		- signal sent to host
 */
int mars_module_host_signal_send(uint64_t watch_point_ea);

/**
 * \ingroup group_mars_workload_module
 * \brief <b>[MPU]</b> Locks a mutex.
 *
 * This function locks a mutex and blocks other requests to lock it.
 * It also loads the mutex instance from the effective address specified
 * into the local mutex instance.
 *
 * \param[in] mutex_ea		- ea of mutex instance to lock
 * \param[in] mutex		- pointer to local mutex instance
 * \return
 *	MARS_SUCCESS		- successfully locked mutex
 * \n	MARS_ERROR_NULL		- ea is 0 or mutex is NULL
 * \n	MARS_ERROR_ALIGN	- ea or mutex not aligned properly
 */
int mars_module_mutex_lock_get(uint64_t mutex_ea, struct mars_mutex *mutex);

/**
 * \ingroup group_mars_workload_module
 * \brief <b>[MPU]</b> Unlocks a mutex.
 *
 * This function unlocks a previously locked mutex to allow other lock requests.
 * It also stores the local mutex instance into the effective address specified.
 *
 * \param[in] mutex_ea		- ea of mutex instance to unlock
 * \param[in] mutex		- pointer to local mutex instance
 * \return
 *	MARS_SUCCESS		- successfully unlocked mutex
 * \n	MARS_ERROR_NULL		- ea is 0 or mutex is NULL
 * \n	MARS_ERROR_ALIGN	- ea or mutex not aligned properly
 * \n	MARS_ERROR_STATE	- instance not in locked state
 */
int mars_module_mutex_unlock_put(uint64_t mutex_ea, struct mars_mutex *mutex);

/**
 * \ingroup group_mars_workload_module
 * \brief <b>[MPU]</b> DMA transfer from host storage to MPU storage.
 *
 * This function begins a DMA transfer request from host storage to MPU storage.
 * Transfer completion is not guaranteed until calling \ref mars_module_dma_wait
 * with the corresponding tag used to request the transfer.
 *
 * \param[in] ls		- address of MPU storage to transfer to
 * \param[in] ea		- ea of host storage to transfer from
 * \param[in] size		- size of dma transfer
 * \param[in] tag		- tag of dma transfer
 * \return
 *	MARS_SUCCESS		- successfully tranferred data
 * \n	MARS_ERROR_PARAMS	- invalid tag specified
 * \n	MARS_ERROR_ALIGN	- ls or ea not aligned properly
 */
int mars_module_dma_get(void *ls, uint64_t ea, uint32_t size, uint32_t tag);

/**
 * \ingroup group_mars_workload_module
 * \brief <b>[MPU]</b> DMA transfer from MPU storage to host storage.
 *
 * This function begins a DMA transfer request from MPU storage to host storage.
 * Transfer completion is not guaranteed until calling \ref mars_module_dma_wait
 * with the corresponding tag used to request the transfer.
 *
 * \param[in] ls		- address of MPU storage to transfer to
 * \param[in] ea		- ea of host storage to transfer from
 * \param[in] size		- size of dma transfer
 * \param[in] tag		- tag of dma transfer
 * \return
 *	MARS_SUCCESS		- successfully tranferred data
 * \n	MARS_ERROR_PARAMS	- invalid tag specified
 * \n	MARS_ERROR_ALIGN	- ls or ea not aligned properly
 */
int mars_module_dma_put(const void *ls, uint64_t ea, uint32_t size,
			uint32_t tag);

/**
 * \ingroup group_mars_workload_module
 * \brief <b>[MPU]</b> Waits for completion of requested DMA transfer.
 *
 * This function waits until completion of all previously started DMA transfer
 * requests with the same tag.
 *
 * \param[in] tag		- tag of dma transfer
 * \return
 *	MARS_SUCCESS		- successfully waited for transfer completion
 * \n	MARS_ERROR_PARAMS	- invalid tag specified
 */
int mars_module_dma_wait(uint32_t tag);

#if defined(__cplusplus)
}
#endif

#endif
