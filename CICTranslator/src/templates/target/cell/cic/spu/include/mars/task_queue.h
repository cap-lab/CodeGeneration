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

#ifndef MARS_TASK_QUEUE_H
#define MARS_TASK_QUEUE_H

/**
 * \file
 * \ingroup group_mars_task_queue
 * \brief <b>[MPU]</b> MARS Task Queue API
 */

#include <stdint.h>
#include <mars/task_queue_types.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * \ingroup group_mars_task_queue
 * \brief <b>[host/MPU]</b> Returns the number of data items in the task queue.
 *
 * This function will return the total number of data items in the queue at the
 * time of the call.
 *
 * \param[in] queue_ea		- ea of initialized queue instance
 * \param[out] count		- pointer to variable to store return count
 * \return
 *	MARS_SUCCESS		- successfully returned count
 * \n	MARS_ERROR_NULL		- ea is 0
 * \n	MARS_ERROR_ALIGN	- ea not aligned properly
 */
int mars_task_queue_count(uint64_t queue_ea, uint32_t *count);

/**
 * \ingroup group_mars_task_queue
 * \brief <b>[host/MPU]</b> Clears the data items in the task queue.
 *
 * This function will clear all data items currently in the queue.
 * Entities waiting to pop data from the queue will remain waiting.
 * Entities waiting to push data into the queue will resume.
 *
 * \param[in] queue_ea		- ea of initialized queue instance
 * \return
 *	MARS_SUCCESS		- successfully cleared queue
 * \n	MARS_ERROR_NULL		- ea is 0
 * \n	MARS_ERROR_ALIGN	- ea not aligned properly
 */
int mars_task_queue_clear(uint64_t queue_ea);

/**
 * \ingroup group_mars_task_queue
 * \brief <b>[host/MPU]</b> Pushes the data specified into the task queue.
 * <b>(Task Switch Call)</b>
 *
 * \note The <b>[MPU]</b> call may result in a task switch and put this
 * task into the waiting state. Understand all the limitations before calling
 * a <b>Task Switch Call</b> (<b>See</b> \ref sec_7_5).
 *
 * This function will push the data specified into the queue.
 * The entity waiting longest to pop data from the queue can resume.
 *
 * If the queue is full at the time of this call, this function will cause the
 * caller task to enter a waiting state until there is room in the queue to push
 * the data.
 *
 * <b>Key Parameters</b>:
 * \n \n
 * \e data
 * - Specify the pointer to user data to copy into the queue.
 * - The size of data should be equal to the size specified at queue creation.
 * - <b>Note:</b> In the Cell B.E. processor, data must be aligned to 16-bytes.
 *
 * \param[in] queue_ea		- ea of initialized queue instance
 * \param[in] data		- address of data to be pushed into queue
 * \return
 *	MARS_SUCCESS		- successfully pushed data into queue
 * \n	MARS_ERROR_NULL		- ea is 0
 * \n	MARS_ERROR_ALIGN	- ea not aligned properly
 * \n	MARS_ERROR_LIMIT	- exceeded limit of max waiting tasks
 * \n	MARS_ERROR_STATE	- invalid direction
 * \n	MARS_ERROR_FORMAT	- no context save area specified
 */
int mars_task_queue_push(uint64_t queue_ea, const void *data);

/**
 * \ingroup group_mars_task_queue
 * \brief <b>[MPU]</b> Begins push operation on a task queue.
 * <b>(Task Switch Call)</b>
 *
 * \note The <b>[MPU]</b> call may result in a task switch and put this
 * task into the waiting state. Understand all the limitations before calling
 * a <b>Task Switch Call</b> (<b>See</b> \ref sec_7_5).
 *
 * This function will begin pushing the data specified into the queue.
 * This only initiates the memory transfer of data into the queue.
 * This function must be completed with a matching call to
 * \ref mars_task_queue_push_end to guarantee the completion of the push.
 *
 * If the queue is full at the time of this call, this function will cause the
 * caller task to enter a waiting state until there is room in the queue to push
 * the data.
 *
 * <b>Key Parameters</b>:
 * \n \n
 * \e data
 * - Specify the pointer to user data to copy into the queue.
 * - The size of data should be equal to the size specified at queue creation.
 * - <b>Note:</b> In the Cell B.E. processor, data must be aligned to 16-bytes.
 *
 * \e tag
 * - Specify a memory transfer tag value between 0-31.
 * - Multiple push operations can be initiated concurrently, and each can be
 * waited for independently if different memory tranfer tags are specified.
 *
 * \param[in] queue_ea		- ea of initialized queue instance
 * \param[in] data		- address of data to be pushed into queue
 * \param[in] tag		- tag identifier for memory transfer
 * \return
 *	MARS_SUCCESS		- successfully pushed data into queue
 * \n	MARS_ERROR_NULL		- ea is 0
 * \n	MARS_ERROR_ALIGN	- ea not aligned properly
 * \n	MARS_ERROR_LIMIT	- exceeded limit of max waiting tasks
 * \n	MARS_ERROR_STATE	- invalid direction
 * \n	MARS_ERROR_PARAMS	- invalid tag
 * \n	MARS_ERROR_FORMAT	- no context save area specified
 */
int mars_task_queue_push_begin(uint64_t queue_ea, const void *data,
			       uint32_t tag);

/**
 * \ingroup group_mars_task_queue
 * \brief <b>[MPU]</b> Completes push operation on a task queue.
 *
 * This function will complete a push operation initiated with
 * \ref mars_task_queue_push_begin or \ref mars_task_queue_try_push_begin.
 * This function must be called in pair for each call to
 * \ref mars_task_queue_push_begin or \ref mars_task_queue_try_push_begin to
 * guarantee the completion of the initiated push operation.
 *
 * <b>Key Parameters</b>:
 * \n \n
 * \e tag
 * - Specify the memory transfer tag specified at push operation initialization.
 * - If multiple push operations were initiated concurrently, this call must
 * wait for all memory transfers initiated with the same tag to complete.
 *
 * \param[in] queue_ea		- ea of initialized queue instance
 * \param[in] tag		- tag identifier for memory transfer
 * \return
 *	MARS_SUCCESS		- successfully pushed data into queue
 * \n	MARS_ERROR_NULL		- ea is 0
 * \n	MARS_ERROR_ALIGN	- ea not aligned properly
 * \n	MARS_ERROR_STATE	- invalid direction
 * \n	MARS_ERROR_PARAMS	- invalid tag
 */
int mars_task_queue_push_end(uint64_t queue_ea, uint32_t tag);

/**
 * \ingroup group_mars_task_queue
 * \brief <b>[host/MPU]</b> Pushes the data specified into the task queue.
 *
 * This function will push the data specified into the queue.
 * The entity waiting longest to pop data from the queue can resume.
 *
 * If the queue is full at the time of this call, this function will return
 * immedately with \ref MARS_ERROR_BUSY.
 *
 * <b>Key Parameters</b>:
 * \n \n
 * \e data
 * - Specify the pointer to user data to copy into the queue.
 * - The size of data should be equal to the size specified at queue creation.
 * - <b>Note:</b> In the Cell B.E. processor, data must be aligned to 16-bytes.
 *
 * \param[in] queue_ea		- ea of initialized queue instance
 * \param[in] data		- address of data to be pushed into queue
 * \return
 *	MARS_SUCCESS		- successfully pushed data into queue
 * \n	MARS_ERROR_NULL		- ea is 0
 * \n	MARS_ERROR_ALIGN	- ea not aligned properly
 * \n	MARS_ERROR_BUSY		- queue is full
 * \n	MARS_ERROR_STATE	- invalid direction
 */
int mars_task_queue_try_push(uint64_t queue_ea, const void *data);

/**
 * \ingroup group_mars_task_queue
 * \brief <b>[MPU]</b> Begins push operation on a task queue.
 *
 * This function will begin pushing the data specified into the queue.
 * This only initiates the memory transfer of data into the queue.
 * This function must be completed with a matching call to
 * \ref mars_task_queue_push_end to guarantee the completion of the push.
 *
 * If the queue is full at the time of this call, this function will return
 * immedately with \ref MARS_ERROR_BUSY.
 *
 * <b>Key Parameters</b>:
 * \n \n
 * \e data
 * - Specify the pointer to user data to copy into the queue.
 * - The size of data should be equal to the size specified at queue creation.
 * - <b>Note:</b> In the Cell B.E. processor, data must be aligned to 16-bytes.
 *
 * \e tag
 * - Specify a memory transfer tag value between 0-31.
 * - Multiple push operations can be initiated concurrently, and each can be
 * waited for independently if different memory tranfer tags are specified.
 *
 * \param[in] queue_ea		- ea of initialized queue instance
 * \param[in] data		- address of data to be pushed into queue
 * \param[in] tag		- tag identifier for memory transfer
 * \return
 *	MARS_SUCCESS		- successfully pushed data into queue
 * \n	MARS_ERROR_NULL		- ea is 0
 * \n	MARS_ERROR_ALIGN	- ea not aligned properly
 * \n	MARS_ERROR_BUSY		- queue is full
 * \n	MARS_ERROR_STATE	- invalid direction
 * \n	MARS_ERROR_PARAMS	- invalid tag
 */
int mars_task_queue_try_push_begin(uint64_t queue_ea, const void *data,
				   uint32_t tag);

/**
 * \ingroup group_mars_task_queue
 * \brief <b>[host/MPU]</b> Pops data from a task queue.
 * <b>(Task Switch Call)</b>
 *
 * \note The <b>[MPU]</b> call may result in a task switch and put this
 * task into the waiting state. Understand all the limitations before calling
 * a <b>Task Switch Call</b> (<b>See</b> \ref sec_7_5).
 *
 * This function will pop data from the queue.
 *
 * If the queue is empty at the time of this call, this function will cause the
 * caller task to enter a waiting state until there is data in the queue to
 * pop.
 *
 * <b>Key Parameters</b>:
 * \n \n
 * \e data
 * - Specify the pointer to some allocated memory to copy the popped data.
 * - The size of the allocated memory area should be equal to the size specified
 * at queue creation.
 * - <b>Note:</b> In the Cell B.E. processor, data must be aligned to 16-bytes.
 *
 * \param[in] queue_ea		- ea of initialized queue instance
 * \param[out] data		- address of data to be popped from queue
 * \return
 *	MARS_SUCCESS		- successfully popped data from queue
 * \n	MARS_ERROR_NULL		- ea or data is 0
 * \n	MARS_ERROR_ALIGN	- ea or data not aligned properly
 * \n	MARS_ERROR_LIMIT	- exceeded limit of max waiting tasks
 * \n	MARS_ERROR_STATE	- invalid direction
 * \n	MARS_ERROR_FORMAT	- no context save area specified
 */
int mars_task_queue_pop(uint64_t queue_ea, void *data);

/**
 * \ingroup group_mars_task_queue
 * \brief <b>[MPU]</b> Begins pop operation on a task queue.
 * <b>(Task Switch Call)</b>
 *
 * \note The <b>[MPU]</b> call may result in a task switch and put this
 * task into the waiting state. Understand all the limitations before calling
 * a <b>Task Switch Call</b> (<b>See</b> \ref sec_7_5).
 *
 * This function will begin popping data from the queue.
 * This only initiates the memory transfer of data from the queue.
 * This function must be completed with a matching call to
 * \ref mars_task_queue_pop_end to guarantee the completion of the pop.
 *
 * If the queue is empty at the time of this call, this function will cause the
 * caller task to enter a waiting state until there is data in the queue to
 * pop.
 *
 * <b>Key Parameters</b>:
 * \n \n
 * \e data
 * - Specify the pointer to some allocated memory to copy the popped data.
 * - The size of the allocated memory area should be equal to the size specified
 * at queue creation.
 * - <b>Note:</b> In the Cell B.E. processor, data must be aligned to 16-bytes.
 *
 * \param[in] queue_ea		- ea of initialized queue instance
 * \param[out] data		- address of data to be popped from queue
 * \param[in] tag		- tag identifier for memory transfer
 * \return
 *	MARS_SUCCESS		- successfully popped data from queue
 * \n	MARS_ERROR_NULL		- ea or data is 0
 * \n	MARS_ERROR_ALIGN	- ea or data not aligned properly
 * \n	MARS_ERROR_LIMIT	- exceeded limit of max waiting tasks
 * \n	MARS_ERROR_STATE	- invalid direction
 * \n	MARS_ERROR_PARAMS	- invalid tag
 * \n	MARS_ERROR_FORMAT	- no context save area specified
 */
int mars_task_queue_pop_begin(uint64_t queue_ea, void *data, uint32_t tag);

/**
 * \ingroup group_mars_task_queue
 * \brief <b>[MPU]</b> Completes pop operation on a task queue.
 *
 * This function will complete a pop operation initiated with
 * \ref mars_task_queue_pop_begin or \ref mars_task_queue_try_pop_begin.
 * This function must be called in pair for each call to
 * \ref mars_task_queue_pop_begin or \ref mars_task_queue_try_pop_begin to
 * guarantee the completion of the initiated pop operation.
 *
 * <b>Key Parameters</b>:
 * \n \n
 * \e tag
 * - Specify the memory transfer tag specified at pop operation initialization.
 * - If multiple pop operations were initiated concurrently, this call must
 * wait for all memory transfers initiated with the same tag to complete.
 *
 * \param[in] queue_ea		- ea of initialized queue instance
 * \param[in] tag		- tag identifier for memory transfer
 * \return
 *	MARS_SUCCESS		- successfully popped data from queue
 * \n	MARS_ERROR_NULL		- ea or data is 0
 * \n	MARS_ERROR_ALIGN	- ea or data not aligned properly
 * \n	MARS_ERROR_STATE	- invalid direction
 * \n	MARS_ERROR_PARAMS	- invalid tag
 */
int mars_task_queue_pop_end(uint64_t queue_ea, uint32_t tag);

/**
 * \ingroup group_mars_task_queue
 * \brief <b>[host/MPU]</b> Pops data from a task queue.
 *
 * This function will pop data from the queue.
 *
 * If the queue is empty at the time of this call, this function will return
 * immedately with \ref MARS_ERROR_BUSY.
 *
 * <b>Key Parameters</b>:
 * \n \n
 * \e data
 * - Specify the pointer to some allocated memory to copy the popped data.
 * - The size of the allocated memory area should be equal to the size specified
 * at queue creation.
 * - <b>Note:</b> In the Cell B.E. processor, data must be aligned to 16-bytes.
 *
 * \param[in] queue_ea		- ea of initialized queue instance
 * \param[out] data		- address for data to be popped from queue
 * \return
 *	MARS_SUCCESS		- successfully popped data from queue
 * \n	MARS_ERROR_NULL		- ea or data is 0
 * \n	MARS_ERROR_ALIGN	- ea or data not aligned properly
 * \n	MARS_ERROR_BUSY		- queue is empty
 * \n	MARS_ERROR_STATE	- invalid direction
 */
int mars_task_queue_try_pop(uint64_t queue_ea, void *data);

/**
 * \ingroup group_mars_task_queue
 * \brief <b>[MPU]</b> Begins pop operation on a task queue.
 *
 * This function will begin popping data from the queue.
 * This only initiates the memory transfer of data from the queue.
 * This function must be completed with a matching call to
 * \ref mars_task_queue_pop_end to guarantee the completion of the pop.
 *
 * If the queue is empty at the time of this call, this function will return
 * immedately with \ref MARS_ERROR_BUSY.
 *
 * <b>Key Parameters</b>:
 * \n \n
 * \e data
 * - Specify the pointer to some allocated memory to copy the popped data.
 * - The size of the allocated memory area should be equal to the size specified
 * at queue creation.
 * - <b>Note:</b> In the Cell B.E. processor, data must be aligned to 16-bytes.
 *
 * \param[in] queue_ea		- ea of initialized queue instance
 * \param[out] data		- address for data to be popped from queue
 * \param[in] tag		- tag identifier for memory transfer
 * \return
 *	MARS_SUCCESS		- successfully popped data from queue
 * \n	MARS_ERROR_NULL		- ea or data is 0
 * \n	MARS_ERROR_ALIGN	- ea or data not aligned properly
 * \n	MARS_ERROR_BUSY		- queue is empty
 * \n	MARS_ERROR_STATE	- invalid direction
 * \n	MARS_ERROR_PARAMS	- invalid tag
 */
int mars_task_queue_try_pop_begin(uint64_t queue_ea, void *data, uint32_t tag);

/**
 * \ingroup group_mars_task_queue
 * \brief <b>[host/MPU]</b> Pops data from a task queue without removing it.
 * <b>(Task Switch Call)</b>
 *
 * \note The <b>[MPU]</b> call may result in a task switch and put this
 * task into the waiting state. Understand all the limitations before calling
 * a <b>Task Switch Call</b> (<b>See</b> \ref sec_7_5).
 *
 * This function will retrieve data from the queue without removing it from the
 * queue.
 *
 * If the queue is empty at the time of this call, this function will cause the
 * caller task to enter a waiting state until there is data in the queue to
 * be retrieved.
 *
 * <b>Key Parameters</b>:
 * \n \n
 * \e data
 * - Specify the pointer to some allocated memory to copy the popped data.
 * - The size of the allocated memory area should be equal to the size specified
 * at queue creation.
 * - <b>Note:</b> In the Cell B.E. processor, data must be aligned to 16-bytes.
 *
 * \param[in] queue_ea		- ea of initialized queue instance
 * \param[out] data		- address of data to be popped from queue
 * \return
 *	MARS_SUCCESS		- successfully popped data from queue
 * \n	MARS_ERROR_NULL		- ea or data is 0
 * \n	MARS_ERROR_ALIGN	- ea or data not aligned properly
 * \n	MARS_ERROR_LIMIT	- exceeded limit of max waiting tasks
 * \n	MARS_ERROR_STATE	- invalid direction
 * \n	MARS_ERROR_FORMAT	- no context save area specified
 */
int mars_task_queue_peek(uint64_t queue_ea, void *data);

/**
 * \ingroup group_mars_task_queue
 * \brief <b>[MPU]</b> Begins peek operation on a task queue.
 * <b>(Task Switch Call)</b>
 *
 * \note The <b>[MPU]</b> call may result in a task switch and put this
 * task into the waiting state. Understand all the limitations before calling
 * a <b>Task Switch Call</b> (<b>See</b> \ref sec_7_5).
 *
 * This function will begin retriveing data from the queue without removing the
 * data from the queue.
 * This only initiates the memory transfer of data from the queue.
 * This function must be completed with a matching call to
 * \ref mars_task_queue_peek_end to guarantee the completion of the peek.
 *
 * If the queue is empty at the time of this call, this function will cause the
 * caller task to enter a waiting state until there is data in the queue to
 * be retrived.
 *
 * <b>Key Parameters</b>:
 * \n \n
 * \e data
 * - Specify the pointer to some allocated memory to copy the popped data.
 * - The size of the allocated memory area should be equal to the size specified
 * at queue creation.
 * - <b>Note:</b> In the Cell B.E. processor, data must be aligned to 16-bytes.
 *
 * \param[in] queue_ea		- ea of initialized queue instance
 * \param[out] data		- address of data to be popped from queue
 * \param[in] tag		- tag identifier for memory transfer
 * \return
 *	MARS_SUCCESS		- successfully popped data from queue
 * \n	MARS_ERROR_NULL		- ea or data is 0
 * \n	MARS_ERROR_ALIGN	- ea or data not aligned properly
 * \n	MARS_ERROR_LIMIT	- exceeded limit of max waiting tasks
 * \n	MARS_ERROR_STATE	- invalid direction
 * \n	MARS_ERROR_PARAMS	- invalid tag
 * \n	MARS_ERROR_FORMAT	- no context save area specified
 */
int mars_task_queue_peek_begin(uint64_t queue_ea, void *data, uint32_t tag);

/**
 * \ingroup group_mars_task_queue
 * \brief <b>[MPU]</b> Completes peek operation on a task queue.
 *
 * This function will complete a peek operation initiated with
 * \ref mars_task_queue_peek_begin or \ref mars_task_queue_try_peek_begin.
 * This function must be called in pair for each call to
 * \ref mars_task_queue_peek_begin or \ref mars_task_queue_try_peek_begin to
 * guarantee the completion of the initiated peek operation.
 *
 * <b>Key Parameters</b>:
 * \n \n
 * \e tag
 * - Specify the memory transfer tag specified at peek operation initialization.
 * - If multiple peek operations were initiated concurrently, this call must
 * wait for all memory transfers initiated with the same tag to complete.
 *
 * \param[in] queue_ea		- ea of initialized queue instance
 * \param[in] tag		- tag identifier for memory transfer
 * \return
 *	MARS_SUCCESS		- successfully popped data from queue
 * \n	MARS_ERROR_NULL		- ea or data is 0
 * \n	MARS_ERROR_ALIGN	- ea or data not aligned properly
 * \n	MARS_ERROR_STATE	- invalid direction
 * \n	MARS_ERROR_PARAMS	- invalid tag
 */
int mars_task_queue_peek_end(uint64_t queue_ea, uint32_t tag);

/**
 * \ingroup group_mars_task_queue
 * \brief <b>[host/MPU]</b> Pops data from a task queue without removing it.
 *
 * This function will retrieve data from the queue without removing it from the
 * queue.
 *
 * If the queue is empty at the time of this call, this function will return
 * immedately with \ref MARS_ERROR_BUSY.
 *
 * <b>Key Parameters</b>:
 * \n \n
 * \e data
 * - Specify the pointer to some allocated memory to copy the popped data.
 * - The size of the allocated memory area should be equal to the size specified
 * at queue creation.
 * - <b>Note:</b> In the Cell B.E. processor, data must be aligned to 16-bytes.
 * .
 * \param[in] queue_ea		- ea of initialized queue instance
 * \param[out] data		- address of data to be popped from queue
 * \return
 *	MARS_SUCCESS		- successfully popped data from queue
 * \n	MARS_ERROR_NULL		- ea or data is 0
 * \n	MARS_ERROR_ALIGN	- ea or data not aligned properly
 * \n	MARS_ERROR_BUSY		- queue is empty
 * \n	MARS_ERROR_STATE	- invalid direction
 */
int mars_task_queue_try_peek(uint64_t queue_ea, void *data);

/**
 * \ingroup group_mars_task_queue
 * \brief <b>[MPU]</b> Begins peek operation on a task queue.
 *
 * This function will begin retriveing data from the queue without removing the
 * data from the queue.
 * This only initiates the memory transfer of data from the queue.
 * This function must be completed with a matching call to
 * \ref mars_task_queue_peek_end to guarantee the completion of the peek.
 *
 * If the queue is empty at the time of this call, this function will return
 * immedately with \ref MARS_ERROR_BUSY.
 *
 * <b>Key Parameters</b>:
 * \n \n
 * \e data
 * - Specify the pointer to some allocated memory to copy the popped data.
 * - The size of the allocated memory area should be equal to the size specified
 * at queue creation.
 * - <b>Note:</b> In the Cell B.E. processor, data must be aligned to 16-bytes.
 *
 * \param[in] queue_ea		- ea of initialized queue instance
 * \param[out] data		- address of data to be popped from queue
 * \param[in] tag		- tag identifier for memory transfer
 * \return
 *	MARS_SUCCESS		- successfully popped data from queue
 * \n	MARS_ERROR_NULL		- ea or data is 0
 * \n	MARS_ERROR_ALIGN	- ea or data not aligned properly
 * \n	MARS_ERROR_BUSY		- queue is empty
 * \n	MARS_ERROR_STATE	- invalid direction
 * \n	MARS_ERROR_PARAMS	- invalid tag
 */
int mars_task_queue_try_peek_begin(uint64_t queue_ea, void *data, uint32_t tag);

#if defined(__cplusplus)
}
#endif

#endif
