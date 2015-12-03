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

#ifndef MARS_CONTEXT_H
#define MARS_CONTEXT_H

/**
 * \file
 * \ingroup group_mars_context
 * \brief <b>[host]</b> MARS Context API
 */

#include <stdint.h>

/**
 * \ingroup group_mars_context
 * \brief MARS context structure
 *
 * An instance of this structure must be created and initialized before
 * using any of the MARS API.
 */
struct mars_context;

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * \ingroup group_mars_context
 * \brief <b>[host]</b> Creates a single MARS context.
 *
 * This function creates a single MARS context. A MARS context must be
 * created before any of the MARS functionality can be used. This will
 * create the MPU contexts that are each loaded with and run the MARS kernel.
 * The MARS kernel on each MPU will continue to run until the MARS context
 * is destroyed through \ref mars_context_destroy.
 *
 * <b>Key Parameters</b>:
 * \n \n
 * \e num_mpus
 * - Specify total number of MPUs to be used by the MARS context
 * - If 0 is specified, MARS will use the maximum number of MPUs available in
 * the system.
 *
 * \e shared
 * - Specify 1 to share the context with other libraries linked into the
 * application that also utilize MARS.
 * - Specify 0 to create an independent MARS context that is not shared with
 * other libraries linked into the application that also utilize MARS.
 * - Sharing a single MARS context within an application with other libraries
 * will maximize the MARS benefits of MPU utilization.
 *
 * \note If there are multiple MARS contexts created in the system, then
 * each MARS context will suffer the large over head of MPU context switches.
 *
 * \param[out] mars		- address of pointer to MARS context
 * \param[in] num_mpus		- number of mpus utilized by MARS context
 * \param[in] shared		- specifies if context is shared or not
 * \return
 *	MARS_SUCCESS		- successfully created MARS context
 * \n	MARS_ERROR_NULL		- null pointer specified
 * \n	MARS_ERROR_PARAMS	- bad MARS params specified
 * \n	MARS_ERROR_MEMORY	- not enough memory
 * \n	MARS_ERROR_INTERNAL	- some internal error occurred
 */
int mars_context_create(struct mars_context **mars, uint32_t num_mpus, uint8_t shared);

/**
 * \ingroup group_mars_context
 * \brief <b>[host]</b> Destroys a single MARS context.
 *
 * This function destroys a single MARS context that was previously
 * created by \ref mars_context_create. In order to successfully destroy
 * a MARS context, all workloads added to the workload queue must be
 * completed and destroyed so that the workload queue is empty.
 *
 * \param[in] mars		- pointer to MARS context
 * \return
 *	MARS_SUCCESS		- successfully destroyed MARS context
 * \n	MARS_ERROR_NULL		- null pointer specified
 * \n	MARS_ERROR_STATE	- workload queue is not empty
 */
int mars_context_destroy(struct mars_context *mars);

#if defined(__cplusplus)
}
#endif

#endif
