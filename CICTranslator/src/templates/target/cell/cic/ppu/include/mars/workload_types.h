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

#ifndef MARS_WORKLOAD_TYPES_H
#define MARS_WORKLOAD_TYPES_H

/**
 * \file
 * \ingroup group_mars_workload_queue
 * \ingroup group_mars_workload_module
 * \brief <b>[host/MPU]</b> MARS Workload Types
 */

#include <stdint.h>

/**
 * \ingroup group_mars_workload_module
 * \brief Base address of workload module
 */
#define MARS_WORKLOAD_MODULE_BASE_ADDR		0x3000

/**
 * \ingroup group_mars_workload_module
 * \brief Size of workload module structure
 */
#define MARS_WORKLOAD_MODULE_SIZE		20

/**
 * \ingroup group_mars_workload_queue
 * \brief Size of workload context structure
 */
#define MARS_WORKLOAD_CONTEXT_SIZE		128

/**
 * \ingroup group_mars_workload_queue
 * \brief Alignment of workload context structure
 */
#define MARS_WORKLOAD_CONTEXT_ALIGN		128

/**
 * \ingroup group_mars_workload_module
 * \brief MARS workload module structure
 *
 * This structure stores information about the workload module executable that
 * needs to be loaded and executed in order to handle processing of a workload
 * context.
 *
 * The workload model implementation is responsible for populating this
 * structure inside the workload context before adding the workload context to
 * the workload queue.
 */
struct mars_workload_module {
	/** ea of exec */
	uint64_t exec_ea;
	/** size of text and data of exec */
	uint32_t exec_size;
	/** size of bss in memory of exec */
	uint32_t bss_size;
	/** entry address of exec */
	uint32_t entry;
} __attribute__((packed));

/**
 * \ingroup group_mars_workload_context
 * \brief MARS workload context structure
 *
 * This structure stores information about a specific workload.
 * The first \ref MARS_WORKLOAD_MODULE_SIZE bytes of the workload context
 * structure is reserved for the \ref mars_workload_module information.
 *
 * The remaining area of the structure can be used by the specific workload
 * model implementation as needed.
 */
struct mars_workload_context {
	/** workload module information */
	struct mars_workload_module module;
	/** workload model specific data */
	uint8_t context[MARS_WORKLOAD_CONTEXT_SIZE - MARS_WORKLOAD_MODULE_SIZE];
} __attribute__((aligned(MARS_WORKLOAD_CONTEXT_ALIGN)));

#endif
