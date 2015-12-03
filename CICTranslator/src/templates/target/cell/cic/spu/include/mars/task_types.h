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

#ifndef MARS_TASK_TYPES_H
#define MARS_TASK_TYPES_H

/**
 * \file
 * \ingroup group_mars_task
 * \brief <b>[host/MPU]</b> MARS Task Types
 */

#include <stdint.h>

/**
 * \ingroup group_mars_task
 * \brief Base address of task
 */
#define MARS_TASK_BASE_ADDR			0x4000

/**
 * \ingroup group_mars_task
 * \brief Max size of context save area
 */
#define MARS_TASK_CONTEXT_SAVE_SIZE_MAX		0x3c000

/**
 * \ingroup group_mars_task
 * \brief Max length of task name
 */

// by iloy 2009.03.02
#ifndef CIC
#define CIC (1)
#endif
#if defined(CIC) && (CIC==1)
#define MARS_TASK_NAME_LEN_MAX			(21-2)
#else
#define MARS_TASK_NAME_LEN_MAX			21
#endif

/**
 * \ingroup group_mars_task
 * \brief MARS task id structure
 *
 * This structure is initialized during MARS task creation and returned when
 * calling \ref mars_task_create.
 *
 * An instance of this structure must be kept until the task is destroyed by
 * calling \ref mars_task_destroy.
 */
struct mars_task_id {
	uint64_t mars_context_ea;
	uint16_t workload_id;
#if defined(CIC) && (CIC==1)
    uint16_t affinity;
#endif
	uint8_t name[MARS_TASK_NAME_LEN_MAX + 1];
#if defined(CIC) && (CIC==1)
} __attribute__((packed));
#else
};
#endif

/**
 * \ingroup group_mars_task
 * \brief MARS task argument structure
 *
 * This structure is initialized by the user and passed into
 * \ref mars_task_schedule for MARS task scheduling.
 *
 * This argument structure is directly passed into the MARS
 * task's \ref mars_task_main function at task execution.
 */
struct mars_task_args {
	union {
		/** array of 32 8-bit unsigned ints */
		uint8_t  u8[32];
		/** array of 16 16-bit unsigned ints */
		uint16_t u16[16];
		/** array of 8 32-bit unsigned ints */
		uint32_t u32[8];
		/** array of 4 64-bit unsigned ints */
		uint64_t u64[4];
	} type;
};

#endif
