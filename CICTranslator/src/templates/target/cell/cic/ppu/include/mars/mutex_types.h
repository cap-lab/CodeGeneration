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

#ifndef MARS_MUTEX_TYPES_H
#define MARS_MUTEX_TYPES_H

/**
 * \file
 * \ingroup group_mars_mutex
 * \brief <b>[host/MPU]</b> MARS Mutex Types
 */

#include <stdint.h>

#define MARS_MUTEX_SIZE		128
#define MARS_MUTEX_ALIGN	128
#define MARS_MUTEX_ALIGN_MASK	0x7f
#define MARS_MUTEX_LOCKED	0x1
#define MARS_MUTEX_UNLOCKED	0x0

/**
 * \ingroup group_mars_mutex
 * \brief MARS mutex structure
 *
 * An instance of this structure must be created when using the MARS Mutex API.
 */

struct mars_mutex_status {
	uint8_t lock;
	uint8_t pad;
	uint8_t current_id;
	uint8_t next_id;
};

struct mars_mutex {
	struct mars_mutex_status status;
	uint8_t pad[124];
} __attribute__((aligned(MARS_MUTEX_ALIGN)));

#endif
