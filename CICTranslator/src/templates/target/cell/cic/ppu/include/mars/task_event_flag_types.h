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

#ifndef MARS_TASK_EVENT_FLAG_TYPES_H
#define MARS_TASK_EVENT_FLAG_TYPES_H

/**
 * \file
 * \ingroup group_mars_task_event_flag
 * \brief <b>[host/MPU]</b> MARS Task Event Flag Types
 */

/**
 * \ingroup group_mars_task_event_flag
 * \brief Event flag direction from PPU to SPU
 */
#define MARS_TASK_EVENT_FLAG_HOST_TO_MPU	0x10

/**
 * \ingroup group_mars_task_event_flag
 * \brief Event flag direction from SPU to PPU
 */
#define MARS_TASK_EVENT_FLAG_MPU_TO_HOST	0x11

/**
 * \ingroup group_mars_task_event_flag
 * \brief Event flag direction from SPU to SPU
 */
#define MARS_TASK_EVENT_FLAG_MPU_TO_MPU		0x12

/**
 * \ingroup group_mars_task_event_flag
 * \brief Event flag clear mode automatic
 */
#define MARS_TASK_EVENT_FLAG_CLEAR_AUTO		0x20

/**
 * \ingroup group_mars_task_event_flag
 * \brief Event flag clear mode manual
 */
#define MARS_TASK_EVENT_FLAG_CLEAR_MANUAL	0x21

/**
 * \ingroup group_mars_task_event_flag
 * \brief Event flag mask mode bitwise OR
 */
#define MARS_TASK_EVENT_FLAG_MASK_OR		0x30

/**
 * \ingroup group_mars_task_event_flag
 * \brief Event flag mask mode bitwise AND
 */
#define MARS_TASK_EVENT_FLAG_MASK_AND		0x31

/**
 * \ingroup group_mars_task_event_flag
 * \brief Maximum tasks allowed to wait on a single event flag
 */
#define MARS_TASK_EVENT_FLAG_WAIT_MAX		15

/**
 * \ingroup group_mars_task_event_flag
 * \brief MARS task event flag structure
 *
 * An instance of this structure must be created when using any of the
 * MARS event flag API.
 */
struct mars_task_event_flag;

#endif
