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

#ifndef MARS_ERROR_H
#define MARS_ERROR_H

/**
 * \file
 * \ingroup group_mars_base
 * \brief <b>[host/MPU]</b> MARS Error Values
 */

/**
 * \brief MARS error values
 *
 * These are the generic error values returned by MARS API functions.
 *
 * For specific reasons for the error, please refer to the specific function
 * reference in the API documentation.
 */
enum {
	MARS_SUCCESS = 0,	/**< successful with no errors */
	MARS_ERROR_NULL,	/**< null pointer passed in */
	MARS_ERROR_PARAMS,	/**< invalid parameter specified */
	MARS_ERROR_INTERNAL,	/**< internal library error */
	MARS_ERROR_MEMORY,	/**< out of memory */
	MARS_ERROR_ALIGN,	/**< bad memory alignment */
	MARS_ERROR_LIMIT,	/**< some limit exceeded */
	MARS_ERROR_STATE,	/**< something is in an invalid state */
	MARS_ERROR_FORMAT,	/**< invalid format specified */
	MARS_ERROR_BUSY		/**< operation returned due to being busy */
};

#endif
