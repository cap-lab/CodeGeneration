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

#ifndef MARS_MUTEX_H
#define MARS_MUTEX_H

/**
 * \file
 * \ingroup group_mars_mutex
 * \brief <b>[host]</b> MARS Mutex API
 */

#include <mars/mutex_types.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * \ingroup group_mars_mutex
 * \brief <b>[host]</b> Creates a mutex.
 *
 * This function creates a mutex instance that can be locked or unlocked
 * from both host and MPU to restrict concurrent accesses.
 *
 * \param[in] mutex_ea		- ea of mutex instance
 * \return
 *	MARS_SUCCESS		- successfully created mutex
 * \n	MARS_ERROR_NULL		- null pointer is specified
 * \n	MARS_ERROR_MEMORY	- instance not aligned properly
 */
int mars_mutex_create(uint64_t *mutex_ea);

/**
 * \ingroup group_mars_mutex
 * \brief <b>[host]</b> Destroys a mutex.
 *
 * This function destroys a mutex instance.
 *
 * \param[in] mutex_ea		- ea of mutex instance
 * \return
 *	MARS_SUCCESS		- successfully destroyed mutex
 * \n	MARS_ERROR_NULL		- null pointer is specified
 * \n	MARS_ERROR_ALIGN	- instance not aligned properly
 */
int mars_mutex_destroy(uint64_t mutex_ea);

/**
 * \ingroup group_mars_mutex
 * \brief <b>[host]</b> Resets a mutex.
 *
 * This function resets a mutex instance and forces it into an unlocked state
 * regardless of whether it is locked or unlocked.
 *
 * \param[in] mutex_ea		- ea of mutex instance
 * \return
 *	MARS_SUCCESS		- successfully reset mutex
 * \n	MARS_ERROR_NULL		- null pointer is specified
 * \n	MARS_ERROR_ALIGN	- instance not aligned properly
 */
int mars_mutex_reset(uint64_t mutex_ea);

/**
 * \ingroup group_mars_mutex
 * \brief <b>[host]</b> Locks a mutex.
 *
 * This function locks a mutex and blocks other requests to lock it.
 *
 * \param[in] mutex_ea		- ea of mutex instance
 * \return
 *	MARS_SUCCESS		- successfully locked mutex
 * \n	MARS_ERROR_NULL		- null pointer is specified
 * \n	MARS_ERROR_ALIGN	- instance not aligned properly
 */
int mars_mutex_lock(uint64_t mutex_ea);

/**
 * \ingroup group_mars_mutex
 * \brief <b>[host]</b> Unlocks a mutex.
 *
 * This function unlocks a previously locked mutex to allow other lock requests.
 *
 * \param[in] mutex_ea		- ea of mutex instance
 * \return
 *	MARS_SUCCESS		- successfully unlocked mutex
 * \n	MARS_ERROR_NULL		- null pointer is specified
 * \n	MARS_ERROR_ALIGN	- instance not aligned properly
 * \n	MARS_ERROR_STATE	- instance not in locked state
 */
int mars_mutex_unlock(uint64_t mutex_ea);

/**
 * \ingroup group_mars_mutex
 * \brief <b>[host]</b> Locks a mutex.
 *
 * This function locks a mutex and blocks other requests to lock it.
 * It also loads the mutex instance from the effective address specified
 * into the local mutex instance.
 *
 * \note This call should only be used when MARS_ENABLE_DISCRETE_SHARED_MEMORY
 * is enabled. Otherwise, the mutex parameter is ignored and the function
 * behaves the same as \ref mars_mutex_lock.
 *
 * \param[in] mutex_ea		- ea of mutex instance to lock
 * \param[in] mutex		- pointer to local mutex instance
 * \return
 *	MARS_SUCCESS		- successfully locked mutex
 * \n	MARS_ERROR_NULL		- ea is 0 or mutex is NULL
 * \n	MARS_ERROR_ALIGN	- ea or mutex not aligned properly
 */
int mars_mutex_lock_get(uint64_t mutex_ea, struct mars_mutex *mutex);

/**
 * \ingroup group_mars_mutex
 * \brief <b>[host]</b> Unlocks a mutex.
 *
 * This function unlocks a previously locked mutex to allow other lock requests.
 * It also stores the local mutex instance into the effective address specified.
 *
 * \note This call should only be used when MARS_ENABLE_DISCRETE_SHARED_MEMORY
 * is enabled. Otherwise, the mutex parameter is ignored and the function
 * behaves the same as \ref mars_mutex_unlock.
 *
 * \param[in] mutex_ea		- ea of mutex instance to unlock
 * \param[in] mutex		- pointer to local mutex instance
 * \return
 *	MARS_SUCCESS		- successfully unlocked mutex
 * \n	MARS_ERROR_NULL		- ea is 0 or mutex is NULL
 * \n	MARS_ERROR_ALIGN	- ea or mutex not aligned properly
 * \n	MARS_ERROR_STATE	- instance not in locked state
 */
int mars_mutex_unlock_put(uint64_t mutex_ea, struct mars_mutex *mutex);

#if defined(__cplusplus)
}
#endif

#endif
