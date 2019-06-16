/*
 * UCPrint.h
 *
 *  Created on: 2018. 9. 3.
 *      Author: chjej202
 */

#ifndef SRC_COMMON_CONSTRAINED_INCLUDE_UCPRINT_H_
#define SRC_COMMON_CONSTRAINED_INCLUDE_UCPRINT_H_


#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Print format for Arduino platform.
 *
 * This function works as printf() function in unconstrained platforms. \n
 *
 * @warning This function costs lots of memory for memory-constrained devices, so minimum usage is required.
 * @warning Floating point printing may not be working well because of device restriction.
 *
 * @param pszFormat print format which is similar to printf().
 * @param ... arguments used in printf().
 */
void UCPrint_format(const char *pszFormat, ... );

#ifdef __cplusplus
}
#endif


#endif /* SRC_COMMON_CONSTRAINED_INCLUDE_UCPRINT_H_ */
