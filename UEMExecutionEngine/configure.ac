#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.68])
AC_INIT([UEM Translated Code], [0.1], [sha@snu.ac.kr])
AC_CONFIG_HEADERS([config.h])
AC_CANONICAL_SYSTEM
AM_INIT_AUTOMAKE()

# Checks for programs.
AC_PROG_CC
AX_PTHREAD


# Checks for libraries.

# Checks for header files.
AC_CHECK_HEADERS([stdlib.h pthread_time.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_SIZEOF([size_t])
AC_CHECK_SIZEOF([void *])

# Checks for library functions.
AC_CHECK_FUNCS(mingw_gettimeofday pthread_timedjoin_np)

AC_FUNC_MALLOC
AC_FUNC_REALLOC

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
