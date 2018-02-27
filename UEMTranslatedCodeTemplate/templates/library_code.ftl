
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "${lib_info.header}"

#define LIBCALL(x, ...) LIBCALL_##x(__VA_ARGS__)
#define LIBCALL_this(f, ...) l_${lib_info.name}_##f(__VA_ARGS__)
#define LIBFUNC(rtype, f, ...) rtype l_${lib_info.name}_##f(__VA_ARGS__)

#include "${lib_info.file}"

#undef LIBFUNC
