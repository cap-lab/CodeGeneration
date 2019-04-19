
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

<#if lib_info.masterPortToLibraryMap??>
#define LIBCALL(x, ...) LIBCALL_##x(__VA_ARGS__)
	<#list lib_info.masterPortToLibraryMap as portName, library>
#define LIBCALL_${portName}(f, ...) l_${library.name}_##f(__VA_ARGS__)
#include "${library.header}"
	</#list>
</#if>

#include "${lib_info.header}"

#define LIBCALL(x, ...) LIBCALL_##x(__VA_ARGS__)
#define LIBCALL_this(f, ...) l_${lib_info.name}_##f(__VA_ARGS__)
#define LIBFUNC(rtype, f, ...) rtype l_${lib_info.name}_##f(__VA_ARGS__)


<#if lib_info.isMasterLanguageC == true>
#ifdef __cplusplus
extern "C"
{
#endif
</#if>

#include "${lib_info.file}"

<#if lib_info.isMasterLanguageC == true>
#ifdef __cplusplus
}
#endif
</#if>

#undef LIBFUNC