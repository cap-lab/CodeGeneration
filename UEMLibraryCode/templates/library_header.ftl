#ifndef __UEM_LIB_HEADER_${lib_info.headerGuard}__
#define __UEM_LIB_HEADER_${lib_info.headerGuard}__

<#list lib_info.extraHeaderSet as headerFile>
#include "${headerFile}"
</#list>

#define LIBFUNC(rtype, f, ...) rtype l_${lib_info.name}_##f(__VA_ARGS__)

<#if lib_info.language == "C" || lib_info.isMasterLanguageC == true>
#ifdef __cplusplus
extern "C"
{
#endif
</#if>

extern LIBFUNC(void, init, void);
extern LIBFUNC(void, wrapup, void);
<#list lib_info.functionList as function>
extern LIBFUNC(${function.returnType}, ${function.name}<#list function.argumentList as argument>, ${argument.type} ${argument.name}</#list>);
</#list>



<#if lib_info.language == "C" || lib_info.isMasterLanguageC == true>
#ifdef __cplusplus
}
#endif
</#if>

#undef LIBFUNC

#endif /* __UEM_LIB_HEADER_${lib_info.headerGuard}__ */


