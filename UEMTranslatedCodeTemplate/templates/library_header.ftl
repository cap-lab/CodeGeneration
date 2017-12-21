#ifndef __UEM_LIB_HEADER_${lib_info.headerGuard}__
#define __UEM_LIB_HEADER_${lib_info.headerGuard}__

extern LIBFUNC(void, init, void);
extern LIBFUNC(void, wrapup, void);
<#list lib_info.functionList as function>
extern LIBFUNC(${function.returnType}, ${function.name}<#list function.argumentList as argument>, ${argument.type} ${argument.name}</#list>);
</#list>

#endif /* __UEM_LIB_HEADER_${lib_info.headerGuard}__ */