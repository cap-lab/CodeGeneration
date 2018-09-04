/* uem_module_data.c made by UEM Translator */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <uem_common.h>

#include <uem_data.h>

<#list module_list as module>
	<#list module.headerList as headerFileName>
#include <${headerFileName}>
	</#list>
</#list>


SAddOnModule g_stModules[] = {
<#list module_list as module>
	{
		${module.initializer}, // module initialization function
		${module.finalizer}, // module finalization function
	},
</#list>
};

<#if (module_list?size > 0) >
int g_nModuleNum = ARRAYLEN(g_stModules);
<#else>
int g_nModuleNum = 0;
</#if>


