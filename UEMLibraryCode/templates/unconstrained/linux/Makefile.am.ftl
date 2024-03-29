
<#list env_var_info as envVar>
${envVar.name}=${envVar.value}
</#list>

MAIN_DIR=src/main

APPLICATION_DIR=src/application
API_DIR=src/api
KERNEL_DIR=src/kernel
COMMON_DIR=src/common
MODULE_DIR=src/module
PLATFORM_DIR=${build_info.platformDir}
DEVICE_RESTRICTION=${build_info.deviceRestriction}

SYSTEM_CFLAGS=${build_info.cflags}

SYSTEM_LDFLAG_LIST=${build_info.ldflags}


bin_PROGRAMS=proc

MAIN_SOURCES=<#list build_info.mainSourceList as source_file><#if (source_file?index > 0)>
	</#if>$(MAIN_DIR)/${source_file}<#if (source_file?index < build_info.mainSourceList?size - 1)>\</#if></#list>

APPLICATION_SOURCES=<#list build_info.taskSourceCodeList as source_file><#if (source_file?index > 0)>
	</#if>$(APPLICATION_DIR)/${source_file}<#if (source_file?index < build_info.taskSourceCodeList?size - 1)>\</#if></#list>

EXTRA_SOURCES=<#if build_info.extraSourceCodeSet??><#list build_info.extraSourceCodeSet as source_file><#if (source_file?index > 0)>
	</#if>$(APPLICATION_DIR)/${source_file}<#if (source_file?index < build_info.extraSourceCodeSet?size - 1)>\</#if></#list></#if>

API_SOURCES=<#list build_info.apiSourceList as source_file><#if (source_file?index > 0)>
	</#if>$(API_DIR)/${source_file}<#if (source_file?index < build_info.apiSourceList?size - 1)>\</#if></#list>


KERNEL_DATA_SOURCES=<#list build_info.kernelDataSourceList as source_file><#if (source_file?index > 0)>
	</#if>$(KERNEL_DIR)/generated/${source_file}<#if (source_file?index < build_info.kernelDataSourceList?size - 1)>\</#if></#list>

KERNEL_SOURCES=<#list build_info.kernelSourceList as source_file><#if (source_file?index > 0)>
	</#if>$(KERNEL_DIR)/${source_file}<#if (source_file?index < build_info.kernelSourceList?size - 1)>\</#if></#list>
			   
KERNEL_DEVICE_SOURCES=<#list build_info.kernelDeviceSourceList as source_file><#if (source_file?index > 0)>
	</#if>$(KERNEL_DIR)/$(DEVICE_RESTRICTION)/${source_file}<#if (source_file?index < build_info.kernelDeviceSourceList?size - 1)>\</#if></#list>

COMMON_SOURCES=<#list build_info.commonSourceList as source_file><#if (source_file?index > 0)>
	</#if>$(COMMON_DIR)/${source_file}<#if (source_file?index < build_info.commonSourceList?size - 1)>\</#if></#list>

MODULE_SOURCES=<#list build_info.moduleSourceList as source_file><#if (source_file?index > 0)>
	</#if>$(MODULE_DIR)/${source_file}<#if (source_file?index < build_info.moduleSourceList?size - 1)>\</#if></#list>
			
proc_SOURCES=$(MAIN_SOURCES) $(APPLICATION_SOURCES) $(EXTRA_SOURCES) $(API_SOURCES) $(KERNEL_DATA_SOURCES) $(KERNEL_SOURCES) $(KERNEL_DEVICE_SOURCES) $(COMMON_SOURCES) $(MODULE_SOURCES)
			 

MAIN_CFLAGS=-I$(MAIN_DIR)/include -I$(MAIN_DIR)/$(DEVICE_RESTRICTION)

API_CFLAGS=-I$(API_DIR)/include

KERNEL_CFLAGS=-I$(KERNEL_DIR)/include\
				-I$(KERNEL_DIR)/include/encryption\
				-I$(KERNEL_DIR)/$(DEVICE_RESTRICTION)/include<#if (build_info.usedPeripheralList?size > 0) >\
				<#list build_info.usedPeripheralList as peripheralName>-I$(KERNEL_DIR)/$(DEVICE_RESTRICTION)/include/${peripheralName}<#if (peripheralName?index < build_info.usedPeripheralList?size - 1)>\
				</#if></#list></#if>

TOP_CFLAGS=-I$(top_srcdir)

COMMON_CFLAGS=-I$(COMMON_DIR)/include\
				-I$(COMMON_DIR)/$(DEVICE_RESTRICTION)/include<#if (build_info.usedPeripheralList?size > 0) >\
				<#list build_info.usedPeripheralList as peripheralName>-I$(COMMON_DIR)/include/${peripheralName}\
				-I$(COMMON_DIR)/$(DEVICE_RESTRICTION)/include/${peripheralName}<#if (peripheralName?index < build_info.usedPeripheralList?size - 1)>\
				</#if></#list></#if>

MODULE_CFLAGS=-I$(MODULE_DIR)/include

CFLAGS_LIST=$(TOP_CFLAGS) $(MAIN_CFLAGS) $(API_CFLAGS) $(KERNEL_CFLAGS) $(COMMON_CFLAGS) $(MODULE_CFLAGS)

<#if build_info.isMappedGPU == true> 
proc_CFLAGS= $(CFLAGS_LIST) $(SYSTEM_CFLAGS)
<#else>
proc_CFLAGS=-Wall $(CFLAGS_LIST) $(SYSTEM_CFLAGS)
	<#if build_info.language=="C++">
proc_CXXFLAGS=-Wall $(CFLAGS_LIST) $(SYSTEM_CFLAGS)
	</#if>
</#if>

MAIN_LDFLAG_LIST=

API_LDFLAG_LIST=

KERNEL_LDFLAG_LIST=

COMMON_LDFLAG_LIST=$(SYSTEM_LDFLAG_LIST)

proc_LDFLAGS=$(MAIN_LDFLAG_LIST) $(API_LDFLAG_LIST) $(KERNEL_LDFLAG_LIST) $(COMMON_LDFLAG_LIST)

<#if build_info.isMappedGPU == true>
.cu.o: 
	$(CC) $(CFLAGS_LIST) $(CUDA_CFLAGS) -dc -o $@ $<
</#if>


