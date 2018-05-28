
MAIN_DIR=src/main

APPLICATION_DIR=src/application
API_DIR=src/api
KERNEL_DIR=src/kernel
COMMON_DIR=src/common
PLATFORM_DIR=${build_info.platformDir}
DEVICE_RESTRICTION=${build_info.deviceRestriction}

SYSTEM_CFLAGS=${build_info.cflags}

SYSTEM_LDADD=${build_info.ldadd}


bin_PROGRAMS=proc

MAIN_SOURCES=<#list build_info.mainSourceList as source_file><#if (source_file?index > 0)>
	</#if>$(MAIN_DIR)/$(PLATFORM_DIR)/${source_file}<#if (source_file?index < build_info.mainSourceList?size - 1)>\</#if></#list>

APPLICATION_SOURCES=<#list build_info.taskSourceCodeList as source_file><#if (source_file?index > 0)>
	</#if>$(APPLICATION_DIR)/${source_file}<#if (source_file?index < build_info.taskSourceCodeList?size - 1)>\</#if></#list>

EXTRA_SOURCES=<#if build_info.extraSourceCodeSet??><#list build_info.extraSourceCodeSet as source_file><#if (source_file?index > 0)>
	</#if>$(APPLICATION_DIR)/${source_file}<#if (source_file?index < build_info.extraSourceCodeSet?size - 1)>\</#if></#list></#if>

API_SOURCES=<#list build_info.apiSourceList as source_file><#if (source_file?index > 0)>
	</#if>$(API_DIR)/${source_file}<#if (source_file?index < build_info.apiSourceList?size - 1)>\</#if></#list>


KERNEL_DATA_SOURCES=<#list build_info.kernelDataSourceList as source_file><#if (source_file?index > 0)>
	</#if>$(KERNEL_DIR)/${source_file}<#if (source_file?index < build_info.kernelDataSourceList?size - 1)>\</#if></#list>

KERNEL_SOURCES=<#list build_info.kernelSourceList as source_file><#if (source_file?index > 0)>
	</#if>$(KERNEL_DIR)/${source_file}<#if (source_file?index < build_info.kernelSourceList?size - 1)>\</#if></#list>
			   
KERNEL_DEVICE_SOURCES=<#list build_info.kernelDeviceSourceList as source_file><#if (source_file?index > 0)>
	</#if>$(KERNEL_DIR)/$(DEVICE_RESTRICTION)/${source_file}<#if (source_file?index < build_info.kernelDeviceSourceList?size - 1)>\</#if></#list>

COMMON_SOURCES=<#list build_info.commonSourceList as source_file><#if (source_file?index > 0)>
	</#if>$(COMMON_DIR)/$(PLATFORM_DIR)/${source_file}<#if (source_file?index < build_info.commonSourceList?size - 1)>\</#if></#list>
	
proc_SOURCES=$(MAIN_SOURCES) $(APPLICATION_SOURCES) $(EXTRA_SOURCES) $(API_SOURCES) $(KERNEL_DATA_SOURCES) $(KERNEL_SOURCES) $(KERNEL_DEVICE_SOURCES) $(COMMON_SOURCES)
			 

MAIN_CFLAGS=-I$(MAIN_DIR)/include

API_CFLAGS=-I$(API_DIR)/include

KERNEL_CFLAGS=-I$(KERNEL_DIR)/include -I$(KERNEL_DIR)/$(DEVICE_RESTRICTION)/include <#list build_info.usedPeripheralList as peripheralName>-I$(KERNEL_DIR)/$(DEVICE_RESTRICTION)/include/${peripheralName}</#list>

TOP_CFLAGS=-I$(top_srcdir)

COMMON_CFLAGS=-I$(COMMON_DIR)/include <#list build_info.usedPeripheralList as peripheralName>-I$(COMMON_DIR)/include/${peripheralName}</#list>

CFLAGS_LIST=$(TOP_CFLAGS) $(MAIN_CFLAGS) $(API_CFLAGS) $(KERNEL_CFLAGS) $(COMMON_CFLAGS)

<#if build_info.isMappedGPU == true> 
proc_CFLAGS= $(CFLAGS_LIST) $(SYSTEM_CFLAGS)
<#else>
proc_CFLAGS=-Wall $(CFLAGS_LIST) $(SYSTEM_CFLAGS)
	<#if build_info.language=="C++">
proc_CXXFLAGS=-Wall $(CFLAGS_LIST) $(SYSTEM_CFLAGS)
	</#if>
</#if>

MAIN_LDADD=

API_LDADD=

KERNEL_LDADD=

COMMON_LDADD=$(SYSTEM_LDADD)

proc_LDADD=$(MAIN_LDADD) $(API_LDADD) $(KERNEL_LDADD) $(COMMON_LDADD)

<#if build_info.isMappedGPU == true>
.cu.o: 
	$(CC) $(CFLAGS_LIST) $(CUDA_CFLAGS) -dc -o $@ $<
</#if>


