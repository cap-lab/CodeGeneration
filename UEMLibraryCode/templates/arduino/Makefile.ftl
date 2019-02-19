
<#list env_var_info as envVar>
${envVar.name}=${envVar.value}
<#if envVar.name = "BOARD_TAG">
<#assign board_tag= envVar.value>
</#if>
</#list>

<#if (used_communication_list?size > 0) && board_tag != "OpenCR">
ARDUINO_LIBS+=SoftwareSerial
</#if>

BOARD_TAG ?= uno

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

<#assign printed=false />
MAIN_C_SOURCES=<#list build_info.mainSourceList as source_file><#if source_file?ends_with(".c") ><#if (printed == true)>
	</#if><#assign printed=true />$(MAIN_DIR)/$(PLATFORM_DIR)/${source_file}<#if (source_file?index < build_info.mainSourceList?size - 1)>\</#if></#if></#list>

<#assign printed=false />
MAIN_CPP_SOURCES=<#list build_info.mainSourceList as source_file><#if source_file?ends_with(".cpp") ><#if (printed == true)>
	</#if><#assign printed=true />$(MAIN_DIR)/$(PLATFORM_DIR)/${source_file}<#if (source_file?index < build_info.mainSourceList?size - 1)>\</#if></#if></#list>

<#assign printed=false />
APPLICATION_C_SOURCES=<#list build_info.taskSourceCodeList as source_file><#if source_file?ends_with(".c") ><#if (printed == true)>
	</#if><#assign printed=true />$(APPLICATION_DIR)/${source_file}<#if (source_file?index < build_info.taskSourceCodeList?size - 1)>\</#if></#if></#list>

<#assign printed=false />
APPLICATION_CPP_SOURCES=<#list build_info.taskSourceCodeList as source_file><#if source_file?ends_with(".cpp") ><#if (printed == true)>
	</#if><#assign printed=true />$(APPLICATION_DIR)/${source_file}<#if (source_file?index < build_info.taskSourceCodeList?size - 1)>\</#if></#if></#list>

<#assign printed=false />
EXTRA_C_SOURCES=<#if build_info.extraSourceCodeSet??><#list build_info.extraSourceCodeSet as source_file><#if source_file?ends_with(".c") ><#if (printed == true)>
	</#if><#assign printed=true />$(APPLICATION_DIR)/${source_file}<#if (source_file?index < build_info.extraSourceCodeSet?size - 1)>\</#if></#if></#list></#if>

<#assign printed=false />
EXTRA_CPP_SOURCES=<#if build_info.extraSourceCodeSet??><#list build_info.extraSourceCodeSet as source_file><#if source_file?ends_with(".cpp") ><#if (printed == true)>
	</#if><#assign printed=true />$(APPLICATION_DIR)/${source_file}<#if (source_file?index < build_info.extraSourceCodeSet?size - 1)>\</#if></#if></#list></#if>
			
<#assign printed=false />
API_C_SOURCES=<#list build_info.apiSourceList as source_file><#if source_file?ends_with(".c") ><#if (printed == true)>
	</#if><#assign printed=true />$(API_DIR)/${source_file}<#if (source_file?index < build_info.apiSourceList?size - 1)>\</#if></#if></#list>

<#assign printed=false />
API_CPP_SOURCES=<#list build_info.apiSourceList as source_file><#if source_file?ends_with(".cpp") ><#if (printed == true)>
	</#if><#assign printed=true />$(API_DIR)/${source_file}<#if (source_file?index < build_info.apiSourceList?size - 1)>\</#if></#if></#list>

<#assign printed=false />
KERNEL_DATA_C_SOURCES=<#list build_info.kernelDataSourceList as source_file><#if source_file?ends_with(".c") ><#if (printed == true)>
	</#if><#assign printed=true />$(KERNEL_DIR)/generated/${source_file}<#if (source_file?index < build_info.kernelDataSourceList?size - 1)>\</#if></#if></#list>
	
<#assign printed=false />
KERNEL_DATA_CPP_SOURCES=<#list build_info.kernelDataSourceList as source_file><#if source_file?ends_with(".cpp") ><#if (printed == true)>
	</#if><#assign printed=true />$(KERNEL_DIR)/generated/${source_file}<#if (source_file?index < build_info.kernelDataSourceList?size - 1)>\</#if></#if></#list>
		
<#assign printed=false />
KERNEL_C_SOURCES=<#list build_info.kernelSourceList as source_file><#if source_file?ends_with(".c") ><#if (printed == true)>
	</#if><#assign printed=true />$(KERNEL_DIR)/${source_file}<#if (source_file?index < build_info.kernelSourceList?size - 1)>\</#if></#if></#list>
	
<#assign printed=false />
KERNEL_CPP_SOURCES=<#list build_info.kernelSourceList as source_file><#if source_file?ends_with(".cpp") ><#if (printed == true)>
	</#if><#assign printed=true />$(KERNEL_DIR)/${source_file}<#if (source_file?index < build_info.kernelSourceList?size - 1)>\</#if></#if></#list>

<#assign printed=false />
KERNEL_DEVICE_C_SOURCES=<#list build_info.kernelDeviceSourceList as source_file><#if source_file?ends_with(".c") ><#if (printed == true)>
	</#if><#assign printed=true />$(KERNEL_DIR)/$(DEVICE_RESTRICTION)/${source_file}<#if (source_file?index < build_info.kernelDeviceSourceList?size - 1)>\</#if></#if></#list>

<#assign printed=false />
KERNEL_DEVICE_CPP_SOURCES=<#list build_info.kernelDeviceSourceList as source_file><#if source_file?ends_with(".cpp") ><#if (printed == true)>
	</#if><#assign printed=true />$(KERNEL_DIR)/$(DEVICE_RESTRICTION)/${source_file}<#if (source_file?index < build_info.kernelDeviceSourceList?size - 1)>\</#if></#if></#list>

<#assign printed=false />
COMMON_C_SOURCES=<#list build_info.commonSourceList as source_file><#if source_file?ends_with(".c") ><#if (printed == true)>
	</#if><#assign printed=true />$(COMMON_DIR)/${source_file}<#if (source_file?index < build_info.commonSourceList?size - 1)>\</#if></#if></#list>

<#assign printed=false />
COMMON_CPP_SOURCES=<#list build_info.commonSourceList as source_file><#if source_file?ends_with(".cpp") ><#if (printed == true)>
	</#if><#assign printed=true />$(COMMON_DIR)/${source_file}<#if (source_file?index < build_info.commonSourceList?size - 1)>\</#if></#if></#list>

<#assign printed=false />
MODULE_C_SOURCES=<#list build_info.moduleSourceList as source_file><#if source_file?ends_with(".c") ><#if (printed == true)>
	</#if><#assign printed=true />$(MODULE_DIR)/${source_file}<#if (source_file?index < build_info.moduleSourceList?size - 1)>\</#if></#if></#list>

<#assign printed=false />
MODULE_CPP_SOURCES=<#list build_info.moduleSourceList as source_file><#if source_file?ends_with(".cpp") ><#if (printed == true)>
	</#if><#assign printed=true />$(MODULE_DIR)/${source_file}<#if (source_file?index < build_info.moduleSourceList?size - 1)>\</#if></#if></#list>

LOCAL_C_SRCS=$(MAIN_C_SOURCES) $(APPLICATION_C_SOURCES) $(EXTRA_C_SOURCES) $(API_C_SOURCES)\
	$(KERNEL_DATA_C_SOURCES) $(KERNEL_C_SOURCES) $(KERNEL_DEVICE_C_SOURCES)\
	$(COMMON_C_SOURCES) $(MODULE_C_SOURCES)

LOCAL_CPP_SRCS=$(MAIN_CPP_SOURCES) $(APPLICATION_CPP_SOURCES) $(EXTRA_CPP_SOURCES) $(API_CPP_SOURCES)\
	$(KERNEL_DATA_CPP_SOURCES) $(KERNEL_CPP_SOURCES) $(KERNEL_DEVICE_CPP_SOURCES)\
	$(COMMON_CPP_SOURCES) $(MODULE_CPP_SOURCES)
			 

MAIN_CFLAGS=-I$(MAIN_DIR)/include

API_CFLAGS=-I$(API_DIR)/include

KERNEL_CFLAGS=-I$(KERNEL_DIR)/include\
				-I$(KERNEL_DIR)/$(DEVICE_RESTRICTION)/include<#if (build_info.usedPeripheralList?size > 0) >\
				<#list build_info.usedPeripheralList as peripheralName>-I$(KERNEL_DIR)/$(DEVICE_RESTRICTION)/include/${peripheralName}<#if (peripheralName?index < build_info.usedPeripheralList?size - 1)>\</#if></#list></#if>

TOP_CFLAGS=-I$(top_srcdir)

COMMON_CFLAGS=-I$(COMMON_DIR)/include\
				-I$(COMMON_DIR)/$(DEVICE_RESTRICTION)/include<#if (build_info.usedPeripheralList?size > 0) >\
				<#list build_info.usedPeripheralList as peripheralName>-I$(COMMON_DIR)/include/${peripheralName}\
				-I$(COMMON_DIR)/$(DEVICE_RESTRICTION)/include/${peripheralName}<#if (peripheralName?index < build_info.usedPeripheralList?size - 1)>\</#if></#list></#if>


MODULE_CFLAGS=-I$(MODULE_DIR)/include

CFLAGS_LIST=$(TOP_CFLAGS) $(MAIN_CFLAGS) $(API_CFLAGS) $(KERNEL_CFLAGS) $(COMMON_CFLAGS) $(MODULE_CFLAGS)

CFLAGS+=-Wall $(CFLAGS_LIST) $(SYSTEM_CFLAGS)
CXXFLAGS+=-Wall $(CFLAGS_LIST) $(SYSTEM_CFLAGS)


MAIN_LDFLAG_LIST=

API_LDFLAG_LIST=

KERNEL_LDFLAG_LIST=

COMMON_LDFLAG_LIST=$(SYSTEM_LDFLAG_LIST)

LDFLAGS+=$(MAIN_LDFLAG_LIST) $(API_LDFLAG_LIST) $(KERNEL_LDFLAG_LIST) $(COMMON_LDFLAG_LIST)


<#if  board_tag == "OpenCR"> 
include OpenCR.mk
<#else> 
include Arduino.mk
</#if>