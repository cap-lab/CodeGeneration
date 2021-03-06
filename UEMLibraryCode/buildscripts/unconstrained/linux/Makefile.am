
MAIN_DIR=src/main

APPLICATION_DIR=src/application
API_DIR=src/api
KERNEL_DIR=src/kernel
COMMON_DIR=src/common
MODULE_DIR=src/module
PLATFORM_DIR=native/linux
DEVICE_RESTRICTION=unconstrained

SYSTEM_CFLAGS=${PTHREAD_CFLAGS}

SYSTEM_LDADD=${PTHREAD_LDADD} -lws2_32


bin_PROGRAMS=proc

MAIN_SOURCES=$(MAIN_DIR)/$(PLATFORM_DIR)/main.c

APPLICATION_SOURCES=$(APPLICATION_DIR)/MatA_0.c\
	$(APPLICATION_DIR)/MatB_0.c\
	$(APPLICATION_DIR)/VecMul_0.c\
	$(APPLICATION_DIR)/Display_0.c

EXTRA_SOURCES=

API_SOURCES=$(API_DIR)/UFControl_deprecated.c\
	$(API_DIR)/UFControl.c\
	$(API_DIR)/UFPort_deprecated.c\
	$(API_DIR)/UFPort.c\
	$(API_DIR)/UFSystem_deprecated.c\
	$(API_DIR)/UFSystem.c\
	$(API_DIR)/UFTask_deprecated.c\
	$(API_DIR)/UFTask.c\
	$(API_DIR)/UFTimer_deprecated.c\
	$(API_DIR)/UFTimer.c\
	$(API_DIR)/UFLoop.c

KERNEL_DATA_SOURCES=$(KERNEL_DIR)/generated/uem_data.c\
					$(KERNEL_DIR)/generated/uem_channel_data.c\
					$(KERNEL_DIR)/generated/uem_module_data.c

KERNEL_SOURCES=$(KERNEL_DIR)/UKChannel.c\
	$(KERNEL_DIR)/UKProcessor.c\
	$(KERNEL_DIR)/UKModeTransition.c\
	$(KERNEL_DIR)/UKLibrary.c\
	$(KERNEL_DIR)/UKTaskParameter.c\
	$(KERNEL_DIR)/UKTask.c\
	$(KERNEL_DIR)/UKTimer.c\
	$(KERNEL_DIR)/UKModule.c
			   
KERNEL_DEVICE_SOURCES=$(KERNEL_DIR)/$(DEVICE_RESTRICTION)/UKCPUTaskManager.c\
	$(KERNEL_DIR)/$(DEVICE_RESTRICTION)/UKCPUCompositeTaskManager.c\
	$(KERNEL_DIR)/$(DEVICE_RESTRICTION)/UKCPUGeneralTaskManager.c\
	$(KERNEL_DIR)/$(DEVICE_RESTRICTION)/UKCPUTaskCommon.c\
	$(KERNEL_DIR)/$(DEVICE_RESTRICTION)/UKTime.c\
	$(KERNEL_DIR)/$(DEVICE_RESTRICTION)/UKChannelMemory.c\
	$(KERNEL_DIR)/$(DEVICE_RESTRICTION)/UKSharedMemoryChannel.c\
	$(KERNEL_DIR)/$(DEVICE_RESTRICTION)/UKHostMemorySystem.c\
	$(KERNEL_DIR)/$(DEVICE_RESTRICTION)/communication/UKUEMProtocol.c\
	$(KERNEL_DIR)/$(DEVICE_RESTRICTION)/communication/UKTCPSocketChannel.c\
	$(KERNEL_DIR)/$(DEVICE_RESTRICTION)/communication/UKTCPServerManager.c\
	$(KERNEL_DIR)/$(DEVICE_RESTRICTION)/UKLoop.c

COMMON_SOURCES=$(COMMON_DIR)/$(PLATFORM_DIR)/UCThread.c\
	$(COMMON_DIR)/$(PLATFORM_DIR)/UCDynamicLinkedList.c\
	$(COMMON_DIR)/$(PLATFORM_DIR)/UCDynamicStack.c\
	$(COMMON_DIR)/$(PLATFORM_DIR)/UCThreadEvent.c\
	$(COMMON_DIR)/$(PLATFORM_DIR)/UCThreadMutex.c\
	$(COMMON_DIR)/$(PLATFORM_DIR)/UCTime.c\
	$(COMMON_DIR)/$(PLATFORM_DIR)/UCString.c\
	$(COMMON_DIR)/$(PLATFORM_DIR)/UCSystem.c\
	$(COMMON_DIR)/$(PLATFORM_DIR)/UCBasic.c\
	$(COMMON_DIR)/$(PLATFORM_DIR)/communication/UCDynamicSocket.c
	
MODULE_SOURCES=
	
proc_SOURCES=$(MAIN_SOURCES) $(APPLICATION_SOURCES) $(EXTRA_SOURCES) $(API_SOURCES) $(KERNEL_DATA_SOURCES) $(KERNEL_SOURCES) $(KERNEL_DEVICE_SOURCES) $(MODULE_SOURCES) $(COMMON_SOURCES)
			 

MAIN_CFLAGS=-I$(MAIN_DIR)/include

API_CFLAGS=-I$(API_DIR)/include

KERNEL_CFLAGS=-I$(KERNEL_DIR)/include -I$(KERNEL_DIR)/$(DEVICE_RESTRICTION)/include -I$(KERNEL_DIR)/$(DEVICE_RESTRICTION)/include/communication 

TOP_CFLAGS=-I$(top_srcdir)

COMMON_CFLAGS=-I$(COMMON_DIR)/include -I$(COMMON_DIR)/include/communication 

MODULE_CFLAGS=-I$(MODULE_DIR)/include

proc_CFLAGS=-Wall $(TOP_CFLAGS) $(MAIN_CFLAGS) $(API_CFLAGS) $(KERNEL_CFLAGS) $(COMMON_CFLAGS) $(MODULE_CFLAGS) $(SYSTEM_CFLAGS)

proc_CXXFLAGS=-Wall $(TOP_CFLAGS) $(MAIN_CFLAGS) $(API_CFLAGS) $(KERNEL_CFLAGS) $(COMMON_CFLAGS) $(MODULE_CFLAGS) $(SYSTEM_CFLAGS)

MAIN_LDADD=

API_LDADD=

KERNEL_LDADD=

COMMON_LDADD=$(SYSTEM_LDADD)

proc_LDADD=$(MAIN_LDADD) $(API_LDADD) $(KERNEL_LDADD) $(COMMON_LDADD)



