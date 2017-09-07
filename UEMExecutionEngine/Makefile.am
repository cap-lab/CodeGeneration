
MAIN_DIR=src/main
APPLICATION_DIR=src/application
API_DIR=src/api
KERNEL_DIR=src/kernel
COMMON_DIR=src/common

PLATFORM_DIR=native/linux

bin_PROGRAMS=proc

MAIN_SOURCES=$(MAIN_DIR)/$(PLATFORM_DIR)/main.c

APPLICATION_SOURCES=$(APPLICATION_DIR)/Display.c\
					$(APPLICATION_DIR)/VecMul.c\
					$(APPLICATION_DIR)/MatA.c\
					$(APPLICATION_DIR)/MatB.c

API_SOURCES=$(API_DIR)/UFControl_deprecated.c\
			$(API_DIR)/UFControl.c\
			$(API_DIR)/UFPort_deprecated.c\
			$(API_DIR)/UFPort.c\
			$(API_DIR)/UFSystem_deprecated.c\
			$(API_DIR)/UFSystem.c\
			$(API_DIR)/UFTask_deprecated.c\
			$(API_DIR)/UFTask.c\
			$(API_DIR)/UFTimer_deprecated.c\
			$(API_DIR)/UFTimer.c

KERNEL_SOURCES=$(KERNEL_DIR)/uem_data.c

COMMON_SOURCES=


proc_SOURCES=$(MAIN_SOURCES) $(APPLICATION_SOURCES) $(API_SOURCES) $(KERNEL_SOURCES) $(COMMON_SOURCES)
			 

MAIN_CFLAGS=-I$(MAIN_DIR)/include

API_CFLAGS=-I$(API_DIR)/include

KERNEL_CFLAGS=-I$(KERNEL_DIR)/include

COMMON_CFLAGS=-I$(COMMON_DIR)/include

proc_CFLAGS=$(MAIN_CFLAGS) $(API_CFLAGS) $(KERNEL_CFLAGS) $(COMMON_CFLAGS)


MAIN_LDADD=

API_LDADD=

KERNEL_LDADD=

COMMON_LDADD=

proc_LDADD=$(MAIN_LDADD) $(API_LDADD) $(KERNEL_LDADD) $(COMMON_LDADD)



