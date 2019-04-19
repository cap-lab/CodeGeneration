package org.snu.cse.cap.translator;

import java.io.File;

public class Constants {
	
	public static final String UEMXML_ALGORITHM_PREFIX = "_algorithm.xml";
	public static final String UEMXML_ARCHITECTURE_PREFIX = "_architecture.xml";
	public static final String UEMXML_MAPPING_PREFIX = "_mapping.xml";
	public static final String UEMXML_CONFIGURATION_PREFIX = "_configuration.xml";
	public static final String UEMXML_CONTROL_PREFIX = "_control.xml";
	public static final String UEMXML_PROFILE_PREFIX = "_profile.xml";
	public static final String UEMXML_SCHEDULE_PREFIX = ",schedule.xml";
	public static final String UEMXML_GPUSETUP_PREFIX = "_gpusetup.xml";
	
	public static final String TOP_TASKGRAPH_NAME = "top";
	public static final String XML_PREFIX = ".xml";
	
	public static final String XML_YES = "Yes";
	public static final String XML_NO = "No";
	
	public static final String SCHEDULE_FOLDER_NAME = "convertedSDF3xml";
	
	public static final String SCHEDULE_FILE_SPLITER = ",";
	
	public static final int INVALID_ID_VALUE = -1;
	public static final int INVALID_VALUE = -1;
	
	public static final String NAME_SPLITER = "/";
	public static final String TEMPLATE_PATH_SEPARATOR = "/";
	
	public static final String DEFAULT_PROPERTIES_FILE_NAME = "translator.properties";
	public static final String DEFAULT_MODULE_XML_FILE_NAME = "module.xml";
	public static final String DEFAULT_PROPERTIES_FILE_PATH = ".." + File.separator + "UEMTranslator" + File.separator + "config" + File.separator + DEFAULT_PROPERTIES_FILE_NAME;
	public static final String DEFAULT_MODULE_XML_PATH = ".." + File.separator + "UEMTranslator" + File.separator + "config" + File.separator + DEFAULT_MODULE_XML_FILE_NAME;
	public static final String DEFAULT_TEMPLATE_DIR = "templates";
	public static final String DEFAULT_TRANSLATED_CODE_TEMPLATE_DIR = ".." + File.separator + "UEMLibraryCode";
	public static final String DEFAULT_MAKEFILE_AM = "Makefile.am";
	public static final String DEFAULT_MAKEFILE = "Makefile";
	
	// template files
	public static final String TEMPLATE_FILE_TASK_CODE = "task_code.ftl";
	public static final String TEMPLATE_FILE_MAKEFILE = "Makefile.ftl";
	public static final String TEMPLATE_FILE_LIBRARY_CODE = "library_code.ftl";
	public static final String TEMPLATE_FILE_LIBRARY_HEADER = "library_header.ftl";
	public static final String TEMPLATE_FILE_EXTENSION = ".ftl";
	
	// tags for uem_data.ftl
	public static final String TEMPLATE_TAG_TASK_MAP = "flat_task";
	public static final String TEMPLATE_TAG_TASK_GRAPH = "task_graph";
	public static final String TEMPLATE_TAG_CHANNEL_LIST = "channel_list";
	public static final String TEMPLATE_TAG_DEVICE_INFO = "device_info";
	public static final String TEMPLATE_TAG_MAPPING_INFO = "mapping_info";
	public static final String TEMPLATE_TAG_STATIC_SCHEDULE_INFO = "schedule_info";
	public static final String TEMPLATE_TAG_PORT_INFO = "port_info";
	public static final String TEMPLATE_TAG_PORT_KEY_TO_INDEX = "port_key_to_index";
	public static final String TEMPLATE_TAG_EXECUTION_TIME = "execution_time";
	public static final String TEMPLATE_TAG_LIBRARY_INFO = "library_info";
	public static final String TEMPLATE_TAG_DEVICE_CONSTRAINED_INFO = "device_constrained_info";
	public static final String TEMPLATE_TAG_USED_COMMUNICATION_LIST = "used_communication_list";
	
	public static final String TEMPLATE_TAG_GPU_USED = "gpu_used";
	public static final String TEMPLATE_TAG_COMMUNICATION_USED = "communication_used";
	public static final String TEMPLATE_TAG_TCP_SERVER_LIST = "tcp_server_list";
	public static final String TEMPLATE_TAG_TCP_CLIENT_LIST = "tcp_client_list";
	public static final String TEMPLATE_TAG_MODULE_LIST = "module_list";
	public static final String TEMPLATE_TAG_BLUETOOTH_MASTER_LIST = "bluetooth_master_list";
	public static final String TEMPLATE_TAG_BLUETOOTH_SLAVE_LIST = "bluetooth_slave_list";
	public static final String TEMPLATE_TAG_SERIAL_MASTER_LIST = "serial_master_list";
	public static final String TEMPLATE_TAG_SERIAL_SLAVE_LIST = "serial_slave_list";
	
	// tags for Makefile.ftl
	public static final String TEMPLATE_TAG_BUILD_INFO = "build_info";
	public static final String TEMPLATE_TAG_ENVIRONMENT_VARIABLE_INFO = "env_var_info";
	public static final String TEMPLATE_TAG_DEVICE_ARCHITECTURE_INFO = "device_architecture_info";
	
	// tags for task_code.ftl
	public static final String TEMPLATE_TAG_TASK_INFO = "task_info";
	public static final String TEMPLATE_TAG_TASK_FUNC_ID = "task_func_id";
	public static final String TEMPLATE_TAG_TASK_GPU_MAPPING_INFO = "task_gpu_mapping_info";
	
	// tags for library_code.ftl/library_header.ftl
	public static final String TEMPLATE_TAG_LIB_INFO = "lib_info";
	
	public static final String COMMANDLINE_OPTION_HELP = "help";
	public static final String COMMANDLINE_OPTION_TEMPLATE_DIR = "template-dir";
	
	public static final String C_FILE_EXTENSION = ".c";
	public static final String CPP_FILE_EXTENSION = ".cpp";
	public static final String CUDA_FILE_EXTENSION = ".cu";
	public static final String HEADER_FILE_EXTENSION = ".h";
	public static final String CIC_HEADER_FILE_EXTENSION = ".cicl.h";
	public static final String CIC_FILE_EXTENSION = ".cic";
	public static final String CICL_FILE_EXTENSION = ".cicl";
	
	public static final String TASK_NAME_FUNC_ID_SEPARATOR = "_";
	
	public static final String DEFAULT_MODE_NAME = "Default";
	
	public static final String FLAG_SEPARATOR = " ";
	
	public static final String FILE_EXTENSION_SEPARATOR = ".";
}
