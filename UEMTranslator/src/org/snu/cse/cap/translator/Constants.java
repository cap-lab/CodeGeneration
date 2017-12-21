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
	
	public static final String TOP_TASKGRAPH_NAME = "top";
	public static final String XML_PREFIX = ".xml";
	
	public static final String XML_YES = "Yes";
	public static final String XML_NO = "No";
	
	public static final String SCHEDULE_FOLDER_NAME = "convertedSDF3xml";
	
	public static final String SCHEDULE_FILE_SPLITER = ",";
	
	public static final int INVALID_ID_VALUE = -1;
	
	public static final String NAME_SPLITER = "/";
	
	public static final String DEFAULT_PROPERTIES_FILE_NAME = "translator.properties";
	public static final String DEFAULT_PROPERTIES_FILE_PATH = "config" + File.separator + DEFAULT_PROPERTIES_FILE_NAME;
	public static final String DEFAULT_TEMPLATE_DIR = "templates";
	public static final String DEFAULT_TRANSLATED_CODE_TEMPLATE_DIR = ".." + File.separator + "UEMTranslatedCodeTemplate";
	
	// template files
	public static final String TEMPLATE_FILE_UEM_DATA = "uem_data.ftl";
	public static final String TEMPLATE_FILE_MAKEFILE = "Makefile.ftl";
	public static final String TEMPLATE_FILE_TASK_CODE = "task_code.ftl";
	public static final String TEMPLATE_FILE_LIBRARY_CODE = "library_code.ftl";
	public static final String TEMPLATE_FILE_LIBRARY_HEADER = "library_header.ftl";
	
	// tags for uem_data.ftl
	public static final String TEMPLATE_TAG_TASK_MAP = "flat_task";
	public static final String TEMPLATE_TAG_TASK_GRAPH = "task_graph";
	public static final String TEMPLATE_TAG_CHANNEL_LIST = "channel_list";
	public static final String TEMPLATE_TAG_DEVICE_INFO = "device_info";
	public static final String TEMPLATE_TAG_MAPPING_INFO = "mapping_info";
	public static final String TEMPLATE_TAG_STATIC_SCHEDULE_INFO = "schedule_info";
	public static final String TEMPLATE_TAG_PORT_INFO = "port_info";
	public static final String TEMPLATE_TAG_PORT_KEY_TO_INDEX = "port_key_to_index";
	
	// tags for Makefile.ftl
	public static final String TEMPLATE_TAG_BUILD_INFO = "build_info";
	
	// tags for task_code.ftl
	public static final String TEMPLATE_TAG_TASK_INFO = "task_info";
	
	// tags for library_code.ftl/library_header.ftl
	public static final String TEMPLATE_TAG_LIB_INFO = "lib_info";
	
	public static final String COMMANDLINE_OPTION_HELP = "help";
	public static final String COMMANDLINE_OPTION_TEMPLATE_DIR = "template-dir";
}
