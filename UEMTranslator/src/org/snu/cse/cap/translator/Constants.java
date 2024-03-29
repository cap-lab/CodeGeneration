package org.snu.cse.cap.translator;

import java.io.File;
import java.util.function.Supplier;

import hopes.cic.exception.CICXMLException;
import hopes.cic.xml.*;

public class Constants {

	public static final String UEMXML_SCHEDULE_PREFIX = ",schedule.xml";
	
	public static final String TOP_TASKGRAPH_NAME = "top";
	public static final String XML_PREFIX = ".xml";
	public static final String DEFAULT_STRING_NAME = "Default";
	
	public static final String XML_YES = "Yes";
	public static final String XML_NO = "No";
	
	public static final String SCHEDULE_FOLDER_NAME = "convertedSDF3xml";
	
	public static final String SCHEDULE_FILE_SPLITER = ",";
	
	public static final int INVALID_ID_VALUE = -1;
	public static final int INVALID_VALUE = -1;
	
	public static final String NAME_SPLITER = "/";
	public static final String TEMPLATE_PATH_SEPARATOR = "/";
	
	public static final String DEFAULT_PROPERTIES_FILE_NAME = "translator.properties";
	public static final String DEFAULT_PROPERTIES_FILE_PATH = "config" + File.separator + DEFAULT_PROPERTIES_FILE_NAME;
	public static final String DEFAULT_MODULE_XML_PATH = "config" + File.separator + "module.xml";
	public static final String DEFAULT_TEMPLATE_DIR = "templates";
	public static final String DEFAULT_TRANSLATED_CODE_TEMPLATE_DIR = ".." + File.separator + "UEMLibraryCode";
	public static final String DEFAULT_DOXYFILE = "Doxyfile";
	public static final String DEFAULT_DOXYGEN_MANUAL = "uem_manual";
	
	// template files
	public static final String TEMPLATE_FILE_TASK_CODE = "task_code.ftl";
	public static final String TEMPLATE_FILE_DOXYFILE= "Doxyfile.ftl";
	public static final String TEMPLATE_FILE_LIBRARY_CODE = "library_code.ftl";
	public static final String TEMPLATE_FILE_LIBRARY_HEADER = "library_header.ftl";
	public static final String TEMPLATE_FILE_EXTENSION = ".ftl";
	
	// tags for uem_data.ftl
	public static final String TEMPLATE_TAG_TASK_MAP = "flat_task";
	public static final String TEMPLATE_TAG_TASK_GRAPH = "task_graph";
	public static final String TEMPLATE_TAG_CHANNEL_LIST = "channel_list";
	public static final String TEMPLATE_TAG_MULTICAST_GROUP_LIST = "multicast_group_list";
	public static final String TEMPLATE_TAG_DEVICE_INFO = "device_info";
	public static final String TEMPLATE_TAG_DEVICE_ID = "device_id";
	public static final String TEMPLATE_TAG_ENCRYPTION_INDEX_INFO = "encryption_index";
	public static final String TEMPLATE_TAG_DEVICE_SCHEDULER = "device_scheduler";
	
	public static final String TEMPLATE_TAG_MAPPING_INFO = "mapping_info";
	public static final String TEMPLATE_TAG_STATIC_SCHEDULE_INFO = "schedule_info";
	public static final String TEMPLATE_TAG_PORT_INFO = "port_info";
	public static final String TEMPLATE_TAG_PORT_KEY_TO_INDEX = "port_key_to_index";
	public static final String TEMPLATE_TAG_EXECUTION_TIME = "execution_time";
	public static final String TEMPLATE_TAG_LIBRARY_INFO = "library_info";
	public static final String TEMPLATE_TAG_DEVICE_CONSTRAINED_INFO = "device_constrained_info";
	public static final String TEMPLATE_TAG_USED_COMMUNICATION_LIST = "used_communication_list";
	public static final String TEMPLATE_TAG_USED_ENCRYPTION_LIST = "used_encryption_list";
	public static final String TEMPLATE_TAG_SUPPORTED_COMMUNICATION_LIST = "supported_communication_list";
	public static final String TEMPLATE_TAG_PLATFORM = "platform";
	
	public static final String TEMPLATE_TAG_GPU_USED = "gpu_used";
	public static final String TEMPLATE_TAG_COMMUNICATION_USED = "communication_used";
	public static final String TEMPLATE_TAG_ENCRYPTION_USED = "encryption_used";
	public static final String TEMPLATE_TAG_ENCRYPTION_LIST = "encryption_list";
	public static final String TEMPLATE_TAG_TCP_SERVER_LIST = "tcp_server_list";
	public static final String TEMPLATE_TAG_TCP_CLIENT_LIST = "tcp_client_list";
	public static final String TEMPLATE_TAG_UDP_LIST = "udp_list";
	public static final String TEMPLATE_TAG_MODULE_LIST = "module_list";
	public static final String TEMPLATE_TAG_BLUETOOTH_MASTER_LIST = "bluetooth_master_list";
	public static final String TEMPLATE_TAG_BLUETOOTH_SLAVE_LIST = "bluetooth_slave_list";
	public static final String TEMPLATE_TAG_SERIAL_MASTER_LIST = "serial_master_list";
	public static final String TEMPLATE_TAG_SERIAL_SLAVE_LIST = "serial_slave_list";
	public static final String TEMPLATE_TAG_SECURE_TCP_SERVER_LIST = "secure_tcp_server_list";
	public static final String TEMPLATE_TAG_SECURE_TCP_CLIENT_LIST = "secure_tcp_client_list";
	public static final String TEMPLATE_TAG_SSL_KEY_INFO_LIST = "ssl_key_info_list";

	// tags for Makefile.ftl and Doxyfile.ftl
	public static final String TEMPLATE_TAG_BUILD_INFO = "build_info";
	public static final String TEMPLATE_TAG_ENVIRONMENT_VARIABLE_INFO = "env_var_info";
	public static final String TEMPLATE_TAG_DEVICE_ARCHITECTURE_INFO = "device_architecture_info";
	
	// tags for uem_manual.ftl
	public static final String TEMPLATE_TAG_MANUAL_DEVICE_INFO = TEMPLATE_TAG_DEVICE_INFO;
	public static final String TEMPLATE_TAG_MANUAL_TASK_GRAPH = TEMPLATE_TAG_TASK_GRAPH;
	public static final String TEMPLATE_TAG_MANUAL_LIBRARY_MAP = "library_map";
	public static final String TEMPLATE_TAG_MANUAL_CHANNEL_LIST = TEMPLATE_TAG_CHANNEL_LIST;
	public static final String TEMPLATE_TAG_MANUAL_TASK_MAP = TEMPLATE_TAG_TASK_MAP;
	public static final String TEMPLATE_TAG_MANUAL_DEVICE_MAP = "device_map";
	public static final String TEMPLATE_TAG_MANUAL_DEVICE_CONNECTION_MAP = "device_connection_map";
	
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

	public enum UEMXML {
		ALGORITHM("_algorithm.xml", CICAlgorithmTypeLoader::new),
		ARCHITECTURE("_architecture.xml", CICArchitectureTypeLoader::new),
		MAPPING("_mapping.xml", CICMappingTypeLoader::new),
		CONFIGURATION("_configuration.xml", CICConfigurationTypeLoader::new),
		CONTROL("_control.xml", CICControlTypeLoader::new), PROFILE("_profile.xml", CICProfileTypeLoader::new),
		GPUSETUP("_gpusetup.xml", CICGPUSetupTypeLoader::new);

		public String prefix;
		private Supplier<ResourceLoader<?>> loader;

		private UEMXML(String prefix, Supplier<ResourceLoader<?>> loader) {
			this.prefix = prefix;
			this.loader = loader;
		}

		@SuppressWarnings("unchecked")
		public <T> T load(String basePath) throws CICXMLException {
			if (!fileExists(basePath)) {
				return null;
			}
			return (T) loader.get().loadResource(basePath + prefix);
		}

		public boolean fileExists(String basePath) {
			return new File(basePath + prefix).isFile();
		}
	}
}
