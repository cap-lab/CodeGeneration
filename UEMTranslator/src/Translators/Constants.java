package Translators;

public class Constants {
	
	public static final String UEMXML_ALGORITHM_PREFIX = "_algorithm.xml";
	public static final String UEMXML_ARCHITECTURE_PREFIX = "_architecture.xml";
	public static final String UEMXML_MAPPING_PREFIX = "_mapping.xml";
	public static final String UEMXML_CONFIGURATION_PREFIX = "_configuration.xml";
	public static final String UEMXML_CONTROL_PREFIX = "_control.xml";
	public static final String UEMXML_PROFILE_PREFIX = "_profile.xml";
	public static final String UEMXML_SCHEDULE_PREFIX = "_schedule.xml";
	
	public static final String TOP_TASKGRAPH_NAME = "top";
	public static final String XML_PREFIX = ".xml";
	
	public static final String XML_YES = "Yes";
	public static final String XML_NO = "No";
	
	public static final String SCHEDULE_FOLDER_NAME = "convertedSDF3xml";
	
	public static final String SCHEDULE_FILE_SPLITER = ",";
	
	public static final int INVALID_ID_VALUE = -1;
	
	public static final String NAME_SPLITER = "/";
	
	public enum PortDirection {
		INPUT("input"),
		OUTPUT("output"),
		;
		
		private final String value;
		
		private PortDirection(String value) {
			this.value = value;
		}
		
		@Override
		public String toString() {
			return value;
		}
		
		public static PortDirection fromValue(String value) {
			 for (PortDirection c : PortDirection.values()) {
				 if (value.equals(value)) {
					 return c;
				 }
			 }
			 throw new IllegalArgumentException(value.toString());
		}
	}
}
