package org.snu.cse.cap.translator;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Properties;

import org.snu.cse.cap.translator.structure.library.Library;
import org.snu.cse.cap.translator.structure.task.Task;

public class CodeOrganizer {
	private String architecture;
	private String platform;
	private String runtime;
	private String deviceRestriction;
	private String platformDir;
	private ArrayList<String> taskSourceCodeList;
	private ArrayList<String> commonSourceList;
	private ArrayList<String> apiSourceList;
	private ArrayList<String> mainSourceList;
	private ArrayList<String> kernelSourceList;
	private ArrayList<String> kernelDeviceSourceList;
	private String cflags;
	private String ldadd;
	
	public static final String MAKEFILE_PATH_SEPARATOR = "/";
	
	public CodeOrganizer(String architecture, String platform, String runtime) {
		this.architecture = architecture;
		this.platform = platform;
		this.runtime = runtime;
		this.commonSourceList = new ArrayList<String>();
		this.apiSourceList = new ArrayList<String>();
		this.mainSourceList = new ArrayList<String>();
		this.kernelSourceList = new ArrayList<String>();
		this.kernelDeviceSourceList = new ArrayList<String>();
		this.taskSourceCodeList = new ArrayList<String>();
	}
	
	private boolean isArchitectureAvailable(String[] architectureList) {
		
		boolean isAvailable = false;
		
		for(String architectureName: architectureList)
		{
			if(this.architecture.equals(architectureName))
			{
				isAvailable = true;
				break;
			}
		}
		
		return isAvailable;
	}
	
	private boolean isPlatformAvailable(String[] platformList) {
		
		boolean isAvailable = false;
		
		for(String platformName: platformList)
		{
			if(this.platform.equals(platformName))
			{
				isAvailable = true;
				break;
			}
		}
		
		return isAvailable;
	}
	
	private boolean isRuntimeAvailable(String[] runtimeList) {
		
		boolean isAvailable = false;
		
		for(String runtimeName: runtimeList)
		{
			if(this.runtime.equals(runtimeName))
			{
				isAvailable = true;
				break;
			}
		}
		
		return isAvailable;
	}
	
	private void makeSourceFileList(String key, Properties translatorProperties, ArrayList<String> list) {
		String sourceFileString = translatorProperties.getProperty(key);
		
		if(sourceFileString != null && sourceFileString.length() > 0)
		{
			String[] sourceFileList = sourceFileString.split(TranslatorProperties.PROPERTY_VALUE_DELIMITER);
			
			for(String sourceFile : sourceFileList)
			{
				list.add(sourceFile);
			}			
		}
	}
	
	private void makeAllSourceFileList(Properties translatorProperties)
	{
		String propertyKey;
		
		propertyKey = TranslatorProperties.PROPERTIES_API_SOURCE_FILE + TranslatorProperties.PROPERTY_DELIMITER + this.platform;
		makeSourceFileList(propertyKey, translatorProperties, this.apiSourceList);
		
		propertyKey = TranslatorProperties.PROPERTIES_COMMON_SOURCE_FILE + TranslatorProperties.PROPERTY_DELIMITER + this.platform;
		makeSourceFileList(propertyKey, translatorProperties, this.commonSourceList);
		
		propertyKey = TranslatorProperties.PROPERTIES_KERNEL_SOURCE_FILE + TranslatorProperties.PROPERTY_DELIMITER + this.platform;
		makeSourceFileList(propertyKey, translatorProperties, this.kernelSourceList);
		
		propertyKey = TranslatorProperties.PROPERTIES_MAIN_SOURCE_FILE + TranslatorProperties.PROPERTY_DELIMITER + this.platform;
		makeSourceFileList(propertyKey, translatorProperties, this.mainSourceList);
		
		propertyKey = TranslatorProperties.PROPERTIES_PLATFORM_RESTRICTION + TranslatorProperties.PROPERTY_DELIMITER + this.platform;
		if(translatorProperties.getProperty(propertyKey).equals(TranslatorProperties.PROPERTY_VALUE_UNCONSTRAINED) == true)
		{
			this.deviceRestriction = TranslatorProperties.PROPERTY_VALUE_UNCONSTRAINED;
			makeSourceFileList(TranslatorProperties.PROPERTIES_UNCONSTRAINED_SOURCE_FILE, translatorProperties, this.kernelDeviceSourceList);
		}
		else
		{
			this.deviceRestriction = TranslatorProperties.PROPERTY_VALUE_CONSTRAINED;
		}
	}
	
	public void extractDataFromProperties(Properties translatorProperties) throws UnsupportedHardwareInformation
	{
		String[] architectureList = translatorProperties.getProperty(TranslatorProperties.PROPERTIES_ARCHITECTURE_LIST).split(TranslatorProperties.PROPERTY_VALUE_DELIMITER);
		String[] platformList = translatorProperties.getProperty(TranslatorProperties.PROPERTIES_PLATFORM_LIST).split(TranslatorProperties.PROPERTY_VALUE_DELIMITER);
		String[] runtimeList = translatorProperties.getProperty(TranslatorProperties.PROPERTIES_RUNTIME_LIST).split(TranslatorProperties.PROPERTY_VALUE_DELIMITER);
		String propertyKey;
		
		if(isArchitectureAvailable(architectureList) == false || isPlatformAvailable(platformList) == false || 
				isRuntimeAvailable(runtimeList) == false)
		{
			throw new UnsupportedHardwareInformation();
		}
		
		makeAllSourceFileList(translatorProperties);
		
		propertyKey = TranslatorProperties.PROPERTIES_CFLAGS + TranslatorProperties.PROPERTY_DELIMITER + this.platform;
		this.cflags = translatorProperties.getProperty(propertyKey);
		propertyKey = TranslatorProperties.PROPERTIES_CFLAGS + TranslatorProperties.PROPERTY_DELIMITER + this.platform + 
					TranslatorProperties.PROPERTY_DELIMITER + this.architecture;
		this.cflags = this.cflags + " " + translatorProperties.getProperty(propertyKey);
		
		propertyKey = TranslatorProperties.PROPERTIES_LDADD + TranslatorProperties.PROPERTY_DELIMITER + this.platform;
		this.ldadd = translatorProperties.getProperty(propertyKey);
		propertyKey = TranslatorProperties.PROPERTIES_LDADD + TranslatorProperties.PROPERTY_DELIMITER + this.platform + 
					TranslatorProperties.PROPERTY_DELIMITER + this.architecture;
		this.ldadd = this.ldadd + " " + translatorProperties.getProperty(propertyKey);
		
		this.platformDir = this.runtime + MAKEFILE_PATH_SEPARATOR + this.platform;
	}
	
	public void fillSourceCodeListFromTaskMap(HashMap<String, Task> taskMap)
	{
		for(Task task : taskMap.values())
		{
			if(task.getChildTaskGraphName() == null)
			{
				this.taskSourceCodeList.add(task.getTaskCodeFile());
			}
		}
	}
	
	public void fillSourceCodeListFromLibraryMap(HashMap<String, Library> libraryMap)
	{
		for(Library library : libraryMap.values())
		{
			this.taskSourceCodeList.add(library.getFile());
		}
	}

	public String getPlatformDir() {
		return platformDir;
	}

	public String getDeviceRestriction() {
		return deviceRestriction;
	}

	public String getArchitecture() {
		return architecture;
	}

	public String getPlatform() {
		return platform;
	}

	public String getRuntime() {
		return runtime;
	}

	public ArrayList<String> getCommonSourceList() {
		return commonSourceList;
	}

	public ArrayList<String> getApiSourceList() {
		return apiSourceList;
	}

	public ArrayList<String> getMainSourceList() {
		return mainSourceList;
	}

	public ArrayList<String> getKernelSourceList() {
		return kernelSourceList;
	}

	public ArrayList<String> getKernelDeviceSourceList() {
		return kernelDeviceSourceList;
	}

	public String getCflags() {
		return cflags;
	}

	public String getLdadd() {
		return ldadd;
	}

	public ArrayList<String> getTaskSourceCodeList() {
		return taskSourceCodeList;
	}
}

