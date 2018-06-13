package org.snu.cse.cap.translator;

import java.io.File;
import java.io.FileFilter;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Properties;
import static java.nio.file.StandardCopyOption.*;

import org.snu.cse.cap.translator.structure.ProgrammingLanguage;
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
	private ArrayList<String> kernelDataSourceList;
	private HashSet<String> extraSourceCodeSet;
	private String cflags;
	private String ldflags;
	private boolean isMappedGPU;
	private boolean useCommunication;
	private HashSet<String> usedPeripheralList;
	private ProgrammingLanguage language;
	
	public static final String MAIN_DIR = "src" + File.separator + "main";
	public static final String API_DIR = "src" + File.separator + "api";
	public static final String KERNEL_DIR = "src" + File.separator + "kernel";
	public static final String KERNEL_GENERATED_DIR = "src" + File.separator + "kernel" + File.separator + "generated";
	public static final String COMMON_DIR = "src" + File.separator + "common";
	public static final String APPLICATION_DIR = "src" + File.separator + "application";
	
	public static final String GPU = "gpu";
	public static final String COMMUNICATION = "communication";
	
	public static final String MAKEFILE_PATH_SEPARATOR = "/";
	
	public CodeOrganizer(String architecture, String platform, String runtime, boolean isMappedGPU, boolean useCommunication) {
		this.architecture = architecture;
		this.platform = platform;
		this.runtime = runtime;
		this.commonSourceList = new ArrayList<String>();
		this.apiSourceList = new ArrayList<String>();
		this.mainSourceList = new ArrayList<String>();
		this.kernelSourceList = new ArrayList<String>();
		this.kernelDeviceSourceList = new ArrayList<String>();
		this.taskSourceCodeList = new ArrayList<String>();
		this.kernelDataSourceList = new ArrayList<String>();
		this.extraSourceCodeSet = new HashSet<String>();
		
		this.cflags = "";
		this.ldflags = "";
		this.isMappedGPU = isMappedGPU;
		this.useCommunication = useCommunication;
		this.usedPeripheralList = new HashSet<String>();
		this.language = ProgrammingLanguage.C;
		
		if(this.isMappedGPU == true)
		{
			this.usedPeripheralList.add(GPU);
		}
		
		if(this.useCommunication == true)
		{
			this.usedPeripheralList.add(COMMUNICATION);
		}
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
	
	private boolean isPeripheralAvailable(String[] peripheralList) {
		
		boolean isAvailable = true;
		
		for(String peripheralUsed: this.usedPeripheralList)
		{
			isAvailable = false;
			for(String peripheralName: peripheralList)
			{
				if(peripheralName.equals(peripheralUsed))
				{
					isAvailable = true;
					break;
				}		
			}
			
			if(isAvailable == false)
			{
				break;
			}
		}
		
		return isAvailable;
	}
	
	
	private void addSourceFileFromSourceString(String prefix, String sourceFileString, ArrayList<String> list)
	{
		if(sourceFileString != null && sourceFileString.length() > 0)
		{
			String[] sourceFileList = sourceFileString.split(TranslatorProperties.PROPERTY_VALUE_DELIMITER);
			
			for(String sourceFile : sourceFileList)
			{
				list.add(prefix + sourceFile);
			}
		}
	}
	
	private void makeSourceFileList(String key, Properties translatorProperties, ArrayList<String> list) {
		String sourceFileString = translatorProperties.getProperty(key);
		String peripheralKey;
		
		addSourceFileFromSourceString("", sourceFileString, list);
				
		for(String peripheralName: this.usedPeripheralList)
		{
			peripheralKey = key + TranslatorProperties.PROPERTY_DELIMITER + peripheralName;
			sourceFileString = translatorProperties.getProperty(peripheralKey);
			
			if(sourceFileString != null)
			{
				addSourceFileFromSourceString(peripheralName + MAKEFILE_PATH_SEPARATOR, sourceFileString, list);	
			}
		}
	}
	
	private void setMissingExtension(ArrayList<String> fileList, String fileExtension)
	{
		int index = 0;
		String fileName;
		
		for(index = 0; index < fileList.size() ; index++) {
			fileName = fileList.get(index);
			if(fileName.contains(Constants.FILE_EXTENSION_SEPARATOR) == false) { // set file extension
				fileList.set(index, fileName + fileExtension);
			}
			else {
				// skip file extension which is already determined				
			}
		}
	}
	
	private void setGenerateKernelData(Properties translatorProperties)
	{
		String propertyKey;
		String fileExtension;
		
		propertyKey = TranslatorProperties.PROPERTIES_GENERATED_KERNEL_DATA_FILE;
		makeSourceFileList(propertyKey, translatorProperties, this.kernelDataSourceList);
		
		if(this.isMappedGPU == true) {
			fileExtension = Constants.CUDA_FILE_EXTENSION; 
		}
		else {
			if(this.language == ProgrammingLanguage.CPP) {
				fileExtension = Constants.CPP_FILE_EXTENSION;
			}
			else {
				fileExtension = Constants.C_FILE_EXTENSION;
			}
		}
		
		setMissingExtension(this.kernelDataSourceList, fileExtension);
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
		
		setGenerateKernelData(translatorProperties);
	}
	
	private String getFlagsFromProperties(String propertyKey, Properties translatorProperties)
	{
		String peripheralKey;
		String architectureKey;
		String flag = "";
		
		if(translatorProperties.getProperty(propertyKey) != null) {
			flag = translatorProperties.getProperty(propertyKey);
		}
		
		architectureKey = propertyKey + TranslatorProperties.PROPERTY_DELIMITER + this.architecture;
		if(translatorProperties.getProperty(architectureKey) != null) {
			flag = flag + " " + translatorProperties.getProperty(architectureKey);
		}
		
		for(String peripheralName: this.usedPeripheralList)
		{
			peripheralKey = propertyKey + TranslatorProperties.PROPERTY_DELIMITER + peripheralName;
			if(translatorProperties.getProperty(peripheralKey) != null) {
				flag = flag + " " + translatorProperties.getProperty(peripheralKey);
			}	
		}
		
		return flag;
	}
	
	public void extractDataFromProperties(Properties translatorProperties) throws UnsupportedHardwareInformation
	{
		String[] architectureList = translatorProperties.getProperty(TranslatorProperties.PROPERTIES_ARCHITECTURE_LIST).split(TranslatorProperties.PROPERTY_VALUE_DELIMITER);
		String[] platformList = translatorProperties.getProperty(TranslatorProperties.PROPERTIES_PLATFORM_LIST).split(TranslatorProperties.PROPERTY_VALUE_DELIMITER);
		String[] runtimeList = translatorProperties.getProperty(TranslatorProperties.PROPERTIES_RUNTIME_LIST).split(TranslatorProperties.PROPERTY_VALUE_DELIMITER);
		String[] peripheralList = translatorProperties.getProperty(TranslatorProperties.PROPERTIES_PERIPHERAL_LIST).split(TranslatorProperties.PROPERTY_VALUE_DELIMITER);
		String propertyKey;
		
		if(isArchitectureAvailable(architectureList) == false || isPlatformAvailable(platformList) == false || 
				isRuntimeAvailable(runtimeList) == false || isPeripheralAvailable(peripheralList) == false)
		{
			throw new UnsupportedHardwareInformation();
		}
		
		makeAllSourceFileList(translatorProperties);
		
		propertyKey = TranslatorProperties.PROPERTIES_CFLAGS + TranslatorProperties.PROPERTY_DELIMITER + this.platform;
		this.cflags = this.cflags + " " + getFlagsFromProperties(propertyKey, translatorProperties);
		
		propertyKey = TranslatorProperties.PROPERTIES_LDADD + TranslatorProperties.PROPERTY_DELIMITER + this.platform;
		this.ldflags = this.ldflags + " " + getFlagsFromProperties(propertyKey, translatorProperties);
		
		this.cflags = this.cflags.trim();
		this.ldflags = this.ldflags.trim();
		
		this.platformDir = this.runtime + MAKEFILE_PATH_SEPARATOR + this.platform;
	}
	
	private void addLibraryFlags(HashMap<String, Task> taskMap, HashMap<String, Library> libraryMap)
	{
		HashSet<String> ldAddSet = new HashSet<String>();
		for(Task task: taskMap.values())
		{
			if(task.getLdFlags() != null && task.getLdFlags().trim().length() > 0)
			{			
				ldAddSet.add(task.getLdFlags().trim());
			}
		}
		
		for(Library library: libraryMap.values())
		{
			if(library.getLdFlags() != null && library.getLdFlags().trim().length() > 0)
			{
				ldAddSet.add(library.getLdFlags().trim());
			}
		}
		
		for(String ldAdd: ldAddSet)
		{
			this.ldflags = this.ldflags + " " + ldAdd;
		}
	}
	
	private void addCFlags(HashMap<String, Task> taskMap, HashMap<String, Library> libraryMap)
	{
		HashSet<String> cFlagsSet = new HashSet<String>();
		for(Task task: taskMap.values())
		{
			if(task.getcFlags() != null && task.getcFlags().trim().length() > 0)
			{
				cFlagsSet.add(task.getcFlags().trim());
			}
		}
		
		for(Library library: libraryMap.values())
		{
			if(library.getcFlags() != null && library.getcFlags().trim().length() > 0)
			{
				cFlagsSet.add(library.getcFlags().trim());
			}
		}
		
		for(String cFlag: cFlagsSet)
		{
			this.cflags = this.cflags + " " + cFlag;
		}
	}
	
	public void extraInfoFromTaskAndLibraryMap(HashMap<String, Task> taskMap, HashMap<String, Library> libraryMap)
	{
		addLibraryFlags(taskMap, libraryMap);
		addCFlags(taskMap, libraryMap);
	}
	
	private void setExtraSources(HashSet<String> extraSourceSet)
	{
		for(String extraSource : extraSourceSet)
		{
			this.extraSourceCodeSet.add(extraSource);
			if(extraSource.endsWith(Constants.CPP_FILE_EXTENSION) == true)
			{
				this.language = ProgrammingLanguage.CPP;
			}
		}
	}
	
	private void fillSourceCodeListFromTaskMap(HashMap<String, Task> taskMap)
	{
		for(Task task : taskMap.values())
		{
			if(task.getChildTaskGraphName() == null)
			{
				for(int i = 0 ; i < task.getTaskFuncNum() ; i++)
				{
					if(task.getLanguage() == ProgrammingLanguage.CPP)
					{
						this.language = ProgrammingLanguage.CPP;
					}
					
					if(isMappedGPU == false){
						this.taskSourceCodeList.add(task.getName() + Constants.TASK_NAME_FUNC_ID_SEPARATOR + i + task.getFileExtension());
	    			}
	    			else{
						this.taskSourceCodeList.add(task.getName() + Constants.TASK_NAME_FUNC_ID_SEPARATOR + i + Constants.CUDA_FILE_EXTENSION);
	    			}
				}
			}
			
			setExtraSources(task.getExtraSourceSet());
		}
	}
	
	private void fillSourceCodeListFromLibraryMap(HashMap<String, Library> libraryMap)
	{
		for(Library library : libraryMap.values())
		{
			if(library.getLanguage() == ProgrammingLanguage.CPP)
			{
				this.language = ProgrammingLanguage.CPP;
			}
			
			this.taskSourceCodeList.add(library.getName() + library.getFileExtension());
			
			setExtraSources(library.getExtraSourceSet());
		}
	}
	
	public void fillSourceCodeListFromTaskAndLibraryMap(HashMap<String, Task> taskMap, HashMap<String, Library> libraryMap)
	{
		fillSourceCodeListFromTaskMap(taskMap);
		fillSourceCodeListFromLibraryMap(libraryMap);
	}
	
	public void copyApplicationCodes(String srcDir, String outputDir) throws IOException
	{
		File source = new File(srcDir);
		File output = new File(outputDir + File.separator + APPLICATION_DIR);
		FileFilter filter = new FileFilter() {
			
			@Override
			public boolean accept(File paramFile) {
				// only copy file with extension .cic/.cicl/.h
				if(paramFile.isFile() == true && 
					(paramFile.getName().endsWith(Constants.CIC_FILE_EXTENSION) || 
					paramFile.getName().endsWith(Constants.CICL_FILE_EXTENSION) || 
					paramFile.getName().endsWith(Constants.C_FILE_EXTENSION) ||
					paramFile.getName().endsWith(Constants.CPP_FILE_EXTENSION) ||
						(paramFile.getName().endsWith(Constants.HEADER_FILE_EXTENSION) && 
						!paramFile.getName().endsWith(Constants.CIC_HEADER_FILE_EXTENSION))
					))
					return true;
				else
					return false;
			}
		};
		
		copyAllFiles(output, source, filter);
	}
	
	public void copyFilesFromLibraryCodeTemplate(String srcDir, String outputDir) throws IOException
	{
		File source = new File(srcDir);
		File output = new File(outputDir);
		FileFilter filter = new FileFilter() {
			
			@Override
			public boolean accept(File paramFile) {
				// add File.separator after the directory path to copy directory itself and avoid internal files
				if(paramFile.getAbsolutePath().contains(srcDir + File.separator + APPLICATION_DIR + File.separator))
					return false;
				else if(paramFile.getAbsolutePath().contains(srcDir + File.separator + KERNEL_GENERATED_DIR + File.separator))
					return false;
				// skip object/executable/temporary/log files
				else if(paramFile.getName().endsWith(".o") || paramFile.getName().endsWith(".log") || paramFile.getName().endsWith("~") || 
					paramFile.getName().startsWith(".")  || paramFile.getName().endsWith(".exe") || paramFile.getName().endsWith(".bak"))
					return false;
				else
					return true;
			}
		};
		
		copyAllFiles(output, source, filter);
	}
		
	private void copyAllFiles(File targetLocation, File sourceLocation, FileFilter fileFilter) throws IOException {
		if (sourceLocation.isDirectory()) {
			if (!targetLocation.exists()) 
				targetLocation.mkdir();

			File[] children = sourceLocation.listFiles(fileFilter);
			for (int i=0; i<children.length; i++)
				copyAllFiles(new File(targetLocation, children[i].getName()), children[i], fileFilter);
		} 
		else {
			// this function is supported from jdk 1.7
			Files.copy(sourceLocation.toPath(), targetLocation.toPath(), REPLACE_EXISTING);
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

	public String getLdflags() {
		return ldflags;
	}

	public ArrayList<String> getTaskSourceCodeList() {
		return taskSourceCodeList;
	}

	public HashSet<String> getExtraSourceCodeSet() {
		return extraSourceCodeSet;
	}

	public boolean getIsMappedGPU() {
		return isMappedGPU;
	}

	public HashSet<String> getUsedPeripheralList() {
		return usedPeripheralList;
	}

	public ProgrammingLanguage getLanguage() {
		return language;
	}

	public ArrayList<String> getKernelDataSourceList() {
		return kernelDataSourceList;
	}
}

