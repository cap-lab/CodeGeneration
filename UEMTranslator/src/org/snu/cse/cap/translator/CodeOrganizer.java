package org.snu.cse.cap.translator;

import java.io.File;
import java.io.FileFilter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;
import static java.nio.file.StandardCopyOption.*;

import org.snu.cse.cap.translator.structure.library.Library;
import org.snu.cse.cap.translator.structure.task.Task;

import freemarker.template.Template;

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
	
	public static final String MAIN_DIR = "src" + File.separator + "main";
	public static final String API_DIR = "src" + File.separator + "api";
	public static final String KERNEL_DIR = "src" + File.separator + "kernel";
	public static final String COMMON_DIR = "src" + File.separator + "common";
	public static final String APPLICATION_DIR = "src" + File.separator + "application";
	
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
		this.cflags = "";
		this.ldadd = "";
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
		if(translatorProperties.getProperty(propertyKey) != null)
		{
			this.cflags = translatorProperties.getProperty(propertyKey);
		}
		propertyKey = TranslatorProperties.PROPERTIES_CFLAGS + TranslatorProperties.PROPERTY_DELIMITER + this.platform + 
					TranslatorProperties.PROPERTY_DELIMITER + this.architecture;
		if(translatorProperties.getProperty(propertyKey) != null)
		{
			this.cflags = this.cflags + " " + translatorProperties.getProperty(propertyKey);
		}
		
		propertyKey = TranslatorProperties.PROPERTIES_LDADD + TranslatorProperties.PROPERTY_DELIMITER + this.platform;
		if(translatorProperties.getProperty(propertyKey) != null)
		{
			this.ldadd = translatorProperties.getProperty(propertyKey);
		}
		propertyKey = TranslatorProperties.PROPERTIES_LDADD + TranslatorProperties.PROPERTY_DELIMITER + this.platform + 
					TranslatorProperties.PROPERTY_DELIMITER + this.architecture;
		if(translatorProperties.getProperty(propertyKey) != null)
		{
			this.ldadd = this.ldadd + " " + translatorProperties.getProperty(propertyKey);
		}
		
		this.platformDir = this.runtime + MAKEFILE_PATH_SEPARATOR + this.platform;
	}
	
	public void fillSourceCodeListFromTaskMap(HashMap<String, Task> taskMap)
	{
		for(Task task : taskMap.values())
		{
			if(task.getChildTaskGraphName() == null)
			{
				for(int i = 0 ; i < task.getTaskFuncNum() ; i++)
				{
					this.taskSourceCodeList.add(task.getName() + i);
				}
			}
		}
	}
	
	public void fillSourceCodeListFromLibraryMap(HashMap<String, Library> libraryMap)
	{
		for(Library library : libraryMap.values())
		{
			this.taskSourceCodeList.add(library.getName());
		}
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
					paramFile.getName().endsWith(Constants.HEADER_FILE_EXTENSION)))
					return true;
				else
					return false;
			}
		};
		
		copyAllFiles(output, source, filter);
	}
	
	public void copyFilesFromTranslatedCodeTemplate(String srcDir, String outputDir) throws IOException
	{
		File source = new File(srcDir);
		File output = new File(outputDir);
		FileFilter filter = new FileFilter() {
			
			@Override
			public boolean accept(File paramFile) {
				// skip object/executable/temporary/log files
				if(paramFile.getName().endsWith(".o") || paramFile.getName().endsWith(".log") || paramFile.getName().endsWith("~") || 
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

	public String getLdadd() {
		return ldadd;
	}

	public ArrayList<String> getTaskSourceCodeList() {
		return taskSourceCodeList;
	}
}

