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
import org.snu.cse.cap.translator.structure.device.DeviceCommunicationType;
import org.snu.cse.cap.translator.structure.device.DeviceEncryptionType;
import org.snu.cse.cap.translator.structure.device.SoftwarePlatformType;
import org.snu.cse.cap.translator.structure.library.Library;
import org.snu.cse.cap.translator.structure.module.Module;
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
	private ArrayList<String> moduleSourceList;
	private ArrayList<String> buildTemplateList;
	private HashSet<String> extraSourceCodeSet;
	private String cflags;
	private String ldflags;
	private boolean isMappedGPU;
	private boolean useCommunication;
	private boolean useEncryption;
	private HashSet<String> usedPeripheralList;
	private HashSet<DeviceCommunicationType> usedCommunicationSet;
	private HashSet<DeviceEncryptionType> usedEncryptionSet;
	private String pathSeparator;

	private ProgrammingLanguage language;

	public static final String MAIN_DIR = "src" + File.separator + "main";
	public static final String API_DIR = "src" + File.separator + "api";
	public static final String KERNEL_DIR = "src" + File.separator + "kernel";
	public static final String KERNEL_GENERATED_DIR = "src" + File.separator + "kernel" + File.separator + "generated";
	public static final String COMMON_DIR = "src" + File.separator + "common";
	public static final String APPLICATION_DIR = "src" + File.separator + "application";
	public static final String MODULE_DIR = "src" + File.separator + "module";
	public static final String TEMPLATES_DIR = "templates";
	public static final String BUILDSCRIPTS_DIR = "buildscripts";

	public static final String GPU = "gpu";
	public static final String COMMUNICATION = "communication";
	public static final String ENCRYPTION = "encryption";

	public static final String STRING_ALL = "*";
	public static final String DEVICE_RESTRICTION_DIR_VARIABLE = "$(DEVICE_RESTRICTION)";
	public static final String PLATFORM_DIR_VARIABLE = "$(PLATFORM_DIR)";

	public CodeOrganizer(String architecture, String platform, String runtime, boolean isMappedGPU,
			HashSet<DeviceCommunicationType> usedCommunicationSet, HashSet<DeviceEncryptionType> usedEncryptionSet) {
		this.architecture = architecture;
		this.platform = platform;
		this.pathSeparator = (this.platform.equals(SoftwarePlatformType.WINDOWS.toString())) ? "\\" : "/";
		this.runtime = runtime;
		this.commonSourceList = new ArrayList<String>();
		this.apiSourceList = new ArrayList<String>();
		this.mainSourceList = new ArrayList<String>();
		this.kernelSourceList = new ArrayList<String>();
		this.kernelDeviceSourceList = new ArrayList<String>();
		this.taskSourceCodeList = new ArrayList<String>();
		this.kernelDataSourceList = new ArrayList<String>();
		this.extraSourceCodeSet = new HashSet<String>();
		this.moduleSourceList = new ArrayList<String>();
		this.buildTemplateList = new ArrayList<String>();

		this.cflags = "";
		this.ldflags = "";
		this.isMappedGPU = isMappedGPU;
		if (usedCommunicationSet.size() > 0) {
			this.useCommunication = true;
		} else {
			this.useCommunication = false;
		}
		
		this.usedCommunicationSet = new HashSet<DeviceCommunicationType>();

		for (DeviceCommunicationType communicationType : usedCommunicationSet) {
			this.usedCommunicationSet.add(communicationType);
		}
		
		if (usedEncryptionSet.size() > 0) {
			this.useEncryption = true;
		} else {
			this.useEncryption = false;
		}

		this.usedEncryptionSet = new HashSet<DeviceEncryptionType>();

		for (DeviceEncryptionType encryptionType : usedEncryptionSet) {
			this.usedEncryptionSet.add(encryptionType);
		}
		
		this.usedPeripheralList = new HashSet<String>();
		this.language = ProgrammingLanguage.C;

		if (this.isMappedGPU == true) {
			this.usedPeripheralList.add(GPU);
		}

		if (this.useCommunication == true) {
			this.usedPeripheralList.add(COMMUNICATION);
		}
		
		if (this.useEncryption == true) {
			this.usedPeripheralList.add(ENCRYPTION);
		}

	}

	private boolean isArchitectureAvailable(String[] architectureList) {

		boolean isAvailable = false;

		for (String architectureName : architectureList) {
			if (this.architecture.equals(architectureName)) {
				isAvailable = true;
				break;
			}
		}

		return isAvailable;
	}

	private boolean isPlatformAvailable(String[] platformList) {

		boolean isAvailable = false;

		for (String platformName : platformList) {
			if (this.platform.equals(platformName)) {
				isAvailable = true;
				break;
			}
		}

		return isAvailable;
	}

	private boolean isRuntimeAvailable(String[] runtimeList) {

		boolean isAvailable = false;

		for (String runtimeName : runtimeList) {
			if (this.runtime.equals(runtimeName)) {
				isAvailable = true;
				break;
			}
		}

		return isAvailable;
	}

	private boolean isCommunicationAvailable(Properties translatorProperties) {
		boolean isAvailable = true;
		String propertyKey = COMMUNICATION + TranslatorProperties.PROPERTY_DELIMITER
				+ TranslatorProperties.PROPERTIES_PERIPHERAL_SUBTYPE;
		String[] peripheralSubTypeList;
		String peripheralSubTypeString = translatorProperties.getProperty(propertyKey);

		if (peripheralSubTypeString == null) {
			return true;
		}

		peripheralSubTypeList = peripheralSubTypeString.split(TranslatorProperties.PROPERTY_VALUE_DELIMITER);

		for (DeviceCommunicationType communicationType : this.usedCommunicationSet) {
			isAvailable = false;
			for (String subType : peripheralSubTypeList) {
				if (subType.equals(communicationType.toString())) {
					isAvailable = true;
					break;
				}
			}

			if (isAvailable == false) {
				break;
			}
		}

		return isAvailable;
	}

	private boolean isEncryptionAvailable(Properties translatorProperties) {
		boolean isAvailable = true;
		String propertyKey = ENCRYPTION + TranslatorProperties.PROPERTY_DELIMITER
				+ TranslatorProperties.PROPERTIES_PERIPHERAL_SUBTYPE;
		String[] peripheralSubTypeList;
		String peripheralSubTypeString = translatorProperties.getProperty(propertyKey);

		if (peripheralSubTypeString == null) {
			return true;
		}

		peripheralSubTypeList = peripheralSubTypeString.split(TranslatorProperties.PROPERTY_VALUE_DELIMITER);

		for (DeviceEncryptionType encryptionType : this.usedEncryptionSet) {
			isAvailable = false;
			for (String subType : peripheralSubTypeList) {
				if (subType.equals(encryptionType.toString())) {
					isAvailable = true;
					break;
				}
			}

			if (isAvailable == false) {
				break;
			}
		}

		return isAvailable;
	}

	private boolean isPeripheralAvailable(String[] peripheralList, Properties translatorProperties) {
		boolean isAvailable = true;

		for (String peripheralUsed : this.usedPeripheralList) {
			isAvailable = false;
			for (String peripheralName : peripheralList) {
				if (peripheralName.equals(peripheralUsed)) {
					if (peripheralName.equals(COMMUNICATION)) {
						isAvailable = isCommunicationAvailable(translatorProperties);
					} else if (peripheralName.equals(ENCRYPTION)) {
						isAvailable = isEncryptionAvailable(translatorProperties);
					} else {
						isAvailable = true;
					}
					break;
				}
			}

			if (isAvailable == false) {
				break;
			}
		}

		return isAvailable;
	}

	private void makeHashSetFromString(String prefix, String sourceFileString, HashSet<String> set) {
		if (sourceFileString != null && sourceFileString.length() > 0) {
			String[] sourceFileList = sourceFileString.split(TranslatorProperties.PROPERTY_VALUE_DELIMITER);

			for (String sourceFile : sourceFileList) {
				set.add(prefix + sourceFile);
			}
		}
	}

	private void addSourceFileFromSourceString(String prefix, String sourceFileString, ArrayList<String> list) {
		if (sourceFileString != null && sourceFileString.length() > 0) {
			String[] sourceFileList = sourceFileString.split(TranslatorProperties.PROPERTY_VALUE_DELIMITER);

			for (String sourceFile : sourceFileList) {
				list.add(prefix + sourceFile);
			}
		}
	}

	private void addCommunicationSourceFileFromSourceString(String prefix, String key, Properties translatorProperties,
			ArrayList<String> list) {
		String sourceFileKey;
		String sourceFileString;
		for (DeviceCommunicationType communicationType : this.usedCommunicationSet) {
			sourceFileKey = key + TranslatorProperties.PROPERTY_DELIMITER + communicationType.toString();
			sourceFileString = translatorProperties.getProperty(sourceFileKey);
			if (sourceFileString != null) {
				addSourceFileFromSourceString(prefix + communicationType.toString() + this.pathSeparator,
						sourceFileString, list);
			}
		}
	}
	
	private void addEncryptionSourceFileFromSourceString(String prefix, String key, Properties translatorProperties,
			ArrayList<String> list) {
		String sourceFileKey;
		String sourceFileString;

		for (DeviceEncryptionType encryptionType : this.usedEncryptionSet) {
			sourceFileKey = key + TranslatorProperties.PROPERTY_DELIMITER + encryptionType.toString().toLowerCase();
			sourceFileString = translatorProperties.getProperty(sourceFileKey);

			if (sourceFileString != null) {
				addSourceFileFromSourceString(prefix + encryptionType.toString().toLowerCase() + this.pathSeparator,
						sourceFileString, list);
			}
		}
	}

	private void makeSourceFileList(String key, String prefix, Properties translatorProperties,
			ArrayList<String> list) {
		String sourceFileString = translatorProperties.getProperty(key);
		String peripheralKey;
		
		addSourceFileFromSourceString(prefix, sourceFileString, list);

		for (String peripheralName : this.usedPeripheralList) {

			peripheralKey = key + TranslatorProperties.PROPERTY_DELIMITER + peripheralName;

			sourceFileString = translatorProperties.getProperty(peripheralKey);

			if (sourceFileString != null) {
				addSourceFileFromSourceString(prefix + peripheralName + this.pathSeparator, sourceFileString,
						list);
				if (peripheralName.equals(COMMUNICATION)) {
					addCommunicationSourceFileFromSourceString(prefix + peripheralName + this.pathSeparator,
							peripheralKey, translatorProperties, list);
				}
			}

			if (sourceFileString == null && peripheralName.equals(ENCRYPTION)) {
				addEncryptionSourceFileFromSourceString(prefix + peripheralName + this.pathSeparator,
							peripheralKey, translatorProperties, list);
			}
		}
	}

	private void makePlatformDependentSourceFileList(String key, Properties translatorProperties, ArrayList<String> list) {
		String sourceFileString = translatorProperties.getProperty(key);
		String subKey;

		// common_source_file
		makeSourceFileList(key, "", translatorProperties, list);

		// common_source_file.[device_restriction]
		subKey = key + TranslatorProperties.PROPERTY_DELIMITER + this.deviceRestriction;
		makeSourceFileList(subKey, DEVICE_RESTRICTION_DIR_VARIABLE + this.pathSeparator,
				translatorProperties, list);

		// common_source_file.[device_restriction].[platform]
		subKey = subKey + TranslatorProperties.PROPERTY_DELIMITER + this.platform;
		makeSourceFileList(subKey,
				DEVICE_RESTRICTION_DIR_VARIABLE + this.pathSeparator + PLATFORM_DIR_VARIABLE + this.pathSeparator,
				translatorProperties, list);


	}

	// API source file is treated differently because it is not target-dependent and
	// no peripheral is used in API layer
	private void makeAPISourceFileList(String key, Properties translatorProperties, ArrayList<String> list) {
		String sourceFileString = translatorProperties.getProperty(key);
		String platformKey;

		addSourceFileFromSourceString("", sourceFileString, list);

		platformKey = key + TranslatorProperties.PROPERTY_DELIMITER + this.platform;
		sourceFileString = translatorProperties.getProperty(platformKey);

		if (sourceFileString != null) {
			addSourceFileFromSourceString("", sourceFileString, list);
		}
	}

	private void setMissingExtension(ArrayList<String> fileList, String fileExtension) {
		int index = 0;
		String fileName;

		for (index = 0; index < fileList.size(); index++) {
			fileName = fileList.get(index);
			if (fileName.contains(Constants.FILE_EXTENSION_SEPARATOR) == false) { // set file extension
				fileList.set(index, fileName + fileExtension);
			} else {
				// skip file extension which is already determined
			}
		}
	}

	private void setGenerateKernelData(Properties translatorProperties) {
		String propertyKey;
		String fileExtension;

		propertyKey = TranslatorProperties.PROPERTIES_GENERATED_KERNEL_DATA_FILE;
		makeSourceFileList(propertyKey, "", translatorProperties, this.kernelDataSourceList);

		if (this.isMappedGPU == true) {
			fileExtension = Constants.CUDA_FILE_EXTENSION;
		} else {
			if (this.language == ProgrammingLanguage.CPP
					|| SoftwarePlatformType.fromValue(this.platform) == SoftwarePlatformType.ARDUINO) {
				fileExtension = Constants.CPP_FILE_EXTENSION;
			} else {
				fileExtension = Constants.C_FILE_EXTENSION;
			}
		}

		setMissingExtension(this.kernelDataSourceList, fileExtension);
	}

	private void makeAllSourceFileList(Properties translatorProperties) {
		String propertyKey;

		propertyKey = TranslatorProperties.PROPERTIES_API_SOURCE_FILE;
		makeAPISourceFileList(propertyKey, translatorProperties, this.apiSourceList);

		propertyKey = TranslatorProperties.PROPERTIES_KERNEL_SOURCE_FILE;
		makeSourceFileList(propertyKey, "", translatorProperties, this.kernelSourceList);

		propertyKey = TranslatorProperties.PROPERTIES_PLATFORM_RESTRICTION + TranslatorProperties.PROPERTY_DELIMITER
				+ this.platform;
		if (translatorProperties.getProperty(propertyKey)
				.equals(TranslatorProperties.PROPERTY_VALUE_UNCONSTRAINED) == true) {
			this.deviceRestriction = TranslatorProperties.PROPERTY_VALUE_UNCONSTRAINED;
			makeSourceFileList(TranslatorProperties.PROPERTIES_UNCONSTRAINED_SOURCE_FILE, "", translatorProperties,
					this.kernelDeviceSourceList);
		} else {
			this.deviceRestriction = TranslatorProperties.PROPERTY_VALUE_CONSTRAINED;
			makeSourceFileList(TranslatorProperties.PROPERTIES_CONSTRAINED_SOURCE_FILE, "", translatorProperties,
					this.kernelDeviceSourceList);
		}

		propertyKey = TranslatorProperties.PROPERTIES_MAIN_SOURCE_FILE;
		makePlatformDependentSourceFileList(propertyKey, translatorProperties, this.mainSourceList);

		propertyKey = TranslatorProperties.PROPERTIES_COMMON_SOURCE_FILE;
		makePlatformDependentSourceFileList(propertyKey, translatorProperties, this.commonSourceList);

		setGenerateKernelData(translatorProperties);
	}

	private String getFlagsFromProperties(String propertyKey, Properties translatorProperties) {
		String peripheralKey;
		String architectureKey;
		String communicationKey;
		String encryptionKey;
		String flag = "";

		if (translatorProperties.getProperty(propertyKey) != null) {
			flag = translatorProperties.getProperty(propertyKey);
		}

		architectureKey = propertyKey + TranslatorProperties.PROPERTY_DELIMITER + this.architecture;
		if (translatorProperties.getProperty(architectureKey) != null) {
			flag = flag + " " + translatorProperties.getProperty(architectureKey);
		}

		for (String peripheralName : this.usedPeripheralList) {
			peripheralKey = propertyKey + TranslatorProperties.PROPERTY_DELIMITER + peripheralName;
			if (translatorProperties.getProperty(peripheralKey) != null) {
				flag = flag + " " + translatorProperties.getProperty(peripheralKey);
			}

			if (peripheralName.equals(COMMUNICATION)) {
				for (DeviceCommunicationType communicationType : this.usedCommunicationSet) {
					communicationKey = peripheralKey + TranslatorProperties.PROPERTY_DELIMITER
							+ communicationType.toString();
					if (translatorProperties.getProperty(communicationKey) != null) {
						flag = flag + " " + translatorProperties.getProperty(communicationKey);
					}
				}
			}
			else if (peripheralName.equals(ENCRYPTION)) {
				for (DeviceEncryptionType encryptionType : this.usedEncryptionSet) {
					encryptionKey = peripheralKey + TranslatorProperties.PROPERTY_DELIMITER
							+ encryptionType.toString();
					if (translatorProperties.getProperty(encryptionKey) != null) {
						flag = flag + " " + translatorProperties.getProperty(encryptionKey);
					}
				}
			}
		}

		return flag;
	}

	public void extractDataFromProperties(Properties translatorProperties) throws UnsupportedHardwareInformation {
		String[] architectureList = translatorProperties.getProperty(TranslatorProperties.PROPERTIES_ARCHITECTURE_LIST)
				.split(TranslatorProperties.PROPERTY_VALUE_DELIMITER);
		String[] platformList = translatorProperties.getProperty(TranslatorProperties.PROPERTIES_PLATFORM_LIST)
				.split(TranslatorProperties.PROPERTY_VALUE_DELIMITER);
		String[] runtimeList = translatorProperties.getProperty(TranslatorProperties.PROPERTIES_RUNTIME_LIST)
				.split(TranslatorProperties.PROPERTY_VALUE_DELIMITER);
		String[] peripheralList = translatorProperties.getProperty(TranslatorProperties.PROPERTIES_PERIPHERAL_LIST)
				.split(TranslatorProperties.PROPERTY_VALUE_DELIMITER);
		String propertyKey;

		if (isArchitectureAvailable(architectureList) == false || isPlatformAvailable(platformList) == false
				|| isRuntimeAvailable(runtimeList) == false
				|| isPeripheralAvailable(peripheralList, translatorProperties) == false) {
			throw new UnsupportedHardwareInformation();
		}

		makeAllSourceFileList(translatorProperties);

		propertyKey = TranslatorProperties.PROPERTIES_CFLAGS + TranslatorProperties.PROPERTY_DELIMITER + this.platform;
		this.cflags = this.cflags + " " + getFlagsFromProperties(propertyKey, translatorProperties);

		propertyKey = TranslatorProperties.PROPERTIES_LDADD + TranslatorProperties.PROPERTY_DELIMITER + this.platform;
		this.ldflags = this.ldflags + " " + getFlagsFromProperties(propertyKey, translatorProperties);

		this.cflags = this.cflags.trim();
		this.ldflags = this.ldflags.trim();

		this.platformDir = this.runtime + this.pathSeparator + this.platform;
	}

	private void addLibraryFlags(HashMap<String, Task> taskMap, HashMap<String, Library> libraryMap) {
		HashSet<String> ldAddSet = new HashSet<String>();
		for (Task task : taskMap.values()) {
			if (task.getLdFlags() != null && task.getLdFlags().trim().length() > 0) {
				ldAddSet.add(task.getLdFlags().trim());
			}
		}

		for (Library library : libraryMap.values()) {
			if (library.getLdFlags() != null && library.getLdFlags().trim().length() > 0) {
				ldAddSet.add(library.getLdFlags().trim());
			}
		}

		for (String ldAdd : ldAddSet) {
			this.ldflags = this.ldflags + " " + ldAdd;
		}
	}

	private void addCFlags(HashMap<String, Task> taskMap, HashMap<String, Library> libraryMap) {
		HashSet<String> cFlagsSet = new HashSet<String>();
		for (Task task : taskMap.values()) {
			if (task.getcFlags() != null && task.getcFlags().trim().length() > 0) {
				cFlagsSet.add(task.getcFlags().trim());
			}
		}

		for (Library library : libraryMap.values()) {
			if (library.getcFlags() != null && library.getcFlags().trim().length() > 0) {
				cFlagsSet.add(library.getcFlags().trim());
			}
		}

		for (String cFlag : cFlagsSet) {
			this.cflags = this.cflags + " " + cFlag;
		}
	}

	public void extraInfoFromTaskAndLibraryMap(HashMap<String, Task> taskMap, HashMap<String, Library> libraryMap) {
		addLibraryFlags(taskMap, libraryMap);
		addCFlags(taskMap, libraryMap);
	}

	private void setExtraSources(HashSet<String> extraSourceSet) {
		for (String extraSource : extraSourceSet) {
			this.extraSourceCodeSet.add(extraSource);
			if (extraSource.endsWith(Constants.CPP_FILE_EXTENSION) == true) {
				this.language = ProgrammingLanguage.CPP;
			}
		}
	}

	private void fillSourceCodeListFromTaskMap(HashMap<String, Task> taskMap) {
		for (Task task : taskMap.values()) {
			if (task.getChildTaskGraphName() == null) {
				for (int i = 0; i < task.getTaskFuncNum(); i++) {
					if (task.getLanguage() == ProgrammingLanguage.CPP) {
						this.language = ProgrammingLanguage.CPP;
					}

					if (isMappedGPU == false) {
						this.taskSourceCodeList.add(
								task.getName() + Constants.TASK_NAME_FUNC_ID_SEPARATOR + i + task.getFileExtension());
					} else {
						this.taskSourceCodeList.add(task.getName() + Constants.TASK_NAME_FUNC_ID_SEPARATOR + i
								+ Constants.CUDA_FILE_EXTENSION);
					}
				}
			}

			setExtraSources(task.getExtraSourceSet());
		}
	}

	private void fillSourceCodeListFromLibraryMap(HashMap<String, Library> libraryMap) {
		for (Library library : libraryMap.values()) {
			if (library.getLanguage() == ProgrammingLanguage.CPP) {
				this.language = ProgrammingLanguage.CPP;
			}

			this.taskSourceCodeList.add(library.getName() + library.getFileExtension());

			setExtraSources(library.getExtraSourceSet());
		}
	}

	public void fillSourceCodeListFromTaskAndLibraryMap(HashMap<String, Task> taskMap,
			HashMap<String, Library> libraryMap) {
		fillSourceCodeListFromTaskMap(taskMap);
		fillSourceCodeListFromLibraryMap(libraryMap);
	}

	public void fillSourceCodeAndFlagsFromModules(ArrayList<Module> moduleList) {
		for (Module module : moduleList) {
			this.moduleSourceList.addAll(module.getSourceList());
			this.cflags = this.cflags + " " + module.getCflags();
			this.ldflags = this.ldflags + " " + module.getLdflags();

			this.cflags = this.cflags.trim();
			this.ldflags = this.ldflags.trim();
		}
	}

	public void copyApplicationCodes(String srcDir, String outputDir) throws IOException {
		File source = new File(srcDir);
		File output = new File(outputDir + File.separator + APPLICATION_DIR);
		FileFilter filter = new FileFilter() {
			@Override
			public boolean accept(File paramFile) {
				// only copy file with extension .cic/.cicl/.h
				if (paramFile.isFile() == true && (paramFile.getName().endsWith(Constants.CIC_FILE_EXTENSION)
						|| paramFile.getName().endsWith(Constants.CICL_FILE_EXTENSION)
						|| paramFile.getName().endsWith(Constants.C_FILE_EXTENSION)
						|| paramFile.getName().endsWith(Constants.CPP_FILE_EXTENSION)
						|| paramFile.getName().endsWith(Constants.CUDA_FILE_EXTENSION)
						|| (paramFile.getName().endsWith(Constants.HEADER_FILE_EXTENSION)
								&& !paramFile.getName().endsWith(Constants.CIC_HEADER_FILE_EXTENSION))))
					return true;
				else
					return false;
			}
		};

		copyAllFiles(output, source, filter);
	}

	public void copyBuildFiles(Properties translatorProperties, String templateDir, String outputDir)
			throws IOException, UnsupportedHardwareInformation {
		File buildScriptDir = new File(templateDir + File.separator + BUILDSCRIPTS_DIR + File.separator + this.deviceRestriction
				+ File.separator + this.platform);
		File output = new File(outputDir);
		FileFilter filter = null;
		HashSet<String> copySet = new HashSet<String>();
		String targetFileList;
		String propertyKey = TranslatorProperties.PROPERTIES_BUILDSCRIPT + TranslatorProperties.PROPERTY_DELIMITER
				+ TranslatorProperties.PROPERTIES_BUILDSCRIPT_SUBFILE + TranslatorProperties.PROPERTY_DELIMITER
				+ this.platform;

		targetFileList = translatorProperties.getProperty(propertyKey);
		if (targetFileList == null) {
			throw new UnsupportedHardwareInformation("build info is not available for target: " + this.platform);
		}

		if (targetFileList.equals(STRING_ALL) == false) { // if STRING_ALL is true, copy all files in buildScriptDir
			makeHashSetFromString(buildScriptDir.getCanonicalPath() + File.separator, targetFileList, copySet);
			filter = new FileFilter() {
				@Override
				public boolean accept(File paramFile) {
					String fullPath;
					try {
						fullPath = paramFile.getCanonicalPath();

						for (String validPath : copySet) {
							if (fullPath.equals(validPath) == true) {
								return true;
							} // copy all the things in the directory
							else if (fullPath.startsWith(validPath + File.separator) == true) {
								return true;
							}
						}
					} catch (IOException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}

					return false;
				}
			};
		}

		copyAllFiles(output, buildScriptDir, filter);
	}

	public void copyFilesFromLibraryCodeTemplate(String srcDir, String outputDir) throws IOException {
		File source = new File(srcDir);
		File output = new File(outputDir);
		FileFilter filter = new FileFilter() {

			@Override
			public boolean accept(File paramFile) {
				// add File.separator after the directory path to copy directory itself and
				// avoid internal files
				if (paramFile.getAbsolutePath().contains(srcDir + File.separator + APPLICATION_DIR + File.separator))
					return false;
				else if (paramFile.getAbsolutePath().contains(srcDir + File.separator + TEMPLATES_DIR))
					return false;
				else if (paramFile.getAbsolutePath().contains(srcDir + File.separator + BUILDSCRIPTS_DIR))
					return false;
				else if (paramFile.getAbsolutePath()
						.contains(srcDir + File.separator + KERNEL_GENERATED_DIR + File.separator))
					return false;
				// skip object/executable/temporary/log files
				else if (paramFile.getName().endsWith(".o") || paramFile.getName().endsWith(".log")
						|| paramFile.getName().endsWith("~") || paramFile.getName().startsWith(".")
						|| paramFile.getName().endsWith(".exe") || paramFile.getName().endsWith(".bak"))
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
			for (int i = 0; i < children.length; i++)
				copyAllFiles(new File(targetLocation, children[i].getName()), children[i], fileFilter);
		} else {
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

	public ArrayList<String> getModuleSourceList() {
		return moduleSourceList;
	}

	public ArrayList<String> getBuildTemplateList() {
		return buildTemplateList;
	}

	public void makeBuildTemplateList(Properties translatorProperties) {
		String propertyKey = TranslatorProperties.PROPERTIES_BUILDSCRIPT + TranslatorProperties.PROPERTY_DELIMITER
				+ TranslatorProperties.PROPERTIES_BUILDSCRIPT_SUBTEMPLATE + TranslatorProperties.PROPERTY_DELIMITER
				+ this.platform;
		String templateListString = translatorProperties.getProperty(propertyKey);

		if (templateListString != null) {
			String[] templateListArray = templateListString.split(TranslatorProperties.PROPERTY_VALUE_DELIMITER);
			for (String template : templateListArray) {
				this.buildTemplateList.add(template.trim());
			}
		}
	}

	public HashSet<DeviceCommunicationType> getUsedCommunicationSet() {
		return usedCommunicationSet;
	}
	
	public HashSet<DeviceEncryptionType> getUsedEncryptionSet() {
		return usedEncryptionSet;
	}

}
