package Translators;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintStream;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.snu.cse.cap.translator.CodeOrganizer;
import org.snu.cse.cap.translator.Constants;
import org.snu.cse.cap.translator.TranslatorProperties;
import org.snu.cse.cap.translator.UEMMetaDataModel;
import org.snu.cse.cap.translator.UnsupportedHardwareInformation;
import org.snu.cse.cap.translator.structure.Application;
import org.snu.cse.cap.translator.structure.InvalidDataInMetadataFileException;
import org.snu.cse.cap.translator.structure.ProgrammingLanguage;
import org.snu.cse.cap.translator.structure.device.Device;
import org.snu.cse.cap.translator.structure.device.EnvironmentVariable;
import org.snu.cse.cap.translator.structure.device.SoftwarePlatformType;
import org.snu.cse.cap.translator.structure.device.connection.InvalidDeviceConnectionException;
import org.snu.cse.cap.translator.structure.library.Library;
import org.snu.cse.cap.translator.structure.task.Task;

import freemarker.template.Configuration;
import freemarker.template.MalformedTemplateNameException;
import freemarker.template.Template;
import freemarker.template.TemplateException;
import freemarker.template.TemplateExceptionHandler;
import freemarker.template.TemplateNotFoundException;
import hopes.cic.exception.CICXMLException;

public class CodeGenerator 
{
    private String mTranslatorPath;
    private String mUEMXMLPath;
    private String mOutputPath;
    private UEMMetaDataModel uemDatamodel;
    private Configuration templateConfig;
    private String templateDir;
    private String libraryCodeTemplateDir;
    private Properties translatorProperties;
    
    private String getCanonicalPath(String path) throws IOException 
    {
    	String canonicalPath;
    	File file = new File(path);
    	
    	canonicalPath = file.getCanonicalPath();
    	
    	return canonicalPath;
    }
    
    public CodeGenerator(String[] args)
    {
    	String translatorRootDir = "";
		this.templateConfig = new Configuration(Configuration.VERSION_2_3_27);
		
		this.templateConfig.setDefaultEncoding("UTF-8");
		this.templateConfig.setTemplateExceptionHandler(TemplateExceptionHandler.RETHROW_HANDLER);
		this.templateConfig.setLogTemplateExceptions(false);
		this.templateConfig.setWrapUncheckedExceptions(true);
		
		this.translatorProperties = new Properties();

		try {
			translatorRootDir = getCanonicalPath(getClass().getProtectionDomain().getCodeSource().getLocation().getFile() + "..");
			
			this.translatorProperties.load(new FileInputStream(translatorRootDir + File.separator + Constants.DEFAULT_PROPERTIES_FILE_PATH));
			
			this.templateDir = this.translatorProperties.getProperty(TranslatorProperties.PROPERTIES_TEMPLATE_CODE_PATH, 
																	Constants.DEFAULT_TEMPLATE_DIR);
			this.templateDir = getCanonicalPath(translatorRootDir + File.separator + this.templateDir);
			
			this.libraryCodeTemplateDir = this.translatorProperties.getProperty(TranslatorProperties.PROPERTIES_TRANSLATED_CODE_TEMPLATE_PATH,
												Constants.DEFAULT_TRANSLATED_CODE_TEMPLATE_DIR);
			this.libraryCodeTemplateDir = getCanonicalPath(translatorRootDir + File.separator + this.libraryCodeTemplateDir);
			
			this.templateConfig.setDirectoryForTemplateLoading(new File(this.templateDir));
			
			//this.tran
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-1);
		}
		
		initMetaData(translatorRootDir, args);
    }
    
    private void changeAllPathSeparator() 
    {
		// Change all separators to system-specific separator
    	mTranslatorPath = mTranslatorPath.replace('\\', File.separatorChar);
    	mTranslatorPath = mTranslatorPath.replace('/', File.separatorChar);
    	mUEMXMLPath = mUEMXMLPath.replace('\\', File.separatorChar);
    	mUEMXMLPath = mUEMXMLPath.replace('/', File.separatorChar);    		
    	mOutputPath = mOutputPath.replace('\\', File.separatorChar);
    	mOutputPath = mOutputPath.replace('/', File.separatorChar);
    }
    
    private void initMetaData(String translatorRootDir, String[] args) 
    {
    	Options options = new Options();
    	HelpFormatter formatter = new HelpFormatter();
    	String[] leftArgs;
    	CommandLine cmd;
    	
    	options.addOption(Constants.COMMANDLINE_OPTION_HELP, false, "print this help");
    	options.addOption("t", Constants.COMMANDLINE_OPTION_TEMPLATE_DIR, true, "set template directory");
    	
    	CommandLineParser parser = new DefaultParser();
    	try {
    		cmd = parser.parse(options, args);
    		
    		leftArgs = cmd.getArgs();
    		
    		if(leftArgs.length < 3)
    		{
    			throw new ParseException("Not enough arguments, at least three arguments are needed ");
    		}
    		    		
    		mTranslatorPath = leftArgs[0];
    		mUEMXMLPath = leftArgs[1];
    		mOutputPath = leftArgs[2];
    		
    		changeAllPathSeparator() ;
    		
    		if(mOutputPath.endsWith(File.separator))
    		{
    			mOutputPath = mOutputPath.substring(0, mOutputPath.length() - 1);
    		}
    		
    		if(cmd.hasOption(Constants.COMMANDLINE_OPTION_HELP)) 
    		{
        		formatter.printHelp("Translator.CodeGenerator [options] <Code generator binary path> <CIC XML file path> <Output file path> ", "UEM to Target C Code Translator", options, "");
    		}
    		
    		if(cmd.hasOption(Constants.COMMANDLINE_OPTION_TEMPLATE_DIR))
    		{
    			this.templateDir = cmd.getOptionValue(Constants.COMMANDLINE_OPTION_TEMPLATE_DIR); 
    		}
    		
    		System.out.println("mTranslatorPath: " + mTranslatorPath + ", mCICXMLPath: " + mUEMXMLPath + ", mOutputPath: " + mOutputPath);
    		
    		this.uemDatamodel = new UEMMetaDataModel(translatorRootDir, mUEMXMLPath, mOutputPath + File.separator + Constants.SCHEDULE_FOLDER_NAME + File.separator);
    	} 
    	catch(ParseException e) {
    		System.out.println("ERROR: " + e.getMessage());
    		formatter.printHelp("Translator.CodeGenerator [options] <Code generator binary path> <CIC XML file path> <Output file path> ", "UEM to Target C Code Translator", options, "");
    	}
    	catch(CICXMLException e) {
    		e.printStackTrace();
    		System.out.println("Cannot load XML metadata information");
    	} catch (InvalidDataInMetadataFileException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InvalidDeviceConnectionException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (CloneNotSupportedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
    
    private void generateMakefile(CodeOrganizer codeOrganizer, Device device, String topDirPath, ArrayList<EnvironmentVariable> envVarList) throws TemplateNotFoundException, MalformedTemplateNameException, 
    																freemarker.core.ParseException, IOException, TemplateException
    {
    	Template makefileTemplate = this.templateConfig.getTemplate(codeOrganizer.getPlatform() + Constants.TEMPLATE_PATH_SEPARATOR + Constants.TEMPLATE_FILE_MAKEFILE);
		// Create the root hash
		Map<String, Object> makefileRootHash = new HashMap<String, Object>();
		String outputFilePath = topDirPath + File.separator;
		
		switch(codeOrganizer.getBuildType())
		{
		case AUTOMAKE:
			outputFilePath += Constants.DEFAULT_MAKEFILE_AM;
			break;
		case MAKEFILE:
			outputFilePath += Constants.DEFAULT_MAKEFILE;
			break;
		default:
			//TODO: must be error
			break;
		}
	
		makefileRootHash.put(Constants.TEMPLATE_TAG_BUILD_INFO, codeOrganizer);
		makefileRootHash.put(Constants.TEMPLATE_TAG_ENVIRONMENT_VARIABLE_INFO, envVarList);
		makefileRootHash.put(Constants.TEMPLATE_TAG_USED_COMMUNICATION_LIST, codeOrganizer.getUsedCommunicationSet());
		//19.04.01 added
		makefileRootHash.put(Constants.TEMPLATE_TAG_DEVICE_ARCHITECTURE_INFO, device.getArchitecture());

		Writer out = new OutputStreamWriter(new PrintStream(new File(outputFilePath)));
		makefileTemplate.process(makefileRootHash, out);
    }
    
    private void generateDoxyFile(CodeOrganizer codeOrganizer, String topDirPath) throws TemplateNotFoundException, MalformedTemplateNameException, 
    freemarker.core.ParseException, IOException, TemplateException
    {
    	Template doxyFileTemplate = this.templateConfig.getTemplate(Constants.TEMPLATE_FILE_DOXYFILE);
    	// Create the root hash
    	Map<String, Object> doxyfileRootHash = new HashMap<String, Object>();
    	String outputFilePath = topDirPath + File.separator;
    	
    	outputFilePath += Constants.DEFAULT_DOXYFILE;

    	doxyfileRootHash.put(Constants.TEMPLATE_TAG_BUILD_INFO, codeOrganizer);

    	Writer out = new OutputStreamWriter(new PrintStream(new File(outputFilePath)));
    	doxyFileTemplate.process(doxyfileRootHash, out);
    }
    
    private void generateDoxygenManual(CodeOrganizer codeOrganizer, Application application, Device device, String topDirPath) throws TemplateNotFoundException, MalformedTemplateNameException, 
    freemarker.core.ParseException, IOException, TemplateException
    {
    	Template doxygenManualTemplate = this.templateConfig.getTemplate(Constants.DEFAULT_DOXYGEN_MANUAL+Constants.TEMPLATE_FILE_EXTENSION);
    	// Create the root hash
    	Map<String, Object> doxygenManualRootHash = new HashMap<String, Object>();
    	String outputFilePath = topDirPath + File.separator + CodeOrganizer.KERNEL_GENERATED_DIR + File.separator;
    	
    	outputFilePath += Constants.DEFAULT_DOXYGEN_MANUAL + Constants.HEADER_FILE_EXTENSION;

    	doxygenManualRootHash.put(Constants.TEMPLATE_TAG_MANUAL_DEVICE_INFO, device);
    	doxygenManualRootHash.put(Constants.TEMPLATE_TAG_MANUAL_TASK_GRAPH, application.getFullTaskGraphMap());
    	doxygenManualRootHash.put(Constants.TEMPLATE_TAG_MANUAL_LIBRARY_MAP, application.getLibraryMap());
    	doxygenManualRootHash.put(Constants.TEMPLATE_TAG_MANUAL_CHANNEL_LIST, application.getChannelList());
    	doxygenManualRootHash.put(Constants.TEMPLATE_TAG_MANUAL_TASK_MAP, application.getTaskMap());
    	doxygenManualRootHash.put(Constants.TEMPLATE_TAG_MANUAL_DEVICE_MAP, application.getDeviceInfo());
    	doxygenManualRootHash.put(Constants.TEMPLATE_TAG_MANUAL_DEVICE_CONNECTION_MAP, application.getDeviceConnectionMap());

    	Writer out = new OutputStreamWriter(new PrintStream(new File(outputFilePath)));
    	doxygenManualTemplate.process(doxygenManualRootHash, out);
    }
    
    private void generateKernelDataCode(CodeOrganizer codeOrganizer, Device device, String topDirPath, ArrayList<EnvironmentVariable> envVarList, ProgrammingLanguage language) throws TemplateNotFoundException, MalformedTemplateNameException, 
	freemarker.core.ParseException, IOException, TemplateException
    {
    	Map<String, Object> uemDataRootHash = new HashMap<String, Object>();
    	
		// Put UEM data model
		uemDataRootHash.put(Constants.TEMPLATE_TAG_TASK_MAP, device.getTaskMap());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_TASK_GRAPH, device.getTaskGraphMap());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_CHANNEL_LIST, device.getChannelList());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_DEVICE_INFO, this.uemDatamodel.getApplication().getDeviceInfo());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_DEVICE_ID, device.getId());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_MAPPING_INFO, device.getGeneralMappingInfo());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_STATIC_SCHEDULE_INFO, device.getStaticScheduleMappingInfo());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_MULTICAST_GROUP_LIST, device.getMulticastGroupMap().values());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_PORT_INFO, device.getPortList());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_PORT_KEY_TO_INDEX, device.getPortKeyToIndex());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_EXECUTION_TIME, this.uemDatamodel.getApplication().getExecutionTime());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_LIBRARY_INFO, device.getLibraryMap());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_GPU_USED, device.isGPUMapped());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_COMMUNICATION_USED, device.useCommunication());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_TCP_CLIENT_LIST, device.getTcpClientList());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_TCP_SERVER_LIST, device.getTcpServerList());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_UDP_LIST, device.getUDPConnectionList());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_BLUETOOTH_MASTER_LIST, device.getBluetoothMasterList());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_BLUETOOTH_SLAVE_LIST, device.getBluetoothUnconstrainedSlaveList());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_SERIAL_MASTER_LIST, device.getSerialMasterList());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_ENVIRONMENT_VARIABLE_INFO, envVarList);
		//
		if(device.getPlatform() == SoftwarePlatformType.ARDUINO)
		{
			uemDataRootHash.put(Constants.TEMPLATE_TAG_SERIAL_SLAVE_LIST, device.getSerialConstrainedSlaveList());	
		}
		else if(device.getPlatform() == SoftwarePlatformType.LINUX)
		{
			uemDataRootHash.put(Constants.TEMPLATE_TAG_SERIAL_SLAVE_LIST, device.getSerialUnconstrainedSlaveList());
		}
		
		uemDataRootHash.put(Constants.TEMPLATE_TAG_MODULE_LIST, device.getModuleList());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_DEVICE_CONSTRAINED_INFO, codeOrganizer.getDeviceRestriction());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_USED_COMMUNICATION_LIST, codeOrganizer.getUsedCommunicationSet());
		
    	for(String outputFileName : codeOrganizer.getKernelDataSourceList())
    	{
    		Template uemDataTemplate;
    		String templateFileName;
    		String outputFilePath = topDirPath + File.separator + CodeOrganizer.KERNEL_GENERATED_DIR + File.separator;
    		
    		// remove file extension of the file name (removes last dot and the following chars ex. abc.x.c => abc.x)
    		templateFileName = outputFileName.replaceFirst("[.][^.]+$", "")  + Constants.TEMPLATE_FILE_EXTENSION;
    		uemDataTemplate = this.templateConfig.getTemplate(codeOrganizer.getPlatform() + Constants.TEMPLATE_PATH_SEPARATOR + templateFileName);
    		outputFilePath += outputFileName;
    		
    		Writer out = new OutputStreamWriter(new PrintStream(new File(outputFilePath)));
    		uemDataTemplate.process(uemDataRootHash, out);
    	}
    }

    
    private void generateTaskCode(Device device, String topDirPath) throws TemplateNotFoundException, MalformedTemplateNameException, 
    												freemarker.core.ParseException, IOException, TemplateException
    {
    	Template taskCodeTemplate = this.templateConfig.getTemplate(Constants.TEMPLATE_FILE_TASK_CODE);
    	
		for(Task task: device.getTaskMap().values())
		{
			if(task.getChildTaskGraphName() == null)
			{
				// Create the root hash
	    		Map<String, Object> taskCodeRootHash = new HashMap<String, Object>();
	    		
	    		taskCodeRootHash.put(Constants.TEMPLATE_TAG_TASK_INFO, task);
				taskCodeRootHash.put(Constants.TEMPLATE_TAG_TASK_GPU_MAPPING_INFO, device.getGpuSetupInfo().get(task.getName()));
	    		    		
	    		for(int loop = 0 ; loop < task.getTaskFuncNum() ; loop++)
	    		{
	    			String outputFilePath = topDirPath + File.separator + CodeOrganizer.APPLICATION_DIR + File.separator + 
	    									task.getName() +  Constants.TASK_NAME_FUNC_ID_SEPARATOR + loop;
	    			
	    			if(device.isGPUMapped() == false){
	    				if(task.getLanguage() == ProgrammingLanguage.CPP) {
	    					outputFilePath += Constants.CPP_FILE_EXTENSION;
	    				}
	    				else {
	    					outputFilePath += Constants.C_FILE_EXTENSION;	
	    				}
	    			}
	    			else{
	    				outputFilePath += Constants.CUDA_FILE_EXTENSION;
	    			}
	    			
	    			if(taskCodeRootHash.containsKey(Constants.TEMPLATE_TAG_TASK_FUNC_ID) == true)
	    			{
	    				taskCodeRootHash.remove(Constants.TEMPLATE_TAG_TASK_FUNC_ID);
	    			}
		    		
		    		taskCodeRootHash.put(Constants.TEMPLATE_TAG_TASK_FUNC_ID, new Integer(loop));
		    		
		    		Writer out = new OutputStreamWriter(new PrintStream(new File(outputFilePath)));
		    		taskCodeTemplate.process(taskCodeRootHash, out);
	    		}
			}
		}
    }
    
    private void generateLibraryCodes(Device device, String topDirPath) throws TemplateNotFoundException, MalformedTemplateNameException, 
    													freemarker.core.ParseException, IOException, TemplateException
    {
		Template libraryCodeTemplate = this.templateConfig.getTemplate(Constants.TEMPLATE_FILE_LIBRARY_CODE);
		Template libraryHeaderTemplate = this.templateConfig.getTemplate(Constants.TEMPLATE_FILE_LIBRARY_HEADER);
		
		for(Library library : device.getLibraryMap().values())
		{
			String outputSourcePath = topDirPath + File.separator + CodeOrganizer.APPLICATION_DIR + File.separator + 
										library.getName();
			String outputHeaderPath = topDirPath + File.separator + CodeOrganizer.APPLICATION_DIR + File.separator + 
										library.getHeader();
			// Create the root hash
			Map<String, Object> libraryRootHash = new HashMap<String, Object>();
			
			if(library.getLanguage() == ProgrammingLanguage.CPP) {
				outputSourcePath += Constants.CPP_FILE_EXTENSION;
			}
			else {
				outputSourcePath += Constants.C_FILE_EXTENSION;	
			}
			
			libraryRootHash.put(Constants.TEMPLATE_TAG_TASK_MAP, device.getTaskMap());
			libraryRootHash.put(Constants.TEMPLATE_TAG_LIB_INFO, library);
			libraryRootHash.put(Constants.TEMPLATE_TAG_LIBRARY_INFO, device.getLibraryMap());
						
			Writer outSource = new OutputStreamWriter(new PrintStream(new File(outputSourcePath)));
			libraryCodeTemplate.process(libraryRootHash, outSource);
			
			Writer outHeader = new OutputStreamWriter(new PrintStream(new File(outputHeaderPath)));
			libraryHeaderTemplate.process(libraryRootHash, outHeader);
		}
    }
    
    public void generateCode()
    {
   		try {
			for(Device device : this.uemDatamodel.getApplication().getDeviceInfo().values())
			{
				CodeOrganizer codeOrganizer = new CodeOrganizer(device.getArchitecture().toString(), 
						device.getPlatform().toString(), device.getRuntime().toString(), device.isGPUMapped(), device.getRequiredCommunicationSet());
				String topSrcDir = this.mOutputPath + File.separator + device.getName();
				
				codeOrganizer.fillSourceCodeListFromTaskAndLibraryMap(device.getTaskMap(), device.getLibraryMap());
				codeOrganizer.extraInfoFromTaskAndLibraryMap(device.getTaskMap(), device.getLibraryMap());
				codeOrganizer.fillSourceCodeAndFlagsFromModules(device.getModuleList());
				codeOrganizer.extractDataFromProperties(this.translatorProperties);
				codeOrganizer.setBuildType(this.translatorProperties);

				codeOrganizer.copyFilesFromLibraryCodeTemplate(this.libraryCodeTemplateDir, topSrcDir);
				codeOrganizer.copyBuildFiles(this.translatorProperties, this.libraryCodeTemplateDir, topSrcDir);
				codeOrganizer.copyApplicationCodes(this.mOutputPath, topSrcDir);
				
				generateMakefile(codeOrganizer, device, topSrcDir, device.getEnvironmentVariableList());
				//generateKernelDataCode(codeOrganizer, device, topSrcDir, codeOrganizer.getLanguage());
				generateKernelDataCode(codeOrganizer, device, topSrcDir, device.getEnvironmentVariableList(), codeOrganizer.getLanguage());
				generateTaskCode(device, topSrcDir);
				generateLibraryCodes(device, topSrcDir);
				generateDoxyFile(codeOrganizer, topSrcDir);
				generateDoxygenManual(codeOrganizer, this.uemDatamodel.getApplication(), device, topSrcDir);
			}			
		} catch (TemplateNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-1);
		} catch (MalformedTemplateNameException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-1);
		} catch (freemarker.core.ParseException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-1);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-1);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-1);
		} catch (TemplateException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-1);
		} catch (UnsupportedHardwareInformation e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-1);
		}
    }

	public static void main(String[] args) 
	{
		// TODO Auto-generated method stub
		CodeGenerator codeGenerator = new CodeGenerator(args); 
		codeGenerator.generateCode();
	}

}
