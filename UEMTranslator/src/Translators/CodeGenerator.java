package Translators;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintStream;
import java.io.Writer;
import java.nio.file.Files;
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
import org.snu.cse.cap.translator.structure.InvalidDataInMetadataFileException;
import org.snu.cse.cap.translator.structure.ProgrammingLanguage;
import org.snu.cse.cap.translator.structure.device.Device;
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
		this.templateConfig = new Configuration(Configuration.VERSION_2_3_27);
		
		this.templateConfig.setDefaultEncoding("UTF-8");
		this.templateConfig.setTemplateExceptionHandler(TemplateExceptionHandler.RETHROW_HANDLER);
		this.templateConfig.setLogTemplateExceptions(false);
		this.templateConfig.setWrapUncheckedExceptions(true);
		
		this.translatorProperties = new Properties();

		try {
			this.translatorProperties.load(new FileInputStream(Constants.DEFAULT_PROPERTIES_FILE_PATH));
			
			this.templateDir = this.translatorProperties.getProperty(TranslatorProperties.PROPERTIES_TEMPLATE_CODE_PATH, 
																	Constants.DEFAULT_TEMPLATE_DIR);
			this.templateDir = getCanonicalPath(this.templateDir);
			
			this.libraryCodeTemplateDir = this.translatorProperties.getProperty(TranslatorProperties.PROPERTIES_TRANSLATED_CODE_TEMPLATE_PATH,
												Constants.DEFAULT_TRANSLATED_CODE_TEMPLATE_DIR);
			this.libraryCodeTemplateDir = getCanonicalPath(this.libraryCodeTemplateDir);
			
			this.templateConfig.setDirectoryForTemplateLoading(new File(this.templateDir));
			
			//this.tran
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-1);
		}
		
		initMetaData(args);
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
    
    private void initMetaData(String[] args) 
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
    		
    		this.uemDatamodel = new UEMMetaDataModel(mUEMXMLPath, mOutputPath + File.separator + Constants.SCHEDULE_FOLDER_NAME + File.separator);
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
		}
    }
    
    private void generateMakefile(CodeOrganizer codeOrganizer, String topDirPath) throws TemplateNotFoundException, MalformedTemplateNameException, 
    																freemarker.core.ParseException, IOException, TemplateException
    {
    	Template makefileTemplate = this.templateConfig.getTemplate(Constants.TEMPLATE_FILE_MAKEFILE);
		// Create the root hash
		Map<String, Object> makefileRootHash = new HashMap<>();
		String outputFilePath = topDirPath + File.separator + Constants.DEFAULT_MAKEFILE_AM;
		
		makefileRootHash.put(Constants.TEMPLATE_TAG_BUILD_INFO, codeOrganizer);

		Writer out = new OutputStreamWriter(new PrintStream(new File(outputFilePath)));
		makefileTemplate.process(makefileRootHash, out);
    }
    
    private void generateUemDataCode(Device device, String topDirPath, ProgrammingLanguage language) throws TemplateNotFoundException, MalformedTemplateNameException, 
    													freemarker.core.ParseException, IOException, TemplateException
    {
    	Template uemDataTemplate = this.templateConfig.getTemplate(Constants.TEMPLATE_FILE_UEM_DATA);
		// Create the root hash
		Map<String, Object> uemDataRootHash = new HashMap<>();
		String outputFilePath = topDirPath + File.separator + CodeOrganizer.KERNEL_DIR + File.separator;
		if(device.getGpuSetupInfo().size() != 0){
			outputFilePath += Constants.DEFAULT_UEM_DATA_CUDA;
			
		}
		else if(language == ProgrammingLanguage.CPP){
			outputFilePath += Constants.DEFAULT_UEM_DATA_CPP;
		}
		else
		{
			outputFilePath += Constants.DEFAULT_UEM_DATA_C;
		}
		
		// Put UEM data model
		uemDataRootHash.put(Constants.TEMPLATE_TAG_TASK_MAP, device.getTaskMap());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_TASK_GRAPH, device.getTaskGraphMap());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_CHANNEL_LIST, device.getChannelList());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_DEVICE_INFO, this.uemDatamodel.getApplication().getDeviceInfo());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_MAPPING_INFO, device.getGeneralMappingInfo());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_STATIC_SCHEDULE_INFO, device.getStaticScheduleMappingInfo());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_PORT_INFO, device.getPortList());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_PORT_KEY_TO_INDEX, device.getPortKeyToIndex());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_EXECUTION_TIME, this.uemDatamodel.getApplication().getExecutionTime());
		uemDataRootHash.put(Constants.TEMPLATE_TAG_LIBRARY_INFO, device.getLibraryMap());
		if(device.getGpuSetupInfo().size() == 0)
		{
			uemDataRootHash.put(Constants.TEMPLATE_TAG_GPU_USED, false);
		}
		else
		{
			uemDataRootHash.put(Constants.TEMPLATE_TAG_GPU_USED, true);
		}
		
		
		Writer out = new OutputStreamWriter(new PrintStream(new File(outputFilePath)));
		uemDataTemplate.process(uemDataRootHash, out);
    }
    
    private void generateTaskCode(Device device, String topDirPath, ProgrammingLanguage language) throws TemplateNotFoundException, MalformedTemplateNameException, 
    												freemarker.core.ParseException, IOException, TemplateException
    {
    	Template taskCodeTemplate = this.templateConfig.getTemplate(Constants.TEMPLATE_FILE_TASK_CODE);
    	
		for(Task task: device.getTaskMap().values())
		{
			if(task.getChildTaskGraphName() == null)
			{
				// Create the root hash
	    		Map<String, Object> taskCodeRootHash = new HashMap<>();
	    		
	    		taskCodeRootHash.put(Constants.TEMPLATE_TAG_TASK_INFO, task);
				taskCodeRootHash.put(Constants.TEMPLATE_TAG_TASK_GPU_MAPPING_INFO, device.getGpuSetupInfo().get(task.getName()));
	    		    		
	    		for(int loop = 0 ; loop < task.getTaskFuncNum() ; loop++)
	    		{
	    			String outputFilePath = topDirPath + File.separator + CodeOrganizer.APPLICATION_DIR + File.separator + 
	    									task.getName() +  Constants.TASK_NAME_FUNC_ID_SEPARATOR + loop;
	    			
	    			if(device.getGpuSetupInfo().size() == 0){
	    				if(language == ProgrammingLanguage.CPP) {
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
    
    private void generateLibraryCodes(Device device, String topDirPath, ProgrammingLanguage language) throws TemplateNotFoundException, MalformedTemplateNameException, 
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
			Map<String, Object> libraryRootHash = new HashMap<>();
			
			if(language == ProgrammingLanguage.CPP) {
				outputSourcePath += Constants.CPP_FILE_EXTENSION;
			}
			else {
				outputSourcePath += Constants.C_FILE_EXTENSION;	
			}
			
			libraryRootHash.put(Constants.TEMPLATE_TAG_LIB_INFO, library);
						
			Writer outSource = new OutputStreamWriter(new PrintStream(new File(outputSourcePath)));
			libraryCodeTemplate.process(libraryRootHash, outSource);
			
			Writer outHeader = new OutputStreamWriter(new PrintStream(new File(outputHeaderPath)));
			libraryHeaderTemplate.process(libraryRootHash, outHeader);
		}
    }
    
	public boolean isMappedGPU(Device device)
	{
		if (device.getGpuSetupInfo().size() == 0)
		{
			return false;
		}
		return true;
	}
    
    public void generateCode()
    {
   		try {
			for(Device device : uemDatamodel.getApplication().getDeviceInfo().values())
			{
				CodeOrganizer codeOrganizer = new CodeOrganizer(device.getArchitecture().toString(), 
						device.getPlatform().toString(), device.getRuntime().toString(), isMappedGPU(device));
				String topSrcDir = this.mOutputPath + File.separator + device.getName();
				
				codeOrganizer.fillSourceCodeListFromTaskAndLibraryMap(device.getTaskMap(), device.getLibraryMap());
				codeOrganizer.extraInfoFromTaskAndLibraryMap(device.getTaskMap(), device.getLibraryMap());
				codeOrganizer.extractDataFromProperties(this.translatorProperties);
				
				codeOrganizer.copyFilesFromLibraryCodeTemplate(this.libraryCodeTemplateDir, topSrcDir);
				codeOrganizer.copyApplicationCodes(this.mOutputPath, topSrcDir);
				
				generateMakefile(codeOrganizer, topSrcDir);
				generateUemDataCode(device, topSrcDir, codeOrganizer.getLanguage());
				generateTaskCode(device, topSrcDir, codeOrganizer.getLanguage());
				generateLibraryCodes(device, topSrcDir, codeOrganizer.getLanguage());
			}			
		} catch (TemplateNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (MalformedTemplateNameException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (freemarker.core.ParseException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (TemplateException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (UnsupportedHardwareInformation e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }

	public static void main(String[] args) 
	{
		// TODO Auto-generated method stub
		CodeGenerator codeGenerator = new CodeGenerator(args); 
		codeGenerator.generateCode();
	}

}
