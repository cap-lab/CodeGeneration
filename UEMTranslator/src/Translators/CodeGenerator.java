package Translators;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
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
import org.snu.cse.cap.translator.UnsupportedHardwareInformation;
import org.snu.cse.cap.translator.structure.InvalidDataInMetadataFileException;
import org.snu.cse.cap.translator.structure.device.Device;
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
    private String templateFolderPath;
    private Properties translatorProperties;
    
    
    public CodeGenerator(String[] args) 
    {
    	this.templateFolderPath = "templates";
		this.templateConfig = new Configuration(Configuration.VERSION_2_3_27);

		try {
			this.templateConfig.setDirectoryForTemplateLoading(new File(this.templateFolderPath));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		this.templateConfig.setDefaultEncoding("UTF-8");
		this.templateConfig.setTemplateExceptionHandler(TemplateExceptionHandler.RETHROW_HANDLER);
		this.templateConfig.setLogTemplateExceptions(false);
		this.templateConfig.setWrapUncheckedExceptions(true);
		this.translatorProperties = new Properties();
		
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
    			this.templateFolderPath = cmd.getOptionValue(Constants.COMMANDLINE_OPTION_TEMPLATE_DIR); 
    		}
    		
    		System.out.println("mTranslatorPath: " + mTranslatorPath + ", mCICXMLPath: " + mUEMXMLPath + ", mOutputPath: " + mOutputPath);
    		
    		this.uemDatamodel = new UEMMetaDataModel(mUEMXMLPath, mOutputPath + File.separator + Constants.SCHEDULE_FOLDER_NAME + File.separator);
    					
			this.translatorProperties.load(new FileInputStream("config" + File.separator + Constants.DEFAULT_PROPERTIES_FILE_NAME));
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
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
    
    public void generateCode()
    {
   		try {
			Template uemDataTemplate = this.templateConfig.getTemplate(Constants.TEMPLATE_FILE_UEM_DATA);
			Template makefileTemplate = this.templateConfig.getTemplate(Constants.TEMPLATE_FILE_MAKEFILE);			
			Template taskCodeTemplate = this.templateConfig.getTemplate(Constants.TEMPLATE_FILE_TASK_CODE);
			Template libraryCodeTemplate = this.templateConfig.getTemplate(Constants.TEMPLATE_FILE_LIBRARY_CODE);
			Template libraryHeaderTemplate = this.templateConfig.getTemplate(Constants.TEMPLATE_FILE_LIBRARY_HEADER);
			
			for(Device device : uemDatamodel.getApplication().getDeviceInfo().values())
			{
				// Create the root hash
				Map<String, Object> uemDataRootHash = new HashMap<>();
				
				// Put UEM data model
				uemDataRootHash.put(Constants.TEMPLATE_TAG_TASK_MAP, device.getTaskMap());
				uemDataRootHash.put(Constants.TEMPLATE_TAG_TASK_GRAPH, device.getTaskGraphMap());
				uemDataRootHash.put(Constants.TEMPLATE_TAG_CHANNEL_LIST, device.getChannelList());
				uemDataRootHash.put(Constants.TEMPLATE_TAG_DEVICE_INFO, uemDatamodel.getApplication().getDeviceInfo());
				uemDataRootHash.put(Constants.TEMPLATE_TAG_MAPPING_INFO, device.getGeneralMappingInfo());
				uemDataRootHash.put(Constants.TEMPLATE_TAG_STATIC_SCHEDULE_INFO, device.getStaticScheduleMappingInfo());
				uemDataRootHash.put(Constants.TEMPLATE_TAG_PORT_INFO, device.getPortList());
				uemDataRootHash.put(Constants.TEMPLATE_TAG_PORT_KEY_TO_INDEX, device.getPortKeyToIndex());
				
				Writer out = new OutputStreamWriter(System.out);
				uemDataTemplate.process(uemDataRootHash, out);
				
				CodeOrganizer codeOrganizer = new CodeOrganizer(device.getArchitecture().toString(), 
						device.getPlatform().toString(), device.getRuntime().toString());
		
				codeOrganizer.extractDataFromProperties(this.translatorProperties);
				codeOrganizer.fillSourceCodeListFromTaskMap(device.getTaskMap());
				codeOrganizer.fillSourceCodeListFromLibraryMap(device.getLibraryMap());
				
				// Create the root hash
				Map<String, Object> makefileRootHash = new HashMap<>();
				
				makefileRootHash.put(Constants.TEMPLATE_TAG_BUILD_INFO, codeOrganizer);
				
				out = new OutputStreamWriter(System.out);
				makefileTemplate.process(makefileRootHash, out);
				
				for(Task task: device.getTaskMap().values())
				{
					if(task.getChildTaskGraphName() == null)
					{
						// Create the root hash
			    		Map<String, Object> taskCodeRootHash = new HashMap<>();
			    		
			    		taskCodeRootHash.put(Constants.TEMPLATE_TAG_TASK_INFO, task);
			    		
			    		out = new OutputStreamWriter(System.out);
			    		taskCodeTemplate.process(taskCodeRootHash, out);
					}
				}
				
				for(Library library : device.getLibraryMap().values())
				{
					// Create the root hash
					Map<String, Object> libraryRootHash = new HashMap<>();
					
					libraryRootHash.put(Constants.TEMPLATE_TAG_LIB_INFO, library);
					
					out = new OutputStreamWriter(System.out);
					libraryCodeTemplate.process(libraryRootHash, out);
					
					out = new OutputStreamWriter(System.out);
					libraryHeaderTemplate.process(libraryRootHash, out);
				}
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
