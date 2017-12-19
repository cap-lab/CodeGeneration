package Translators;

import java.io.File;
import java.io.FileInputStream;
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
import freemarker.template.Template;
import freemarker.template.TemplateException;
import freemarker.template.TemplateExceptionHandler;
import hopes.cic.exception.CICXMLException;

public class CodeGenerator 
{
    private String mTranslatorPath;
    private String mUEMXMLPath;
    private String mOutputPath;
    private UEMMetaDataModel uemDatamodel;
    private Configuration templateConfig;
    private String templateFolderPath;
    
    
    public CodeGenerator() 
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
    
    public void initMetaData(String[] args) 
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
    		
    		uemDatamodel = new UEMMetaDataModel(mUEMXMLPath, mOutputPath + File.separator + Constants.SCHEDULE_FOLDER_NAME + File.separator);
 
    		Template temp = this.templateConfig.getTemplate(Constants.TEMPLATE_FILE_UEM_DATA);
    		
    		for(Device device : uemDatamodel.getApplication().getDeviceInfo().values())
    		{
    			// Create the root hash
        		Map<String, Object> root = new HashMap<>();
        		
        		// Put UEM data model
        		root.put(Constants.TEMPLATE_TAG_TASK_MAP, device.getTaskMap());
        		root.put(Constants.TEMPLATE_TAG_TASK_GRAPH, device.getTaskGraphMap());
        		root.put(Constants.TEMPLATE_TAG_CHANNEL_LIST, device.getChannelList());
        		root.put(Constants.TEMPLATE_TAG_DEVICE_INFO, uemDatamodel.getApplication().getDeviceInfo());
        		root.put(Constants.TEMPLATE_TAG_MAPPING_INFO, device.getGeneralMappingInfo());
        		root.put(Constants.TEMPLATE_TAG_STATIC_SCHEDULE_INFO, device.getStaticScheduleMappingInfo());
        		
        		Writer out = new OutputStreamWriter(System.out);
        		temp.process(root, out);
    		}
    		
    		Properties prop = new Properties();
    		
    		prop.load(new FileInputStream("config" + File.separator + Constants.DEFAULT_PROPERTIES_FILE_NAME));
    		
    		temp = this.templateConfig.getTemplate(Constants.TEMPLATE_FILE_MAKEFILE);
    		
    		for(Device device : uemDatamodel.getApplication().getDeviceInfo().values())
    		{
    			CodeOrganizer codeOrganizer = new CodeOrganizer(device.getArchitecture().toString(), 
    							device.getPlatform().toString(), device.getRuntime().toString());
    			
    			codeOrganizer.extractDataFromProperties(prop);
    			codeOrganizer.fillSourceCodeListFromTaskMap(device.getTaskMap());
    			codeOrganizer.fillSourceCodeListFromLibraryMap(device.getLibraryMap());
    			
    			// Create the root hash
        		Map<String, Object> root = new HashMap<>();
        		
        		root.put(Constants.TEMPLATE_TAG_BUILD_INFO, codeOrganizer);
        		
        		Writer out = new OutputStreamWriter(System.out);
        		temp.process(root, out);
    		}
    		
    		temp = this.templateConfig.getTemplate(Constants.TEMPLATE_FILE_TASK_CODE);
    		
    		for(Device device : uemDatamodel.getApplication().getDeviceInfo().values())
    		{
    			for(Task task: device.getTaskMap().values())
    			{
    				if(task.getChildTaskGraphName() == null)
    				{
	        			// Create the root hash
	            		Map<String, Object> root = new HashMap<>();
	            		
	            		root.put(Constants.TEMPLATE_TAG_TASK_INFO, task);
	            		
	            		Writer out = new OutputStreamWriter(System.out);
	            		temp.process(root, out);
    				}
    			}
    		}
    		
    		temp = this.templateConfig.getTemplate(Constants.TEMPLATE_FILE_LIBRARY_CODE);
    		
    		for(Device device : uemDatamodel.getApplication().getDeviceInfo().values())
    		{
    			for(Library library : device.getLibraryMap().values())
    			{
    				// Create the root hash
    				Map<String, Object> root = new HashMap<>();
            		
            		root.put(Constants.TEMPLATE_TAG_LIB_INFO, library);
            		
            		Writer out = new OutputStreamWriter(System.out);
            		temp.process(root, out);
    			}
    		}
    		
    		temp = this.templateConfig.getTemplate(Constants.TEMPLATE_FILE_LIBRARY_HEADER);
    		
    		for(Device device : uemDatamodel.getApplication().getDeviceInfo().values())
    		{
    			for(Library library : device.getLibraryMap().values())
    			{
    				// Create the root hash
    				Map<String, Object> root = new HashMap<>();
            		
            		root.put(Constants.TEMPLATE_TAG_LIB_INFO, library);
            		
            		Writer out = new OutputStreamWriter(System.out);
            		temp.process(root, out);
    			}
    		}
    	} 
    	catch(ParseException e) {
    		System.out.println("ERROR: " + e.getMessage());
    		formatter.printHelp("Translator.CodeGenerator [options] <Code generator binary path> <CIC XML file path> <Output file path> ", "UEM to Target C Code Translator", options, "");
    	}
    	catch(CICXMLException e) {
    		e.printStackTrace();
    		System.out.println("Cannot load XML metadata information");
    	} catch (TemplateException e) {
			// TODO Auto-generated catch block
    		System.out.println("Error during parsing template");
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InvalidDataInMetadataFileException e) {
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
		CodeGenerator codeGenerator = new CodeGenerator(); 
		codeGenerator.initMetaData(args);
	}

}
