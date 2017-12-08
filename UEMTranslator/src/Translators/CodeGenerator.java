package Translators;

import java.io.File;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

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
    private UEMMetaDataModel mModel;
    
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
    	String[] leftArgs;
    	CommandLine cmd;
    	
    	options.addOption("help", false, "print this help");
    	HelpFormatter formatter = new HelpFormatter();
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
    		
    		if(cmd.hasOption("help")) 
    		{
        		formatter.printHelp("Translator.CodeGenerator [options] <Code generator binary path> <CIC XML file path> <Output file path> ", "UEM to Target C Code Translator", options, "");
    		}
    		
    		System.out.println("mTranslatorPath: " + mTranslatorPath + ", mCICXMLPath: " + mUEMXMLPath + ", mOutputPath: " + mOutputPath);
    		
    		mModel = new UEMMetaDataModel(mUEMXMLPath, mOutputPath + File.separator + Constants.SCHEDULE_FOLDER_NAME + File.separator);
    		
    		Configuration cfg = new Configuration(Configuration.VERSION_2_3_27);

    		cfg.setDirectoryForTemplateLoading(new File("templates"));

    		cfg.setDefaultEncoding("UTF-8");

    		cfg.setTemplateExceptionHandler(TemplateExceptionHandler.RETHROW_HANDLER);

    		cfg.setLogTemplateExceptions(false);

    		cfg.setWrapUncheckedExceptions(true);

    		Template temp = cfg.getTemplate("uem_data.ftl");

    		// Create the root hash. We use a Map here, but it could be a JavaBean too.
    		Map<String, Object> root = new HashMap<>();

    		// Put string "user" into the root
    		root.put("flat_task", mModel.getApplication().getTaskMap());
    		root.put("task_graph", mModel.getApplication().getTaskGraphMap());
    		root.put("channel_list", mModel.getApplication().getChannelList());
    		root.put("device_info", mModel.getApplication().getDeviceInfo());
    		root.put("mapping_info", mModel.getApplication().getMappingInfo());


    		Writer out = new OutputStreamWriter(System.out);
    		temp.process(root, out);
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
		}
    }

	public static void main(String[] args) 
	{
		// TODO Auto-generated method stub
		CodeGenerator codeGenerator = new CodeGenerator(); 
		codeGenerator.initMetaData(args);
		
		Template temp;

	}

}
