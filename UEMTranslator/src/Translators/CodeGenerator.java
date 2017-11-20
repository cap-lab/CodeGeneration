package Translators;

import java.io.File;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import freemarker.template.Template;
import hopes.cic.exception.CICXMLException;
import hopes.cic.xml.CICAlgorithmType;
import hopes.cic.xml.CICAlgorithmTypeLoader;
import hopes.cic.xml.CICArchitectureType;
import hopes.cic.xml.CICArchitectureTypeLoader;
import hopes.cic.xml.CICConfigurationType;
import hopes.cic.xml.CICConfigurationTypeLoader;
import hopes.cic.xml.CICControlType;
import hopes.cic.xml.CICControlTypeLoader;
import hopes.cic.xml.CICMappingType;
import hopes.cic.xml.CICMappingTypeLoader;
import hopes.cic.xml.CICProfileType;
import hopes.cic.xml.CICProfileTypeLoader;
import hopes.cic.xml.CICScheduleType;
import hopes.cic.xml.CICScheduleTypeLoader;


public class CodeGenerator 
{
    private CICAlgorithmType mAlgorithm = null;
    private CICArchitectureType mArchitecture = null;
    private CICMappingType mMapping = null;
    private CICControlType mControl = null;
    private CICConfigurationType mConfiguration = null;
    private CICProfileType mProfile = null;
    private CICScheduleType mSchedule = null;
    private String mTranslatorPath;
    private String mUEMXMLPath;
    private String mOutputPath;

    
    public void parseArguments(String[] args) 
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
    		
    		if(cmd.hasOption("help")) 
    		{
        		formatter.printHelp("Translator.CodeGenerator [options] <Code generator binary path> <CIC XML file path> <Output file path> ", "UEM to Target C Code Translator", options, "");
    		}
    		
    		System.out.println("mTranslatorPath: " + mTranslatorPath + ", mCICXMLPath: " + mUEMXMLPath + ", mOutputPath: " + mOutputPath);
    	}
    	catch(ParseException e) {
    		System.out.println("ERROR: " + e.getMessage());
    		formatter.printHelp("Translator.CodeGenerator [options] <Code generator binary path> <CIC XML file path> <Output file path> ", "UEM to Target C Code Translator", options, "");
    	}
    
    }
    
    public void parseXMLFile()
    {
        CICAlgorithmTypeLoader algorithmLoader = new CICAlgorithmTypeLoader();
        CICArchitectureTypeLoader architectureLoader = new CICArchitectureTypeLoader();
        CICMappingTypeLoader mappingLoader = new CICMappingTypeLoader();
        CICConfigurationTypeLoader configurationLoader = new CICConfigurationTypeLoader();
        CICControlTypeLoader controlLoader = new CICControlTypeLoader();
        CICProfileTypeLoader profileLoader = new CICProfileTypeLoader();
        CICScheduleTypeLoader scheduleLoader = new CICScheduleTypeLoader();
        
        try {
        	// Mandatory XML Files
        	mAlgorithm = algorithmLoader.loadResource(mUEMXMLPath + Constants.UEMXML_ALGORITHM_PREFIX);
        	mArchitecture = architectureLoader.loadResource(mUEMXMLPath + Constants.UEMXML_ARCHITECTURE_PREFIX);
        	
        	// Optional XML files
        	if(new File(mUEMXMLPath + Constants.UEMXML_MAPPING_PREFIX).isFile() == true)
        	{
        		mMapping = mappingLoader.loadResource(mUEMXMLPath + Constants.UEMXML_MAPPING_PREFIX);
        	}
        	
        	if(new File(mUEMXMLPath + Constants.UEMXML_CONFIGURATION_PREFIX).isFile() == true)
        	{
        		mConfiguration = configurationLoader.loadResource(mUEMXMLPath + Constants.UEMXML_CONFIGURATION_PREFIX);
        	}
        	
        	if(new File(mUEMXMLPath + Constants.UEMXML_CONTROL_PREFIX).isFile() == true)
        	{
        		mControl = controlLoader.loadResource(mUEMXMLPath + Constants.UEMXML_CONTROL_PREFIX);
        	}
        	
        	if(new File(mUEMXMLPath + Constants.UEMXML_PROFILE_PREFIX).isFile() == true)
        	{
        		mProfile = profileLoader.loadResource(mUEMXMLPath + Constants.UEMXML_PROFILE_PREFIX);
        	}
        	
        	if(new File(mUEMXMLPath + Constants.UEMXML_SCHEDULE_PREFIX).isFile() == true)
        	{
        		mSchedule = scheduleLoader.loadResource(mUEMXMLPath + Constants.UEMXML_SCHEDULE_PREFIX);
        	}
        }
        catch(CICXMLException e) {
        	e.printStackTrace();
        }
    }

	public static void main(String[] args) 
	{
		// TODO Auto-generated method stub
		CodeGenerator codeGenerator = new CodeGenerator();
		
		codeGenerator.parseArguments(args);
		codeGenerator.parseXMLFile();
		
		Template temp;

	}

}
