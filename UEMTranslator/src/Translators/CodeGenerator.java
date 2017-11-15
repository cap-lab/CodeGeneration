package Translators;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import hopes.cic.xml.CICAlgorithmType;
import hopes.cic.xml.CICArchitectureType;
import hopes.cic.xml.CICConfigurationType;
import hopes.cic.xml.CICControlType;
import hopes.cic.xml.CICMappingType;
import hopes.cic.xml.CICProfileType;
import hopes.cic.xml.CICScheduleType;


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
    private String mCICXMLPath;
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
    		mCICXMLPath = leftArgs[1];
    		mOutputPath = leftArgs[2];
    		
    		if(cmd.hasOption("help")) 
    		{
        		formatter.printHelp("Translator.CodeGenerator <options> <Code generator binary path> <CIC XML file path> <Output file path> ", "UEM to Target C Code Translator", options, "");
    		}
    		
    		System.out.println("mTranslatorPath: " + mTranslatorPath + ", mCICXMLPath: " + mCICXMLPath + ", mOutputPath: " + mOutputPath);
    	}
    	catch(ParseException e) {
    		System.out.println("ERROR: " + e.getMessage());
    		formatter.printHelp("Translator.CodeGenerator <merong>", "UEM to Target C Code Translator", options, "");
    	}
    
    }

	public static void main(String[] args) 
	{
		// TODO Auto-generated method stub
		CodeGenerator codeGenerator = new CodeGenerator();
		
		codeGenerator.parseArguments(args);

	}

}
