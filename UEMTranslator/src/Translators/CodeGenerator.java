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

public class CodeGenerator 
{
    private String mTranslatorPath;
    private String mUEMXMLPath;
    private String mOutputPath;
    private UEMMetaDataModel mModel;
    
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
    		
    		if(cmd.hasOption("help")) 
    		{
        		formatter.printHelp("Translator.CodeGenerator [options] <Code generator binary path> <CIC XML file path> <Output file path> ", "UEM to Target C Code Translator", options, "");
    		}
    		
    		System.out.println("mTranslatorPath: " + mTranslatorPath + ", mCICXMLPath: " + mUEMXMLPath + ", mOutputPath: " + mOutputPath);
    		
    		mModel = new UEMMetaDataModel(mUEMXMLPath);
    	}
    	catch(ParseException e) {
    		System.out.println("ERROR: " + e.getMessage());
    		formatter.printHelp("Translator.CodeGenerator [options] <Code generator binary path> <CIC XML file path> <Output file path> ", "UEM to Target C Code Translator", options, "");
    	}
    	catch(CICXMLException e) {
    		e.printStackTrace();
    		System.out.println("Cannot load XML metadata information");
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
