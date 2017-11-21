package Translators;

import java.io.File;

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

public class UEMMetaDataModel {
    private CICAlgorithmType mAlgorithm = null;
    private CICArchitectureType mArchitecture = null;
    private CICMappingType mMapping = null;
    private CICControlType mControl = null;
    private CICConfigurationType mConfiguration = null;
    private CICProfileType mProfile = null;
    private CICScheduleType mSchedule = null;
    
    public UEMMetaDataModel(String mUEMXMLPath) throws CICXMLException
    {
    	parseXMLFile(mUEMXMLPath);
    }
	
    private void parseXMLFile(String mUEMXMLPath) throws CICXMLException
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
}
