package org.snu.cse.cap.translator;

import java.io.File;

import org.snu.cse.cap.translator.structure.Application;
import org.snu.cse.cap.translator.structure.InvalidDataInMetadataFileException;

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

public class UEMMetaDataModel {
    private CICAlgorithmType algorithmMetadata = null;
    private CICArchitectureType architectureMetadata = null;
    private CICMappingType mappingMetadata = null;
    private CICControlType controlMetadata = null;
    private CICConfigurationType configurationMetadata = null;
    private CICProfileType profileMetadata = null;
    private String schedulePath = null;
    //private CICScheduleType mSchedule = null;
    
    private Application application = null;
    
    public UEMMetaDataModel(String uemXMLPath, String scheduleFileFolderPath) throws CICXMLException, InvalidDataInMetadataFileException
    {
    	parseXMLFile(uemXMLPath);
    	this.schedulePath = scheduleFileFolderPath;
    	makeApplicationDataModel();
    }
	
    private void parseXMLFile(String uemXMLPath) throws CICXMLException
    {
        CICAlgorithmTypeLoader algorithmLoader = new CICAlgorithmTypeLoader();
        CICArchitectureTypeLoader architectureLoader = new CICArchitectureTypeLoader();
        CICMappingTypeLoader mappingLoader = new CICMappingTypeLoader();
        CICConfigurationTypeLoader configurationLoader = new CICConfigurationTypeLoader();
        CICControlTypeLoader controlLoader = new CICControlTypeLoader();
        CICProfileTypeLoader profileLoader = new CICProfileTypeLoader();
        
        try {
        	// Mandatory XML Files
        	algorithmMetadata = algorithmLoader.loadResource(uemXMLPath + Constants.UEMXML_ALGORITHM_PREFIX);
        	architectureMetadata = architectureLoader.loadResource(uemXMLPath + Constants.UEMXML_ARCHITECTURE_PREFIX);
        	
        	// Optional XML files
        	if(new File(uemXMLPath + Constants.UEMXML_MAPPING_PREFIX).isFile() == true)
        	{
        		mappingMetadata = mappingLoader.loadResource(uemXMLPath + Constants.UEMXML_MAPPING_PREFIX);
        	}
        	
        	if(new File(uemXMLPath + Constants.UEMXML_CONFIGURATION_PREFIX).isFile() == true)
        	{
        		configurationMetadata = configurationLoader.loadResource(uemXMLPath + Constants.UEMXML_CONFIGURATION_PREFIX);
        	}
        	
        	if(new File(uemXMLPath + Constants.UEMXML_CONTROL_PREFIX).isFile() == true)
        	{
        		controlMetadata = controlLoader.loadResource(uemXMLPath + Constants.UEMXML_CONTROL_PREFIX);
        	}
        	
        	if(new File(uemXMLPath + Constants.UEMXML_PROFILE_PREFIX).isFile() == true)
        	{
        		profileMetadata = profileLoader.loadResource(uemXMLPath + Constants.UEMXML_PROFILE_PREFIX);
        	}
        }
        catch(CICXMLException e) {
        	System.out.println("XML Parse Error: " + e.getMessage());
        	e.printStackTrace();
        }
    }
    
    private void makeApplicationDataModel() throws InvalidDataInMetadataFileException
    {
    	this.application = new Application();
    	
    	this.application.makeDeviceInformation(architectureMetadata);
    	
    	this.application.makeTaskInformation(algorithmMetadata);
    	this.application.makeMappingAndTaskInformationPerDevices(mappingMetadata, profileMetadata, configurationMetadata, this.schedulePath);
    	this.application.makeChannelInformation(algorithmMetadata);
    	this.application.makeLibraryInformation(algorithmMetadata);
    	this.application.makeConfigurationInformation(configurationMetadata);
    }

	public Application getApplication() {
		return application;
	}
}
