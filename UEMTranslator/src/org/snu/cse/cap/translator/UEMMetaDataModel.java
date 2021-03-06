package org.snu.cse.cap.translator;

import java.io.File;
import java.util.HashMap;

import org.snu.cse.cap.translator.structure.Application;
import org.snu.cse.cap.translator.structure.InvalidDataInMetadataFileException;
import org.snu.cse.cap.translator.structure.device.connection.InvalidDeviceConnectionException;
import org.snu.cse.cap.translator.structure.module.Module;

import hopes.cic.exception.CICXMLException;
import hopes.cic.xml.CICAlgorithmType;
import hopes.cic.xml.CICAlgorithmTypeLoader;
import hopes.cic.xml.CICArchitectureType;
import hopes.cic.xml.CICArchitectureTypeLoader;
import hopes.cic.xml.CICConfigurationType;
import hopes.cic.xml.CICConfigurationTypeLoader;
import hopes.cic.xml.CICControlType;
import hopes.cic.xml.CICControlTypeLoader;
import hopes.cic.xml.CICGPUSetupType;
import hopes.cic.xml.CICGPUSetupTypeLoader;
import hopes.cic.xml.CICMappingType;
import hopes.cic.xml.CICMappingTypeLoader;
import hopes.cic.xml.CICModuleType;
import hopes.cic.xml.CICModuleTypeLoader;
import hopes.cic.xml.CICProfileType;
import hopes.cic.xml.CICProfileTypeLoader;
import hopes.cic.xml.FileSourceType;
import hopes.cic.xml.SoftwareModuleType;

public class UEMMetaDataModel {
    private CICAlgorithmType algorithmMetadata = null;
    private CICArchitectureType architectureMetadata = null;
    private CICMappingType mappingMetadata = null;
    private CICControlType controlMetadata = null;
    private CICConfigurationType configurationMetadata = null;
    private CICProfileType profileMetadata = null;
    private CICGPUSetupType gpusetupMetadata = null;
    private String schedulePath = null;
    private HashMap<String, Module> moduleMap;
    //private CICScheduleType mSchedule = null;
    
    private Application application = null;
    
    public UEMMetaDataModel(String translatorRootDir, String uemXMLPath, String scheduleFileFolderPath) throws CICXMLException, InvalidDataInMetadataFileException, InvalidDeviceConnectionException, CloneNotSupportedException
    {
    	this.moduleMap = new HashMap<String, Module>();
    	this.schedulePath = scheduleFileFolderPath;
    	parseXMLFile(translatorRootDir, uemXMLPath);
    	makeApplicationDataModel();
    }
	
    private void parseXMLFile(String translatorRootDir, String uemXMLPath) throws CICXMLException
    {
    	CICModuleType moduleMetadata;
    	
        CICAlgorithmTypeLoader algorithmLoader = new CICAlgorithmTypeLoader();
        CICArchitectureTypeLoader architectureLoader = new CICArchitectureTypeLoader();
        CICMappingTypeLoader mappingLoader = new CICMappingTypeLoader();
        CICConfigurationTypeLoader configurationLoader = new CICConfigurationTypeLoader();
        CICControlTypeLoader controlLoader = new CICControlTypeLoader();
        CICProfileTypeLoader profileLoader = new CICProfileTypeLoader();
        CICGPUSetupTypeLoader gpusetupLoader = new CICGPUSetupTypeLoader();    
        CICModuleTypeLoader moduleLoader = new CICModuleTypeLoader();
        
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
        	
        	if(new File(uemXMLPath + Constants.UEMXML_GPUSETUP_PREFIX).isFile() == true)
        	{
        		gpusetupMetadata = gpusetupLoader.loadResource(uemXMLPath + Constants.UEMXML_GPUSETUP_PREFIX);
        	}
        	
        	if(new File(translatorRootDir + File.separator + Constants.DEFAULT_MODULE_XML_PATH).isFile() == true) 
        	{
        		moduleMetadata = moduleLoader.loadResource(translatorRootDir + File.separator + Constants.DEFAULT_MODULE_XML_PATH);
        		insertSupportedModuleInfo(moduleMetadata);
        	}
        }
        catch(CICXMLException e) {
        	System.out.println("XML Parse Error: " + e.getMessage());
        	e.printStackTrace();
        }
    }
    
    private void insertSupportedModuleInfo(CICModuleType moduleMetadata)
    {
    	for (SoftwareModuleType moduleXML : moduleMetadata.getModule())
    	{
    		Module module = new Module(moduleXML.getName(), moduleXML.getCflags(), moduleXML.getLdflags(), 
    									moduleXML.getInitializer(), moduleXML.getFinalizer());
    		
    		for(FileSourceType sourceFileName : moduleXML.getSources().getFile())
    		{
    			module.putSourceFile(sourceFileName.getName());
    		}
    		
    		for(FileSourceType headerFileName : moduleXML.getHeaders().getFile())
    		{
    			module.putHeaderFile(headerFileName.getName());
    		}
    		
    		this.moduleMap.put(moduleXML.getName(), module);
    	}
    }
    
    private void makeApplicationDataModel() throws InvalidDataInMetadataFileException, InvalidDeviceConnectionException, CloneNotSupportedException
    {
    	this.application = new Application();
    	
    	this.application.makeDeviceInformation(architectureMetadata, this.moduleMap);
    	
    	this.application.makeTaskInformation(algorithmMetadata);
    	this.application.makeMappingAndTaskInformationPerDevices(mappingMetadata, profileMetadata, configurationMetadata, this.schedulePath, gpusetupMetadata);
    	this.application.makeMulticastGroupInformation(algorithmMetadata);
    	this.application.makeChannelInformation(algorithmMetadata);
    	this.application.makeLibraryInformation(algorithmMetadata);
    	this.application.makeConnectionMappingInfo(mappingMetadata);
    	this.application.makeConfigurationInformation(configurationMetadata);
    }

	public Application getApplication() {
		return application;
	}
}
