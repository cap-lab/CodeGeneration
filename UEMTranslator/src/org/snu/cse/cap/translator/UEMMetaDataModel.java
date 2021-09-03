package org.snu.cse.cap.translator;

import java.io.File;
import java.util.HashMap;

import org.snu.cse.cap.translator.Constants.UEMXML;
import org.snu.cse.cap.translator.structure.Application;
import org.snu.cse.cap.translator.structure.InvalidDataInMetadataFileException;
import org.snu.cse.cap.translator.structure.device.connection.InvalidDeviceConnectionException;
import org.snu.cse.cap.translator.structure.module.Module;

import hopes.cic.exception.CICXMLException;
import hopes.cic.xml.*;

public class UEMMetaDataModel {
	private CICAlgorithmType algorithmMetadata;
	private CICArchitectureType architectureMetadata;
	private CICMappingType mappingMetadata;
	private CICConfigurationType configurationMetadata;
	private CICProfileType profileMetadata;
	private CICGPUSetupType gpusetupMetadata;
	private String schedulePath;
	private HashMap<String, Module> moduleMap;

	private Application application = null;

	public UEMMetaDataModel(String translatorRootDir, String uemXMLPath, String scheduleFileFolderPath)
			throws CICXMLException, InvalidDataInMetadataFileException, InvalidDeviceConnectionException,
			CloneNotSupportedException {
		this.moduleMap = new HashMap<String, Module>();
		this.schedulePath = scheduleFileFolderPath;
		parseXMLFile(translatorRootDir, uemXMLPath);
		makeApplicationDataModel();
	}

	private void parseXMLFile(String translatorRootDir, String uemXMLPath) throws CICXMLException {
		try {
			// Mandatory XML Files
			algorithmMetadata = UEMXML.ALGORITHM.load(uemXMLPath);
			architectureMetadata = UEMXML.ARCHITECTURE.load(uemXMLPath);

			// Optional XML files, can be null
			mappingMetadata = UEMXML.MAPPING.load(uemXMLPath);
			configurationMetadata = UEMXML.CONFIGURATION.load(uemXMLPath);
			profileMetadata = UEMXML.PROFILE.load(uemXMLPath);
			gpusetupMetadata = UEMXML.GPUSETUP.load(uemXMLPath);

			String defaultModuleXml = translatorRootDir + File.separator + Constants.DEFAULT_MODULE_XML_PATH;
			if (new File(defaultModuleXml).isFile()) {
				CICModuleType moduleMetadata = new CICModuleTypeLoader().loadResource(defaultModuleXml);
				insertSupportedModuleInfo(moduleMetadata);
			}
		} catch (CICXMLException e) {
			System.out.println("XML Parse Error: " + e.getMessage());
			e.printStackTrace();
		}
	}

	private void insertSupportedModuleInfo(CICModuleType moduleMetadata) {
		for (SoftwareModuleType moduleXML : moduleMetadata.getModule()) {
			Module module = new Module(moduleXML.getName(), moduleXML.getCflags(), moduleXML.getLdflags(),
					moduleXML.getInitializer(), moduleXML.getFinalizer());

			for (FileSourceType sourceFileName : moduleXML.getSources().getFile()) {
				module.putSourceFile(sourceFileName.getName());
			}

			for (FileSourceType headerFileName : moduleXML.getHeaders().getFile()) {
				module.putHeaderFile(headerFileName.getName());
			}

			this.moduleMap.put(moduleXML.getName(), module);
		}
	}

	private void makeApplicationDataModel()
			throws InvalidDataInMetadataFileException, InvalidDeviceConnectionException, CloneNotSupportedException {
		application = new Application();
		application.makeDeviceInformation(architectureMetadata, moduleMap);
		application.makeTaskInformation(algorithmMetadata);
		application.makeMappingAndTaskInformationPerDevices(mappingMetadata, profileMetadata, configurationMetadata,
				schedulePath, gpusetupMetadata);
		application.makeMulticastGroupInformation(algorithmMetadata);
		application.makeChannelInformation(algorithmMetadata);
		application.makeLibraryInformation(algorithmMetadata);
		application.makeConnectionMappingInfo(mappingMetadata);
		application.makeConfigurationInformation(configurationMetadata);
	}

	public Application getApplication() {
		return application;
	}
}
