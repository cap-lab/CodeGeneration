package Translators;

import java.io.*;
import java.util.*;

import hopes.cic.exception.CICXMLException;
import hopes.cic.xml.CICAlgorithmType;
import hopes.cic.xml.CICAlgorithmTypeLoader;
import hopes.cic.xml.CICArchitectureType;
import hopes.cic.xml.CICArchitectureTypeLoader;
import hopes.cic.xml.CICMappingType;
import hopes.cic.xml.CICMappingTypeLoader;
import hopes.cic.xml.CICConfigurationType;
import hopes.cic.xml.CICConfigurationTypeLoader;
import hopes.cic.xml.CICControlType;
import hopes.cic.xml.CICControlTypeLoader;
import hopes.cic.xml.CICProfileType;
import hopes.cic.xml.CICProfileTypeLoader;
import hopes.cic.xml.CICScheduleType;
import hopes.cic.xml.CICScheduleTypeLoader;
import hopes.cic.xml.CICDeviceIOType;
import hopes.cic.xml.CICGPUSetupType;
import hopes.cic.xml.CICGPUSetupTypeLoader;
import CommonLibraries.Util;
import InnerDataStructures.*;
import InnerDataStructures.Queue;

public class CodeGenerator
{
	private String[] mOptions;
	private String[] mArguments;
	private String mCICXMLFile;
	private String mOutputPath;
	private String mTranslatorPath;
	private String mRootPath;
	private CICAlgorithmType mAlgorithm;
	private CICArchitectureType mArchitecture;
	private CICMappingType mMapping;
	private CICControlType mControl;
	private CICConfigurationType mConfiguration;
	private CICProfileType mProfile;
	private CICGPUSetupType mGpusetup;
	private CICScheduleType mSchedule;
	private CICDeviceIOType mDeviceIO;
	private String mTarget;
	
	private Map<String, Task> mTask;
	private Map<String, Library> mLibrary;
	private Map<Integer, Queue> mQueue;
	private Map<Integer, Processor> mProcessor;
	private List<Communication> mCommunication;
	private Map<String, LibraryStub> mLibraryStub;
	
	private Map<Integer, List<Task>> mConnectedTaskGraph;
	private Map<Integer, List<List<Task>>> mConnectedSDFTaskSet;
	private Map<String, Task> mVTask;
	private Map<String, Task> mPVTask;
	
	private String mGlobalPeriodMetric;
	private int mGlobalPeriod;
	private int mTotalControlQueue;
	
	private String mGraphType;
	private String mThreadVer;
	private String mCodeGenType;
	private String mLanguage;
		
	CodeGenerator()
	{
		mOptions = null;
		mArguments = null;
		mCICXMLFile = null;
		mOutputPath = null;
		mTranslatorPath = null;
		mRootPath = null;
		mAlgorithm = null;
		mArchitecture = null;
		mMapping = null;
		mControl = null;
		mConfiguration = null;
		mProfile = null;
		mGpusetup = null;
		mSchedule = null;
		mTarget = null;
		
		mTask = null;
		mLibrary = null;
		mQueue = null;
		mProcessor = null;
		mLibraryStub = null;
		
		mConnectedTaskGraph = null;
		mConnectedSDFTaskSet = null;
		
		mGlobalPeriodMetric = null;
		mGlobalPeriod = 0;
		mTotalControlQueue = 0;	
		
		mGraphType = "";	// DataFlow, ProcessNetwork, Hybrid
		mThreadVer = "m";	// s: single thread, m: multi thread
		mCodeGenType = "p";	// a: thread per app, t: thread per task, p: thread per proc
		mLanguage = "c";
	}
	
	public CodeGenerator(String cicxml, String outputpath, String rootpath){
		mTranslatorPath = "";
		mCICXMLFile = cicxml;
		mOutputPath = outputpath;
		mRootPath = rootpath;
	}
	
	public void execute(){
		// Parse XML files
		System.out.print("Step #2 Parse XML ... ");
		String cicxmlfile = parseXMLFile();
		System.out.print("OK! (XML file: " + cicxmlfile + ".xml)\n");
		
		// Extended XML Generation
		System.out.print("Step #3 Extending XML ... ");
		//codeGenerator.extendXML();
		System.out.print("OK!\n");
		
		// Make inner data structure
		System.out.print("Step #4 Build inner data structures ... ");
		makeInnerDataStructures();
		System.out.print("OK!\n");
		
		// Target code generation
		System.out.print("Step #5 Generate Target Code ... ");
		String target = generateTargetCode();
		if(target != null)	System.out.print("OK! (Target: " + target + ")\n");
	}

	public void parseArguments(String[] theArguments)
	{
		int i,index=0;
		mArguments = new String[4];
		for(i=0; i<theArguments.length; i++) {
			if(theArguments[i].charAt(0)=='-') continue;
			else {
				mArguments[index++] = theArguments[i];
			}
		}

		if(index < 4) {
			System.out.print("Incorrect number of arguments\n");
			// error handling
			System.exit(-1);
		} 
		else if (index == 4){
			mTranslatorPath = mArguments[0];
			mCICXMLFile = mArguments[1];
			mOutputPath = mArguments[2];
			mRootPath = mArguments[3];
			
			mTranslatorPath = mTranslatorPath + "\\";
			mOutputPath = mOutputPath + "\\";
		}
	}

	public String parseXMLFile()
	{
		File f;
		
		CICAlgorithmTypeLoader algorithmLoader = new CICAlgorithmTypeLoader();
		CICArchitectureTypeLoader architectureLoader = new CICArchitectureTypeLoader();
		CICMappingTypeLoader mappingLoader = new CICMappingTypeLoader();
		CICConfigurationTypeLoader configurationLoader = new CICConfigurationTypeLoader();
		CICControlTypeLoader controlLoader = new CICControlTypeLoader();
		CICProfileTypeLoader profileLoader = new CICProfileTypeLoader();
		CICGPUSetupTypeLoader gpusetupLoader = new CICGPUSetupTypeLoader();
		CICScheduleTypeLoader scheduleLoader = new CICScheduleTypeLoader();
				
		try {
			mAlgorithm = algorithmLoader.loadResource(mCICXMLFile + "_algorithm.xml");
			mArchitecture = architectureLoader.loadResource(mCICXMLFile + "_architecture.xml");
			mMapping = mappingLoader.loadResource(mCICXMLFile + "_mapping.xml");
			mConfiguration = configurationLoader.loadResource(mCICXMLFile + "_configuration.xml");
			f = new File(mCICXMLFile + "_control.xml");
			if(f.exists())	mControl = controlLoader.loadResource(mCICXMLFile + "_control.xml");
			f = new File(mCICXMLFile + "_profile.xml");
			if(f.exists())	mProfile = profileLoader.loadResource(mCICXMLFile + "_profile.xml");
			f = new File(mCICXMLFile + "_gpusetup.xml");
			if(f.exists())	mGpusetup = gpusetupLoader.loadResource(mCICXMLFile + "_gpusetup.xml");
			f = new File("_schedule.xml");
			if(f.exists())	mSchedule = scheduleLoader.loadResource(mCICXMLFile + "_schedule.xml");
		} catch (CICXMLException e) {
			e.printStackTrace();
		}
		
		return mCICXMLFile;
	}
	
	public void makeInnerDataStructures(){
		mGraphType = mAlgorithm.getProperty();
		BuildInnerDataStructures builder = new BuildInnerDataStructures();
		mProcessor = builder.makeProcessors(mArchitecture);
		mCommunication = builder.makeCommunications(mArchitecture);
		mTask = builder.makeTasks(mOutputPath, mAlgorithm);
		builder.fillMappingForTask(mTask, mProcessor, mMapping, mOutputPath, mGraphType);
		//builder.fillTaskEntry(mProcessor,mTask);
		mQueue = builder.makeQueues(mAlgorithm, mTask);
		
		mTarget = mArchitecture.getTarget();
		
		copyTaskCode();
			
		mGlobalPeriodMetric = mConfiguration.getSimulation().getExecutionTime().getMetric().value();
		mGlobalPeriod = mConfiguration.getSimulation().getExecutionTime().getValue().intValue();
		
		if(mTarget.toUpperCase().contains("THREAD")){
			if(mGraphType.equals("DataFlow") || mGraphType.equals("Hybrid")){
				mConnectedTaskGraph = builder.findConnectedTaskGraph(mTask);
				mConnectedSDFTaskSet = new HashMap<Integer, List<List<Task>>>();
				//System.out.println(mConnectedTaskGraph);
				for(int i=0; i<mConnectedTaskGraph.size(); i++){
					List<Task> connected_task_graph = mConnectedTaskGraph.get(i);
					List<List<Task>> taskSet = null;
					taskSet = builder.findSDFTaskSet(mTask, connected_task_graph);
					mConnectedSDFTaskSet.put(i, taskSet);
				}
			
				// Make virtual tasks for top-level sdf graphs
				mVTask = builder.modifyTaskStructure(mTask, mQueue, mConnectedTaskGraph, mConnectedSDFTaskSet, mAlgorithm.getProperty(), mGlobalPeriod, mGlobalPeriodMetric);
				mPVTask = new HashMap<String, Task>(); 
			}
			else if(mGraphType.equals("ProcessNetwork")){
				mTask = builder.removeParentTask(mTask);
				mConnectedSDFTaskSet = new HashMap<Integer, List<List<Task>>>();
				mVTask = new HashMap<String, Task>(); 
				mPVTask = new HashMap<String, Task>(); 
			}
			else{
				System.out.println("Graph property is something wrong!");
				System.exit(-1);
			}
		}
		else{
			if(mGraphType.equals("DataFlow") || mGraphType.equals("Hybrid")){
				mConnectedTaskGraph = builder.findConnectedTaskGraph(mTask);
				mConnectedSDFTaskSet = new HashMap<Integer, List<List<Task>>>();
				//System.out.println(mConnectedTaskGraph);
				for(int i=0; i<mConnectedTaskGraph.size(); i++){
					List<Task> connected_task_graph = mConnectedTaskGraph.get(i);
					List<List<Task>> taskSet = null;
					taskSet = builder.findSDFTaskSet(mTask, connected_task_graph);
					mConnectedSDFTaskSet.put(i, taskSet);
				}
				//System.out.println(mConnectedSDFTaskSet);
				
				// Make virtual tasks for top-level sdf graphs
				mVTask = builder.modifyTaskStructure(mTask, mQueue, mConnectedTaskGraph, mConnectedSDFTaskSet, mAlgorithm.getProperty(), mGlobalPeriod, mGlobalPeriodMetric);
				if(mCodeGenType.equals("p")){
					mPVTask = builder.addProcessorVirtualTask(mTask, mQueue, mProcessor, mConnectedTaskGraph, mConnectedSDFTaskSet, mAlgorithm.getProperty(), mGlobalPeriod, mGlobalPeriodMetric, mVTask, mOutputPath);
				}
				else	mPVTask = new HashMap<String, Task>(); 
			}
			else if(mGraphType.equals("ProcessNetwork")){
				mTask = builder.removeParentTask(mTask);
				mConnectedSDFTaskSet = new HashMap<Integer, List<List<Task>>>();
				mVTask = new HashMap<String, Task>(); 
				mPVTask = new HashMap<String, Task>(); 
			}
			else{
				System.out.println("Graph property is something wrong!");
				System.exit(-1);
			}
		}
	
		//if(mProfile != null)	builder.fillExecutionTimeInfo(mProfile, mProcessor, mTask);
		
		if(mControl != null){
			builder.checkSlaveTask(mTask, mControl);
			mTotalControlQueue = builder.setControlQueueIndex(mTask, mProcessor);
		}
		
		// DeviceIO는 현재 구현하지 않은 상태 - 호근이와 논의후 결정
		
		if(mAlgorithm.getLibraries() != null)	mLibrary = builder.fillLibraryMapping(mAlgorithm, mMapping, mProcessor, mTask, mCICXMLFile, mOutputPath);

	}
	
	public void copyTaskCode(){
		// Make Output Directory
		File f = new File(mOutputPath);	
		
		if(!f.exists())	f.mkdir();
		
	    Util.copyExtensionFiles(mOutputPath,"./", ".h");
	    Util.copyExtensionFiles(mOutputPath,"./", ".c");
	    Util.copyExtensionFiles(mOutputPath,"./", ".cic");
	    Util.copyExtensionFiles(mOutputPath,"./", ".cicl");
	    Util.copyExtensionFiles(mOutputPath,"./", ".cicl.h");
	    Util.copyExtensionFiles(mOutputPath,"./", ".mtm");
	    Util.copyExtensionFiles(mOutputPath,"./", ".xml");
	    
	    for(Task t: mTask.values()){
	    	if(t.getCICFile() != null && t.getCICFile().endsWith(".xml")){
	    		int index = t.getCICFile().lastIndexOf("/");
	    		String path = t.getCICFile().substring(0, index);
	    		File n = new File(path);
	    		try {
					Util.copyAllFiles(f, n);
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
	    	}	
	    }
	}

	public String generateTargetCode()
	{
		if(mTarget.toUpperCase().contains("THREAD")) {
			
			if(mTarget.toUpperCase().contains("_S"))	mThreadVer = "s";
			if(mTarget.contains("_C++"))				mLanguage = "c++";
			
			CommonLibraries.CIC.generateCommonCode("Single", mOutputPath, mTranslatorPath, mTask, mQueue, mLibrary, mThreadVer, mAlgorithm, mControl);
						
			CICTargetCodeTranslator translator = new CICPthreadTranslator();
			try {
				translator.generateCode(mTarget, mTranslatorPath, mOutputPath, mRootPath, mProcessor, mTask, mQueue, mLibrary, mLibrary, mGlobalPeriod, mGlobalPeriodMetric, mCICXMLFile, mLanguage, mThreadVer, mAlgorithm, mControl, mSchedule, mGpusetup, mMapping, mConnectedTaskGraph, mConnectedSDFTaskSet, mVTask, mPVTask, mCodeGenType);
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		else if(mTarget.toUpperCase().contains("CELL")) {
			int ret = 0;
			
			CommonLibraries.CIC.generateCommonCode("Single", mOutputPath, mTranslatorPath, mTask, mQueue, mLibrary, mThreadVer, mAlgorithm, mControl);
			
			CICTargetCodeTranslator translator = new CICCellTranslator();
			try {
				ret = translator.generateCode(mTarget, mTranslatorPath, mOutputPath, mRootPath, mProcessor, mTask, mQueue, mLibrary, mLibrary, mGlobalPeriod, mGlobalPeriodMetric, mCICXMLFile, mLanguage, mThreadVer, mAlgorithm, mControl, mSchedule, mGpusetup, mMapping, mConnectedTaskGraph, mConnectedSDFTaskSet, mVTask, mPVTask, mCodeGenType);
				if(ret == -1){
					CICTargetCodeTranslator translator_pthread = new CICPthreadTranslator();
					translator_pthread.generateCode(mTarget, mTranslatorPath, mOutputPath, mRootPath, mProcessor, mTask, mQueue, mLibrary, mLibrary, mGlobalPeriod, mGlobalPeriodMetric, mCICXMLFile, mLanguage, mThreadVer, mAlgorithm, mControl, mSchedule, mGpusetup, mMapping, mConnectedTaskGraph, mConnectedSDFTaskSet, mVTask, mPVTask, mCodeGenType);
				}
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		else if(mTarget.toUpperCase().contains("CUDA")) {
			int ret = 0;
			
			CommonLibraries.CIC.generateCommonCode("Single", mOutputPath, mTranslatorPath, mTask, mQueue, mLibrary, mThreadVer, mAlgorithm, mControl);
			
			CICTargetCodeTranslator translator = new CICCudaTranslator();
			try {
				ret = translator.generateCode(mTarget, mTranslatorPath, mOutputPath, mRootPath, mProcessor, mTask, mQueue, mLibrary, mLibrary, mGlobalPeriod, mGlobalPeriodMetric, mCICXMLFile, mLanguage, mThreadVer, mAlgorithm, mControl, mSchedule, mGpusetup, mMapping, mConnectedTaskGraph, mConnectedSDFTaskSet, mVTask, mPVTask, mCodeGenType);
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		else if(mTarget.toUpperCase().contains("MULTICOREHOST")) {
			int ret = 0;
			
			CommonLibraries.CIC.generateCommonCode("Single", mOutputPath, mTranslatorPath, mTask, mQueue, mLibrary, mThreadVer, mAlgorithm, mControl);
			
			CICTargetCodeTranslator translator = new CICMulticoreTranslator();
			try {
				ret = translator.generateCode(mTarget, mTranslatorPath, mOutputPath, mRootPath, mProcessor, mTask, mQueue, mLibrary, mLibrary, mGlobalPeriod, mGlobalPeriodMetric, mCICXMLFile, mLanguage, mThreadVer, mAlgorithm, mControl, mSchedule, mGpusetup, mMapping, mConnectedTaskGraph, mConnectedSDFTaskSet, mVTask, mPVTask, mCodeGenType);
				if(ret == -1){
					CICTargetCodeTranslator translator_pthread = new CICPthreadTranslator();
					translator_pthread.generateCode(mTarget, mTranslatorPath, mOutputPath, mRootPath, mProcessor, mTask, mQueue, mLibrary, mLibrary, mGlobalPeriod, mGlobalPeriodMetric, mCICXMLFile, mLanguage, mThreadVer, mAlgorithm, mControl, mSchedule, mGpusetup, mMapping, mConnectedTaskGraph, mConnectedSDFTaskSet, mVTask, mPVTask, mCodeGenType);
				}
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		else if(mTarget.toUpperCase().contains("HSIM")) {
			int ret = 0, index = 0;
			
			for(Processor proc: mProcessor.values()){
				if(proc.getSupportOS().contains("uC-OS")){
					CommonLibraries.CIC.generateCommonCode("Single", mOutputPath, mTranslatorPath, mTask, mQueue, mLibrary, mThreadVer, mAlgorithm, mControl);
					CICHSimUcosTranslator translator = new CICHSimUcosTranslator();
					try {
						String t_mTarget = Integer.toString(index);
						String t_mOutputPath = mOutputPath + "/proc." + t_mTarget + "/";
						ret = translator.generateCode(t_mTarget, mTranslatorPath, t_mOutputPath, mRootPath, mProcessor, mTask, mQueue, mLibrary, mLibrary, mGlobalPeriod, mGlobalPeriodMetric, mCICXMLFile, mLanguage, mThreadVer, mAlgorithm, mControl, mSchedule, mGpusetup, mMapping, mConnectedTaskGraph, mConnectedSDFTaskSet, mVTask, mPVTask, mCodeGenType);
					} catch (FileNotFoundException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
				index++;
			}

			String fileOut = mOutputPath+"top.cpp";
			String templateFile = mTranslatorPath + "templates/target/hsim/hsim.template";
			CICHSimUcosTranslator.generateHSimCode(fileOut, templateFile, mProcessor);
			
			fileOut = mOutputPath+"Makefile";
			CICHSimUcosTranslator.generateGlobalMakefile(fileOut, mTranslatorPath +  "templates/target/hsim/Makefile.template", mProcessor);
			
			CICHSimUcosTranslator.copyOSandLibrary(mOutputPath, mTranslatorPath);
			CICHSimUcosTranslator.wrapup(mOutputPath);
		}

		else if(mTarget.toUpperCase().contains("XEONPHI")){
			int ret = 0;
			
			CICXeonPhiTranslator translator = new CICXeonPhiTranslator();
			try {
				ret = translator.generateCode(mTarget, mTranslatorPath, mOutputPath, mRootPath, mProcessor, mTask, mQueue, mLibrary, mLibrary, mGlobalPeriod, mGlobalPeriodMetric, mCICXMLFile, mLanguage, mThreadVer, mAlgorithm, mControl, mSchedule, mGpusetup, mMapping, mConnectedTaskGraph, mConnectedSDFTaskSet, mVTask, mPVTask, mCodeGenType);
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		return mTarget;
	}
	
	public static void main(String[] args)
	{
		// USAGE : %prog [options] translator_path input_cic_xml dest_dir root"
		CodeGenerator codeGenerator = new CodeGenerator();

		// Parse arguments
		System.out.print("Step #1 Parse Arg ... ");
		codeGenerator.parseArguments(args);
		System.out.print("OK!\n");

		// Parse XML files
		System.out.print("Step #2 Parse XML ... ");
		String cicxmlfile = codeGenerator.parseXMLFile();
		System.out.print("OK! (XML file: " + cicxmlfile + ".xml)\n");
		
		// Extended XML Generation
		System.out.print("Step #3 Extending XML ... ");
		//codeGenerator.extendXML();
		System.out.print("OK!\n");
		
		// Make inner data structure
		System.out.print("Step #4 Build inner data structures ... ");
		codeGenerator.makeInnerDataStructures();
		System.out.print("OK!\n");
		
		// Target code generation
		System.out.print("Step #5 Generate Target Code ... ");
		String target = codeGenerator.generateTargetCode();
		if(target != null)	System.out.print("OK! (Target: " + target + ")\n");
	}
}

