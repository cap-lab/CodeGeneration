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
import CommonLibraries.HopesInterface;
import CommonLibraries.Util;
import InnerDataStructures.*;
import InnerDataStructures.Queue;

public class CodeGenerator
{
	private BuildInnerDataStructures mBuilder = null;
	
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
	
	private String mFuncSimPeriodMetric;
	private int mFuncSimPeriod;
	private int mTotalControlQueue;
	
	private String mGraphType;
	private String mRuntimeExecutionPolicy;
	private String mCodeGenerationStyle;
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
		
		mFuncSimPeriodMetric = null;
		mFuncSimPeriod = 0;
		mTotalControlQueue = 0;	
		
		mGraphType = "ProcessNetwork";	// DataFlow, ProcessNetwork, Hybrid
		mRuntimeExecutionPolicy = HopesInterface.RuntimeExecutionPolicy_SelfTimed;	//style: a: thread per app, t: thread per task, p: thread per proc
		mCodeGenerationStyle = HopesInterface.CodeGenerationPolicy_FunctionCall; 
		mLanguage = "c";
	}
	
	public CodeGenerator(String cicxml, String outputpath, String rootpath){
		mTranslatorPath = "";
		mCICXMLFile = cicxml;
		mOutputPath = outputpath;
		mRootPath = rootpath;
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
			if (System.getProperty("os.name").contains("Windows")){
				mTranslatorPath = mTranslatorPath + "\\";
				mOutputPath = mOutputPath + "\\";
			}
			else{
				mTranslatorPath = mTranslatorPath + "/";
				mOutputPath = mOutputPath + "/";
			}
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
			System.out.println("--- " + mCICXMLFile + "_algorithm.xml");
			System.out.println(mAlgorithm);
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
//		mThreadVer = mConfiguration.getCodeGeneration().getThread();
		mRuntimeExecutionPolicy = mConfiguration.getCodeGeneration().getRuntimeExecutionPolicy();
		mCodeGenerationStyle = mConfiguration.getCodeGeneration().getThreadOrFunctioncall();
		
		mBuilder = new BuildInnerDataStructures();
		mProcessor = mBuilder.makeProcessors(mArchitecture);
		mCommunication = mBuilder.makeCommunications(mArchitecture);
		mTask = mBuilder.makeTasks(mOutputPath, mAlgorithm);
//		mBuilder.fillMappingForTask(mTask, mProcessor, mMapping, mOutputPath, mGraphType);
		mQueue = mBuilder.makeQueues(mAlgorithm, mTask);
		
		mTarget = mArchitecture.getTarget();
		
		copyTaskCode();
			
		mFuncSimPeriodMetric = mConfiguration.getSimulation().getExecutionTime().getMetric().value();
		mFuncSimPeriod = mConfiguration.getSimulation().getExecutionTime().getValue().intValue();
		
		//[CODE_REVIEW]: hshong(4/21):delete func.sim - need to merge 
		if(mTarget.toUpperCase().contains("THREAD")){ //functional simulation
			if(mGraphType.equals("DataFlow") || mGraphType.equals("Hybrid")){
				mConnectedTaskGraph = mBuilder.findConnectedTaskGraph(mTask);
				mConnectedSDFTaskSet = new HashMap<Integer, List<List<Task>>>();
				for(int i=0; i<mConnectedTaskGraph.size(); i++){
					List<Task> connected_task_graph = mConnectedTaskGraph.get(i);
					List<List<Task>> taskSet = null;
					taskSet = mBuilder.findSDFTaskSet(mTask, connected_task_graph);
					mConnectedSDFTaskSet.put(i, taskSet);
				}
			
				// Make virtual tasks for top-level sdf graphs
				mVTask = mBuilder.makeVirtualTask(mTask, mQueue, mConnectedSDFTaskSet, mAlgorithm.getProperty(), mFuncSimPeriod, mFuncSimPeriodMetric);
				mBuilder.modifyTaskStructure_runRate(mTask, mVTask, mQueue);
				mBuilder.fillMappingForTask(mTask, mVTask, mProcessor, mMapping, mOutputPath, mGraphType);
				mPVTask = new HashMap<String, Task>(); 
			}
			else if(mGraphType.equals("ProcessNetwork")){
				mTask = mBuilder.removeParentTask(mTask);
				mConnectedSDFTaskSet = new HashMap<Integer, List<List<Task>>>();
				mVTask = new HashMap<String, Task>(); 
				mBuilder.fillMappingForTask(mTask, mVTask, mProcessor, mMapping, mOutputPath, mGraphType);
				mPVTask = new HashMap<String, Task>(); 
			}
			else{
				System.out.println("Graph property is something wrong!");
				System.exit(-1);
			}
		}
		else{
			if(mGraphType.equals("DataFlow") || mGraphType.equals("Hybrid")){
				mConnectedTaskGraph = mBuilder.findConnectedTaskGraph(mTask);
				mConnectedSDFTaskSet = new HashMap<Integer, List<List<Task>>>();
				
				for(int i=0; i<mConnectedTaskGraph.size(); i++){
					List<Task> connected_task_graph = mConnectedTaskGraph.get(i);
					List<List<Task>> taskSet = null;
					taskSet = mBuilder.findSDFTaskSet(mTask, connected_task_graph);
					mConnectedSDFTaskSet.put(i, taskSet);
				}
				
				// Make virtual tasks for top-level sdf graphs
				mVTask = mBuilder.makeVirtualTask(mTask, mQueue, mConnectedSDFTaskSet, mAlgorithm.getProperty(), mFuncSimPeriod, mFuncSimPeriodMetric);
				mBuilder.modifyTaskStructure_runRate(mTask, mVTask, mQueue);
				mBuilder.fillMappingForTask(mTask, mVTask, mProcessor, mMapping, mOutputPath, mGraphType);
			}
			else if(mGraphType.equals("ProcessNetwork")){
				mTask = mBuilder.removeParentTask(mTask);
				mConnectedSDFTaskSet = new HashMap<Integer, List<List<Task>>>();
				mVTask = new HashMap<String, Task>(); 
				mBuilder.fillMappingForTask(mTask, mVTask, mProcessor, mMapping, mOutputPath, mGraphType);
			}
			else{
				System.out.println("Graph property is something wrong!");
				System.exit(-1);
			}
		}
	
		if(mProfile != null)	mBuilder.fillExecutionTimeInfo(mProfile, mTask);
		
		if(mControl != null){
			mBuilder.checkSlaveTask(mTask, mControl);
			mTotalControlQueue = mBuilder.setControlQueueIndex(mTask, mProcessor);
		}
		
		if(mAlgorithm.getLibraries() != null)	
			mLibrary = mBuilder.fillLibraryMapping(mAlgorithm, mMapping, mProcessor, mTask, mCICXMLFile, mOutputPath);

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
	
	public Map<String, Task> generateProcessorVirtualTask()
	{
		Map<String, Task> pVTask;
		
		if(mGraphType.equals("ProcessNetwork")){
			pVTask = new HashMap<String, Task>();
		}
		else if(mGraphType.equals("DataFlow") || mGraphType.equals("Hybrid")){
			if(mCodeGenerationStyle.equals(HopesInterface.CodeGenerationPolicy_FunctionCall))
				pVTask = mBuilder.addProcessorVirtualTask(mTask, mQueue, mProcessor, mConnectedTaskGraph, mConnectedSDFTaskSet, mAlgorithm.getProperty(), mVTask, mOutputPath);
			else
				pVTask = new HashMap<String, Task>();
			
			//need to delete
			if(mRuntimeExecutionPolicy.equals("Partitioned")){
				pVTask = mBuilder.addProcessorVirtualTask(mTask, mQueue, mProcessor, mConnectedTaskGraph, mConnectedSDFTaskSet, mAlgorithm.getProperty(), mVTask, mOutputPath);
			}
		}
		else
			pVTask = new HashMap<String, Task>();		
						
		return pVTask;
	}
	
	public String generateTargetCode()
	{
		if(mTarget.toUpperCase().contains("THREAD")) {
			//[CODE_REVIEW]: hshong(4/21): multi + makefile + debug 
			if(mTarget.contains("_C++"))				mLanguage = "c++";
			
			CommonLibraries.CIC.generateCommonCode(mOutputPath, mTranslatorPath);
						
			CICTargetCodeTranslator translator = new CICPthreadTranslator();
			try {
				translator.generateCode(mTarget, mTranslatorPath, mOutputPath, mRootPath, mProcessor, mTask, mQueue, mLibrary, mLibrary, mFuncSimPeriod, mFuncSimPeriodMetric, mCICXMLFile, mLanguage, mAlgorithm, mControl, mSchedule, mGpusetup, mMapping, mConnectedTaskGraph, mConnectedSDFTaskSet, mVTask, mPVTask, mGraphType, mRuntimeExecutionPolicy, mCodeGenerationStyle);
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		else if(mTarget.toUpperCase().contains("CELL")) {
			int ret = 0;
			
			CommonLibraries.CIC.generateCommonCode(mOutputPath, mTranslatorPath);
			
			CICTargetCodeTranslator translator = new CICCellTranslator();
			try {
				ret = translator.generateCode(mTarget, mTranslatorPath, mOutputPath, mRootPath, mProcessor, mTask, mQueue, mLibrary, mLibrary, mFuncSimPeriod, mFuncSimPeriodMetric, mCICXMLFile, mLanguage, mAlgorithm, mControl, mSchedule, mGpusetup, mMapping, mConnectedTaskGraph, mConnectedSDFTaskSet, mVTask, mPVTask, mGraphType, mRuntimeExecutionPolicy, mCodeGenerationStyle);
				if(ret == -1){
					CICTargetCodeTranslator translator_pthread = new CICPthreadTranslator();
					translator_pthread.generateCode(mTarget, mTranslatorPath, mOutputPath, mRootPath, mProcessor, mTask, mQueue, mLibrary, mLibrary, mFuncSimPeriod, mFuncSimPeriodMetric, mCICXMLFile, mLanguage, mAlgorithm, mControl, mSchedule, mGpusetup, mMapping, mConnectedTaskGraph, mConnectedSDFTaskSet, mVTask, mPVTask, mGraphType, mRuntimeExecutionPolicy, mCodeGenerationStyle);
				}
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		else if(mTarget.toUpperCase().contains("CUDA")) {
			int ret = 0;
			
			CommonLibraries.CIC.generateCommonCode(mOutputPath, mTranslatorPath);
			
			CICTargetCodeTranslator translator = new CICCudaTranslator();
			try {
				ret = translator.generateCode(mTarget, mTranslatorPath, mOutputPath, mRootPath, mProcessor, mTask, mQueue, mLibrary, mLibrary, mFuncSimPeriod, mFuncSimPeriodMetric, mCICXMLFile, mLanguage, mAlgorithm, mControl, mSchedule, mGpusetup, mMapping, mConnectedTaskGraph, mConnectedSDFTaskSet, mVTask, mPVTask, mGraphType, mRuntimeExecutionPolicy, mCodeGenerationStyle);
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		else if(mTarget.toUpperCase().contains("MULTICOREHOST")) {
			int ret = 0;
			
			mPVTask = generateProcessorVirtualTask();			
			CommonLibraries.CIC.generateCommonCode(mOutputPath, mTranslatorPath);
			
			CICTargetCodeTranslator translator = new CICMulticoreTranslator();
			try {
				ret = translator.generateCode(mTarget, mTranslatorPath, mOutputPath, mRootPath, mProcessor, mTask, mQueue, mLibrary, mLibrary, mFuncSimPeriod, mFuncSimPeriodMetric, mCICXMLFile, mLanguage, mAlgorithm, mControl, mSchedule, mGpusetup, mMapping, mConnectedTaskGraph, mConnectedSDFTaskSet, mVTask, mPVTask, mGraphType, mRuntimeExecutionPolicy, mCodeGenerationStyle);
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		else if(mTarget.toUpperCase().contains("HSIM")) {
			int ret = 0, index = 0;
			
			for(Processor proc: mProcessor.values()){
				if(proc.getSupportOS().contains("uC-OS")){
					CommonLibraries.CIC.generateCommonCode(mOutputPath, mTranslatorPath);
					CICHSimUcosTranslator translator = new CICHSimUcosTranslator();
					try {
						String t_mTarget = Integer.toString(index);
						String t_mOutputPath = mOutputPath + "/proc." + t_mTarget + "/";
						ret = translator.generateCode(t_mTarget, mTranslatorPath, t_mOutputPath, mRootPath, mProcessor, mTask, mQueue, mLibrary, mLibrary, mFuncSimPeriod, mFuncSimPeriodMetric, mCICXMLFile, mLanguage, mAlgorithm, mControl, mSchedule, mGpusetup, mMapping, mConnectedTaskGraph, mConnectedSDFTaskSet, mVTask, mPVTask, mGraphType, mRuntimeExecutionPolicy, mCodeGenerationStyle);
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
				ret = translator.generateCode(mTarget, mTranslatorPath, mOutputPath, mRootPath, mProcessor, mTask, mQueue, mLibrary, mLibrary, mFuncSimPeriod, mFuncSimPeriodMetric, mCICXMLFile, mLanguage, mAlgorithm, mControl, mSchedule, mGpusetup, mMapping, mConnectedTaskGraph, mConnectedSDFTaskSet, mVTask, mPVTask, mGraphType, mRuntimeExecutionPolicy, mCodeGenerationStyle);
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

