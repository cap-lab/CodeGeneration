package Translators;

import java.io.*;
import java.math.*;
import java.util.*;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import mapss.dif.csdf.sdf.SDFEdgeWeight;
import mapss.dif.csdf.sdf.SDFGraph;
import mapss.dif.csdf.sdf.SDFNodeWeight;
import mocgraph.Edge;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import CommonLibraries.*;
import InnerDataStructures.*;
import InnerDataStructures.Communication.VRepSharedMemComm;
import InnerDataStructures.Communication.VRepSharedNode;
import InnerDataStructures.Library;
import InnerDataStructures.Queue;
import hopes.cic.xml.*;
import mapss.dif.csdf.sdf.SDFEdgeWeight;
import mapss.dif.csdf.sdf.SDFGraph;
import mapss.dif.csdf.sdf.SDFNodeWeight;
import mapss.dif.csdf.sdf.sched.APGANStrategy;
import mapss.dif.csdf.sdf.sched.DLCStrategy;
import mapss.dif.csdf.sdf.sched.FlatStrategy;
import mapss.dif.csdf.sdf.sched.MinBufferStrategy;
import mapss.dif.csdf.sdf.sched.ProcedureStrategy;
import mapss.dif.csdf.sdf.sched.TwoNodeStrategy;
import mocgraph.Edge;
import mocgraph.Node;
import mocgraph.sched.Firing;
import mocgraph.sched.Schedule;
import mocgraph.sched.ScheduleElement;

// * [hshong, 2015/03/20]: VRepSim 추가
public class CICVRepSimTranslator implements CICTargetCodeTranslator 
{	
	public static final int TYPE_BLUETOOTH = 0;
	public static final int TYPE_I2C = 1;
	public static final int TYPE_WIFI = 2;
	public static final int TYPE_VREPSHAREDBUS = 3;
	
	private String mTarget;
	private String mTranslatorPath;
	private String mOutputPath; 
	private String mRootPath;
	private String mCICXMLFile;
	private int mGlobalPeriod;
	private String mGlobalPeriodMetric;
	private String mThreadVer;
	private String mCodeGenType;
	private String mLanguage;
	
	private Map<String, Task> mTask;
	private Map<Integer, Queue> mQueue;
	private Map<String, Library> mLibrary;
	private Map<String, Library> mGlobalLibrary;
	private Map<Integer, Processor> mProcessor;
	private List<Communication> mCommunication;
	
	private Map<String, Task> mVTask;
	private Map<String, Task> mPVTask;
	
	private CICAlgorithmType mAlgorithm;
	private CICControlType mControl;
	private CICScheduleType mSchedule;
	private CICGPUSetupType mGpusetup;
	private CICMappingType mMapping;
	
	private ArrayList<Library> mLibraryStubList;
	private ArrayList<Library> mLibraryWrapperList;
	
	private Processor mMyProcessor;
	
	@Override
	public int generateCode(String target, String translatorPath, String outputPath, String rootPath, Map<Integer, Processor> processor, Map<String, Task> task, Map<Integer, Queue> queue, Map<String, Library> library, Map<String, Library> globalLibrary, int globalPeriod, String globalPeriodMetric, String cicxmlfile, String language, String threadVer, CICAlgorithmType algorithm, CICControlType control, CICScheduleType schedule, CICGPUSetupType gpusetup, CICMappingType mapping, Map<Integer, List<Task>> connectedtaskgraph,  Map<Integer, List<List<Task>>> connectedsdftaskset, Map<String, Task> vtask, Map<String, Task> pvtask, String codegentype) throws FileNotFoundException
	{	
		int ret = 0;
		mProcessor = processor;
		System.out.println("[V-REP] generateCode();");
		for(Processor proc: mProcessor.values())
		{
			mTarget = target;
			mTranslatorPath = translatorPath;
			mOutputPath = outputPath;
			mRootPath = rootPath;
			mCICXMLFile = cicxmlfile;
			mGlobalPeriod = globalPeriod;
			mGlobalPeriodMetric = globalPeriodMetric;
			mThreadVer = threadVer;
			mCodeGenType = codegentype;
			mLanguage = language;
			
			mTask = task;
			mQueue = queue;
			mLibrary = library;
			mGlobalLibrary = globalLibrary;
			mProcessor = processor;
			
			mVTask = vtask;
			mPVTask = pvtask;
			
			mAlgorithm = algorithm;
			mControl = control;
			mGpusetup = gpusetup;
			mMapping = mapping;
						
			if(proc.getPoolName().contains("BubbleRob"))
			{
				mMyProcessor = proc;
				
				Map<String, Task> mBubbleRobTask = new HashMap<String, Task>();
				Map<Integer, Queue> mBubbleRobQueue = new HashMap<Integer, Queue>();
				Map<String, Library> mBubbleRobLibrary= new HashMap<String, Library>();
				
				seperateDataStructure(proc.getPoolName(), mBubbleRobTask, mBubbleRobQueue, mBubbleRobLibrary);
				String mBubbleRobOutputPath = mOutputPath + proc.getPoolName() + "/";				
				File BubbleRobF = new File(mBubbleRobOutputPath);			
				BubbleRobF.mkdir();

				try {
					ret = generateRobotCode(mTarget, mTranslatorPath, mBubbleRobOutputPath, mRootPath, mProcessor, mMyProcessor, mBubbleRobTask, mBubbleRobQueue, mBubbleRobLibrary, mLibrary, mGlobalPeriod, mGlobalPeriodMetric, mCICXMLFile, mLanguage, mThreadVer, mAlgorithm, mControl, mSchedule, mGpusetup, mMapping, null, null, mVTask);
				} catch (FileNotFoundException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
			else if(proc.getPoolName().contains("Drone"))
			{
				mMyProcessor = proc;
				
				Map<String, Task> mDroneTask = new HashMap<String, Task>();
				Map<Integer, Queue> mDroneQueue = new HashMap<Integer, Queue>();
				Map<String, Library> mDroneLibrary= new HashMap<String, Library>();
				
				seperateDataStructure(proc.getPoolName(), mDroneTask, mDroneQueue, mDroneLibrary);
				String mDroneOutputPath = mOutputPath + proc.getPoolName() + "/";				
				File BubbleRobF = new File(mDroneOutputPath);			
				BubbleRobF.mkdir();

				try {
					ret = generateRobotCode(mTarget, mTranslatorPath, mDroneOutputPath, mRootPath, mProcessor, mMyProcessor, mDroneTask, mDroneQueue, mDroneLibrary, mLibrary, mGlobalPeriod, mGlobalPeriodMetric, mCICXMLFile, mLanguage, mThreadVer, mAlgorithm, mControl, mSchedule, mGpusetup, mMapping, null, null, mVTask);
				} catch (FileNotFoundException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
			else if(proc.getPoolName().contains("4LegWalker"))
			{
				mMyProcessor = proc;
				
				Map<String, Task> mDroneTask = new HashMap<String, Task>();
				Map<Integer, Queue> mDroneQueue = new HashMap<Integer, Queue>();
				Map<String, Library> mDroneLibrary= new HashMap<String, Library>();
				
				seperateDataStructure(proc.getPoolName(), mDroneTask, mDroneQueue, mDroneLibrary);
				String mDroneOutputPath = mOutputPath + proc.getPoolName() + "/";				
				File BubbleRobF = new File(mDroneOutputPath);			
				BubbleRobF.mkdir();

				try {
					ret = generateRobotCode(mTarget, mTranslatorPath, mDroneOutputPath, mRootPath, mProcessor, mMyProcessor, mDroneTask, mDroneQueue, mDroneLibrary, mLibrary, mGlobalPeriod, mGlobalPeriodMetric, mCICXMLFile, mLanguage, mThreadVer, mAlgorithm, mControl, mSchedule, mGpusetup, mMapping, null, null, mVTask);
				} catch (FileNotFoundException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
	    
	    return 0;
	}
	
	public void seperateDataStructure(String target, Map<String, Task> mTargetTask, Map<Integer, Queue> mTargetQueue, Map<String, Library> mTargetLibrary){
		for(Task t: mTargetTask.values()){
			Map<String, Map<String, List<Integer>>> plmapmap = t.getProc();
			for(Map<String, List<Integer>> plmap: plmapmap.values()){
				for(List<Integer> pl: plmap.values()){
					for(int p: pl){
						if(mProcessor.get(p).getPoolName().contains(target)){
							mTargetTask.put(t.getName(), t);
						}
					}
				}
			}
		}
		/*
		for(Processor proc: mProcessor.values()){
			List<Task> taskList = proc.getTask();
			if(proc.getPoolName().contains(target)){
				int index = 0;
				for(Task t: taskList){
					t.setIndex(index++);
					mTargetTask.put(t.getName(), t);
				}
			}
		}
		*/
		if(mLibrary != null){
			for(Library lib: mLibrary.values()){
				if(mProcessor.get(lib.getProc()).getPoolName().contains(target))			
					mTargetLibrary.put(lib.getName(), lib);
			}
		}
		
		for(Queue q: mQueue.values()){
			if(mTargetTask.containsKey(q.getSrc()) || mTargetTask.containsKey(q.getDst()))	
			{
				mTargetQueue.put(Integer.parseInt(q.getIndex()), q);
			}
		}
	}
	
	public int generateRobotCode(String target, String translatorPath, String outputPath, String rootPath, Map<Integer, Processor> processor, Processor myProcessor, Map<String, Task> task, Map<Integer, Queue> queue, Map<String, Library> library, Map<String, Library> globalLibrary, int globalPeriod, String globalPeriodMetric, String cicxmlfile, String language, String threadVer, CICAlgorithmType algorithm, CICControlType control, CICScheduleType schedule, CICGPUSetupType gpusetup, CICMappingType mapping, Map<Integer, List<Task>> connectedtaskgraph,  Map<Integer, List<List<Task>>> connectedsdftaskset, Map<String, Task> vtask) throws FileNotFoundException
	{
		mTarget = target;
		mTranslatorPath = translatorPath;
		mOutputPath = outputPath;
		mRootPath = rootPath;
		mCICXMLFile = cicxmlfile;
		mGlobalPeriod = globalPeriod;
		mGlobalPeriodMetric = globalPeriodMetric;
		mThreadVer = threadVer;
		mLanguage = language;
		
		mTask = task;
		mQueue = queue;
		mLibrary = library;
		mGlobalLibrary = globalLibrary;
		mProcessor = processor;
		
		mVTask = vtask;
		
		mAlgorithm = algorithm;
		mControl = control;
		mGpusetup = gpusetup;
		mMapping = mapping;
		
		mLibraryStubList = new ArrayList<Library>();
		mLibraryWrapperList = new ArrayList<Library>();
		
		mMyProcessor = myProcessor;
		
		CommonLibraries.CIC.generateCommonCode("Single", mOutputPath, mTranslatorPath, mTask, mQueue, mLibrary, mThreadVer, mAlgorithm, mControl);
		
		Util.copyFile(mOutputPath+"target_task_model.h", mTranslatorPath + "templates/common/task_model/pthread.template");
		Util.copyFile(mOutputPath+"target_system_model.h", mTranslatorPath + "templates/common/system_model/general_linux.template");
		Util.copyFile(mOutputPath+"CIC_robot.h", mTranslatorPath + "templates/target/VRepSim/CIC_robot.h");
		
		//target 마다 생성해줘야 하는 library 추가
		/*
		if(mMyProcessor.getPoolName().contains("4LegWalker"))
		{
			Util.copyFile(mOutputPath+"target_library.h", mTranslatorPath + "templates/target/VRepSim/4LegWalker/target_library.h");			
		}
		*/
				
		
		String fileOut = null;	
		String templateFile = "";

		// generate task_def.h
		fileOut = mOutputPath + "task_def.h";
		templateFile = mTranslatorPath + "templates/common/common_template/task_def.h.template";
		CommonLibraries.CIC.generateTaskDataStructure(fileOut, templateFile, mTask, mGlobalPeriod, mGlobalPeriodMetric, mThreadVer, mCodeGenType, mVTask, mPVTask);
		
		String srcExtension = "";
		//this target consider only cpp... 
		/* test for 0617
		if(mLanguage.equals("c++"))	srcExtension = ".cpp";
		else						srcExtension = ".c";
		*/
		srcExtension = ".cpp";
		
		// generate task_name.c (include task_name.cic)
		// this target does not consider multi-mode(mtm)
		for(Task t: mTask.values()){
			fileOut = mOutputPath + t.getName() + srcExtension;			
			templateFile = mTranslatorPath + "templates/target/VRepSim/task_code_template.c";
			
			generateTaskCode(fileOut, templateFile, t, mAlgorithm);
		}
					
		// generate target_specific.h : to register Handle 
		fileOut = mOutputPath + "target_specific.h";
		templateFile = mTranslatorPath + "templates/target/VRepSim/target_specific_template.h";
		generateTargetSpecificHeader(fileOut, templateFile, mProcessor);
		
		//generate childscript.lua
		generateChildScript(mProcessor);
		
		//generate readme.txt
		Util.copyFile(mOutputPath+"readme.txt", mTranslatorPath + "templates/target/VRepSim/readme_template");
				
		//generateLibrary
		generateLibraryCode();
		
		//generateComm
		generateCommCode();
		
		copyTaskFilesToTargetFolder();
		
		// generate proc.c or proc.cpp
		fileOut = mOutputPath+"proc" + srcExtension;
		if(mThreadVer.equals("s"))
			templateFile = mTranslatorPath + "templates/common/common_template/proc.c.template";	// NEED TO FIX
		else
			templateFile = mTranslatorPath + "templates/common/common_template/proc.c.template";
		
		generateProcCode(fileOut, templateFile);
	
	    // generate Makefile
	    fileOut = mOutputPath + "Makefile";
	    generateMakefile(fileOut);
		
		return 0;
	}
	public void generateTaskCode(String mDestFile, String mTemplateFile, Task task, CICAlgorithmType mAlgorithm)
	{
		File fileOut = new File(mDestFile);
		File templateFile = new File(mTemplateFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);
			
			outstream.write(translateTaskCode(content, task, mAlgorithm).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public String translateTaskCode(String mContent, Task task, CICAlgorithmType mAlgorithm)
	{
		String code = mContent;
		String libraryDef="";
		String sysportDef="";
		String cicInclude="";
		String includeHeaders="";
		
		if(mAlgorithm.getLibraries() != null){
			libraryDef += "#define LIBCALL(x, ...) LIBCALL_##x(__VA_ARGS__)\n";
			for(TaskLibraryConnectionType taskLibCon: mAlgorithm.getLibraryConnections().getTaskLibraryConnection()){
				if(taskLibCon.getMasterTask().equals(task.getName())){
					libraryDef += "\n#include \"" + taskLibCon.getSlaveLibrary()+".h\"\n";
					libraryDef += "#define LIBCALL_" + taskLibCon.getMasterPort() + "(f, ...) l_" + taskLibCon.getSlaveLibrary() + "_##f(__VA_ARGS__)\n";
				}
			}
		}
		
		sysportDef = "#define SYS_REQ(x, ...) SYS_REQ_##x(__VA_ARGS__)\n ";
						
		includeHeaders += "#include \"CIC_port.h\"\n";
		includeHeaders += "#include \"target_specific.h\"\n";
		includeHeaders += "#include \"CIC_robot.h\"\n";
		
		cicInclude = "#include \"" + task.getCICFile() + "\"\n";
		
		code = code.replace("##LIBRARY_DEFINITION", libraryDef);
		code = code.replace("##INCLUDE_HEADERS", includeHeaders);	    
	    code = code.replace("##SYSPORT_DEFINITION", sysportDef);
	    code = code.replace("##CIC_INCLUDE", cicInclude);
	    code = code.replace("##TASK_NAME", task.getName());
		
		return code;
	}
	
	public void generateTargetSpecificHeader(String mDestFile, String mTemplateFile, Map<Integer, Processor> mProcessor)
	{
		File fileOut = new File(mDestFile);
		File templateFile = new File(mTemplateFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);
			
			outstream.write(translateTargetSpecificHeader(content, mProcessor).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public String translateTargetSpecificHeader(String mContent, Map<Integer, Processor> mProcessor)
	{
		String code = mContent;
		String templateFile;
		String specificHandler = "";
		String specificDefineNum = "";
		
		if(mMyProcessor.getPoolName().contains("BubbleRob"))
		{
			//TARGET_SPECIFIC_HANDLER //
			templateFile = mTranslatorPath + "templates/target/VRepSim/BubbleRob/target_specific_template";
			specificHandler = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TARGET_SPECIFIC_HANDLER");
			specificDefineNum = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TARGET_DEPENDENT_SPECIFIC_DEFINE_NUM");	
			
		}
		else if(mMyProcessor.getPoolName().contains("Drone"))
		{
			//TARGET_SPECIFIC_HANDLER //
			templateFile = mTranslatorPath + "templates/target/VRepSim/Drone/target_specific_template";
			specificHandler = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TARGET_SPECIFIC_HANDLER");
			specificDefineNum = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TARGET_DEPENDENT_SPECIFIC_DEFINE_NUM");			
		}
		else if(mMyProcessor.getPoolName().contains("4LegWalker"))
		{
			//TARGET_SPECIFIC_HANDLER //
			templateFile = mTranslatorPath + "templates/target/VRepSim/4LegWalker/target_specific_template";
			specificHandler = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TARGET_SPECIFIC_HANDLER");
			specificDefineNum = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TARGET_DEPENDENT_SPECIFIC_DEFINE_NUM");			
		}
		
		code = code.replace("##TARGET_SPECIFIC_HANDLER", specificHandler);
		code = code.replace("##TARGET_DEPENDENT_SPECIFIC_DEFINE_NUM", specificDefineNum);
		
		
		return code;
	}
	
	public void generateChildScript(Map<Integer, Processor> mProcessor)
	{
		String fileOut = null;	
		String templateFile = null;  
		String subTemplateFile = "";
				
		if(mMyProcessor.getPoolName().contains("BubbleRob"))
		{
			templateFile = mTranslatorPath + "templates/target/VRepSim/BubbleRob/childscript_template.lua";
			fileOut = mOutputPath+"BubbleRob.lua";
			subTemplateFile = mTranslatorPath + "templates/target/VRepSim/BubbleRob/target_specific_template";
			generateBubbleRobScript(fileOut, templateFile, subTemplateFile);
		}
		else if(mMyProcessor.getPoolName().contains("Drone"))
		{
			//need to fix! 
			templateFile = mTranslatorPath + "templates/target/VRepSim/Drone/childscript_template.lua";
			fileOut = mOutputPath+"Drone.lua";
			subTemplateFile = mTranslatorPath + "templates/target/VRepSim/Drone/target_specific_template";
			generateDroneScript(fileOut, templateFile, subTemplateFile);
		}
		else if(mMyProcessor.getPoolName().contains("4LegWalker"))
		{ 
			templateFile = mTranslatorPath + "templates/target/VRepSim/4LegWalker/childscript_template.lua";
			fileOut = mOutputPath+"4LegWalker.lua";
			subTemplateFile = mTranslatorPath + "templates/target/VRepSim/4LegWalker/target_specific_template";
			generate4LegWalkerScript(fileOut, templateFile, subTemplateFile);
		}
		
	}
	
	public void generateBubbleRobScript(String mDestFile, String mTemplateFile, String mSubTemplateFile)
	{
		File fileOut = new File(mDestFile);
		File templateFile = new File(mTemplateFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);
			
			outstream.write(generateBubbleRobScript(content, mSubTemplateFile).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	// 로봇의 종류가 다르더라도 기본적으로 child script는 같지 않을까 싶어 일단 이렇게 만들었다. 추후에 script 형태가 바뀔 경우 BubbleRob의 script가 될 것임 
	public String generateBubbleRobScript(String mContent, String mTemplateFile)
	{
		String code = mContent;
		//String templateFile = mTranslatorPath + "templates/target/VRepSim/BubbleRob/target_specific_template";
		String getHandleMotorSensor = "";
		String remoteApiStartWPort = ""; 
		String launchClientApp = "";
		String displayErrorMsg = "";
				
		getHandleMotorSensor = CommonLibraries.Util.getCodeFromTemplate(mTemplateFile, "##GET_HANDLE_MOTOR_SENSOR");
		remoteApiStartWPort = CommonLibraries.Util.getCodeFromTemplate(mTemplateFile, "##REMOTE_API_START_W_PORT");		
		launchClientApp = CommonLibraries.Util.getCodeFromTemplate(mTemplateFile, "##LAUNCH_CLIENT_APP");
		displayErrorMsg = CommonLibraries.Util.getCodeFromTemplate(mTemplateFile, "##DISPLAY_ERROR_MESSAGE");
				
		code = code.replace("##GET_HANDLE_MOTOR_SENSOR", getHandleMotorSensor);
		code = code.replace("##REMOTE_API_START_W_PORT", remoteApiStartWPort);
		code = code.replace("##LAUNCH_CLIENT_APP", launchClientApp);
		code = code.replace("##DISPLAY_ERROR_MESSAGE", displayErrorMsg);
		
		return code;
	}
	
	public void generateDroneScript(String mDestFile, String mTemplateFile, String mSubTemplateFile)
	{
		File fileOut = new File(mDestFile);
		File templateFile = new File(mTemplateFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);
			
			outstream.write(generateDroneScript(content, mSubTemplateFile).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	// 로봇의 종류가 다르더라도 기본적으로 child script는 같지 않을까 싶어 일단 이렇게 만들었다. 추후에 script 형태가 바뀔 경우 BubbleRob의 script가 될 것임 
	public String generateDroneScript(String mContent, String mTemplateFile)
	{
		String code = mContent;
		//String templateFile = mTranslatorPath + "templates/target/VRepSim/Drone/target_specific_template";
		String getHandleMotorSensor = "";
		String remoteApiStartWPort = ""; 
		String launchClientApp = "";
		String displayErrorMsg = "";
		String handleCmdFromProc = "";
				
		getHandleMotorSensor = CommonLibraries.Util.getCodeFromTemplate(mTemplateFile, "##GET_HANDLE_MOTOR_SENSOR");
		remoteApiStartWPort = CommonLibraries.Util.getCodeFromTemplate(mTemplateFile, "##REMOTE_API_START_W_PORT");		
		launchClientApp = CommonLibraries.Util.getCodeFromTemplate(mTemplateFile, "##LAUNCH_CLIENT_APP");
		displayErrorMsg = CommonLibraries.Util.getCodeFromTemplate(mTemplateFile, "##DISPLAY_ERROR_MESSAGE");
		handleCmdFromProc = CommonLibraries.Util.getCodeFromTemplate(mTemplateFile, "##HANDLE_CMD_FROM_PROC");
						
		code = code.replace("##GET_HANDLE_MOTOR_SENSOR", getHandleMotorSensor);
		code = code.replace("##REMOTE_API_START_W_PORT", remoteApiStartWPort);
		code = code.replace("##LAUNCH_CLIENT_APP", launchClientApp);
		code = code.replace("##DISPLAY_ERROR_MESSAGE", displayErrorMsg);
		code = code.replace("##HANDLE_CMD_FROM_PROC", handleCmdFromProc);
		
		return code;
	}
	
	public void generate4LegWalkerScript(String mDestFile, String mTemplateFile, String mSubTemplateFile)
	{
		File fileOut = new File(mDestFile);
		File templateFile = new File(mTemplateFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);
			
			outstream.write(generate4LegWalkerScript(content, mSubTemplateFile).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	// 로봇의 종류가 다르더라도 기본적으로 child script는 같지 않을까 싶어 일단 이렇게 만들었다. 추후에 script 형태가 바뀔 경우 BubbleRob의 script가 될 것임 
	public String generate4LegWalkerScript(String mContent, String mTemplateFile)
	{
		String code = mContent;
		//String templateFile = mTranslatorPath + "templates/target/VRepSim/4LegWalker/target_specific_template";
		String getHandleMotorSensor = "";
		String remoteApiStartWPort = ""; 
		String launchClientApp = "";
		String displayErrorMsg = "";
				
		getHandleMotorSensor = CommonLibraries.Util.getCodeFromTemplate(mTemplateFile, "##GET_HANDLE_MOTOR_SENSOR");
		remoteApiStartWPort = CommonLibraries.Util.getCodeFromTemplate(mTemplateFile, "##REMOTE_API_START_W_PORT");		
		launchClientApp = CommonLibraries.Util.getCodeFromTemplate(mTemplateFile, "##LAUNCH_CLIENT_APP");
		displayErrorMsg = CommonLibraries.Util.getCodeFromTemplate(mTemplateFile, "##DISPLAY_ERROR_MESSAGE");
				
		code = code.replace("##GET_HANDLE_MOTOR_SENSOR", getHandleMotorSensor);
		code = code.replace("##REMOTE_API_START_W_PORT", remoteApiStartWPort);
		code = code.replace("##LAUNCH_CLIENT_APP", launchClientApp);
		code = code.replace("##DISPLAY_ERROR_MESSAGE", displayErrorMsg);
		
		return code;
	}
	
	
	public void generateLibraryCode()
	{
		//Libraries are mapped on the same target proc
		for(Library library: mLibrary.values())
		{
			int procId = library.getProc();
			Processor proc = mProcessor.get(procId);
			if(mMyProcessor.getPoolName().equals(proc.getPoolName()))
			{
				boolean hasRemoteConn = false;
				for(int i=0; i<mAlgorithm.getLibraryConnections().getTaskLibraryConnection().size(); i++){
					TaskLibraryConnectionType taskLibCon = mAlgorithm.getLibraryConnections().getTaskLibraryConnection().get(i);
					if(!mTask.containsKey(taskLibCon.getMasterTask())){
						hasRemoteConn = true;
						break;
					}
				}
				CommonLibraries.Library.generateLibraryCode(mOutputPath, library, mAlgorithm);
				if(hasRemoteConn)
				{	
					//바뀌어야 할 듯 
					CommonLibraries.Library.generateLibraryWrapperCode(mOutputPath, library, mAlgorithm);
					
					mLibraryWrapperList.add(library);
				}
			}
		}
		
		// Libraries are mapped on other target procs
		for(Task t: mTask.values())
		{
			String taskName = t.getName();
			String libPortName = "";
			String libName = "";
			Library library = null;
			if(t.getLibraryPortList().size() != 0)
			{
				List<LibraryMasterPortType> libportList = t.getLibraryPortList();
				for(int i=0; i<libportList.size(); i++)
				{
					LibraryMasterPortType libPort = libportList.get(i);
					libPortName = libPort.getName();
					break;
				}
				if(libPortName == "")
				{ 
					System.out.println("Library task does not exist!");
					System.exit(-1);
				}
				else
				{
					for(int i=0; i<mAlgorithm.getLibraryConnections().getTaskLibraryConnection().size(); i++)
					{
						TaskLibraryConnectionType taskLibCon = mAlgorithm.getLibraryConnections().getTaskLibraryConnection().get(i);
						if(taskLibCon.getMasterTask().equals(taskName) && taskLibCon.getMasterPort().equals(libPortName))
						{
							libName = taskLibCon.getSlaveLibrary();
							break;
						}
					}
					if(!mLibrary.containsKey(libName))
					{
						for(Library lib: mGlobalLibrary.values())
						{
							if(lib.getName().equals(libName))
							{
								library = lib;
								break;
							}
						}
						if(library != null)
						{
							Util.copyFile(mOutputPath + "/"+ library.getHeader(), mOutputPath + "/../" + library.getHeader());
							//바꿔야 할 듯
							CommonLibraries.Library.generateLibraryStubCode(mOutputPath, library, mAlgorithm, false);
							mLibraryStubList.add(library);
						}
					}
				}
			}
		}
		
		if(mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0 )
		{
			//System.out.println("stub: " + mLibraryStubList.size() + ", wrapper: " + mLibraryWrapperList.size() + ", lib: " + mLibrary.size()+"\n");
			String fileOut = mOutputPath + "lib_channels.h";
			//template 바뀜 
			String templateFile = mTranslatorPath + "templates/target/VRepSim/lib_channels.h";
			CommonLibraries.Library.generateLibraryChannelHeader(fileOut, templateFile, mLibraryStubList, mLibraryWrapperList);
	
			fileOut = mOutputPath + "libchannel_def.h";
			templateFile = mTranslatorPath + "templates/common/library/libchannel_def.h.template";
			CommonLibraries.Library.generateLibraryChannelDef(fileOut, templateFile, mLibraryStubList, mLibraryWrapperList);
		}
		if(mLibraryWrapperList.size() > 0 )
		{
			String fileOut = mOutputPath + "lib_wrapper.h";
			String templateFile = mTranslatorPath + "templates/common/library/lib_wrapper.h";
			CommonLibraries.Util.copyFile(fileOut, templateFile);
	
			fileOut = mOutputPath + "libwrapper_def.h";
			templateFile = mTranslatorPath + "templates/common/library/libwrapper_def.h.template";
			CommonLibraries.Library.generateLibraryWrapperDef(fileOut, templateFile, mLibraryWrapperList, null);
		}
	}	

	public void generateCommCode()
	{		
		if(mCommunication.size() > 0)
		{
			String fileOut = mOutputPath + "cic_conmap.h";
			String templateFile = mTranslatorPath + "templates/common/common_template/cic_conmap.h";
			CommonLibraries.Util.copyFile(fileOut, templateFile);
	
			fileOut = mOutputPath + "conmap_def.h";
			templateFile = mTranslatorPath + "templates/common/common_template/conmap_def.h.template";
			CommonLibraries.OutComm.generateConmapHeader(fileOut, templateFile, mTask, mQueue, mCommunication, mProcessor, mMyProcessor, mMapping);
		}
	}
	
	public void copyTaskFilesToTargetFolder(){

		String src = mOutputPath + "../";
		String dst = mOutputPath;
		
		for(Task et: mTask.values()){
			if(!et.getCICFile().endsWith(".xml")){
				CommonLibraries.Util.copyFile(dst + "/" + et.getCICFile(), src + "/" + et.getCICFile());
			}
		}
		
		for(Library el: mLibrary.values()){
			CommonLibraries.Util.copyFile(dst + "/" + el.getFile(), src + "/" + el.getFile());
			CommonLibraries.Util.copyFile(dst + "/" + el.getHeader(), src + "/" + el.getHeader());
		}
	}
	
	public void generateProcCode(String mDestFile, String mTemplateFile)
	{
		File fileOut = new File(mDestFile);
		File templateFile = new File(mTemplateFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);
			
			outstream.write(translateProcCode(content).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public String translateProcCode(String mContent)
	{
		String code = mContent;
		String templateFile = "";
	
		// OS_DEPENDENT_INCLUDE_HEADERS //
		Util.copyFile(mOutputPath+"includes.h", mTranslatorPath + "templates/target/VRepSim/includes_template.h");
		String os_dep_includeHeader = "";
		os_dep_includeHeader = "#include \"includes.h\"\n";
		code = code.replace("##OS_DEPENDENT_INCLUDE_HEADERS", os_dep_includeHeader);
		/////////////////////////////////
				
		// TARGET_DEPENDENT_HEADER_INCLUDE //
		templateFile = mTranslatorPath + "templates/target/VRepSim/target_dependent_template";
		String target_dep_includeHeader = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TARGET_DEPENDENT_INCLUDE_HEADERS");
		code = code.replace("##TARGET_DEPENDENT_INCLUDE_HEADERS", target_dep_includeHeader);
		/////////////////////////////////

		// EXTERNAL_GLOBAL_HEADERS //
		String externalGlobalHeaders = "";
		if(mAlgorithm.getHeaders()!=null){
			for(String header: mAlgorithm.getHeaders().getHeaderFile())
				externalGlobalHeaders += "#include\"" + header +"\"\n";
		}
		code = code.replace("##EXTERNAL_GLOBAL_HEADERS", externalGlobalHeaders);
		////////////////////////////
		
		// LIB_INCLUDE //
		String libInclude = "";
		if(mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0 ){
			libInclude += "#include \"LIB_port.h\"\n";
			libInclude += "#include \"lib_channels.h\"\n";
			libInclude += "#include <sys/shm.h>\n";
			
			libInclude += "#define num_libchannels (int)(sizeof(lib_channels)/sizeof(lib_channels[0]))\n";
			
			libInclude += "#define KEY_NUM 9011\n";
			libInclude += "int shm_id;\n";
			libInclude += "void* shm_memory;\n";
			libInclude += "struct shmid_ds shm_info;\n";
			
			String channelSize="";
			int ichannelSize = 0;
			if(mLibraryStubList.size() > 0)
				ichannelSize += mLibraryStubList.size() * 2;
			if(mLibraryWrapperList.size() > 0)
				ichannelSize += mLibraryWrapperList.size() * 2;
			channelSize = String.valueOf(ichannelSize);
			
			libInclude += "LIB_CHANNEL* lib_channels[" + channelSize + "];\n";
		}
		if(mLibraryWrapperList.size() > 0 )	{
			libInclude += "#include \"lib_wrapper.h\"\n#include \"libwrapper_def.h\"\n";
			libInclude += "#define num_libwrappers (int)(sizeof(lib_wrappers)/sizeof(lib_wrappers[0]))\n";
			libInclude += "#define LIB_WRAPPER 1\n";
		}
		code = code.replace("##LIB_INCLUDES", libInclude);
		//////////////
		
		// CONN_INCLUDES //
		String conInclude = CommonLibraries.OutComm.generateConnIncludes(mCommunication, mLibraryStubList.size(), mLibraryWrapperList.size());
		code = code.replace("##CONN_INCLUDES", conInclude);
		//////////////
				
		
		// TARGET_DEPENDENT_IMPLEMENTATION //
		templateFile = mTranslatorPath + "templates/target/VRepSim/target_dependent_template";
		String targetDependentImpl = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TARGET_DEPENDENT_IMPLEMENTATION");
		code = code.replace("##TARGET_DEPENDENT_IMPLEMENTATION", targetDependentImpl);		
		/////////////////////////////////////
		
		// LIB_INIT_WRAPUP //
		String libraryInitWrapup="";
		if(mLibrary != null){
			for(Library library: mLibrary.values()){
				libraryInitWrapup += "extern void l_"+library.getName()+"_init(void);\n";
				libraryInitWrapup += "extern void l_"+library.getName()+"_wrapup(void);\n\n";
			}
		}
	
		libraryInitWrapup += "static void init_libs() {\n";
		if(mLibrary != null)
		{
			for(Library library: mLibrary.values())
				libraryInitWrapup += "\tl_" + library.getName() + "_init();\n";
		}
		libraryInitWrapup += "}\n\n";
		
		libraryInitWrapup += "static void wrapup_libs() {\n";
		if(mLibrary != null){
			for(Library library: mLibrary.values())
				libraryInitWrapup += "\tl_" + library.getName() + "_wrapup();\n";
		}
		libraryInitWrapup += "}\n\n";
		code = code.replace("##LIB_INIT_WRAPUP",  libraryInitWrapup);	
		/////////////////////
				
		// SCHEDULE_CODE //
		code = code.replace("##SCHEDULE_CODE", "");
		///////////////////
		
		// COMM_CODE //
		String commCode = CommonLibraries.OutComm.generateCommCode(mTranslatorPath, mMyProcessor.getPoolName(), mCommunication);		
		code = code.replace("##COMM_CODE", commCode);
		///////////////////
		
		// COMM_SENDER_ROUTINE //
		String commSenderRoutine = CommonLibraries.OutComm.generateSenderCode(mTranslatorPath, mCommunication);
		code = code.replace("##COMM_SENDER_ROUTINE", commSenderRoutine);
		///////////////////
		
		// COMM_RECEIVER_ROUTINE //
		String commReceiverRoutine = CommonLibraries.OutComm.generateReceiverCode(mTranslatorPath, mCommunication);
		code = code.replace("##COMM_RECEIVER_ROUTINE", commReceiverRoutine);
		///////////////////	
			
		// DEBUG_CODE //
		String debugCode = "";
		code = code.replace("##DEBUG_CODE_IMPLEMENTATION", debugCode);
		///////////////////////////
		
		// DATA_TASK_ROUTINE //
		templateFile = mTranslatorPath + "templates/common/task_execution/timed_top_pn_th_bottom_df_func.template";
		String dataTaskRoutine = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##DATA_TASK_ROUTINE");
		code = code.replace("##DATA_TASK_ROUTINE", dataTaskRoutine);
		//////////////////////////
		
		// TIME_TASK_ROUTINE //
		templateFile = mTranslatorPath + "templates/common/task_execution/timed_top_pn_th_bottom_df_func.template";
		String timeTaskRoutine = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TIME_TASK_ROUTINE");
		code = code.replace("##TIME_TASK_ROUTINE", timeTaskRoutine);
		//////////////////////////
		
		// EXECUTE_TASKS //
		templateFile = mTranslatorPath + "templates/common/task_execution/timed_top_pn_th_bottom_df_func.template";
		String executeTasks = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##EXECUTE_TASKS");
		code = code.replace("##EXECUTE_TASKS", executeTasks);
		//////////////////////////
		
		// INIT_WRAPUP_TASK_CHANNELS //
		templateFile = mTranslatorPath + "templates/common/channel_manage/general_linux.template";
		String initWrapupTaskChannels = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##INIT_WRAPUP_TASK_CHANNELS");
		code = code.replace("##INIT_WRAPUP_TASK_CHANNELS", initWrapupTaskChannels);
		//////////////////////////
			
		// INIT_WRAPUP_LIB_CHANNELS //
		String initWrapupLibChannels = "";
		templateFile = mTranslatorPath + "templates/target/VRepSim/lib_channel.template";
		//바뀌어야 할 듯
		if(mLibraryWrapperList.size() > 0 )
		{
			initWrapupLibChannels = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##INIT_WRAPUP_WRAPPER_LIBRARY_CHANNELS");
			code = code.replace("##INIT_WRAPUP_LIBRARY_CHANNELS", initWrapupLibChannels);
			
			String mapShmMemory = "";
			String libEntries = "";
			int index = 0;
			for(InnerDataStructures.Library library: mLibraryStubList)
			{
				mapShmMemory += "lib_channels[" + index + "] = (LIB_CHANNEL*)(shm_memory+ sizeof(LIB_CHANNEL) * " + index++ +");\n";
				libEntries += "\t\tlib_channels[i]->channel_id = " + library.getIndex() + ";\n";
				libEntries += "\t\tstrncpy(lib_channels[i]->lib_name, \"" + library.getName() + "\", strlen(\"" + library.getName() + "\"));\n";
				libEntries += "\t\tif((i%2) == 0)\n";
				libEntries += "\t\t{\n";
				libEntries += "\t\tlib_channels[i]->op = 'r';\n";
				libEntries += "\t\tlib_channels[i]->max_size = sizeof(" + library.getName() + "_func_data);\n";
				libEntries += "\t\tlib_channels[i]->sampleSize = sizeof(" + library.getName() + "_func_data);\n";
				libEntries += "\t\t}\n";
				libEntries += "\t\telse\n";
				libEntries += "\t\t{\n";
				libEntries += "\t\tlib_channels[i]->op = 'w';\n";
				libEntries += "\t\tlib_channels[i]->max_size = sizeof(" + library.getName() + "_func_data);\n";
				libEntries += "\t\tlib_channels[i]->sampleSize = sizeof(" + library.getName() + "_func_data);\n";
				libEntries += "\t\t}\n";
			}
			
			code = code.replace("##LIB_CHANNEL_MAP_SHM_MEMORY", mapShmMemory);
			code = code.replace("##LIBCHANNEL_ENTRIES", mapShmMemory);
		}
		else if(mLibraryStubList.size() > 0)
		{
			initWrapupLibChannels = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##INIT_WRAPUP_STUB_LIBRARY_CHANNELS");
			code = code.replace("##INIT_WRAPUP_LIBRARY_CHANNELS", initWrapupLibChannels);
			
			String mapShmMemory = "";
			int index = 0;
			for(InnerDataStructures.Library library: mLibraryStubList)
			{
				mapShmMemory += "\t\tlib_channels[" + index + "] = (LIB_CHANNEL*)(shm_memory+ sizeof(LIB_CHANNEL) * " + index +");\n";
				mapShmMemory += "\t\tlib_channels[" + (index+1) + "] = (LIB_CHANNEL*)(shm_memory+ sizeof(LIB_CHANNEL) * " + (index+1) +");\n";
				
				index = index + 2;
			}
			
			code = code.replace("##LIB_CHANNEL_MAP_SHM_MEMORY", mapShmMemory);
		}		
		else
		{
			code = code.replace("##INIT_WRAPUP_LIBRARY_CHANNELS", initWrapupLibChannels);
		}
		//////////////
		
		// INIT_WRAPUP_CHANNELS //
		templateFile = mTranslatorPath + "templates/common/channel_manage/general_linux.template";
		String initWrapupChannels = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##INIT_WRAPUP_CHANNELS");
		code = code.replace("##INIT_WRAPUP_CHANNELS", initWrapupChannels);
		//////////////////////////
		
		// READ_WRITE_PORT //
		templateFile = mTranslatorPath + "templates/common/channel_manage/general_linux.template";
		String readWritePort = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##READ_WRITE_PORT");
		code = code.replace("##READ_WRITE_PORT", readWritePort);
		/////////////////////
		
		// READ_WRITE_AC_PORT //
		templateFile = mTranslatorPath + "templates/common/channel_manage/general_linux.template";
		String readWriteACPort = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##READ_WRITE_AC_PORT");
		code = code.replace("##READ_WRITE_AC_PORT", readWriteACPort);
		////////////////////////
		
		// READ_WRITE_BUF_PORT //
		templateFile = mTranslatorPath + "templates/common/channel_manage/general_linux.template";
		String readWriteBufPort = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##READ_WRITE_BUF_PORT");
		code = code.replace("##READ_WRITE_BUF_PORT", readWriteBufPort);
		/////////////////////
			
		// READ_WRITE_LIB_PORT //
		String readWriteLibPort = "";
		templateFile = mTranslatorPath + "templates/target/VRepSim/lib_channel.template";
		if(mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0 )
			readWriteLibPort = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##READ_WRITE_LIB_PORT");
		code = code.replace("##READ_WRITE_LIB_PORT", readWriteLibPort);
		//////////////
		
		// GET_CURRENT_TIME_BASE //
		templateFile = mTranslatorPath + "templates/common/time_code/general_linux.template";
		String getCurrentTimeBase = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##GET_CURRENT_TIME_BASE");
		code = code.replace("##GET_CURRENT_TIME_BASE", getCurrentTimeBase);
		///////////////////////////
		
		// CONTROL_RUN_TASK //
		templateFile = mTranslatorPath + "templates/common/task_execution/timed_top_pn_th_bottom_df_func.template";
		String controlRunTask = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##CONTROL_RUN_TASK");
		code = code.replace("##CONTROL_RUN_TASK", controlRunTask);
		//////////////////////////
		
		// CONTROL_STOP_TASK //
		templateFile = mTranslatorPath + "templates/common/task_execution/timed_top_pn_th_bottom_df_func.template";
		String controlStopTask = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##CONTROL_STOP_TASK");
		code = code.replace("##CONTROL_STOP_TASK", controlStopTask);
		//////////////////////////
				
		// CONTROL_RESUME_TASK //
		templateFile = mTranslatorPath + "templates/common/task_execution/timed_top_pn_th_bottom_df_func.template";
		String controlResumeTask = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##CONTROL_RESUME_TASK");
		code = code.replace("##CONTROL_RESUME_TASK", controlResumeTask);
		//////////////////////////
		
		// CONTROL_SUSPEND_TASK //
		templateFile = mTranslatorPath + "templates/common/task_execution/timed_top_pn_th_bottom_df_func.template";
		String controlSuspendTask = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##CONTROL_SUSPEND_TASK");
		code = code.replace("##CONTROL_SUSPEND_TASK", controlSuspendTask);
		//////////////////////////
		
		// CHANGE_TIME_UNIT //
		templateFile = mTranslatorPath + "templates/common/time_code/general_linux.template";
		String changeTimeUnit = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##CHANGE_TIME_UNIT");
		code = code.replace("##CHANGE_TIME_UNIT", changeTimeUnit);
		///////////////////////////
		
		// MAIN_FUNCTION //
		String mainFunc = "int main(int argc, char *argv[])";
		code = code.replace("##MAIN_FUNCTION", mainFunc);
		//////////////
		
		// TARGET_DEPENDENT_INIT_CALL //
		String targetDependentInit = "\ttarget_dependent_init(argc, argv);";
		code = code.replace("##TARGET_DEPENDENT_INIT_CALL", targetDependentInit);
		////////////////////////////////
		
		// LIB_INIT //
		String libInit = "init_libs();\n";
		if(mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0 )	
			libInit = "\tinit_lib_channel();\n\tinit_libs();\n";
		else																
			libInit = "\tinit_libs();\n";	//해야해?
		code = code.replace("##LIB_INIT", libInit);
		//////////////
		
		// LIB_WRAPUP //
		String libWrapup = "wrapup_libs();";
		if(mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0 )	
			libWrapup = "\twrapup_libs();\n\twrapup_lib_channel();\n";
		else																
			libWrapup = "\twrapup_libs();\n";	//해야해?
		code = code.replace("##LIB_WRAPUP", libWrapup);
		//////////////
		
		// OUT_CONN_INIT //
		String outConnInit = "";
		if(mCommunication.size() > 0 )	
			outConnInit = "OutConnMasterRun();\n";
		code = code.replace("##OUT_CONN_INIT", outConnInit);
		//////////////
		
		// OUT_CONN_WRAPUP //
		String outConnwrapup = "";
		if(mCommunication.size() > 0 )	
			outConnwrapup = "//to do out_conn_wrapup! ";
		code = code.replace("##OUT_CONN_WRAPUP", outConnwrapup);
		//////////////
		
		// TARGET_DEPENDENT_WRAPUP_CALL //
		String targetDependentWrapup = "\ttarget_dependent_wrapup();\n\nreturn EXIT_SUCCESS;";
		code = code.replace("##TARGET_DEPENDENT_WRAPUP_CALL", targetDependentWrapup);
		//////////////////////////////////
						
		// SET_PROC //
		String setProc= "";
		code = code.replace("##SET_PROC", setProc);
		//////////////
		
		//robot마다 다른 부분을 채워주기 위해서	
		String target_spe_includeHeader = "";
		String specificInclude = "";
		String specificDefineNum = "";
		String specificInitImplement = "";
		String specificImplement = "";
		String specificApiGo = "";
		String specificApiStop = "";
		String specificApiGoBack = "";
		String specificApiTurn = "";
		String specificApiLocatePoint = "";
		String specificApiMeetObstacle = "";

		if(mMyProcessor.getPoolName().contains("BubbleRob"))
		{
			templateFile = mTranslatorPath + "templates/target/VRepSim/BubbleRob/target_specific_template";				
		}
		else if(mMyProcessor.getPoolName().contains("Drone"))
		{
			templateFile = mTranslatorPath + "templates/target/VRepSim/Drone/target_specific_template";				
		}
		else if(mMyProcessor.getPoolName().contains("4LegWalker"))
		{
			templateFile = mTranslatorPath + "templates/target/VRepSim/4LegWalker/target_specific_template";				
		}
		
		target_spe_includeHeader = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TARGET_SPECIFIC_INCLUDE_HEADER");
		specificInclude = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TARGET_DEPENDENT_SPECIFIC_INCLUDE");
		specificDefineNum = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TARGET_DEPENDENT_SPECIFIC_DEFINE_NUM");
		specificInitImplement = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TARGET_DEPENDENT_SPECIFIC_INIT_IMPLEMENTATION");
		specificImplement = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TARGET_DEPENDENT_SPECIFIC_IMPLEMENTATION");
		
		specificApiGo = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TARGET_DEPENDENT_SPECIFIC_API_GO_IMPLEMENTATION");
		specificApiStop = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TARGET_DEPENDENT_SPECIFIC_API_STOP_IMPLEMENTATION");
		specificApiGoBack = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TARGET_DEPENDENT_SPECIFIC_API_GOBACK_IMPLEMENTATION");
		specificApiTurn = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TARGET_DEPENDENT_SPECIFIC_API_TURN_IMPLEMENTATION");
		specificApiLocatePoint = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TARGET_DEPENDENT_SPECIFIC_API_LOCATE_POINT_IMPLEMENTATION");
		specificApiMeetObstacle = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TARGET_DEPENDENT_SPECIFIC_API_MEET_OBSTACLE_IMPLEMENTATION");
		
		code = code.replace("##TARGET_SPECIFIC_INCLUDE_HEADER", target_spe_includeHeader);
		code = code.replace("##TARGET_DEPENDENT_SPECIFIC_INCLUDE", specificInclude);
		code = code.replace("##TARGET_DEPENDENT_SPECIFIC_DEFINE_NUM", specificDefineNum);
		code = code.replace("##TARGET_DEPENDENT_SPECIFIC_INIT_IMPLEMENTATION", specificInitImplement);
		code = code.replace("##TARGET_DEPENDENT_SPECIFIC_IMPLEMENTATION", specificImplement);
		
		code = code.replace("##TARGET_DEPENDENT_SPECIFIC_API_GO_IMPLEMENTATION", specificApiGo);
		code = code.replace("##TARGET_DEPENDENT_SPECIFIC_API_STOP_IMPLEMENTATION", specificApiStop);
		code = code.replace("##TARGET_DEPENDENT_SPECIFIC_API_GOBACK_IMPLEMENTATION", specificApiGoBack);
		code = code.replace("##TARGET_DEPENDENT_SPECIFIC_API_TURN_IMPLEMENTATION", specificApiTurn);
		code = code.replace("##TARGET_DEPENDENT_SPECIFIC_API_LOCATE_POINT_IMPLEMENTATION", specificApiLocatePoint);
		code = code.replace("##TARGET_DEPENDENT_SPECIFIC_API_MEET_OBSTACLE_IMPLEMENTATION", specificApiMeetObstacle);
		
		return code;
	}
	
	public void generateMakefile(String mDestFile)
	{
		Map<String, String> extraSourceList = new HashMap<String, String>();
		Map<String, String> extraHeaderList = new HashMap<String, String>();
		Map<String, String> extraLibSourceList = new HashMap<String, String>();
		Map<String, String> extraLibHeaderList = new HashMap<String, String>();
		Map<String, String> ldFlagList = new HashMap<String, String>();
		
		try {
			FileOutputStream outstream = new FileOutputStream(mDestFile);
			
			for(Task task: mTask.values()){
				for(String extraSource: task.getExtraSource())
					extraSourceList.put(extraSource, extraSource.substring(0, extraSource.length()-2));
				for(String extraHeader: task.getExtraHeader())
					extraHeaderList.put(extraHeader, extraHeader.substring(0, extraHeader.length()-2));
			}
			
			if(mAlgorithm.getLibraries() != null){
				for(Library library: mLibrary.values()){
					for(String extraSource: library.getExtraSource())
						extraLibSourceList.put(extraSource, extraSource.substring(0, extraSource.length()-2));
					for(String extraHeader: library.getExtraHeader())
						extraLibHeaderList.put(extraHeader, extraHeader.substring(0, extraHeader.length()-2));
				}
			}
			
			for(Task task: mTask.values())
				if(!task.getLDflag().isEmpty())		ldFlagList.put(task.getLDflag(), task.getLDflag());

			/* -- test for 0617 
			String srcExtension = null;
			if(mLanguage.equals("c++")){
				srcExtension = ".cpp";
				outstream.write("CC=g++\n".getBytes());
				outstream.write("LD=g++\n".getBytes());
			}
			else{
				srcExtension = ".c";
				outstream.write("CC=gcc\n".getBytes());
				outstream.write("LD=gcc\n".getBytes());
			}
			*/
			String srcExtension = null;			
			srcExtension = ".cpp";
			outstream.write("CC=g++\n".getBytes());
			outstream.write("LD=g++\n".getBytes());			
			
		    mRootPath = mRootPath.replace("\\","/");
		    mRootPath = mRootPath.replace("C:", "/cygdrive/C");
		    mRootPath = mRootPath.replace("D:", "/cygdrive/D");
		    mRootPath = mRootPath.replace("E:", "/cygdrive/E");
		    mRootPath = mRootPath.replace("F:", "/cygdrive/F");
		    mRootPath = mRootPath.replace("G:", "/cygdrive/G");
		    
		    outstream.write("# -- For Linux & Debug --\n".getBytes());
		    outstream.write("#CFLAGS=-Wall -O0 -g -DDISPLAY\n".getBytes());
		    outstream.write("# -- For Linux & !Debug --\n".getBytes());
	        outstream.write("CFLAGS=-O2 -DDISPLAY\n".getBytes());
	        
	        //for vrep cflags should be added 
	        //outstream.write("CFLAGS+= -I../remoteApi -I../include -DNON_MATLAB_PARSING -DMAX_EXT_API_CONNECTIONS=255\n\n".getBytes());
	        outstream.write("CFLAGS+= -I../remoteApi -I../include -DNON_MATLAB_PARSING -DMAX_EXT_API_CONNECTIONS=255 \n\n".getBytes());

		    outstream.write("# -- For Linux --\n".getBytes());
		    //outstream.write("LDFLAGS= -lpthread -lm -lX11 ".getBytes());
		    outstream.write("LDFLAGS= -lpthread -lm -lX11 ".getBytes());
		    
		    for(String ldflag: ldFlagList.values())
		        outstream.write((" " + ldflag).getBytes());
		    outstream.write("\n\n".getBytes());
		    
		    outstream.write("all: proc\n\n".getBytes());
		    
		    outstream.write("proc:".getBytes());
		    for(Task task: mTask.values())
		    	outstream.write((" " + task.getName() + ".o").getBytes());
		    
		    for(String extraSource: extraSourceList.values())
		    	outstream.write((" " + extraSource + ".o").getBytes());

		    for(String extraLibSource: extraLibSourceList.values())
		    	outstream.write((" " + extraLibSource + ".o").getBytes());
		    
		    if(mAlgorithm.getLibraries() != null)
		    {
		    	for(Library library: mLibrary.values())
		    		outstream.write((" " + library.getName() + ".o").getBytes());
		    	for(Library library: mLibraryWrapperList){
		    		String wrapper = library.getName() + "_wrapper";
		    		outstream.write((" " + wrapper + ".o").getBytes());
		    	}
		    	for(Library library: mLibraryStubList){
		    		String stub = library.getName() + "_stub";
		    		outstream.write((" " + stub + ".o").getBytes());
		    	}
		    }
		    
		    //for vrep 
		    outstream.write(" ../remoteApi/extApi.o ../remoteApi/extApiPlatform.o".getBytes());
		    
		    outstream.write(" proc.o\n".getBytes());
		    outstream.write("\t$(LD) $^ -o proc $(LDFLAGS)\n\n".getBytes());
		    
		    outstream.write(("proc.o: proc" + srcExtension + " CIC_port.h CIC_robot.h").getBytes());
		    
		    if(mAlgorithm.getHeaders() != null)
		    	for(String headerFile: mAlgorithm.getHeaders().getHeaderFile())
		    		outstream.write((" " + headerFile).getBytes());
		    
		    outstream.write("\n".getBytes());
		    outstream.write(("\t$(CC) $(CFLAGS) -c proc" + srcExtension + " -o proc.o\n").getBytes());
		    
		    //for vrep
		    /*
		    outstream.write(("\t$(CC) $(CFLAGS) -c ../remoteApi/extApi.c -o ../remoteApi/extApi.o\n").getBytes());
		    outstream.write(("\t$(CC) $(CFLAGS) -c ../remoteApi/extApiPlatform.c -o ../remoteApi/extApiPlatform.o\n\n").getBytes());
			*/
		    outstream.write(("\tgcc $(CFLAGS) -c ../remoteApi/extApi.c -o ../remoteApi/extApi.o\n").getBytes());
		    outstream.write(("\tgcc $(CFLAGS) -c ../remoteApi/extApiPlatform.c -o ../remoteApi/extApiPlatform.o\n\n").getBytes());
			
		    
		    for(Task task: mTask.values()){
		    	if(task.getCICFile().endsWith(".cic"))
		    		outstream.write((task.getName() + ".o: " + task.getName() + srcExtension + " " + task.getCICFile() + " CIC_port.h CIC_robot.h").getBytes());
		    	else if(task.getCICFile().endsWith(".xml"))
		    		outstream.write((task.getName() + ".o: " + task.getName() + srcExtension + " " + " CIC_port.h ").getBytes());
		    	for(String header: task.getExtraHeader())
		    		outstream.write((" " + header).getBytes());
		    	outstream.write("\n".getBytes());
		    	outstream.write(("\t$(CC) $(CFLAGS) " + task.getCflag() + " -c " + task.getName() + srcExtension + " -o " + task.getName() + ".o\n\n").getBytes());
		    	 
		    }
		    
		    for(String extraSource: extraSourceList.keySet()){
		    	outstream.write((extraSourceList.get(extraSource) + ".o: " + extraSourceList.get(extraSource) + ".c").getBytes());
		    	for(String extraHeader: extraHeaderList.keySet())
		    		outstream.write((" " + extraHeader).getBytes());
		    	outstream.write("\n".getBytes());
		    	outstream.write(("\t$(CC) $(CFLAGS) -c " + extraSourceList.get(extraSource) + ".c -o " + extraSourceList.get(extraSource) + ".o\n\n").getBytes());
		    }
		    
		    if(mAlgorithm.getLibraries() != null){
		    	for(Library library: mLibrary.values()){
		    		outstream.write((library.getName() + ".o: " + library.getName() + ".c").getBytes());
		    		for(String extraHeader: library.getExtraHeader())
		    			outstream.write((" " + extraHeader).getBytes());
		    		outstream.write("\n".getBytes());
		    		if(!library.getCflag().isEmpty())
		    			outstream.write(("\t$(CC) $(CFLAGS) " + library.getCflag()).getBytes());
		    		else
		    			outstream.write(("\t$(CC) $(CFLAGS)").getBytes());
		    		outstream.write((" -c " + library.getName() + ".c -o " + library.getName() + ".o\n").getBytes());
		    	}
		    }
		    outstream.write("\n".getBytes());
		    
		    for(Library library: mLibraryWrapperList){
	    		String wrapper = library.getName() + "_wrapper";
	    		outstream.write((wrapper + ".o: " + wrapper + ".c").getBytes());
	    		for(String extraHeader: library.getExtraHeader())
	    			outstream.write((" " + extraHeader).getBytes());
	    		outstream.write("\n".getBytes());
	    		if(!library.getCflag().isEmpty())
	    			outstream.write(("\t$(CC) $(CFLAGS) " + library.getCflag()).getBytes());
	    		else
	    			outstream.write(("\t$(CC) $(CFLAGS)").getBytes());
	    		outstream.write((" -c " + wrapper + ".c -o " + wrapper + ".o\n").getBytes());
	    	}
		    outstream.write("\n".getBytes());
		    
	    	for(Library library: mLibraryStubList){
	    		String stub = "l_" + library.getName() + "_stub";
	    		outstream.write((stub + ".o: " + stub + ".c").getBytes());
	    		for(String extraHeader: library.getExtraHeader())
	    			outstream.write((" " + extraHeader).getBytes());
	    		outstream.write("\n".getBytes());
	    		if(!library.getCflag().isEmpty())
	    			outstream.write(("\t$(CC) $(CFLAGS) " + library.getCflag()).getBytes());
	    		else
	    			outstream.write(("\t$(CC) $(CFLAGS)").getBytes());
	    		outstream.write((" -c " + stub + ".c -o " + stub + ".o\n").getBytes());
	    	}
		    outstream.write("\n".getBytes());
		    
		    for(String extraSource: extraLibSourceList.keySet()){
		    	outstream.write((extraLibSourceList.get(extraSource) + ".o: " + extraSource).getBytes());
		    	for(String extraHeader: extraLibHeaderList.values())
		    		outstream.write((" " + extraHeader).getBytes());
		    	outstream.write("\n".getBytes());
		    	outstream.write(("\t$(CC) $(CFLAGS) -c " + extraSource + " -o " + extraSourceList.get(extraSource) + ".o\n").getBytes());
		    }
		    outstream.write("\n".getBytes());
		    
		    outstream.write("test: all\n".getBytes());
		    outstream.write("\t./proc\n".getBytes());
		    outstream.write("\n".getBytes());
		    
		    outstream.write("graph:\n".getBytes());
		    outstream.write("\tdot -Tps portGraph.dot -o portGraph.ps\n".getBytes());
		    outstream.write("\n".getBytes());
		    
		    outstream.write("tags:\n".getBytes());
		    outstream.write("\tctags -R --c-types=cdefglmnstuv\n".getBytes());
		    outstream.write("\n".getBytes());
		    
		    outstream.write("clean:\n".getBytes());
		    outstream.write("\trm -f proc *.o\n".getBytes());
		    
		    //for vrep
		    outstream.write("\trm -f ../remoteApi/*.o\n".getBytes());
		    
		    outstream.write("\n\n".getBytes());
		    
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}	
	}

	@Override
	public int generateCodeWithComm(String target, String translatorPath, String outputPath, String rootPath, Map<Integer, Processor> processor, List<Communication> communication, Map<String, Task> task, Map<Integer, Queue> queue, Map<String, Library> library, Map<String, Library> globalLibrary, int globalPeriod, String globalPeriodMetric, String cicxmlfile, String language, String threadVer, CICAlgorithmType algorithm, CICControlType control, CICScheduleType schedule, CICGPUSetupType gpusetup, CICMappingType mapping, Map<Integer, List<Task>> connectedTaskGraph, Map<Integer, List<List<Task>>> connectedSDFTaskSet, Map<String, Task> vtask, Map<String, Task> pvtask, String codegentype) throws FileNotFoundException {
		// TODO Auto-generated method stub
		int ret = 0;
		mProcessor = processor;
		System.out.println("[V-REP] generateCodeWithComm();");
		for(Processor proc: mProcessor.values())
		{
			mTarget = target;
			mTranslatorPath = translatorPath;
			mOutputPath = outputPath;
			mRootPath = rootPath;
			mCICXMLFile = cicxmlfile;
			mGlobalPeriod = globalPeriod;
			mGlobalPeriodMetric = globalPeriodMetric;
			mThreadVer = threadVer;
			mLanguage = language;
			
			mTask = task;
			mQueue = queue;
			mLibrary = library;
			mGlobalLibrary = globalLibrary;
			mProcessor = processor;
			mCommunication = communication;
			
			mVTask = vtask;
			
			mAlgorithm = algorithm;
			mControl = control;
			mGpusetup = gpusetup;
			mMapping = mapping;
							
			System.out.println("===================================== ");
			if(proc.getPoolName().contains("BubbleRob"))
			{
				mMyProcessor = proc;
								
				System.out.println("### mMyProcessor.getPoolName()::" + mMyProcessor.getPoolName());
				Map<String, Task> mBubbleRobTask = new HashMap<String, Task>();
				Map<Integer, Queue> mBubbleRobQueue = new HashMap<Integer, Queue>();
				Map<String, Library> mBubbleRobLibrary= new HashMap<String, Library>();
				
				seperateDataStructure(proc.getPoolName(), mBubbleRobTask, mBubbleRobQueue, mBubbleRobLibrary);
				String mBubbleRobOutputPath = mOutputPath + proc.getPoolName() + "/";				
				File BubbleRobF = new File(mBubbleRobOutputPath);			
				BubbleRobF.mkdir();

				try {
					ret = generateRobotCode(mTarget, mTranslatorPath, mBubbleRobOutputPath, mRootPath, mProcessor, mMyProcessor, mBubbleRobTask, mBubbleRobQueue, mBubbleRobLibrary, mLibrary, mGlobalPeriod, mGlobalPeriodMetric, mCICXMLFile, mLanguage, mThreadVer, mAlgorithm, mControl, mSchedule, mGpusetup, mMapping, null, null, mVTask);
				} catch (FileNotFoundException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
			else if(proc.getPoolName().contains("Drone"))
			{
				mMyProcessor = proc;
				
				Map<String, Task> mDroneTask = new HashMap<String, Task>();
				Map<Integer, Queue> mDroneQueue = new HashMap<Integer, Queue>();
				Map<String, Library> mDroneLibrary= new HashMap<String, Library>();
				
				seperateDataStructure(proc.getPoolName(), mDroneTask, mDroneQueue, mDroneLibrary);
				String mDroneOutputPath = mOutputPath + proc.getPoolName() + "/";				
				File BubbleRobF = new File(mDroneOutputPath);			
				BubbleRobF.mkdir();

				try {
					ret = generateRobotCode(mTarget, mTranslatorPath, mDroneOutputPath, mRootPath, mProcessor, mMyProcessor, mDroneTask, mDroneQueue, mDroneLibrary, mLibrary, mGlobalPeriod, mGlobalPeriodMetric, mCICXMLFile, mLanguage, mThreadVer, mAlgorithm, mControl, mSchedule, mGpusetup, mMapping, null, null, mVTask);
				} catch (FileNotFoundException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
			else if(proc.getPoolName().contains("4LegWalker"))
			{
				mMyProcessor = proc;
				
				Map<String, Task> mDroneTask = new HashMap<String, Task>();
				Map<Integer, Queue> mDroneQueue = new HashMap<Integer, Queue>();
				Map<String, Library> mDroneLibrary= new HashMap<String, Library>();
				
				seperateDataStructure(proc.getPoolName(), mDroneTask, mDroneQueue, mDroneLibrary);
				String mDroneOutputPath = mOutputPath + proc.getPoolName() + "/";				
				File BubbleRobF = new File(mDroneOutputPath);			
				BubbleRobF.mkdir();

				try {
					ret = generateRobotCode(mTarget, mTranslatorPath, mDroneOutputPath, mRootPath, mProcessor, mMyProcessor, mDroneTask, mDroneQueue, mDroneLibrary, mLibrary, mGlobalPeriod, mGlobalPeriodMetric, mCICXMLFile, mLanguage, mThreadVer, mAlgorithm, mControl, mSchedule, mGpusetup, mMapping, null, null, mVTask);
				} catch (FileNotFoundException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
	    
	    return 0;
	}
}
