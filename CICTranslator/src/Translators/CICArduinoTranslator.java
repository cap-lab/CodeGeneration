package Translators;

import hopes.cic.xml.CICAlgorithmType;
import hopes.cic.xml.CICControlType;
import hopes.cic.xml.CICGPUSetupType;
import hopes.cic.xml.CICMappingType;
import hopes.cic.xml.CICScheduleType;
import hopes.cic.xml.ControlTaskType;
import hopes.cic.xml.LibraryMasterPortType;
import hopes.cic.xml.TaskLibraryConnectionType;
import hopes.cic.xml.TaskParameterType;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import CommonLibraries.Util;
import InnerDataStructures.Library;
import InnerDataStructures.Processor;
import InnerDataStructures.Communication;
import InnerDataStructures.Queue;
import InnerDataStructures.Task;

public class CICArduinoTranslator implements CICTargetCodeTranslator {
	private String mTarget;
	private String mTranslatorPath;
	private String mOutputPath;
	private String mRootPath;
	private String mCICXMLFile;
	private int mGlobalPeriod;
	private String mGlobalPeriodMetric;
	private String mThreadVer;
	private String mLanguage;
	
	private Map<String, Task> mTask;
	private Map<Integer, Queue> mQueue;
	private Map<String, Library> mLibrary;
	private Map<Integer, Processor> mProcessor;
	private List<Communication> mCommunication;
	
	private Map<String, Library> mGlobalLibrary;
		
	private Map<String, Task> mVTask;
	
	private CICAlgorithmType mAlgorithm;
	private CICControlType mControl;
	private CICScheduleType mSchedule;
	private CICGPUSetupType mGpusetup;
	private CICMappingType mMapping;

	private String strategy;
	
	private String mArduinoOutputPath;
	
	private ArrayList<Library> mLibraryStubList;
	private ArrayList<Library> mLibraryWrapperList;
	
	private Processor mMyProcessor;
	
	public int generateCode(String target, String translatorPath, String outputPath, String rootPath, Map<Integer, Processor> processor, Map<String, Task> task, Map<Integer, Queue> queue, Map<String, Library> library, Map<String, Library> globalLibrary, int globalPeriod, String globalPeriodMetric, String cicxmlfile, String language, String threadVer, CICAlgorithmType algorithm, CICControlType control, CICScheduleType schedule, CICGPUSetupType gpusetup, CICMappingType mapping, Map<Integer, List<Task>> connectedtaskgraph,  Map<Integer, List<List<Task>>> connectedsdftaskset, Map<String, Task> vtask, Map<String, Task> pvtask, String codegentype) throws FileNotFoundException
	{
		for(Processor proc: processor.values())
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
			mMyProcessor = proc;
			
			mVTask = vtask;
			
			mAlgorithm = algorithm;
			mControl = control;
			mSchedule = schedule;
			mGpusetup = gpusetup;
			mMapping = mapping;
					
			mLibraryStubList = new ArrayList<Library>();
			mLibraryWrapperList = new ArrayList<Library>();
				
			mArduinoOutputPath = mOutputPath + "Arduino";
			
			if(!mTarget.toUpperCase().equals("ARDUINO"))	mOutputPath += "../";
			
			File arduF = new File(mArduinoOutputPath);

			arduF.mkdir();
			
			generateProcIno(mArduinoOutputPath + "/Arduino.ino", mTranslatorPath + "templates/target/arduino/proc.ino.general.template");
			
			/* 아직 라이브러리 고려 안함 - 통신 또한 고려안함 
			 * for(Library l: mLibraryStubList){
				File fileOut = new File(mArduinoOutputPath + "/" + l.getName() + "_data_structure.h");
				CommonLibraries.Library.generateLibraryDataStructureHeader_16bit(fileOut, l);
			}
			for(Library l: mLibraryWrapperList){
				File fileOut = new File(mArduinoOutputPath + "/" + l.getName() + "_data_structure.h");
				CommonLibraries.Library.generateLibraryDataStructureHeader_16bit(fileOut, l);
			}
			*/
			
			for(Task t: mTask.values()){
				generateTaskIno(mArduinoOutputPath + "/" + t.getName() + ".ino", mOutputPath + "/" + t.getCICFile(), t);
			}
		}		
				 
	    return 0;
	}
	
	public int generateCodeWithComm(String target, String translatorPath, String outputPath, String rootPath, Map<Integer, Processor> processor, List<Communication> communication, Map<String, Task> task, Map<Integer, Queue> queue, Map<String, Library> library, Map<String, Library> globalLibrary, int globalPeriod, String globalPeriodMetric, String cicxmlfile, String language, String threadVer, CICAlgorithmType algorithm, CICControlType control, CICScheduleType schedule, CICGPUSetupType gpusetup, CICMappingType mapping, Map<Integer, List<Task>> connectedtaskgraph,  Map<Integer, List<List<Task>>> connectedsdftaskset, Map<String, Task> vtask) throws FileNotFoundException
	{
		//2015.05. 아직 테스트 안한 상황 -- TI에 있는 것 그대로임 
		/*mTarget = target;
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
		mSchedule = schedule;
		mGpusetup = gpusetup;
		mMapping = mapping;
		
		mArduinoTask = new HashMap<String, Task>();
		mEvalbotTask = new HashMap<String, Task>();
		
		mArduinoLibrary = new HashMap<String, Library>();
		mEvalbotLibrary = new HashMap<String, Library>();
		
		mArduinoQueue = new HashMap<Integer, Queue>();
		mEvalbotQueue = new HashMap<Integer, Queue>();
		
		mLibraryStubList = new ArrayList<Library>();
		mLibraryWrapperList = new ArrayList<Library>();
			
		mEvalbotOutputPath = mOutputPath + "TIEvalbot";
		mArduinoOutputPath = mOutputPath + "Arduino";
		
		if(!mTarget.toUpperCase().equals("TIEVALBOT"))	mOutputPath += "../";
		
		File evalF = new File(mEvalbotOutputPath);
		File arduF = new File(mArduinoOutputPath);

		evalF.mkdir();
		arduF.mkdir();
		
		seperateDataStructure();
		
		generateTIEvalbot();		
		generateArduino();
		
		copyTaskFilesToTargetFolder();
		*/
	    return 0;
	}		
	
	public void generateTaskIno(String mDestFile, String mTaskFile, Task task){
		File fileOut = new File(mDestFile);
		File templateFile = new File(mTaskFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);

			outstream.write(translateTaskIno(content, task).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public String translateTaskIno(String mTaskCode, Task task){
		String code = "";
		
		code += "#ifndef " + task.getName() + "_\n";
		code += "#define " + task.getName() + "_\n\n";
		
		code += "#define TASK_NAME \"" + task.getName() + "\"\n";
		code += "#define TASK_INIT void " + task.getName() + "_init(int TASK_ID)\n";
		code += "#define TASK_GO void " + task.getName() + "_go()\n";
		code += "#define TASK_WRAPUP void " + task.getName() + "_wrapup()\n";
		
		code += "#define BUF_SEND sendData\n";
		code += "#define BUF_RECEIVE receiveData\n";
		
		code += "#define SYS_REQ_GET_PARAM_INT getParamInt\n";
		code += "#define SYS_REQ_SET_PARAM_INT setParamInt\n";
		
		code += "#define SYS_REQ_CALL_TASK callTask\n";
		
		code += "#define SYS_REQ_SET_TIMER(a, b) set_timer(TASK_NAME, a, b)\n";
		code += "#define SYS_REQ_GET_TIMER_ALARMED(a) get_timer_alarmed(TASK_NAME, a)\n";
		
		code += "#define STATIC static\n";
		
		for(String port: task.getPortList().keySet())
			code += "#define port_" + port + " " + task.getName() + "_port_" + port + "\n";
		
		code += "\n\n";
		code += mTaskCode;
		code += "\n\n";
		
		code += "#undef TASK_NAME\n";
		code += "#undef TASK_INIT\n";
		code += "#undef TASK_GO\n";
		code += "#undef TASK_WRAPUP\n";
		code += "#undef BUF_SEND\n";

		for(String port: task.getPortList().keySet())
			code += "#undef port_" + port + "\n";
		
		code += "#endif\n";
		
		code = code.replace("TASK_CODE_BEGIN", "");
		code = code.replace("TASK_CODE_END", "");
		
		return code;
	}
	
	public void generateProcIno(String mDestFile, String mTemplateFile){
		File fileOut = new File(mDestFile);
		File templateFile = new File(mTemplateFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);

			outstream.write(translateProcIno(content).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public String translateProcIno(String mContent){
		String code = mContent;
		String templateFile;
		
		// TARGET_DEPENDENT_HEADER_INCLUDE //
		String target_dep_headerInclude = "";
		if(mMyProcessor.getPoolName().contains("DASH"))
		{
			templateFile = mTranslatorPath + "templates/target/arduino/DashRobot/DashRobot.template";		
			target_dep_headerInclude = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TARGET_DEPENDENT_INCLUDE_HEADER");
		}
		code = code.replace("##TARGET_DEPENDENT_INCLUDE_HEADER", target_dep_headerInclude);
		/////////////////////////////////
		
		// TARGET_DEPENDENT_CALL_SETUP //
		String target_dep_callSetup = "";
		if(mMyProcessor.getPoolName().contains("DASH"))
		{
			templateFile = mTranslatorPath + "templates/target/arduino/DashRobot/DashRobot.template";
			target_dep_callSetup = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TARGET_DEPENDENT_CALL_SETUP");
		}		
		code = code.replace("##TARGET_DEPENDENT_CALL_SETUP", target_dep_callSetup);
		/////////////////////////////////////////////////
		
		String taskStructure = "";
		
		for(Task task: mTask.values()){
			taskStructure += "{";
			taskStructure += task.getIndex() + ", ";
			
			taskStructure += "\"" + task.getName() + "\", ";
			taskStructure += "\"" + task.getTaskType() + "\", ";
			taskStructure += "\"" + task.getRunCondition() + "\", ";			
			
			int result = 0;
			int period = Integer.parseInt(task.getPeriod());
			String unit = task.getPeriodMetric();
			
			if(unit.equalsIgnoreCase("MS"))			result = period;
			else if(unit.equalsIgnoreCase("S"))		result = period * 1000;
			else if(unit.equalsIgnoreCase("US"))	result = 1;
			
			taskStructure += result + ", 0},\n";
		}
		code = code.replace("##TASK_STRUCTURE", taskStructure);
		
		String controlChannelStructure = "";
		String controlParamStructure = "";
		if(mControl.getControlTasks() != null)
		{
			for(ControlTaskType controlTask: mControl.getControlTasks().getControlTask())
			{
					if(mTask.containsKey(controlTask.getTask()))
					{
						for(Task task: mTask.values())
						{
							if(task.getName().equals(controlTask.getTask()))
							{
								controlChannelStructure += "\t{" + task.getIndex() + ", {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {0,}},\n";
							}
						}
					}
			}
			
			for(Task task: mTask.values())
			{
				int index = 0;
				for(TaskParameterType  parameter:task.getParameter())
				{
					controlParamStructure += "\t{" + task.getIndex() + ", " + (index++) + ", \"" + parameter.getName() + "\", " + parameter.getValue() + "},\n";
				}
			}
			
		}
		code = code.replace("##CONTROL_CHANNEL_STRUCTURE", controlChannelStructure);
		code = code.replace("##PARAM_STRUCTURE", controlParamStructure);
		
		String numTasks = "static int num_tasks = " + mTask.size() + ";\n";
		code = code.replace("##NUM_TASKS", numTasks);
		
		String channelStructure = "";
		int channelIndex = 0;
		
		String portmapStructure = "";
		for(Queue queue: mQueue.values()){
			channelStructure += "\t{" + channelIndex++ + ", 0},\n";
			if(mTask.get(queue.getSrc()) != null)
				portmapStructure += "\t{" + mTask.get(queue.getSrc()).getIndex() +","
					+ "\"" + queue.getSrcPortName() +"\","
					+ queue.getIndex()+ ",'w'},\n";
			if(mTask.get(queue.getDst()) != null)
				portmapStructure += "\t{" + mTask.get(queue.getDst()).getIndex() +","
				+ "\"" + queue.getDstPortName() +"\","
				+ queue.getIndex()+ ",'r'},\n";
		}	
		code = code.replace("##CHANNEL_STRUCTURE", channelStructure);
		code = code.replace("##PORTMAP_STRUCTURE", portmapStructure);
		
		String numChannels = "static int num_channels = " + mQueue.size() + ";\n";
		code = code.replace("##NUM_CHANNELS", numChannels);
		
		/*
		String numLibChannels = "static int num_libchannels = " + ((mLibraryStubList.size() + mLibraryWrapperList.size()) * 2) + ";\n";
		code = code.replace("##NUM_LIB_CHANNELS", numLibChannels);
		
		String libDataStructureInclude = "";
		String libChannels = "";
		for(Library l: mLibraryStubList){
			libDataStructureInclude += "#include \"" + l.getName() + "_data_structure.h\"\n";
			libChannels += "\t{" + l.getIndex() + ", 'w', sizeof(" + l.getName() + "_func_data)},\n";
			libChannels += "\t{" + l.getIndex() + ", 'r', sizeof(" + l.getName() + "_ret_data)},\n";
		}
		for(Library l: mLibraryWrapperList){
			libDataStructureInclude += "#include \"" + l.getName() + "_data_structure.h\"\n";
			libChannels += "\t{" + l.getIndex() + ", 'w', sizeof(" + l.getName() + "_func_data)},\n";
			libChannels += "\t{" + l.getIndex() + ", 'r', sizeof(" + l.getName() + "_ret_data)},\n";
		}
		code = code.replace("##LIB_DATA_STRUCTURE_INCLUDE", libDataStructureInclude);
		code = code.replace("##LIB_CHANNEL_STRUCTURE", libChannels);
		*/
		
		String taskFuncDecl = "";
		String taskInitCall = "";
		String taskGoCall = "";
		String taskInitGoWrapupCall = "";
		boolean is2ndTaskInitGoWrapupCall = false;
		
		for(Task task: mTask.values()){
			taskFuncDecl += "void " + task.getName() + "_init(int);\n";
			taskFuncDecl += "void " + task.getName() + "_go();\n";
			taskFuncDecl += "void " + task.getName() + "_wrapup();\n\n";
			
			taskInitCall += task.getName() + "_init(" + task.getIndex() + ");\n";
			taskGoCall += "\tif(tasks[i].index == " + task.getIndex() +")  " + task.getName() + "_go();\n";
			
			if(task.getRunCondition().equals("CONTROL_DRIVEN"))
			{
				if(is2ndTaskInitGoWrapupCall == true)
					taskInitGoWrapupCall += "\telse ";
				else
					taskInitGoWrapupCall += "\t";
				taskInitGoWrapupCall += "if(tasks[task_index].index == " + task.getIndex() + ")\n";
				taskInitGoWrapupCall += "\t{\n";
				taskInitGoWrapupCall += "\t\t" + task.getName() + "_init(" + task.getIndex() + ");\n";
				taskInitGoWrapupCall += "\t\t" + task.getName() + "_go();\n";
				taskInitGoWrapupCall += "\t\t" + task.getName() + "_wrapup();\n";
				taskInitGoWrapupCall += "\t}";
				is2ndTaskInitGoWrapupCall = true;
			}
		}
		
		code = code.replace("##TASK_INIT_CALL", taskInitCall);
		code = code.replace("##TASK_GO_CALL", taskGoCall);
		code = code.replace("##TASK_FUNCTION_DECLARATION", taskFuncDecl);
		code = code.replace("##TASK_INIT_GO_WRAPUP_CALL", taskInitGoWrapupCall);
		
		return code;
	}

	@Override
	public int generateCodeWithComm(String mTarget, String mTranslatorPath, String mOutputPath, String mRootPath,
			Map<Integer, Processor> mProcessor, List<Communication> mCommunication, Map<String, Task> mTask,
			Map<Integer, Queue> mQueue, Map<String, Library> mLibrary, Map<String, Library> mGlobalLibrary,
			int mGlobalPeriod, String mGlbalPeriodMetric, String mCICXMLFile, String language, String threadVer,
			CICAlgorithmType mAlgorithm, CICControlType mControl, CICScheduleType mSchedule, CICGPUSetupType mGpusetup,
			CICMappingType mMapping, Map<Integer, List<Task>> mConnectedTaskGraph,
			Map<Integer, List<List<Task>>> mConnectedSDFTaskSet, Map<String, Task> mVTask, Map<String, Task> mPVTask, String mCodeGenType)
					throws FileNotFoundException {
		// TODO Auto-generated method stub
		return 0;
	}

}
