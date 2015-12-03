package Translators;

import hopes.cic.xml.CICAlgorithmType;
import hopes.cic.xml.CICControlType;
import hopes.cic.xml.CICGPUSetupType;
import hopes.cic.xml.CICMappingType;
import hopes.cic.xml.CICScheduleType;
import hopes.cic.xml.LibraryMasterPortType;
import hopes.cic.xml.TaskLibraryConnectionType;

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

public class CICTIEvalbotTranslator implements CICTargetCodeTranslator {
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
	private Map<Integer, Processor> mProcessor;
	private List<Communication> mCommunication;
	
	private Map<String, Task> mEvalbotTask;
	private Map<Integer, Queue> mEvalbotQueue;
	private Map<String, Library> mEvalbotLibrary;
	private Map<String, Library> mGlobalLibrary;
	
	private Map<String, Task> mArduinoTask;
	private Map<Integer, Queue> mArduinoQueue;
	private Map<String, Library> mArduinoLibrary;
	
	private Map<String, Task> mVTask;
	private Map<String, Task> mPVTask;
	
	private CICAlgorithmType mAlgorithm;
	private CICControlType mControl;
	private CICScheduleType mSchedule;
	private CICGPUSetupType mGpusetup;
	private CICMappingType mMapping;

	private String strategy;
	
	private String mEvalbotOutputPath;
	private String mArduinoOutputPath;
	
	private ArrayList<Library> mLibraryStubList;
	private ArrayList<Library> mLibraryWrapperList;
	
	public int generateCode(String target, String translatorPath, String outputPath, String rootPath, Map<Integer, Processor> processor, Map<String, Task> task, Map<Integer, Queue> queue, Map<String, Library> library, Map<String, Library> globalLibrary, int globalPeriod, String globalPeriodMetric, String cicxmlfile, String language, String threadVer, CICAlgorithmType algorithm, CICControlType control, CICScheduleType schedule, CICGPUSetupType gpusetup, CICMappingType mapping, Map<Integer, List<Task>> connectedtaskgraph,  Map<Integer, List<List<Task>>> connectedsdftaskset, Map<String, Task> vtask, Map<String, Task> pvtask, String codegentype) throws FileNotFoundException
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
		 
	    return 0;
	}
	
	public int generateCodeWithComm(String target, String translatorPath, String outputPath, String rootPath, Map<Integer, Processor> processor, List<Communication> communication, Map<String, Task> task, Map<Integer, Queue> queue, Map<String, Library> library, Map<String, Library> globalLibrary, int globalPeriod, String globalPeriodMetric, String cicxmlfile, String language, String threadVer, CICAlgorithmType algorithm, CICControlType control, CICScheduleType schedule, CICGPUSetupType gpusetup, CICMappingType mapping, Map<Integer, List<Task>> connectedtaskgraph,  Map<Integer, List<List<Task>>> connectedsdftaskset, Map<String, Task> vtask) throws FileNotFoundException
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
		 
	    return 0;
	}
	
	public void copyTaskFilesToTargetFolder(){
		String src = mOutputPath;
		String dst = mEvalbotOutputPath + "/EvalBoards/TI/LM3S9B92-EVALBOT/IAR/uCOS-III-App/";
		
		for(Task et: mEvalbotTask.values()){
			if(!et.getCICFile().endsWith(".xml")){
				CommonLibraries.Util.copyFile(dst + "/" + et.getCICFile(), src + "/" + et.getCICFile());
			}
		}
		
		for(Library el: mEvalbotLibrary.values()){
			CommonLibraries.Util.copyFile(dst + "/" + el.getFile(), src + "/" + el.getFile());
			CommonLibraries.Util.copyFile(dst + "/" + el.getHeader(), src + "/" + el.getHeader());
		}
		
		/*
		dst = mArduinoOutputPath;
		
		for(Task at: mArduinoTask.values())
			CommonLibraries.Util.copyFile(dst + "/" + at.getCICFile(), src + "/" + at.getCICFile());
		
		for(Library al: mArduinoLibrary.values()){
			CommonLibraries.Util.copyFile(dst + "/" + al.getFile(), src + "/" + al.getFile());
			CommonLibraries.Util.copyFile(dst + "/" + al.getHeader(), src + "/" + al.getHeader());
		}
		*/
	}
	
	public void seperateDataStructure(){
		for(Task t: mTask.values()){
			Map<String, Map<String, List<Integer>>> plmapmap = t.getProc();
			for(Map<String, List<Integer>> plmap: plmapmap.values()){
				for(List<Integer> pl: plmap.values()){
					for(int p: pl){
						if(mProcessor.get(p).getPoolName().contains("Arduino")){
							mArduinoTask.put(t.getName(), t);
						}
						else if(mProcessor.get(p).getPoolName().contains("TIEvalbot")){
							mArduinoTask.put(t.getName(), t);
						}
					}
				}
			}
		}
		/*
		for(Processor proc: mProcessor.values()){
			List<Task> taskList = proc.getTask();
			if(proc.getPoolName().contains("Arduino")){
				int index = 0;
				for(Task t: taskList){
					t.setIndex(index++);
					mArduinoTask.put(t.getName(), t);
				}
			}
			else if(proc.getPoolName().contains("TIEvalbot")){
				int index = 0;
				for(Task t: taskList){
					t.setIndex(index++);
					mEvalbotTask.put(t.getName(), t);
				}
			}
		}
		*/
		
		// For parent task! Maybe need to fix...
		int index = mEvalbotTask.size();
		for(Task t: mTask.values()){
			if(!mEvalbotTask.containsKey(t.getName()) && !mArduinoTask.containsKey(t.getName())){
				t.setIndex(index++);
				mEvalbotTask.put(t.getName(), t);
			}
		}

		if(mLibrary != null){
			for(Library lib: mLibrary.values()){
				if(mProcessor.get(lib.getProc()).getPoolName().contains("Arduino"))			mArduinoLibrary.put(lib.getName(), lib);
				else if(mProcessor.get(lib.getProc()).getPoolName().contains("TIEvalbot"))	mEvalbotLibrary.put(lib.getName(), lib);
			}
		}
		
		int a_index = 0, t_index = 0;
		for(Queue q: mQueue.values()){
			if(mArduinoTask.containsKey(q.getSrc()) || mArduinoTask.containsKey(q.getDst()))	{mArduinoQueue.put(Integer.parseInt(q.getIndex()), q);}
			if(mEvalbotTask.containsKey(q.getSrc()) || mEvalbotTask.containsKey(q.getDst()))	{mEvalbotQueue.put(Integer.parseInt(q.getIndex()), q);}
		}
	}
	
	public void generateArduino(){
		generateProcIno(mArduinoOutputPath + "/Arduino.ino", mTranslatorPath + "templates/target/arduino/proc.ino.template");
		
		for(Library l: mLibraryStubList){
			File fileOut = new File(mArduinoOutputPath + "/" + l.getName() + "_data_structure.h");
			CommonLibraries.Library.generateLibraryDataStructureHeader_16bit(fileOut, l);
		}
		for(Library l: mLibraryWrapperList){
			File fileOut = new File(mArduinoOutputPath + "/" + l.getName() + "_data_structure.h");
			CommonLibraries.Library.generateLibraryDataStructureHeader_16bit(fileOut, l);
		}
		
		for(Task t: mArduinoTask.values()){
			generateTaskIno(mArduinoOutputPath + "/" + t.getName() + ".ino", mOutputPath + "/" + t.getCICFile(), t);
		}
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
		
		code += "#define TASK_INIT void " + task.getName() + "_init(int TASK_ID)\n";
		code += "#define TASK_GO void " + task.getName() + "_go()\n";
		code += "#define TASK_WRAPUP void " + task.getName() + "_wrapup()\n";
		
		code += "#define BUF_SEND sendData\n";
		code += "#define MQ_SEND sendData\n";
		code += "#define STATIC static\n";
		
		for(String port: task.getPortList().keySet())
			code += "#define port_" + port + " " + task.getName() + "_port_" + port + "\n";
		
		code += "\n\n";
		code += mTaskCode;
		code += "\n\n";
		
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
		
		String taskStructure = "";
		for(Task task: mArduinoTask.values()){
			taskStructure += "{";
			taskStructure += task.getIndex() + ", ";
			
			int result = 0;
			int period = Integer.parseInt(task.getPeriod());
			String unit = task.getPeriodMetric();
			
			if(unit.equalsIgnoreCase("MS"))			result = period;
			else if(unit.equalsIgnoreCase("S"))		result = period * 1000;
			else if(unit.equalsIgnoreCase("US"))	result = 1;
			
			taskStructure += result + ", 0},\n";
		}
		code = code.replace("##TASK_STRUCTURE", taskStructure);
		
		String numTasks = "static int num_tasks = " + mArduinoTask.size() + ";\n";
		code = code.replace("##NUM_TASKS", numTasks);
		
		String channelStructure = "";
		for(Queue queue: mArduinoQueue.values()){
			if(mArduinoTask.get(queue.getSrc()) != null)
				channelStructure += "\t{" + mArduinoTask.get(queue.getSrc()).getIndex() +","
					+ "\"" + queue.getSrcPortName() +"\","
					+ queue.getIndex()+ ",'w'},\n";
			if(mArduinoTask.get(queue.getDst()) != null)
				channelStructure += "\t{" + mArduinoTask.get(queue.getDst()).getIndex() +","
				+ "\"" + queue.getDstPortName() +"\","
				+ queue.getIndex()+ ",'r'},\n";
		}	
		code = code.replace("##CHANNEL_STRUCTURE", channelStructure);
		
		String numChannels = "static int num_channels = " + mArduinoQueue.size() + ";\n";
		code = code.replace("##NUM_CHANNELS", numChannels);
		
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
	
		String taskFuncDecl = "";
		String taskInitCall = "";
		String taskGoCall = "";
		
		for(Task task: mArduinoTask.values()){
			taskFuncDecl += "void " + task.getName() + "_init(int);\n";
			taskFuncDecl += "void " + task.getName() + "_go();\n";
			taskFuncDecl += "void " + task.getName() + "_wrapup();\n\n";
			
			taskInitCall += task.getName() + "_init(" + task.getIndex() + ");\n";
			taskGoCall += "\tif(tasks[i].index == " + task.getIndex() +")  " + task.getName() + "_go();\n";
		}
		
		code = code.replace("##TASK_INIT_CALL", taskInitCall);
		code = code.replace("##TASK_GO_CALL", taskGoCall);
		code = code.replace("##TASK_FUNCTION_DECLARATION", taskFuncDecl);
		
		return code;
	}
	
	public void generateTaskIno(){

	}
	
	public void generateTIEvalbot(){
		String newOutputPath = mEvalbotOutputPath + "/EvalBoards/TI/LM3S9B92-EVALBOT/IAR/uCOS-III-App/";

		// Make Output Directory
		File f = new File(mEvalbotOutputPath);
		
		if(!f.exists())	f.mkdir();
		
		// Copy uCOS folder
		File s = new File(mTranslatorPath + "templates/target/tievalbot/uCOS");
		try {
			CommonLibraries.Util.copyAllFiles(f, s);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		// generate common files
		CommonLibraries.CIC.generateCommonCode("TIEvalbot", newOutputPath, mTranslatorPath, mEvalbotTask, mEvalbotQueue, mEvalbotLibrary, mThreadVer, mAlgorithm, mControl);
		
		Util.copyFile(newOutputPath+"target_task_model.h", mTranslatorPath + "templates/common/task_model/ucos_3.template");
		Util.copyFile(newOutputPath+"target_system_model.h", mTranslatorPath + "templates/common/system_model/ucos_3_evalbot.template");
		
		// generate proc.c or proc.cpp
		String fileOut = null;	
		String templateFile = "";
		
		// generate cic_tasks.h
		fileOut = newOutputPath + "task_def.h";
		templateFile = mTranslatorPath + "templates/common/common_template/task_def.h.template";
		CommonLibraries.CIC.generateTaskDataStructure(fileOut, templateFile, mTask, mGlobalPeriod, mGlobalPeriodMetric, mThreadVer, mCodeGenType, mVTask, mPVTask);
		
		String srcExtension = "";
		if(mLanguage.equals("c++"))	srcExtension = ".cpp";
		else						srcExtension = ".c";
		
		// generate task_name.c (include task_name.cic)
		for(Task t: mEvalbotTask.values()){
			if(t.getHasSubgraph().equalsIgnoreCase("Yes") && t.getHasMTM().equalsIgnoreCase("Yes")){
				fileOut = newOutputPath + t.getName() + srcExtension;
				templateFile = mTranslatorPath + "templates/common/common_template/task_mtm_code_template.c";
				CommonLibraries.CIC.generateTaskMTMCode(fileOut, templateFile, t, mAlgorithm);
			}
			else{
				fileOut = newOutputPath + t.getName() + srcExtension;
				if(mLanguage.equals("c++"))	templateFile = mTranslatorPath + "templates/common/common_template/task_code_template.cpp";
				else						templateFile = mTranslatorPath + "templates/common/common_template/task_code_template.c";
				CommonLibraries.CIC.generateTaskCode(fileOut, templateFile, t, mAlgorithm, mControl);
			}
		}
				
		// generate mtm files from xml files	
		for(Task t: mEvalbotTask.values()){
			if(t.getHasMTM().equalsIgnoreCase("Yes")){
				templateFile = mTranslatorPath + "templates/common/common_template/task_code_template.mtm";			
				CommonLibraries.CIC.generateMTMFile(newOutputPath, templateFile, t, mAlgorithm, mEvalbotTask, mPVTask, mEvalbotQueue, mCodeGenType);
			}
		}
		
		/////////////////////// Library ////////////////////////////
		
		// Libraries are mapped on the same target proc
		for(Library library: mLibrary.values()){
			int procId = library.getProc();
			Processor proc = mProcessor.get(procId);
			if(proc.getPoolName().contains("Evalbot")){
				// case 1: local only
				// if()
					CommonLibraries.Library.generateLibraryCode(newOutputPath, library, mAlgorithm);
				// case 2: local + remote 
			    // else if()
					//CommonLibraries.Library.generateLibraryWrapperCode(newOutputPath, mTranslatorPath, library, mAlgorithm);
					//mLibraryWrapperList.add(library);
			}
		}
		
		// Libraries are mapped on other target procs
		for(Task t: mTask.values()){
			String taskName = t.getName();
			String libPortName = "";
			String libName = "";
			Library library = null;
			if(t.getLibraryPortList().size() != 0){
				List<LibraryMasterPortType> libportList = t.getLibraryPortList();
				for(int i=0; i<libportList.size(); i++){
					LibraryMasterPortType libPort = libportList.get(i);
					libPortName = libPort.getName();
					break;
				}
				if(libPortName == ""){
					System.out.println("Library task does not exist!");
					System.exit(-1);
				}
				else{
					for(int i=0; i<mAlgorithm.getLibraryConnections().getTaskLibraryConnection().size(); i++){
						TaskLibraryConnectionType taskLibCon = mAlgorithm.getLibraryConnections().getTaskLibraryConnection().get(i);
						if(taskLibCon.getMasterTask().equals(taskName) && taskLibCon.getMasterPort().equals(libPortName)){
							libName = taskLibCon.getSlaveLibrary();
							break;
						}
					}
					if(!mLibrary.containsKey(libName)){
						for(Library lib: mGlobalLibrary.values()){
							if(lib.getName().equals(libName)){
								library = lib;
								break;
							}
						}
						if(library != null){
							Util.copyFile(newOutputPath + "/"+ library.getHeader(), mOutputPath + "/" + library.getHeader());
							CommonLibraries.Library.generateLibraryStubCode(newOutputPath, library, mAlgorithm, false);
							mLibraryStubList.add(library);
						}
					}
				}
			}
		}
		
		if(mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0 ){
			fileOut = newOutputPath + "lib_channels.h";
			templateFile = mTranslatorPath + "templates/common/library/lib_channels.h";
			CommonLibraries.Library.generateLibraryChannelHeader(fileOut, templateFile, mLibraryStubList, mLibraryWrapperList);
	
			fileOut = newOutputPath + "libchannel_def.h";
			templateFile = mTranslatorPath + "templates/common/library/libchannel_def.h.template";
			CommonLibraries.Library.generateLibraryChannelDef(fileOut, templateFile, mLibraryStubList, mLibraryWrapperList);
		}
		//////////////////////////////////////////////////////////////
		
		fileOut = newOutputPath+"proc" + srcExtension;
		if(mThreadVer.equals("s"))
			templateFile = mTranslatorPath + "templates/common/common_template/proc.c.template";	// NEED TO FIX
		else
			templateFile = mTranslatorPath + "templates/common/common_template/proc.c.template";
	
		generateProcCode(fileOut, templateFile);
		
		fileOut = newOutputPath + "uCOS-III-App.ewp";
		templateFile = mTranslatorPath + "templates/target/tievalbot/template/uCOS-III-App.ewp.template";
		generateProjectFile(fileOut, templateFile);
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
	
		// LIB_INIT_WRAPUP //
		String libraryInitWrapup="";
		if(mLibrary != null){
			for(Library library: mLibrary.values()){
				libraryInitWrapup += "extern void l_"+library.getName()+"_init(void);\n";
				libraryInitWrapup += "extern void l_"+library.getName()+"_wrapup(void);\n\n";
			}
		}
		if(mLibraryStubList != null){
			for(Library library: mLibraryStubList){
				libraryInitWrapup += "extern void l_"+library.getName()+"_init(void);\n";
				libraryInitWrapup += "extern void l_"+library.getName()+"_wrapup(void);\n\n";
			}
		}
		
		libraryInitWrapup += "static void init_libs() {\n";
		if(mLibrary != null){
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
		
		// EXTERNAL_GLOBAL_HEADERS //
		String externalGlobalHeaders = "";
		if(mAlgorithm.getHeaders()!=null){
			for(String header: mAlgorithm.getHeaders().getHeaderFile())
				externalGlobalHeaders += "#include\"" + header +"\"\n";
		}
		code = code.replace("##EXTERNAL_GLOBAL_HEADERS", externalGlobalHeaders);
		////////////////////////////
		
		// SCHEDULE_CODE //
		String outPath = mOutputPath + "/convertedSDF3xml/";
		String staticScheduleCode = CommonLibraries.Schedule.generateSingleProcessorStaticScheduleCode(outPath, mTask, mVTask);
		code = code.replace("##SCHEDULE_CODE", staticScheduleCode);
		///////////////////
		
		// COMM_CODE //
		String commCode = "";
		code = code.replace("##COMM_CODE", commCode);
		///////////////////
		
		// COMM_SENDER_ROUTINE //
		String commSenderRoutine = "";
		code = code.replace("##COMM_SENDER_ROUTINE", commSenderRoutine);
		///////////////////
		
		// COMM_RECEIVER_ROUTINE //
		String commReceiverRoutine = "";
		code = code.replace("##COMM_RECEIVER_ROUTINE", commReceiverRoutine);
		///////////////////	
		
		// OS_DEPENDENT_INCLUDE_HEADERS //
		String os_dep_includeHeader = "";
		os_dep_includeHeader = "";
		code = code.replace("##OS_DEPENDENT_INCLUDE_HEADERS", os_dep_includeHeader);
		/////////////////////////////////
		
		// TARGET_DEPENDENT_HEADER_INCLUDE //
		String target_dep_includeHeader = "";
		code = code.replace("##TARGET_DEPENDENT_INCLUDE_HEADERS", target_dep_includeHeader);
		/////////////////////////////////

		// TARGET_DEPENDENT_IMPLEMENTATION //
		templateFile = mTranslatorPath + "templates/target/tievalbot/template/target_dependent_impl.template";
		String targetDependentImpl = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TARGET_DEPENDENT_IMPLEMENTATION");
		code = code.replace("##TARGET_DEPENDENT_IMPLEMENTATION", targetDependentImpl);
		/////////////////////////////////////
		
		// TARGET_DEPENDENT_INIT_CALL //
		String targetDependentInit = "";
		targetDependentInit = "target_dependent_init();\n";
		code = code.replace("##TARGET_DEPENDENT_INIT_CALL", targetDependentInit);
		////////////////////////////////
				
		// TARGET_DEPENDENT_WRAPUP_CALL //
		String targetDependentWrapup = "target_dependent_wrapup();\n";
		code = code.replace("##TARGET_DEPENDENT_WRAPUP_CALL", targetDependentWrapup);
		//////////////////////////////////
		
		templateFile = mTranslatorPath + "templates/common/channel_manage/ucos.template";
		// INIT_WRAPUP_CHANNELS //
		String initWrapupChannels = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##INIT_WRAPUP_CHANNELS");
		code = code.replace("##INIT_WRAPUP_CHANNELS", initWrapupChannels);
		//////////////////////////
		
		// INIT_WRAPUP_CHANNELS //
		String initWrapupTaskChannels = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##INIT_WRAPUP_TASK_CHANNELS");
		code = code.replace("##INIT_WRAPUP_TASK_CHANNELS", initWrapupTaskChannels);
		//////////////////////////
		
		// READ_WRITE_PORT //
		String readWritePort = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##READ_WRITE_PORT");
		code = code.replace("##READ_WRITE_PORT", readWritePort);
		/////////////////////
		
		// READ_WRITE_BUF_PORT //
		String readWriteBufPort = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##READ_WRITE_BUF_PORT");
		code = code.replace("##READ_WRITE_BUF_PORT", readWriteBufPort);
		/////////////////////
		
		// READ_WRITE_AC_PORT //
		code = code.replace("##READ_WRITE_AC_PORT", "");
		////////////////////////
		
		templateFile = mTranslatorPath + "templates/target/tievalbot/template/task_execution.template";
		// DATA_TASK_ROUTINE //
		String dataTaskRoutine = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TASK_ROUTINE");
		code = code.replace("##DATA_TASK_ROUTINE", dataTaskRoutine);
		//////////////////////////
		
		// TIME_TASK_ROUTINE //
		code = code.replace("##TIME_TASK_ROUTINE", "");
		//////////////////////////
		
		// EXECUTE_TASKS //
		String executeTasks = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##EXECUTE_TASKS");
		code = code.replace("##EXECUTE_TASKS", executeTasks);
		//////////////////////////
		
		// CONTROL_RUN_TASK //
		String controlRunTask = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##CONTROL_RUN_TASK");
		code = code.replace("##CONTROL_RUN_TASK", controlRunTask);
		//////////////////////////
		
		// CONTROL_STOP_TASK //
		String controlStopTask = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##CONTROL_STOP_TASK");
		code = code.replace("##CONTROL_STOP_TASK", controlStopTask);
		//////////////////////////
				
		// CONTROL_RESUME_TASK //
		String controlResumeTask = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##CONTROL_RESUME_TASK");
		code = code.replace("##CONTROL_RESUME_TASK", controlResumeTask);
		//////////////////////////
		
		// CONTROL_SUSPEND_TASK //
		String controlSuspendTask = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##CONTROL_SUSPEND_TASK");
		code = code.replace("##CONTROL_SUSPEND_TASK", controlSuspendTask);
		//////////////////////////
			
		// SET_PROC //
		String setProc= "";
		code = code.replace("##SET_PROC", setProc);
		//////////////
			
		// GET_CURRENT_TIME_BASE //
		templateFile = mTranslatorPath + "templates/common/time_code/ucos_3.template";
		String getCurrentTimeBase = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##GET_CURRENT_TIME_BASE");
		code = code.replace("##GET_CURRENT_TIME_BASE", getCurrentTimeBase);
		///////////////////////////
		
		// CHANGE_TIME_UNIT //
		String changeTimeUnit = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##CHANGE_TIME_UNIT");
		code = code.replace("##CHANGE_TIME_UNIT", changeTimeUnit);
		///////////////////////////
			
		// DEBUG_CODE //
		code = code.replace("##DEBUG_CODE_IMPLEMENTATION", "");
		///////////////////////////
		
		// MAIN_FUNCTION //
		templateFile = mTranslatorPath + "templates/target/tievalbot/template/target_dependent_impl.template";
		String mainFunc = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##MAIN_FUNCTION");
		code = code.replace("##MAIN_FUNCTION", mainFunc);
		//////////////
		
		// LIB_INCLUDE //
		String libInclude = "";
		if(mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0 ){
			libInclude += "#include \"lib_channels.h\"\n#include \"libchannel_def.h\"\n";
			libInclude += "#define num_libchannels (int)(sizeof(lib_channels)/sizeof(lib_channels[0]))\n";
			libInclude += "static int *lib_read_waiting_list[num_libchannels];\n";
			libInclude += "static int *lib_write_waiting_list[num_libchannels];\n";
			libInclude += "#define LIB_CONN 1";
		}
		code = code.replace("##LIB_INCLUDES", libInclude);
		//////////////
		
		// CONN_INCLUDES //
		String conInclude = "";
		code = code.replace("##CONN_INCLUDES", conInclude);
		//////////////
		
		templateFile = mTranslatorPath + "templates/common/lib_channel_manage/ucos_3.template";
		// INIT_WRAPUP_LIB_CHANNELS //
		String initWrapupLibChannels = "";
		if(mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0 )
			initWrapupLibChannels = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##INIT_WRAPUP_LIBRARY_CHANNELS");
		code = code.replace("##INIT_WRAPUP_LIBRARY_CHANNELS", initWrapupLibChannels);
		//////////////
		
		// READ_WRITE_LIB_PORT //
		String readWriteLibPort = "";
		if(mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0 )
			readWriteLibPort = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##READ_WRITE_LIB_PORT");
		code = code.replace("##READ_WRITE_LIB_PORT", readWriteLibPort);
		//////////////
		
		// LIB_INIT //
		String libInit = "";
		if(mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0 )	libInit = "\tinit_lib_channel();\n\tinit_libs();\n";
		else																libInit = "\tinit_libs();\n";
		code = code.replace("##LIB_INIT", libInit);
		//////////////
		
		// LIB_WRAPUP //
		String libWrapup = "";
		if(mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0 )	libWrapup = "\twrapup_libs();\n\twrapup_lib_channel();\n";
		else																libWrapup = "\twrapup_libs();\n";
		code = code.replace("##LIB_WRAPUP", libWrapup);
		//////////////
		
		// OUT_CONN_INIT // 
		String outConnInit = "";				
		code = code.replace("##OUT_CONN_INIT", outConnInit);
		//////////////
		
		// OUT_CONN_WRAPUP //
		String outConnwrapup = "";				
		code = code.replace("##OUT_CONN_WRAPUP", outConnwrapup);
		//////////////
		
		return code;
	}
	
	public void generateProjectFile(String mDestFile, String mTemplateFile)
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

			outstream.write(translateProjectCode(content).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public String translateProjectCode(String mContent)
	{
		String code = mContent;
		String taskFiles = "";
		
		for(Task t: mEvalbotTask.values()){
			taskFiles += "\t\t\t<file>\n"
					  + "\t\t\t\t<name>$PROJ_DIR$\\" + t.getName() + ".c</name>\n"
					  + "\t\t\t</file>\n";
		}
		
		for(Library l: mEvalbotLibrary.values()){
			taskFiles += "\t\t\t<file>\n"
					  + "\t\t\t\t<name>$PROJ_DIR$\\" + l.getName() + ".c</name>\n"
					  + "\t\t\t</file>\n";
		}
		
		for(Library ls: mLibraryStubList){
			taskFiles += "\t\t\t<file>\n"
					  + "\t\t\t\t<name>$PROJ_DIR$\\" + ls.getName() + "_stub.c</name>\n"
					  + "\t\t\t</file>\n";
		}
		
		for(Library lw: mLibraryWrapperList){
			taskFiles += "\t\t\t<file>\n"
					  + "\t\t\t\t<name>$PROJ_DIR$\\" + "l_" + lw.getName() + "_wrapper.c</name>\n"
					  + "\t\t\t</file>\n";
		}
		
		code = code.replace("##CIC_TASK_FILES", taskFiles);
		
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
