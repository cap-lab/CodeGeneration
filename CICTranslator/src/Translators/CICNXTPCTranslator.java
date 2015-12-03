package Translators;

import hopes.cic.xml.CICAlgorithmType;
import hopes.cic.xml.CICControlType;
import hopes.cic.xml.CICGPUSetupType;
import hopes.cic.xml.CICMappingType;
import hopes.cic.xml.CICScheduleType;
import hopes.cic.xml.ControlTaskType;
import hopes.cic.xml.ExclusiveControlTasksType;
import hopes.cic.xml.LibraryMasterPortType;
import hopes.cic.xml.TaskLibraryConnectionType;
import hopes.cic.xml.TaskParameterType;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import CommonLibraries.Util;
import InnerDataStructures.Argument;
import InnerDataStructures.Function;
import InnerDataStructures.Library;
import InnerDataStructures.Processor;
import InnerDataStructures.Communication;
import InnerDataStructures.Queue;
import InnerDataStructures.Task;

//hshong : using ROBOTC platform 
public class CICNXTPCTranslator implements CICTargetCodeTranslator {
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
	private Map<String, Library> mGlobalLibrary;
	private Map<Integer, Processor> mProcessor;
	List<Communication> mCommunication;
	
	private Map<String, Task> mVTask;
	
	private CICAlgorithmType mAlgorithm;
	private CICControlType mControl;
	private CICScheduleType mSchedule;
	private CICGPUSetupType mGpusetup;

	private String strategy;
	
	private String newOutputPath;
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
		mLanguage = language;
		
		mTask = task;
		mQueue = queue;
		mLibrary = library;
		mGlobalLibrary = globalLibrary;
		mProcessor = processor;
		
		mVTask = vtask;
		
		mAlgorithm = algorithm;
		mControl = control;
		mSchedule = schedule;
		mGpusetup = gpusetup;
		
		mLibraryStubList = new ArrayList<Library>();
		mLibraryWrapperList = new ArrayList<Library>();
		
		// Make Output Directory
		File f = new File(mOutputPath);	
		File nxt_f = new File(mOutputPath + "nxt/");	
		File host_f = new File(mOutputPath + "host/");	
		
		f.mkdir();
		nxt_f.mkdir();
		host_f.mkdir();
		
		newOutputPath = mOutputPath + "nxt/";
		
		// generate header files: cic_constant, cic_thread
		Util.copyFile(mOutputPath+"nxt/cic_thread.h", mTranslatorPath + "templates/target/NXT/nxtOSEK/thread.template");
		Util.copyFile(mOutputPath + "nxt/cic_control.h", mTranslatorPath + "templates/target/NXT/nxtOSEK/cic_control.template");
		
		String fileOut = null;
		String templateFile = "";
		
		fileOut = mOutputPath + "nxt/cic_control_def.h";
		templateFile = mTranslatorPath + "templates/target/NXT/nxtOSEK/control_def.h.template";
		generateControlDefHeader(fileOut, templateFile, mTask, mControl);
		
		Util.copyFile(mOutputPath + "nxt/cic_tasks.h", mTranslatorPath + "templates/target/NXT/nxtOSEK/cic_tasks.template");
		
		fileOut = mOutputPath + "nxt/cic_tasks_def.h";
		templateFile = mTranslatorPath + "templates/target/NXT/nxtOSEK/tasks_def.h.template";
		generateTaskDefHeader(fileOut, templateFile, mTask);
		
		fileOut = mOutputPath + "nxt/nxt_task_set.h";
		templateFile = mTranslatorPath + "templates/target/NXT/nxtOSEK/nxt_task_set.template";
		generateTaskSetHeader(fileOut, templateFile, mTask);
		
		Util.copyFile(mOutputPath + "nxt/CIC_port.h", mTranslatorPath + "templates/target/NXT/nxtOSEK/CIC_port.template");
				
		fileOut = mOutputPath + "nxt/include_cic_task.h";
		//templateFile = mTranslatorPath + "templates/target/NXT/nxtOSEK/nxt_task_set.template";
		generateIncludeCICTaskHeader(fileOut, mTask);
		
		fileOut = mOutputPath + "nxt/task_handler.h";
		generateTaskHandlerHeader(fileOut, mTask);
		
		Util.copyFile(mOutputPath + "nxt/cic_channels.h", mTranslatorPath + "templates/target/NXT/nxtOSEK/cic_channels.template");
		
		fileOut = mOutputPath + "nxt/cic_channels_def.h";
		templateFile = mTranslatorPath + "templates/target/NXT/nxtOSEK/channels_def.h.template";
		generateChannelDefHeader(fileOut, templateFile, mQueue);
		
		Util.copyFile(mOutputPath + "nxt/cic_portmap.h", mTranslatorPath + "templates/target/NXT/nxtOSEK/cic_portmap.template");
		
		fileOut = mOutputPath + "nxt/cic_portmap_def.h";
		templateFile = mTranslatorPath + "templates/target/NXT/nxtOSEK/portmap_def.h.template";
		generatePortmapDefHeader(fileOut, templateFile, mQueue);
		
		String srcExtension = ".c";
		
		//generate task_name.c (include task_name.cic)
		for(Task t: mTask.values())
		{
			//no mtm assumed
			fileOut = mOutputPath + "nxt/"+ t.getName() + srcExtension;
			templateFile = mTranslatorPath + "templates/target/NXT/nxtOSEK/task_code_template.c";
			generateTaskCode(fileOut, templateFile, t, mAlgorithm);
		}
		
		generateLibraryCode();
		
		//generate proc.c
		fileOut = mOutputPath + "nxt/proc" + srcExtension;
		templateFile = mTranslatorPath + "templates/target/NXT/nxtOSEK/proc.c.template";
		
		generateProcCode(fileOut, templateFile);
		
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
		
		mLibraryStubList = new ArrayList<Library>();
		mLibraryWrapperList = new ArrayList<Library>();
		
		// Make Output Directory
		File f = new File(mOutputPath);	
		File nxt_f = new File(mOutputPath + "nxt/");	
		File host_f = new File(mOutputPath + "host/");	
		
		f.mkdir();
		nxt_f.mkdir();
		host_f.mkdir();
		
		newOutputPath = mOutputPath + "nxt/";
		
		// generate header files: cic_constant, cic_thread
		Util.copyFile(mOutputPath+"nxt/cic_thread.h", mTranslatorPath + "templates/target/NXT/nxtOSEK/thread.template");
		Util.copyFile(mOutputPath + "nxt/cic_control.h", mTranslatorPath + "templates/target/NXT/nxtOSEK/cic_control.template");
		
		String fileOut = null;
		String templateFile = "";
		
		fileOut = mOutputPath + "nxt/cic_control_def.h";
		templateFile = mTranslatorPath + "templates/target/NXT/nxtOSEK/control_def.h.template";
		generateControlDefHeader(fileOut, templateFile, mTask, mControl);
		
		Util.copyFile(mOutputPath + "nxt/cic_tasks.h", mTranslatorPath + "templates/target/NXT/nxtOSEK/cic_tasks.template");
		
		fileOut = mOutputPath + "nxt/cic_tasks_def.h";
		templateFile = mTranslatorPath + "templates/target/NXT/nxtOSEK/tasks_def.h.template";
		generateTaskDefHeader(fileOut, templateFile, mTask);
		
		fileOut = mOutputPath + "nxt/nxt_task_set.h";
		templateFile = mTranslatorPath + "templates/target/NXT/nxtOSEK/nxt_task_set.template";
		generateTaskSetHeader(fileOut, templateFile, mTask);
		
		Util.copyFile(mOutputPath + "nxt/CIC_port.h", mTranslatorPath + "templates/target/NXT/nxtOSEK/CIC_port.template");
				
		fileOut = mOutputPath + "nxt/include_cic_task.h";
		//templateFile = mTranslatorPath + "templates/target/NXT/nxtOSEK/nxt_task_set.template";
		generateIncludeCICTaskHeader(fileOut, mTask);
		
		fileOut = mOutputPath + "nxt/task_handler.h";
		generateTaskHandlerHeader(fileOut, mTask);
		
		Util.copyFile(mOutputPath + "nxt/cic_channels.h", mTranslatorPath + "templates/target/NXT/nxtOSEK/cic_channels.template");
		
		fileOut = mOutputPath + "nxt/cic_channels_def.h";
		templateFile = mTranslatorPath + "templates/target/NXT/nxtOSEK/channels_def.h.template";
		generateChannelDefHeader(fileOut, templateFile, mQueue);
		
		Util.copyFile(mOutputPath + "nxt/cic_portmap.h", mTranslatorPath + "templates/target/NXT/nxtOSEK/cic_portmap.template");
		
		fileOut = mOutputPath + "nxt/cic_portmap_def.h";
		templateFile = mTranslatorPath + "templates/target/NXT/nxtOSEK/portmap_def.h.template";
		generatePortmapDefHeader(fileOut, templateFile, mQueue);
		
		String srcExtension = ".c";
		
		//generate task_name.c (include task_name.cic)
		for(Task t: mTask.values())
		{
			//no mtm assumed
			fileOut = mOutputPath + "nxt/"+ t.getName() + srcExtension;
			templateFile = mTranslatorPath + "templates/target/NXT/nxtOSEK/task_code_template.c";
			generateTaskCode(fileOut, templateFile, t, mAlgorithm);
		}
		
		generateLibraryCode();
		
		//generate proc.c
		fileOut = mOutputPath + "nxt/proc" + srcExtension;
		templateFile = mTranslatorPath + "templates/target/NXT/nxtOSEK/proc.c.template";
		
		generateProcCode(fileOut, templateFile);
		
		copyTaskFilesToTargetFolder();
		
		return 0;
	}
	
	public void generateLibraryCode(){
	/////////////////////// Library ////////////////////////////
		String fileOut = "";
		String templateFile = "";
		
		// Libraries are mapped on the same target proc
		for(Library library: mLibrary.values()){
			int procId = library.getProc();
			Processor proc = mProcessor.get(procId);
			if(proc.getPoolName().contains("NXT")){
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
							Util.copyFile(newOutputPath + "/"+ library.getHeader(), mOutputPath + "../" + library.getHeader());
							generateLibraryStubCode(newOutputPath, mTranslatorPath, library, mAlgorithm);
							mLibraryStubList.add(library);
						}
					}
				}
			}
		}
		
		if(mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0 ){
			Util.copyFile(mOutputPath + "nxt/LIB_port.h", mTranslatorPath + "templates/target/NXT/nxtOSEK/LIB_port.template");			
						
			fileOut = newOutputPath + "lib_channels.h";
			templateFile = mTranslatorPath + "templates/target/NXT/nxtOSEK/lib_channels.h.template";
			generateLibraryChannelHeader(fileOut, templateFile, mLibraryStubList, mLibraryWrapperList);
	
			fileOut = newOutputPath + "libchannel_def.h";
			templateFile = mTranslatorPath + "templates/target/NXT/nxtOSEK/libchannel_def.h.template";
			generateLibraryChannelDef(fileOut, templateFile, mLibraryStubList, mLibraryWrapperList);
		}
		//////////////////////////////////////////////////////////////
	}
	
	public void generateLibraryChannelHeader(String file, String mTemplateFile, ArrayList<InnerDataStructures.Library> mStubList, ArrayList<InnerDataStructures.Library> mWrapperList)
	{
		File fileOut = new File(file);
		File templateFile = new File(mTemplateFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);
			
			outstream.write(translateLibraryChannelHeader(content, mStubList, mWrapperList).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public String translateLibraryChannelHeader(String mContent, ArrayList<InnerDataStructures.Library> mStubList, ArrayList<InnerDataStructures.Library> mWrapperList)
	{
		String code = mContent;
		String headerIncludeCode="";
		String channelEntriesCode="";
		
		for(InnerDataStructures.Library library: mStubList)
			headerIncludeCode += "#include \"" + library.getName()+ "_data_structure.h\"\n";
		for(InnerDataStructures.Library library: mWrapperList)
			headerIncludeCode += "#include \"" + library.getName()+ "_data_structure.h\"\n";

		code = code.replace("##HEADER_INCLUDE", headerIncludeCode);
		
		return code;
	}
	
	public void generateLibraryChannelDef(String file, String mTemplateFile, ArrayList<InnerDataStructures.Library> mStubList, ArrayList<InnerDataStructures.Library> mWrapperList)
	{
		File fileOut = new File(file);
		File templateFile = new File(mTemplateFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);
			
			outstream.write(translateLibraryChannelDef(content, mStubList, mWrapperList).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public String translateLibraryChannelDef(String mContent, ArrayList<InnerDataStructures.Library> mStubList, ArrayList<InnerDataStructures.Library> mWrapperList)
	{
		String code = mContent;
		String channelSize="";
		int ichannelSize = 0;
		
		ichannelSize = (mStubList.size() + mWrapperList.size())*2;
		channelSize = String.valueOf(ichannelSize);
		
		code = code.replace("##lib_channels_size", channelSize);
		
		return code;
	}
	
	public void generateLibraryStubCode(String mDestFile, String mTranslatorPath, InnerDataStructures.Library library, CICAlgorithmType mAlgorithm){		
		File fileOut = new File(mDestFile + "l_" + library.getName() + "_stub.c");
		generateLibraryStubFile(fileOut, 0, library, mAlgorithm);
		
		fileOut = new File(mDestFile + library.getName() + "_data_structure.h");
		generateLibraryDataStructureHeader(fileOut, library);
	}
	


	public static void generateLibraryStubFile(File fileOut, int taskLibraryFlag, InnerDataStructures.Library library, CICAlgorithmType mAlgorithm)
	{

		try {
			FileOutputStream outstream = new FileOutputStream(fileOut);
			outstream.write(translateLibraryStubFile(taskLibraryFlag, library, mAlgorithm).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}


	public static String translateLibraryStubFile(int taskLibraryFlag, InnerDataStructures.Library library, CICAlgorithmType mAlgorithm){
		int index = 0;
		String content = new String();	
	    
	    content += "#include \"" + library.getName() + "_data_structure.h\"\n";

	    for(Function func: library.getFuncList()){
	    	content += func.getReturnType() + " " + func.getFunctionName() + "(";
	    	//content += "LIBFUNC(" + func.getReturnType() + ", " + func.getFunctionName();
	    	int count = 0;
	    	for(Argument arg: func.getArgList()){
	    		content += arg.getType() + " " + arg.getVariableName();
	    		count++;
	    		if(func.getArgList().size() > count) content += ", ";
	    	}
	    	
	    	content += ")\n{\n";
	    	content += "\t" + library.getName() + "_func_data send_data;\n";
	    	if(!func.getReturnType().equals("void"))	content += "\t" + library.getName() + "_ret_data receive_data;\n\t";
	    	
	    	if(!func.getReturnType().equals("void"))	content += func.getReturnType() + " ret;\n\n";
	    	
	    	content += "\tint write_channel_id = init_lib_port(" + library.getIndex() + ", 'w');\n\n";
	    	if(!func.getReturnType().equals("void"))	
	    		content += "\tint read_channel_id = init_lib_port(" + library.getIndex() + ", 'r');\n\n";
	    	
	    	if(taskLibraryFlag == 1){
	    		content += "\tsend_data.task_library = 1;\n";
	    		content += "\tsend_data.task_id = get_mytask_id();\n";
	    	}
	    	else if(taskLibraryFlag == 2){
	    		content += "\tsend_data.task_library = 2;\n";
	    		content += "\tsend_data.task_id = " + library.getIndex() + ";\n";
	    	}
	    	content += "\tsend_data.func_num = " + func.getIndex() + ";\n"; 
	    	
	    	for(Argument arg: func.getArgList())
	    		content += "\tsend_data.func." + func.getFunctionName() + "." + arg.getVariableName() + " = " + arg.getVariableName() + ";\n";
	    	
	    	content += "\n\t// write port\n";
	    	//content += "\tlock_lib_channel(channel_id);\n";
	    	
	    	content += "\tLIB_SEND(write_channel_id, &send_data, sizeof(" + library.getName() + "_func_data));\n";
    	
	    	if(!func.getReturnType().equals("void")){
		    	content += "\t// read port\n";
		    	content += "\tLIB_RECEIVE(read_channel_id, &receive_data, sizeof(" + library.getName() + "_ret_data));\n\n";
		    	//content += "\tunlock_lib_channel(channel_id);\n";
		    	
		    	content += "\tif(receive_data.func_num == " + func.getIndex() + ")\n";
	    		content += "\t\tret = receive_data.ret.ret_" + func.getFunctionName() + ";\n";
	    		content += "\t//else	//for bluetooth error, but now it makes error sometimes.. \n";
	    		content += "\t\t//ret = 0;\n\n";
	    		content += "\treturn ret;\n";
	    	}
	    	
	    	content += "}\n\n";
	    	
	    }
	    content += "#undef LIBFUNC\n\n";
	    
		return content;
	}
	

	public static void generateLibraryDataStructureHeader(File fileOut, InnerDataStructures.Library library)
	{
		try {
			FileOutputStream outstream = new FileOutputStream(fileOut);
			outstream.write(translateLibraryDataStructureHeader(library).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
				e.printStackTrace();
		}
	}
	
	public static String translateLibraryDataStructureHeader(InnerDataStructures.Library library){
		int index = 0;
		String content = new String();
		
		content += "#ifndef _" + library.getName().toUpperCase() + "_DATA_STRUCTURE_H_\n";
		content += "#define _" + library.getName().toUpperCase() + "_DATA_STRUCTURE_H_\n\n";
								
		for(String extraHeader: library.getExtraHeader())
			content += "#include \"" + extraHeader + "\"\n\n";
		
		//LIBCALL을 위해서 define이 필요! 단 매개변수 개수에 맞춰서 불러줘야함. 
		//이는 #define LIBCALL(a, b, ...) b(__VA_ARGS__)를 지원해주지 않는 RobotC때문에 해야하는 일인데 
		//매개변수가 0개짜리가 쓰이면  #define LIBCALL(a, b) b()로
		//매개 변수가 1개까지가 쓰이면 #define LIBCALL(a, b, c) b(c)로 선언을 해주어야함. 
		String defineLibCall = "";
		for(Function func: library.getFuncList())
		{
			int num_args = func.getArgList().size();
			if(num_args >= 0)
			{
				defineLibCall += "#define LIBCALL(a, b";
				for(int i = 0; i < num_args; i++)
				{					
					int ascii = (int)'c' + i;
					char retAlpha = (char)ascii;
					defineLibCall += ", " + retAlpha; 					
				}
				defineLibCall += ") b(";
				for(int i = 0; i < num_args; i++)
				{					
					int ascii = (int)'c' + i;
					char retAlpha = (char)ascii;
					if(i > 0)
						defineLibCall += ", ";
					defineLibCall += retAlpha; 					
				}
				defineLibCall += ")\n";
			}
		}
		content += defineLibCall + "\n";
		
		String stringtypetemp = "";
		for(Function func: library.getFuncList()){			
			if(func.getArgList().size() > 0){		
				content += "typedef struct {\n";
				for(Argument arg: func.getArgList()){
					if(arg.getType().equals("int"))
						stringtypetemp = "long";
					else
						stringtypetemp = arg.getType();
					content += "\t" + stringtypetemp + " " + arg.getVariableName() + ";\n";
					index++;
				}
				if(index == 0)	content += "\tint temp;\n";
				content += "} " + func.getFunctionName().toUpperCase() + ";\n";
			}
		}
		
		content += "union FUNC {\n";
		for(Function func: library.getFuncList()){
			if(func.getArgList().size() > 0){				
				content += "\t" + func.getFunctionName().toUpperCase() + " " + func.getFunctionName() + ";\n";
			}
		}
		content += "} Func;\n";
		
		content += "typedef struct {\n";
		content += "\tlong task_id;\n";
		content += "\tlong func_num;\n";
		content += "\tFunc func;\n";
		content += "} " + library.getName() + "_func_data;\n\n";
						
		index = 0;
		content += "union RET {\n";
		for(Function func: library.getFuncList()){
			if(func.getReturnType().equals("int"))
				stringtypetemp = "long";
			else
				stringtypetemp = func.getReturnType();
			if(func.getReturnType().equals("void"))	content += "\tlong ret_" + func.getFunctionName() + ";\n";
			else		content += "\t" + stringtypetemp + " ret_" + func.getFunctionName() + ";\n";
			index++;
		}
		if(index == 0)	content += "\t\tint temp;\n";
		content += "} Ret;\n";
		
		content += "typedef struct {\n";
		content += "\tlong task_id;\n";
		content += "\tlong func_num;\n";
		content += "\tRet ret;\n";
		content += "} " + library.getName() + "_ret_data;\n\n";
				
//				
//		
//		content += "\t";
//		
//		content += "typedef struct {\n" + "\tint func_num;\n" + "\tunion {\n";
//		for(Function func: library.getFuncList()){
//			if(func.getReturnType().equals("void"))	content += "\t\tint ret_" + func.getFunctionName() + ";\n";
//			else		content += "\t\t" + func.getReturnType() + " ret_" + func.getFunctionName() + ";\n";
//			index++;
//		}
//		if(index == 0)	content += "\t\t\tint temp;\n";
//		
//		content += "\t} ret;\n";
//		content += "} " + library.getName() + "_ret_data;\n\n";
		content += "\n#endif\n";
	    	    
		return content;
	}

	
	
	public void copyTaskFilesToTargetFolder(){
		String src = mOutputPath + "../";
		String dst = mOutputPath + "nxt/";
		
		System.out.println("output: " + mOutputPath);
		
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
	
	public void generateControlDefHeader(String mDestFile, String mTemplateFile, Map<String, Task> mTask, CICControlType mControl)
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
			
			outstream.write(translateControlDefHeader(content, mTask, mControl).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public String translateControlDefHeader(String mContent, Map<String, Task> mTask, CICControlType mControl)
	{
		String code = mContent;
		
		String paramSize="";
		int iparamSize = 0;
		if(mControl.getControlTasks() != null)
		{			
			for(Task task: mTask.values())
			{
				for(TaskParameterType parameter: task.getParameter())
					iparamSize++;
			}
			
			paramSize = String.valueOf(iparamSize);
			
		}
		
		String controlGroupCount = "1";
		if(mControl.getExclusiveControlTasksList() != null)	controlGroupCount = Integer.toString(mControl.getExclusiveControlTasksList().getExclusiveControlTasks().size());
		controlGroupCount = "#define CONTROL_GROUP_COUNT " + controlGroupCount;
		
		String controlChannelCount = "1";
		if(mControl.getControlTasks() != null)	controlChannelCount = Integer.toString(mControl.getControlTasks().getControlTask().size());
		controlChannelCount = "#define CONTROL_CHANNEL_COUNT " + controlChannelCount;
		
		code = code.replace("##param_size", paramSize);
		code = code.replace("##CONTROL_GROUP_COUNT", controlGroupCount);
		code = code.replace("##CONTROL_CHANNEL_COUNT", controlChannelCount);
		
		return code;
	}
	
	public void generateTaskDefHeader(String mDestFile, String mTemplateFile, Map<String, Task> mTask)
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
			
			outstream.write(translateTaskDefHeader(content, mTask).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public String translateTaskDefHeader(String mContent, Map<String, Task> mTask)
	{
		String code = mContent;
		
		String taskSize="";
		int itaskSize = 0;
		
		itaskSize = mTask.size();
		taskSize = String.valueOf(itaskSize);
				
		String mtmSize ="";
		int imtmSize = 1;
		
		for(Task task: mTask.values())
		{
			if(task.getHasMTM().equalsIgnoreCase("YES"))
				imtmSize++;
		}
		mtmSize = String.valueOf(imtmSize);
		
		code = code.replace("##tasks_size", taskSize);
		code = code.replace("##mtm_size", mtmSize);
		
		return code;
	}
	
	public void generateTaskSetHeader(String mDestFile, String mTemplateFile, Map<String, Task> mTask)
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
			
			outstream.write(translateTaskSetHeader(content, mTask).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public String translateTaskSetHeader(String mContent, Map<String, Task> mTask)
	{
		String code = mContent;
		
		String task_code;
		int index = 0;
		while(index < mTask.size())
		{
			task_code = "";
			Task task = null;
			for(Task t: mTask.values())
			{
				if(Integer.parseInt(t.getIndex()) == index)
				{
					task = t;
					break;
				}
			}
			
			task_code += "\ntask ";
			task_code += task.getName();
			task_code += "()\n";
			task_code += "{\n";
			task_code += "\tint task_index = ";
			task_code += task.getIndex();
			task_code += ";\n";
			if(task.getRunCondition().equals("DATA_DRIVEN") || task.getRunCondition().equals("CONTROL_DRIVEN"))
				task_code += "\tdata_task_routine(task_index);\n";
			else if(task.getRunCondition().equals("TIME_DRIVEN"))
				task_code += "\ttime_task_routine(task_index);\n";
			task_code += "}\n";
			code += task_code;
			index++;
		}
		
		return code;
	}
	
	public void generateIncludeCICTaskHeader(String mDestFile, Map<String, Task> mTask)
	{
		File fileOut = new File(mDestFile);
		try {
			FileOutputStream outstream = new FileOutputStream(fileOut);
			
			String content = "";//new String(buffer);
			
			outstream.write(translateIncludeCICTaskHeader(content, mTask).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public String translateIncludeCICTaskHeader(String mContent, Map<String, Task> mTask)
	{
		String code = mContent;
		
		String task_code = "";
		int index = 0;
		task_code += "#undef TASK_CODE_BEGIN\n#undef TASK_CODE_END\n#undef TASK_NAME\n#undef TASK_INIT\n#undef TASK_GO\n#undef TASK_WRAPUP\n#undef STATIC\n\n";
		while(index < mTask.size())
		{
			Task task = null;
			for(Task t: mTask.values())
			{
				if(Integer.parseInt(t.getIndex()) == index)
				{
					task = t;
					break;
				}
			}
			
			task_code += "#include \"";
			task_code += task.getName();
			task_code += ".c\"\n";
			task_code += "\n#undef Run\n#undef Stop\n#undef Wait\n\n";
			task_code += "#undef TASK_CODE_BEGIN\n#undef TASK_CODE_END\n#undef TASK_NAME\n#undef TASK_INIT\n#undef TASK_GO\n#undef TASK_WRAPUP\n#undef STATIC\n\n";
						
			code += task_code;
			index++;
		}
		
		return code;
	}
	
	public void generateTaskHandlerHeader(String mDestFile, Map<String, Task> mTask)
	{
		File fileOut = new File(mDestFile);
		try {
			FileOutputStream outstream = new FileOutputStream(fileOut);
			
			String content = "";			
			outstream.write(translateTaskHandlerHeader(content, mTask).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public String translateTaskHandlerHeader(String mContent, Map<String, Task> mTask)
	{
		String code = mContent;
		
		String task_code;
		int index = 0;
		task_code = "";
		task_code += "void TASK_INIT_CALLER(int index)\n";
		task_code += "{\n";
		task_code += "\tswitch(index)\n";
		task_code += "\t{\n";
		
		while(index < mTask.size())
		{
			Task task = null;
			for(Task t: mTask.values())
			{
				if(Integer.parseInt(t.getIndex()) == index)
				{
					task = t;
					break;
				}
			}
			
			task_code += "\t\tcase " + task.getIndex() + ":\n";
			task_code += "\t\t{\n";
			task_code += "\t\t\t";
			task_code += task.getName() + "_init(index);\n";
			task_code += "\t\t\tbreak;\n";
			task_code += "\t\t}\n";
			
			code += task_code;
			index++;
			task_code = "";
		}
		
		task_code += "\t}\n";
		task_code += "}\n\n";
		//
		index = 0;
		task_code += "void TASK_GO_CALLER(int index)\n";
		task_code += "{\n";
		task_code += "\tswitch(index)\n";
		task_code += "\t{\n";
		
		while(index < mTask.size())
		{
			Task task = null;
			for(Task t: mTask.values())
			{
				if(Integer.parseInt(t.getIndex()) == index)
				{
					task = t;
					break;
				}
			}
			
			task_code += "\t\tcase " + task.getIndex() + ":\n";
			task_code += "\t\t{\n";
			task_code += "\t\t\t";
			task_code += task.getName() + "_go(index);\n";
			task_code += "\t\t\tbreak;\n";
			task_code += "\t\t}\n";
			
			code += task_code;
			index++;
			task_code = "";
		}
		
		task_code += "\t}\n";
		task_code += "}\n\n";
		//
		index = 0;
		task_code += "void TASK_WRAPUP_CALLER(int index)\n";
		task_code += "{\n";
		task_code += "\tswitch(index)\n";
		task_code += "\t{\n";
		
		while(index < mTask.size())
		{
			Task task = null;
			for(Task t: mTask.values())
			{
				if(Integer.parseInt(t.getIndex()) == index)
				{
					task = t;
					break;
				}
			}
			
			task_code += "\t\tcase " + task.getIndex() + ":\n";
			task_code += "\t\t{\n";
			task_code += "\t\t\t";
			task_code += task.getName() + "_wrapup(index);\n";
			task_code += "\t\t\tbreak;\n";
			task_code += "\t\t}\n";
			
			code += task_code;
			index++;
			task_code = "";
		}
		
		task_code += "\t}\n";
		task_code += "}\n\n";
		//
		index = 0;
		task_code += "void TASK_CREATER(int index)\n";
		task_code += "{\n";
		task_code += "\tswitch(index)\n";
		task_code += "\t{\n";
		
		while(index < mTask.size())
		{
			Task task = null;
			for(Task t: mTask.values())
			{
				if(Integer.parseInt(t.getIndex()) == index)
				{
					task = t;
					break;
				}
			}
			
			task_code += "\t\tcase " + task.getIndex() + ":\n";
			task_code += "\t\t{\n";
			task_code += "\t\t\t";
			task_code += "StartTask(" + task.getName() + ");\n";
			task_code += "\t\t\tbreak;\n";
			task_code += "\t\t}\n";
			
			code += task_code;
			index++;
			task_code = "";
		}
		
		task_code += "\t}\n";
		task_code += "}\n\n";
		
		//
		index = 0;
		task_code += "void TASK_CANCELER(int index)\n";
		task_code += "{\n";
		task_code += "\tswitch(index)\n";
		task_code += "\t{\n";
		
		while(index < mTask.size())
		{
			Task task = null;
			for(Task t: mTask.values())
			{
				if(Integer.parseInt(t.getIndex()) == index)
				{
					task = t;
					break;
				}
			}
			
			task_code += "\t\tcase " + task.getIndex() + ":\n";
			task_code += "\t\t{\n";
			task_code += "\t\t\t";
			task_code += "StopTask(" + task.getName() + ");\n";
			task_code += "\t\t\tbreak;\n";
			task_code += "\t\t}\n";
			
			code += task_code;
			index++;
			task_code = "";
		}
		
		task_code += "\t}\n";
		task_code += "}\n\n";
		
		code += task_code;
		
		return code;
	}
	
	public void generateChannelDefHeader(String mDestFile, String mTemplateFile, Map<Integer, Queue> mQueue)
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
			
			outstream.write(translateChannelDefHeader(content, mQueue).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public String translateChannelDefHeader(String mContent, Map<Integer, Queue> mQueue)
	{
		String code = mContent;
		
		String channelSize="";
		int ichannelSize = 0;
		
		ichannelSize = mQueue.size();
		channelSize = String.valueOf(ichannelSize);				
				
		code = code.replace("##channel_size", channelSize);
		
		return code;
	}
	
	public void generatePortmapDefHeader(String mDestFile, String mTemplateFile, Map<Integer, Queue> mQueue)
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
			
			outstream.write(translatePortmapDefHeader(content, mQueue).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public String translatePortmapDefHeader(String mContent, Map<Integer, Queue> mQueue)
	{
		String code = mContent;
		
		String portmapSize="";
		int iportmapSize = 0;
		
		iportmapSize = 2*mQueue.size();	//get할 때마다 두개씩 쓰니깐!//CIC.java 의  translatePortmapHeader참고
		portmapSize = String.valueOf(iportmapSize);				
				
		code = code.replace("##portmap_size", portmapSize);
		
		return code;
	}
	
	public void generateTaskCode(String mDestFile, String mTemplateFile, Task task, CICAlgorithmType mAlgorithm){
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
	
	public String translateTaskCode(String mContent, Task task, CICAlgorithmType mAlgorithm){
		String code = mContent;
		String parameterDef="";
		String libraryDef="";
		String sysportDef="";
		String mtmDef="";
		String cicInclude="";
		String includeHeaders = "";

		if(mAlgorithm.getLibraries() != null){
			for(TaskLibraryConnectionType taskLibCon: mAlgorithm.getLibraryConnections().getTaskLibraryConnection()){
				if(taskLibCon.getMasterTask().equals(task.getName())){
					//libraryDef += "#define LIBCALL(a, b) __b(\n";
					//libraryDef += "#define LIBCALL(a, b, __b(\n";
				}
			}
		}
			
		cicInclude += "#include \"" + task.getCICFile() + "\"\n";
		
		if(task.getHasMTM().equalsIgnoreCase("Yes")){
			mtmDef += "#define GET_CURRENT_MODE_NAME char* ##TASK_NAME_get_current_mode_name()\n" +
					  "#define GET_CURRENT_MODE_ID int ##TASK_NAME_get_current_mode_id()\n" +
					  "#define GET_MODE_NAME char* ##TASK_NAME_get_mode_name(int id)\n" +
					  "#define GET_VARIABLE_INT int ##TASK_NAME_get_variable_int(char* name)\n" +
					  "#define SET_VARIABLE_INT void ##TASK_NAME_set_variable_int(char* name, int value)\n" +
					  "#define GET_VARIABLE_STRING char* ##TASK_NAME_get_variable_string(char* name)\n" +
   					  "#define SET_VARIABLE_STRING void ##TASK_NAME_set_variable_string(char* name, char* value)\n" +
					  "#define TRANSITION void ##TASK_NAME_transition()\n";
			cicInclude += "#include \"" + task.getName() + ".mtm\"\n"; 
		}
				
		code = code.replace("##INCLUDE_HEADERS", includeHeaders);
		code = code.replace("##LIBRARY_DEFINITION", libraryDef);    
	    code = code.replace("##SYSPORT_DEFINITION", sysportDef);
	    code = code.replace("##CIC_INCLUDE", cicInclude);
	    code = code.replace("##MTM_DEFINITION", mtmDef);
	    code = code.replace("##TASK_NAME", task.getName());
		
		return code;
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
		//TARGET_DEPENDENT_SENSOR_ACTUATOR_DECLARATION
		String targetSenseActuator = "";
		
		targetSenseActuator += "#pragma config(Sensor, S1,     DIST,           sensorSONAR) \n";
		targetSenseActuator += "#pragma config(Sensor, S2,     COLOR,          sensorCOLORFULL) \n";
		targetSenseActuator += "#pragma config(Sensor, S3,     SOUND,          sensorSoundDBA)\n";
		targetSenseActuator += "#pragma config(Motor,  motorB,           ,             tmotorNXT, openLoop) \n";
		targetSenseActuator += "// *!!Code automatically generated by 'ROBOTC' configuration wizard               !!*//";
		
		code = code.replace("##TARGET_DEPENDENT_SENSOR_ACTUATOR_DECLARATION", targetSenseActuator);
		//OS_DEPENDENT_HEADER_INCLUDE
		String OsDependentHeader = "";
		code = code.replace("##OS_DEPENDENT_HEADER_INCLUDE", OsDependentHeader);
		//EXTERN_FUNCTION_DECLARATION
		String externDeclaration = "";
		externDeclaration += "#include \"cic_thread.h\" \n";
		externDeclaration += "#include \"cic_control.h\" \n";
		externDeclaration += "#include \"cic_control_def.h\" \n";
		externDeclaration += "#include \"cic_tasks.h\" \n";
		externDeclaration += "#include \"cic_tasks_def.h\" \n";
		externDeclaration += "#include \"nxt_task_set.h\"\n";
		externDeclaration += "#include \"CIC_port.h\"\n";
		if(mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0 )
		{
			externDeclaration += "#include \"LIB_port.h\"\n";
			externDeclaration += "#include \"lib_channels.h\" \n";
			externDeclaration += "#include \"libchannel_def.h\" \n";
			
			for(Task t: mTask.values())
			{
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
								externDeclaration += "#include \"l_" + library.getName() + "_stub.c\"\n";
							}
						}
					}
				}
			}
		}		
		externDeclaration += "#include \"include_cic_task.h\"\n";
		externDeclaration += "#include \"task_handler.h\"\n";
		externDeclaration += "#include \"cic_channels.h\" \n";
		externDeclaration += "#include \"cic_channels_def.h\" \n";
		externDeclaration += "#include \"cic_portmap.h\" \n";
		externDeclaration += "#include \"cic_portmap_def.h\" \n";
		
		
		code = code.replace("##EXTERN_FUNCTION_DECLARATION", externDeclaration);		
		//SLEEP_MACRO
		String sleepMacro = "";
		code = code.replace("##SLEEP_MACRO", sleepMacro);	
		
		//LIBRARY_NUM_DECLARATION
		String libraryNumDeclaration = "";
		if(mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0 )
		{
			libraryNumDeclaration += "#define num_lib_channels (int)(sizeof(lib_channels)/sizeof(lib_channels[0]))\n";
		}
		code = code.replace("##LIBRARY_NUM_DECLARATION", libraryNumDeclaration);	
				
		//EXTERNAL_GLOBAL_HEADERS
		String ExternalGlobalHeader = "";
		code = code.replace("##EXTERNAL_GLOBAL_HEADERS", ExternalGlobalHeader);	
		//ALL_INIT
		String AllInit = "";
		
		String TaskInit = "";
		TaskInit += "void task_init()\n{\n";
		int index = 0;
		while(index < mTask.size())
		{
			Task task = null;
			for(Task t: mTask.values()){
				if(Integer.parseInt(t.getIndex()) == index){
					task = t;
					break;
				}
			}
			String taskDrivenType = null;
			String state = null;
			String runState = null;
			if(task.getRunCondition().equals("DATA_DRIVEN")) {
				taskDrivenType = "DataDriven";
				state = "Run";
				runState = "Running";
			} else if(task.getRunCondition().equals("TIME_DRIVEN")) {
				taskDrivenType = "TimeDriven";
				state = "Run";
				runState = "Running";
			} else if(task.getRunCondition().equals("CONTROL_DRIVEN")) {
				taskDrivenType = "ControlDriven";
				state = "Stop";
				runState = "Running";
			} else {
				System.out.println(task.getRunCondition() + " is not supported!");
				System.exit(-1);
			}
			
			int globalPeriod = 0;
			if(mGlobalPeriodMetric.equalsIgnoreCase("h"))			globalPeriod = mGlobalPeriod * 3600 * 1000 * 1000;
			else if(mGlobalPeriodMetric.equalsIgnoreCase("m"))		globalPeriod = mGlobalPeriod * 60 * 1000 * 1000;
			else if(mGlobalPeriodMetric.equalsIgnoreCase("s"))		globalPeriod = mGlobalPeriod * 1000 * 1000;
			else if(mGlobalPeriodMetric.equalsIgnoreCase("ms"))		globalPeriod = mGlobalPeriod * 1000;
			else if(mGlobalPeriodMetric.equalsIgnoreCase("us"))		globalPeriod = mGlobalPeriod * 1;
			else{
				globalPeriod = mGlobalPeriod * 1;
				//System.out.println("[makeTasks] Not supported metric of period");
				//System.exit(-1);
			}
			
			String taskType = task.getTaskType();
			
			String hasMTM = "";
			if(task.getHasMTM().equalsIgnoreCase("Yes"))	hasMTM = "true";
			else											hasMTM = "false";
			
			String hasSubgraph = "";
			if(task.getHasSubgraph().equalsIgnoreCase("Yes"))	hasSubgraph = "true";
			else												hasSubgraph = "false";
			
			String isChildTask = "false";
			int parentTaskId = -1;
			
			if(hasSubgraph.equals("false")){
				if(!task.getName().equals(task.getParentTask())){
					for(Task t: mTask.values()){
						if(task.getParentTask().equals(t.getName())){
							isChildTask = "true";
							parentTaskId = Integer.parseInt(t.getIndex());
							break;
						}
					}
					for(Task t: mVTask.values()){
						if(task.getParentTask().equals(t.getName())){
							isChildTask = "true";
							parentTaskId = Integer.parseInt(t.getIndex());
							break;
						}
					}
				}
				else{
					isChildTask = "false";
					parentTaskId = index;
				}
			}
			else{
				isChildTask = "false";
				parentTaskId = index;
			}
			
			if(parentTaskId == -1)
			{
				parentTaskId = index;
			}
			// check isSrcTask
			String isSrcTask = "true";
			for(Queue taskQueue: task.getQueue()){
				if(taskQueue.getDst().equals(task.getName()) && Integer.parseInt(taskQueue.getInitData()) == 0){
					isSrcTask = "false";
					break;
				}
			}
			
			TaskInit += "\ttasks[" + index + "].task_id = " + task.getIndex() + ";\n";
			TaskInit += "\ttasks[" + index + "].name = \"" + task.getName() + "\";\n";
			TaskInit += "\ttasks[" + index + "].task_type = " + taskType + ";\n";
			TaskInit += "\ttasks[" + index + "].driven_type = " + taskDrivenType + ";\n";
			TaskInit += "\ttasks[" + index + "].state = " + state + ";\n";
			TaskInit += "\ttasks[" + index + "].run_state = " + runState + ";\n";
			TaskInit += "\ttasks[" + index + "].p_metric = " + task.getPeriodMetric().toUpperCase() + ";\n";
			TaskInit += "\ttasks[" + index + "].run_rate = " + task.getRunRate() + ";\n";
			TaskInit += "\ttasks[" + index + "].period = " + task.getPeriod() + ";\n";
			TaskInit += "\ttasks[" + index + "].globalPeriod = " + globalPeriod + ";\n";
			TaskInit += "\ttasks[" + index + "].run_count = " + Integer.toString(globalPeriod / Integer.parseInt(task.getPeriod())) + ";\n";
			TaskInit += "\ttasks[" + index + "].hasMTM = " + hasMTM + ";\n";
			TaskInit += "\ttasks[" + index + "].hasSubgraph = " + hasSubgraph + ";\n";
			TaskInit += "\ttasks[" + index + "].isChildtask = " + isChildTask + ";\n";
			TaskInit += "\ttasks[" + index + "].parent_task_id = " + parentTaskId + ";\n";
			TaskInit += "\tMUTEX_INIT(tasks[" + index + "].p_mutex);\n";
			TaskInit += "\t*tasks[" + index + "].p_cond = -1;\n\n";
						
			index++;
		}
		TaskInit += "}\n\n";
		
		String ChannelInit = "";
		ChannelInit += "void channel_init()\n{\n";
		index = 0;
		HashMap<String, ArrayList<Queue>> history = new HashMap<String, ArrayList<Queue>>();
		for(Queue queue: mQueue.values())
		{
			int nextChannelIndex = CommonLibraries.CIC.getNextChannelIndex(Integer.parseInt(queue.getIndex()), mQueue, history);
			
			if(history.containsKey(queue.getSrc() + "_" + queue.getSrcPortId())){
				history.get(queue.getSrc() + "_" + queue.getSrcPortId()).add(queue);
			}
			else{
				ArrayList<Queue> tmp = new ArrayList<Queue>();
				tmp.add(queue);
				history.put(queue.getSrc() + "_" + queue.getSrcPortId(), tmp);
			}
			
			String queueSize = queue.getSize() + "*" + queue.getSampleSize();
			String initData = queue.getInitData() + "*" + queue.getSampleSize();
			String sampleSize = queue.getSampleSize();
			String sampleType = queue.getSampleType();
			if(sampleType=="")	sampleType = "unsigned char";
			
			if(queue.getSampleType() != ""){
				queueSize += "*sizeof(" + queue.getSampleType() + ")";
				sampleSize += "*sizeof(" + queue.getSampleType() + ")";
			}
	
			ChannelInit += "\tchannels["+ index +"].channel_id = " + queue.getIndex() + ";\n";
			ChannelInit += "\tchannels["+ index +"].next_channel_index = " + nextChannelIndex + ";\n";
			ChannelInit += "\tchannels["+ index +"].type = " + queue.getTypeName() + ";\n";
			ChannelInit += "\tchannels["+ index +"].buf = NULL;\n";
			ChannelInit += "\tchannels["+ index +"].start = NULL;\n";
			ChannelInit += "\tchannels["+ index +"].end = NULL;\n";
			ChannelInit += "\tchannels["+ index +"].max_size = " + queueSize + ";\n";
			ChannelInit += "\tchannels["+ index +"].cur_size = -1;\n";
			ChannelInit += "\tchannels["+ index +"].head = NULL;\n";
			ChannelInit += "\tchannels["+ index +"].avail_index_start = NULL;\n";
			ChannelInit += "\tchannels["+ index +"].avail_index_end = NULL;\n";
			ChannelInit += "\tchannels["+ index +"].initData = " + initData + ";\n";
			ChannelInit += "\tchannels["+ index +"].sampleSize = " + sampleSize + ";\n";
			ChannelInit += "\tchannels["+ index +"].sampleType = \"" + sampleType + "\";\n";
			ChannelInit += "\tchannels["+ index +"].request_read = false;\n";
			ChannelInit += "\tchannels["+ index +"].request_write = false;\n";
			ChannelInit += "\tchannels["+ index +"].source_port = " + queue.getSrcPortId() + ";\n";
			ChannelInit += "\tchannels["+ index +"].sink_port = " + queue.getDstPortId() + ";\n";
			ChannelInit += "\tchannels["+ index +"].isWatch = false;\n";
			ChannelInit += "\tchannels["+ index +"].isBreak = false;\n";
			ChannelInit += "\tMUTEX_INIT(channels["+ index +"].mutex);\n\n";
			index++;
		}
		ChannelInit += "}\n\n";
		
		String AddressInit = "";
		index = 0;

		AddressInit += "void addressmap_init()\n{\n";
		for(Queue queue: mQueue.values()){
			AddressInit += "\taddressmap[" + index + "].task_id = " + mTask.get(queue.getSrc()).getIndex() + ";\n";
			AddressInit += "\taddressmap[" + index + "].port_id = " + queue.getSrcPortId() + ";\n";
			AddressInit += "\taddressmap[" + index + "].port_name = \"" + queue.getSrcPortName() + "\";\n";
			AddressInit += "\taddressmap[" + index + "].channel_id = " + queue.getIndex() + ";\n";
			AddressInit += "\taddressmap[" + index + "].op = 'w';\n\n";
			
			index++;
			
			AddressInit += "\taddressmap[" + index + "].task_id = " + mTask.get(queue.getDst()).getIndex() + ";\n";
			AddressInit += "\taddressmap[" + index + "].port_id = " + queue.getDstPortId() + ";\n";
			AddressInit += "\taddressmap[" + index + "].port_name = \"" + queue.getDstPortName() + "\";\n";
			AddressInit += "\taddressmap[" + index + "].channel_id = " + queue.getIndex() + ";\n";
			AddressInit += "\taddressmap[" + index + "].op = 'r';\n\n";
						
			index++;
		}
		AddressInit += "}\n\n";
		
		String ParamInit = "";
		index = 0;
		ParamInit += "void param_init()\n{\n";
		if(mControl.getControlTasks() != null){
			for(Task task: mTask.values()){
				for(TaskParameterType parameter: task.getParameter()){
					ParamInit += "\tparam_list[" + index + "].task_id = " + task.getIndex() + ";\n";
					ParamInit += "\tparam_list[" + index + "].param_id = " + index + ";\n";
					ParamInit += "\tparam_list[" + index + "].task_name = \"" + task.getName() + "\";\n";
					ParamInit += "\tparam_list[" + index + "].param_name = \"" + parameter.getName() + "\";\n";
					ParamInit += "\tparam_list[" + index + "].param_value = (void*)" + parameter.getValue() + ";\n\n";
					
					index++;
				}
			}
		}
		ParamInit += "}\n\n";
		
		int innerindex = 0;
		String ControlChannelInit = "";
		index = 0;
		ControlChannelInit += "void control_channel_init()\n{\n";
		if(mControl.getControlTasks() != null){
			int groupIndex = 0;
			for(ControlTaskType controlTask: mControl.getControlTasks().getControlTask())
			{
				if(mTask.containsKey(controlTask.getTask()))
				{
					String slaveProcId = null;
					String masterProcId = null;
					String taskID = null;
					int groupFlag = 0;
					
					for(Task task: mTask.values()){
						if(task.getName().equals(controlTask.getTask())){
							taskID = task.getIndex();
							masterProcId = task.getProc().get(0).toString();
							break;
						}
					}
					for(Task task: mTask.values()){
						if(task.getHasSubgraph().equalsIgnoreCase("No")){
							if(task.getName().equals(controlTask.getSlaveTask().get(0))){
								slaveProcId = task.getProc().get(0).toString();
								break;
							} 
						}
					}
					
					innerindex = 0;
					int groupId = 0;
					
					for(ExclusiveControlTasksType exclusive: mControl.getExclusiveControlTasksList().getExclusiveControlTasks()){
						for(String conTask: exclusive.getControlTask()){
							if(controlTask.getTask().equals(conTask)){
								groupId = innerindex;
								groupFlag = 1;
								break;
							}
						}
						innerindex++;
						if(groupFlag == 0){
							groupId = groupIndex;
							groupIndex++;
						}
					}
					
					ControlChannelInit += "\tcontrol_channel[" + index + "].control_task_id = " + taskID + ";\n";
					ControlChannelInit += "\tcontrol_channel[" + index + "].control_priority = " + Integer.toString(controlTask.getPriority().intValue()) + ";\n";
					ControlChannelInit += "\tcontrol_channel[" + index + "].control_group_id = " + groupId + ";\n";
					ControlChannelInit += "\tcontrol_channel[" + index + "].empty_slot_index = 0;\n";
					ControlChannelInit += "\tcontrol_channel[" + index + "].empty_base_index = 0;\n\n";
					
					index++;
				}				
			}
		}
		ControlChannelInit += "}\n\n";
		
		String LibraryChannelInit = "";
		if(mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0 )
		{
			index = 0;
			LibraryChannelInit += "void lib_channel_init()\n{\n";
			for(InnerDataStructures.Library library: mLibraryStubList)
			{
				LibraryChannelInit += "\tlib_channels[" + index + "].channel_id = "+ library.getIndex() + ";\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].lib_name = \""+ library.getName() + "\";\n"; 
				LibraryChannelInit += "\tlib_channels[" + index + "].op = 'w';\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].buf = NULL;\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].start = NULL;\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].end = NULL;\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].max_size = sizeof(" + library.getName() + "_func_data);\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].cur_size = 0;\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].sampleSize = sizeof(" + library.getName() + "_func_data);\n";
				LibraryChannelInit += "\tMUTEX_INIT(lib_channels["+ index +"].mutex);\n";
				LibraryChannelInit += "\t*lib_channels[" + index + "].cond = -1;\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].isFull = 0;\n\n";
				
				index++;
				
				LibraryChannelInit += "\tlib_channels[" + index + "].channel_id = "+ library.getIndex() + ";\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].lib_name = \""+ library.getName() + "\";\n"; 
				LibraryChannelInit += "\tlib_channels[" + index + "].op = 'r';\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].buf = NULL;\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].start = NULL;\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].end = NULL;\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].max_size = sizeof(" + library.getName() + "_ret_data);\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].cur_size = 0;\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].sampleSize = sizeof(" + library.getName() + "_ret_data);\n";
				LibraryChannelInit += "\tMUTEX_INIT(lib_channels["+ index +"].mutex);\n";
				LibraryChannelInit += "\t*lib_channels[" + index + "].cond = -1;\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].isFull = 0;\n\n";
				
				index++;
			}
			
			for(InnerDataStructures.Library library: mLibraryWrapperList)
			{
				LibraryChannelInit += "\tlib_channels[" + index + "].channel_id = "+ library.getIndex() + ";\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].lib_name = \""+ library.getName() + "\";\n"; 
				LibraryChannelInit += "\tlib_channels[" + index + "].op = 'w';\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].buf = NULL;\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].start = NULL;\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].end = NULL;\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].max_size = sizeof(" + library.getName() + "_ret_data);\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].cur_size = 0;\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].sampleSize = sizeof(" + library.getName() + "_ret_data);\n";
				LibraryChannelInit += "\tMUTEX_INIT(lib_channels["+ index +"].mutex);\n";
				LibraryChannelInit += "\t*lib_channels[" + index + "].cond = -1;\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].isFull = 0;\n\n";
				
				index++;
				
				LibraryChannelInit += "\tlib_channels[" + index + "].channel_id = "+ library.getIndex() + ";\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].lib_name = \""+ library.getName() + "\";\n"; 
				LibraryChannelInit += "\tlib_channels[" + index + "].op = 'r';\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].buf = NULL;\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].start = NULL;\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].end = NULL;\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].max_size = sizeof(" + library.getName() + "_func_data);\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].cur_size = 0;\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].sampleSize = sizeof(" + library.getName() + "_func_data);\n";
				LibraryChannelInit += "\tMUTEX_INIT(lib_channels["+ index +"].mutex);\n";
				LibraryChannelInit += "\t*lib_channels[" + index + "].cond = -1;\n";
				LibraryChannelInit += "\tlib_channels[" + index + "].isFull = 0;\n\n";
				
				index++;				
			}
			LibraryChannelInit += "}\n\n";
		}		
		
		String globalSemaphoreInit = "";
		globalSemaphoreInit += "void globalSemaphoreInit()\n{\n";
		globalSemaphoreInit += "\tMUTEX_INIT(global_mutex);\n";
		globalSemaphoreInit += "\tMUTEX_INIT(time_mutex);\n";
		globalSemaphoreInit += "\t*time_cond = -1;\n";
		globalSemaphoreInit += "\t*global_cond = -1;\n";
		globalSemaphoreInit += "}\n";
		
		String ControlSempahoreInit = "";
		ControlSempahoreInit += "void controlSempahoreInit()\n{\n";
		ControlSempahoreInit += "\tMUTEX_INIT(control_task_count_lock);\n";
		ControlSempahoreInit += "}\n";			
		
		AllInit += TaskInit;
		AllInit += ChannelInit;
		AllInit += AddressInit;
		AllInit += ParamInit;
		AllInit += ControlChannelInit;
		AllInit += LibraryChannelInit;
		AllInit += globalSemaphoreInit;
		AllInit += ControlSempahoreInit;
		
		code = code.replace("##ALL_INIT", AllInit);	
		
		// LIB_INIT_WRAPUP //
		String libraryInitWrapup="";
		if(mLibrary.size() != 0)
		{
			if(mAlgorithm.getLibraries() != null){
				for(Library library: mLibrary.values()){
					libraryInitWrapup += "extern void l_"+library.getName()+"_init(void);\n";
					libraryInitWrapup += "extern void l_"+library.getName()+"_wrapup(void);\n\n";
				}
			}
			
			libraryInitWrapup += "static void init_libs() {\n";
			if(mAlgorithm.getLibraries() != null)
			{
				for(Library library: mLibrary.values())
					libraryInitWrapup += "\tl_" + library.getName() + "_init();\n";
			}
			libraryInitWrapup += "}\n\n";
			
			libraryInitWrapup += "static void wrapup_libs() {\n";
			if(mAlgorithm.getLibraries() != null)
			{
				for(Library library: mLibrary.values())
					libraryInitWrapup += "\tl_" + library.getName() + "_wrapup();\n";
			}
			libraryInitWrapup += "}\n\n";
		}		
		code = code.replace("##LIB_INIT_WRAPUP",  libraryInitWrapup);	
		
		//SCHEDULE_CODE
		String scheduleCode = "";
		code = code.replace("##SCHEDULE_CODE", scheduleCode);	
		
		// COMM_CODE //
		String commCode = "";
		code = code.replace("##COMM_CODE", commCode);
		
		// COMM_SENDER_ROUTINE //
		String commSenderRoutine = "";
		code = code.replace("##COMM_SENDER_ROUTINE", commSenderRoutine);
		
		// COMM_RECEIVER_ROUTINE //
		String commReceiverRoutine = "";
		code = code.replace("##COMM_RECEIVER_ROUTINE", commReceiverRoutine);
		
		//TARGET_DEPENDENT_IMPLEMENTATION
		String targetDependent = "";
		int idx = 0;
		
		for(Queue queue:mQueue.values())
		{
			int queueSize = Integer.parseInt(queue.getSize()) * Integer.parseInt(queue.getSampleSize());
			
			targetDependent += "#define SIZOEOFARRAY " + Integer.toString(queueSize) +"\n" ;		
			targetDependent += "unsigned char normalchannel_" + idx + "[SIZOEOFARRAY];\n" ;		
			targetDependent += "#undef SIZOEOFARRAY\n" ;	
			
			idx++;
		}			
				
		
		if(mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0)
		{
			idx = 0;
	
			for(InnerDataStructures.Library library: mLibraryStubList)
			{
				targetDependent += "#define SIZOEOFARRAY " + "sizeof(" + library.getName() + "_func_data)\n" ;				
				targetDependent += "unsigned char normallibchannel_" + idx + "[SIZOEOFARRAY];\n" ;		
				targetDependent += "#undef SIZOEOFARRAY\n" ;	
				
				idx++;
				
				targetDependent += "#define SIZOEOFARRAY " + "sizeof(" + library.getName() + "_ret_data)\n" ;
				targetDependent += "unsigned char normallibchannel_" + idx + "[SIZOEOFARRAY];\n" ;		
				targetDependent += "#undef SIZOEOFARRAY\n" ;	
				idx++;
			}
			
			for(InnerDataStructures.Library library: mLibraryWrapperList)
			{
				targetDependent += "#define SIZOEOFARRAY " + "sizeof(" + library.getName() + "_func_data)\n" ;				
				targetDependent += "unsigned char normallibchannel_" + idx + "[SIZOEOFARRAY];\n" ;		
				targetDependent += "#undef SIZOEOFARRAY\n" ;	
				
				idx++;
				
				targetDependent += "#define SIZOEOFARRAY " + "sizeof(" + library.getName() + "_ret_data)\n" ;
				targetDependent += "unsigned char normallibchannel_" + idx + "[SIZOEOFARRAY];\n" ;		
				targetDependent += "#undef SIZOEOFARRAY\n" ;	
				idx++;
			}
			
		}
		code = code.replace("##TARGET_DEPENDENT_IMPLEMENTATION", targetDependent);	
		
		//DEBUG_CODE_IMPLEMENTATION
		String debugCode = "";
		code = code.replace("##DEBUG_CODE_IMPLEMENTATION", debugCode);	
		
		//WRAPUP_TASK_ROUTINE
		templateFile = mTranslatorPath + "templates/target/NXT/nxtOSEK/nxt_TaskRelatedFunction.template";
		String wrapupTaskRoutine = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##WRAPUP_TASK_ROUTINE");
		code = code.replace("##WRAPUP_TASK_ROUTINE", wrapupTaskRoutine);
		
		//DATA_TASK_ROUTINE
		String dataRoutine = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##DATA_TASK_ROUTINE");
		code = code.replace("##DATA_TASK_ROUTINE", dataRoutine);
		
		//CONTROL_ONCE_TASK_ROUTINE
		String controlOnceRoutine = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##CONTROL_ONCE_TASK_ROUTINE");
		code = code.replace("##CONTROL_ONCE_TASK_ROUTINE", controlOnceRoutine);
		
		//TIME_TASK_ROUTINE
		String timeRoutine = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TIME_TASK_ROUTINE");
		code = code.replace("##TIME_TASK_ROUTINE", timeRoutine);
		
		//EXECUTE_TASKS
		String executeTask = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##EXECUTE_TASKS");
		code = code.replace("##EXECUTE_TASKS", executeTask);
		
		//INIT_WRAPUP_TASK_CHANNELS
		templateFile = mTranslatorPath + "templates/target/NXT/nxtOSEK/nxt_ChannelRelatedFunction.template";
		String initWrapupTaskChannels = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##INIT_WRAPUP_TASK_CHANNELS");
		code = code.replace("##INIT_WRAPUP_TASK_CHANNELS", initWrapupTaskChannels);
		
		//INIT_WRAPUP_CHANNELS
		String initWrapupChannels = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##INIT_WRAPUP_CHANNELS");
		code = code.replace("##INIT_WRAPUP_CHANNELS", initWrapupChannels);
		String bufferInit = "";
		
		for(int i = 0; i < mQueue.size(); i++)
		{
			bufferInit += "\t\t\t\t\t\t\tcase " + i + ":\n";
			bufferInit += "\t\t\t\t\t\t\t\tchannels[i].buf = &normalchannel_" + i + "[0];\n";
			bufferInit += "\t\t\t\t\t\t\t\tchannels[i].start = &normalchannel_" + i + "[0];\n";
			bufferInit += "\t\t\t\t\t\t\t\tmemset(&normalchannel_" + i + "[0], 0x0, channels[i].initData);\n";
			bufferInit += "\t\t\t\t\t\t\t\tchannels[i].end = &normalchannel_" + i + "[0] + channels[i].initData;\n";
			bufferInit += "\t\t\t\t\t\t\t\tchannels[i].cur_size = channels[i].initData;\n";
			bufferInit += "\t\t\t\t\t\t\t\tbreak;\n";
		}
		
		code = code.replace("##BUFFER_INITIALIZE", bufferInit);
		
		
		//INIT_WRAPUP_LIB_CHANNELS
		if(mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0)
		{
			String initWrapupLibChannels = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##INIT_WRAPUP_LIB_CHANNELS");
			code = code.replace("##INIT_WRAPUP_LIB_CHANNELS", initWrapupLibChannels);
		}
		else
		{
			code = code.replace("##INIT_WRAPUP_LIB_CHANNELS", "");
		}
		
		String LchannelInit = "";
		
		if(mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0)
		{
			for(int i = 0; i < mLibraryStubList.size(); i++)
			{
				LchannelInit += "\t\t\tcase " + (i*2) + ":\n";
				LchannelInit += "\t\t\t\tlib_channels[i].buf = &normallibchannel_" + (i*2) + "[0];\n";
				LchannelInit += "\t\t\t\tlib_channels[i].start = &normallibchannel_" + (i*2) + "[0];\n";
				LchannelInit += "\t\t\t\tlib_channels[i].end = &normallibchannel_" + (i*2) + "[0];\n";
				LchannelInit += "\t\t\t\tlib_channels[i].cur_size = 0;\n";
				LchannelInit += "\t\t\t\tbreak;\n";
				
				LchannelInit += "\t\t\tcase " + (i*2+1) + ":\n";
				LchannelInit += "\t\t\t\tlib_channels[i].buf = &normallibchannel_" + (i*2+1) + "[0];\n";
				LchannelInit += "\t\t\t\tlib_channels[i].start = &normallibchannel_" + (i*2+1) + "[0];\n";
				LchannelInit += "\t\t\t\tlib_channels[i].end = &normallibchannel_" + (i*2+1) + "[0];\n";
				LchannelInit += "\t\t\t\tlib_channels[i].cur_size = 0;\n";
				LchannelInit += "\t\t\t\tbreak;\n";
			}
			
			for(int i = 0; i < mLibraryWrapperList.size(); i++)
			{
				LchannelInit += "\t\t\tcase " + ((i+mLibraryStubList.size())*2) + ":\n";
				LchannelInit += "\t\t\t\tlib_channels[i].buf = &normallibchannel_" + ((i+mLibraryStubList.size())*2) + "[0];\n";
				LchannelInit += "\t\t\t\tlib_channels[i].start = &normallibchannel_" + ((i+mLibraryStubList.size())*2) + "[0];\n";
				LchannelInit += "\t\t\t\tlib_channels[i].end = &normallibchannel_" + ((i+mLibraryStubList.size())*2) + "[0];\n";
				LchannelInit += "\t\t\t\tlib_channels[i].cur_size = 0;\n";
				LchannelInit += "\t\t\t\tbreak;\n";
				
				LchannelInit += "\t\t\tcase " + ((i+mLibraryStubList.size())*2+1) + ":\n";
				LchannelInit += "\t\t\t\tlib_channels[i].buf = &normallibchannel_" + ((i+mLibraryStubList.size())*2+1) + "[0];\n";
				LchannelInit += "\t\t\t\tlib_channels[i].start = &normallibchannel_" + ((i+mLibraryStubList.size())*2+1) + "[0];\n";
				LchannelInit += "\t\t\t\tlib_channels[i].end = &normallibchannel_" + ((i+mLibraryStubList.size())*2+1) + "[0];\n";
				LchannelInit += "\t\t\t\tlib_channels[i].cur_size = 0;\n";
				LchannelInit += "\t\t\t\tbreak;\n";
			}
			
		}		
		
		code = code.replace("##L_CHANNEL_INITIALIZE", LchannelInit);
				
		//READ_WRITE_PORT
		String rwport = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##READ_WRITE_PORT");
		code = code.replace("##READ_WRITE_PORT", rwport);
		
		//READ_WRITE_AC_PORT
		String rwacport = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##READ_WRITE_AC_PORT");
		code = code.replace("##READ_WRITE_AC_PORT", rwacport);
		
		//READ_WRITE_BUF_PORT
		String rwbufport = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##READ_WRITE_BUF_PORT");
		code = code.replace("##READ_WRITE_BUF_PORT", rwbufport);
		
		//READ_WRITE_LIB_PORT
		if(mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0)
		{
			String rwlibport = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##READ_WRITE_LIB_PORT");
			code = code.replace("##READ_WRITE_LIB_PORT", rwlibport);
		}
		else
		{
			code = code.replace("##READ_WRITE_LIB_PORT", "");
		}
		
		//GET_CURRENT_TIME_BASE
		templateFile = mTranslatorPath + "templates/target/NXT/nxtOSEK/nxt_TaskRelatedFunction.template";
		String getCurrentTime = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##GET_CURRENT_TIME_BASE");
		code = code.replace("##GET_CURRENT_TIME_BASE", getCurrentTime);
		
		//CONTROL_RUN_TASK
		String controlRun = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##CONTROL_RUN_TASK");
		code = code.replace("##CONTROL_RUN_TASK", controlRun);
		
		//CONTROL_CALL_TASK
		String controlCall = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##CONTROL_CALL_TASK");
		code = code.replace("##CONTROL_CALL_TASK", controlCall);
		
		//CONTROL_STOP_TASK
		String controlStop = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##CONTROL_STOP_TASK");
		code = code.replace("##CONTROL_STOP_TASK", controlStop);
		
		//CONTROL_RESUME_TASK
		String controlResume = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##CONTROL_RESUME_TASK");
		code = code.replace("##CONTROL_RESUME_TASK", controlResume);
		
		//CONTROL_SUSPEND_TASK
		String controlSuspend = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##CONTROL_SUSPEND_TASK");
		code = code.replace("##CONTROL_SUSPEND_TASK", controlSuspend);
		
		//LIB_WRAPPER_DECLARATION
		templateFile = mTranslatorPath + "templates/target/NXT/nxtOSEK/nxt_ChannelRelatedFunction.template";
		String libWrapperDeclar = "";
		if(mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0 )
		{
			libWrapperDeclar = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##LIB_WRAPPER_DECLARATION");
		}
		code = code.replace("##LIB_WRAPPER_DECLARATION", libWrapperDeclar);
		
		//LIB_WRAPPER_TASK
		String libWrapperTask = "";
		String libWrapperTaskSend = "";
		String libWrapperTaskReceive = "";
		if(mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0 )
		{
			libWrapperTask = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##LIB_WRAPPER_TASK");

			
			
			int lib_index = 0;
			for(InnerDataStructures.Library library: mLibraryStubList)
			{
				if(lib_index > 0)
				{
					libWrapperTaskSend += "else ";
				}				
				libWrapperTaskSend += "if(i == " + library.getIndex() + ")\n\t\t\t\t\t{\n";
				libWrapperTaskSend += "\t\t\t\t\t\tnxtWriteRawBluetooth((ubyte*)&lib_channels[i].sampleSize, sizeof(ubyte));\n";
				libWrapperTaskSend += "\t\t\t\t\t\tGlobalState_func_data data;\n";
				libWrapperTaskSend += "\t\t\t\t\t\tread_lib_port(i, (unsigned char*)&data, lib_channels[i].sampleSize);\n";
				libWrapperTaskSend += "\t\t\t\t\t\tnxtWriteRawBluetooth((ubyte*)&data, lib_channels[i].sampleSize);\n";
				libWrapperTaskSend += "\t\t\t\t\t}\n";							
				
				lib_index++;
			}
			
			for(InnerDataStructures.Library library: mLibraryWrapperList)
			{
				if(lib_index > 0)
				{
					libWrapperTaskSend += "else ";
				}
				libWrapperTaskSend += "if(i == " + library.getIndex() + ")\n\t\t\t\t\t{\n";
				libWrapperTaskSend += "\t\t\t\t\t\tnxtWriteRawBluetooth((ubyte*)&lib_channels[i].sampleSize, sizeof(ubyte));\n";
				libWrapperTaskSend += "\t\t\t\t\t\tGlobalState_func_data data;\n";
				libWrapperTaskSend += "\t\t\t\t\t\tread_lib_port(i, (unsigned char*)&data, lib_channels[i].sampleSize);\n";
				libWrapperTaskSend += "\t\t\t\t\t\tnxtWriteRawBluetooth((ubyte*)&data, lib_channels[i].sampleSize);\n";
				libWrapperTaskSend += "\t\t\t\t\t}\n";							
				
				lib_index++;
			}
			
			lib_index = 0;
			for(InnerDataStructures.Library library: mLibraryStubList)
			{
				if(lib_index > 0)
				{
					libWrapperTaskReceive += "else ";
				}				
				libWrapperTaskReceive += "if(library_id == " + library.getIndex() + ")\n\t\t{\n";
				libWrapperTaskReceive += "\t\t\tGlobalState_ret_data data;\n";
				libWrapperTaskReceive += "\t\t\tchannel_id = init_lib_port((int)library_id, 'r');\n";
				libWrapperTaskReceive += "\t\t\tdo\n";
				libWrapperTaskReceive += "\t\t\t{\n";
				libWrapperTaskReceive += "\t\t\t\treturnValue = nxtReadRawBluetooth((ubyte*)&data, lib_channels[channel_id].sampleSize);\n";
				libWrapperTaskReceive += "\t\t\t} while(returnValue == 0);\n\n";
				libWrapperTaskReceive += "\t\t\twrite_lib_port(channel_id, (unsigned char*)&data, lib_channels[channel_id].sampleSize);\n";				
				libWrapperTaskReceive += "\t\t}\n";							
				
				lib_index++;
			}
			
			for(InnerDataStructures.Library library: mLibraryWrapperList)
			{
				if(lib_index > 0)
				{
					libWrapperTaskReceive += "else ";
				}
				libWrapperTaskReceive += "if(library_id == " + library.getIndex() + ")\n\t\t{\n";
				libWrapperTaskReceive += "\t\t\tGlobalState_ret_data data;\n";
				libWrapperTaskReceive += "\t\t\tint channel_id = init_lib_port((int)library_id, 'r');\n";
				libWrapperTaskReceive += "\t\t\tnxtReadRawBluetooth(&data, lib_channels[channel_id].sampleSize);\n";
				libWrapperTaskReceive += "\t\t\twrite_lib_port(channel_id, (unsigned char*)&data, lib_channels[i].sampleSize);\n";				
				libWrapperTaskReceive += "\t\t}\n";								
				
				lib_index++;
			}
			
		}
		code = code.replace("##LIB_WRAPPER_TASK", libWrapperTask);	
		if(mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0 )
		{
			code = code.replace("##LIB_WRAPPER_TASK_LIB_SEND", libWrapperTaskSend);
			code = code.replace("##LIB_WRAPPER_TASK_LIB_RECEIVE", libWrapperTaskReceive);
		}
		
		//TARGET_DEPENDENT_INIT_CALL
		String targetInitCall = "";
		targetInitCall += "\teraseDisplay();\n";
		targetInitCall += "\ttask_init();\n";
		targetInitCall += "\tchannel_init();\n";
		targetInitCall += "\taddressmap_init();\n";
		targetInitCall += "\tparam_init();\n";
		if(mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0 )
		{
			targetInitCall += "\tlib_channel_init();\n";
		}
		targetInitCall += "\tcontrol_channel_init();\n";
		targetInitCall += "\tglobalSemaphoreInit();\n";
		targetInitCall += "\tcontrolSempahoreInit();\n";
		code = code.replace("##TARGET_DEPENDENT_INIT_CALL", targetInitCall);
		
		//TARGET_INIT_LIB_CHANNEL
		String targetInitLibChannel = "";
		if(mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0 )
		{
			targetInitLibChannel += "\tinit_lib_channel();\n";
		}
		code = code.replace("##TARGET_INIT_LIB_CHANNEL", targetInitLibChannel);
		
		//TARGET_DEPENDENT_LIB_CALL
		String targetDependentLibCall = "";
		if(mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0 )
		{
			targetDependentLibCall += "\tbluetoothInit();\n";
			targetDependentLibCall += "\tStartTask(BluetoothWrapperSenderRoutine);\n";
			targetDependentLibCall += "\tStartTask(BluetoothWrapperReceiverRoutine);\n";					
		}
		if(mLibrary.size() != 0)
		{
			targetDependentLibCall += "\tinit_libs();\n";
		}
		code = code.replace("##TARGET_DEPENDENT_LIB_CALL", targetDependentLibCall);
		
		//TARGET_WRAPUP_LIB_CHANNEL
		String targetWrapupLibChannel = "";
		
		if(mLibrary.size() != 0)
		{
			targetDependentLibCall += "\twrapup_libs();\n";
		}
		if(mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0 )
		{
			targetWrapupLibChannel += "\twrapup_lib_channel();\n";
		}
		code = code.replace("##TARGET_WRAPUP_LIB_CHANNEL", targetWrapupLibChannel);
		
		
		//TARGET_DEPENDENT_WRAPUP_CALL
		String targetWrapupCall = "";
		code = code.replace("##TARGET_DEPENDENT_WRAPUP_CALL", targetWrapupCall);
		
		// OUT_CONN_INIT // // to do check in here 
		String outConnInit = "";				
		code = code.replace("##OUT_CONN_INIT", outConnInit);
		//////////////
		
		// OUT_CONN_WRAPUP // // to do check in here 
		String outConnwrapup = "";				
		code = code.replace("##OUT_CONN_WRAPUP", outConnwrapup);
		//////////////
		
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
