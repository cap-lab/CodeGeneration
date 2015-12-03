package Translators;

import java.io.*;
import java.math.*;
import java.util.*;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.Document;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import CommonLibraries.*;
import InnerDataStructures.*;
import InnerDataStructures.Communication.BluetoothComm;
import InnerDataStructures.Communication.BluetoothNode;
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

public class CICIRobotCreateTranslator implements CICTargetCodeTranslator 
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
	private Processor mMyProcessor;
	
	private Map<String, Task> mVTask;
	private Map<String, Task> mPVTask;
	
	private CICAlgorithmType mAlgorithm;
	private CICControlType mControl;
	private CICScheduleType mSchedule;
	private CICGPUSetupType mGpusetup;
	private CICMappingType mMapping;

	private String strategy;
	
	private ArrayList<Library> mLibraryStubList;
	private ArrayList<Library> mLibraryWrapperList;

	@Override
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
		
		for(Processor proc: mProcessor.values())
		{
			if(proc.getPoolName().contains("IRobotCreate"))
				mMyProcessor = proc;
		}
		
		mVTask = vtask;
		mPVTask = pvtask;
		
		mAlgorithm = algorithm;
		mControl = control;
		mSchedule = schedule;
		mGpusetup = gpusetup;
		mMapping = mapping;
		
		mLibraryStubList = new ArrayList<Library>();
		mLibraryWrapperList = new ArrayList<Library>();
		
		CommonLibraries.CIC.generateCommonCode("Single", mOutputPath, mTranslatorPath, mTask, mQueue, mLibrary, mThreadVer, mAlgorithm, mControl);
			
		Util.copyFile(mOutputPath+"target_task_model.h", mTranslatorPath + "templates/common/task_model/pthread.template");
		Util.copyFile(mOutputPath+"target_system_model.h", mTranslatorPath + "templates/common/system_model/general_linux.template");
		
		// generate proc.c or proc.cpp
		String fileOut = null;	
		String templateFile = "";
		
		// generate cic_tasks.h
		fileOut = mOutputPath + "task_def.h";
		templateFile = mTranslatorPath + "templates/common/common_template/task_def.h.template";
		CommonLibraries.CIC.generateTaskDataStructure(fileOut, templateFile, mTask, mGlobalPeriod, mGlobalPeriodMetric, mThreadVer, mCodeGenType, mVTask, mPVTask);
		
		String srcExtension = "";
		if(mLanguage.equals("c++"))	srcExtension = ".cpp";
		else						srcExtension = ".c";
		
		// generate task_name.c (include task_name.cic)
		for(Task t: mTask.values()){
			if(t.getHasSubgraph().equalsIgnoreCase("Yes") && t.getHasMTM().equalsIgnoreCase("Yes")){
				fileOut = mOutputPath + t.getName() + srcExtension;
				templateFile = mTranslatorPath + "templates/common/common_template/task_mtm_code_template.c";
				CommonLibraries.CIC.generateTaskMTMCode(fileOut, templateFile, t, mAlgorithm);
			}
			else{
				fileOut = mOutputPath + t.getName() + srcExtension;
				if(mLanguage.equals("c++"))	templateFile = mTranslatorPath + "templates/common/common_template/task_code_template.cpp";
				else						templateFile = mTranslatorPath + "templates/common/common_template/task_code_template.c";
				CommonLibraries.CIC.generateTaskCode(fileOut, templateFile, t, mAlgorithm, mControl);
			}
		}
				
		// generate mtm files from xml files	
		for(Task t: mTask.values()){
			if(t.getHasMTM().equalsIgnoreCase("Yes")){
				templateFile = mTranslatorPath + "templates/common/common_template/task_code_template.mtm";			
				CommonLibraries.CIC.generateMTMFile(mOutputPath, templateFile, t, mAlgorithm, mTask, mPVTask, mQueue, mCodeGenType);
			}
		}
		
		// generateLibrary
		generateLibraryCode();
		
		if(!mTarget.toUpperCase().equals("IROBOTCREATE"))	copyTaskFilesToTargetFolder();
		
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
	
	@Override
	public int generateCodeWithComm(String target, String translatorPath, String outputPath, String rootPath, Map<Integer, Processor> processor, List<Communication> communication, Map<String, Task> task, Map<Integer, Queue> queue, Map<String, Library> library, Map<String, Library> globalLibrary, int globalPeriod, String globalPeriodMetric, String cicxmlfile, String language, String threadVer, CICAlgorithmType algorithm, CICControlType control, CICScheduleType schedule, CICGPUSetupType gpusetup, CICMappingType mapping, Map<Integer, List<Task>> connectedtaskgraph,  Map<Integer, List<List<Task>>> connectedsdftaskset, Map<String, Task> vtask, Map<String, Task> pvtask, String codegentype) throws FileNotFoundException
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
		
		for(Processor proc: mProcessor.values())
		{
			if(proc.getPoolName().contains("IRobotCreate"))
				mMyProcessor = proc;
		}
		
		mVTask = vtask;
		
		mAlgorithm = algorithm;
		mControl = control;
		mSchedule = schedule;
		mGpusetup = gpusetup;
		mMapping = mapping;
		
		mLibraryStubList = new ArrayList<Library>();
		mLibraryWrapperList = new ArrayList<Library>();
		
		CommonLibraries.CIC.generateCommonCode("Single", mOutputPath, mTranslatorPath, mTask, mQueue, mLibrary, mThreadVer, mAlgorithm, mControl);
			
		Util.copyFile(mOutputPath+"target_task_model.h", mTranslatorPath + "templates/common/task_model/pthread.template");
		Util.copyFile(mOutputPath+"target_system_model.h", mTranslatorPath + "templates/common/system_model/general_linux.template");
		
		// generate proc.c or proc.cpp
		String fileOut = null;	
		String templateFile = "";
		
		// generate cic_tasks.h
		fileOut = mOutputPath + "task_def.h";
		templateFile = mTranslatorPath + "templates/common/common_template/task_def.h.template";
		CommonLibraries.CIC.generateTaskDataStructure(fileOut, templateFile, mTask, mGlobalPeriod, mGlobalPeriodMetric, mThreadVer, mCodeGenType, mVTask, mPVTask);
		
		String srcExtension = "";
		if(mLanguage.equals("c++"))	srcExtension = ".cpp";
		else						srcExtension = ".c";
		
		// generate task_name.c (include task_name.cic)
		for(Task t: mTask.values()){
			if(t.getHasSubgraph().equalsIgnoreCase("Yes") && t.getHasMTM().equalsIgnoreCase("Yes")){
				fileOut = mOutputPath + t.getName() + srcExtension;
				templateFile = mTranslatorPath + "templates/common/common_template/task_mtm_code_template.c";
				CommonLibraries.CIC.generateTaskMTMCode(fileOut, templateFile, t, mAlgorithm);
			}
			else{
				fileOut = mOutputPath + t.getName() + srcExtension;
				if(mLanguage.equals("c++"))	templateFile = mTranslatorPath + "templates/common/common_template/task_code_template.cpp";
				else						templateFile = mTranslatorPath + "templates/common/common_template/task_code_template.c";
				CommonLibraries.CIC.generateTaskCode(fileOut, templateFile, t, mAlgorithm, mControl);
			}
		}
				
		// generate mtm files from xml files	
		for(Task t: mTask.values()){
			if(t.getHasMTM().equalsIgnoreCase("Yes")){
				templateFile = mTranslatorPath + "templates/common/common_template/task_code_template.mtm";			
				CommonLibraries.CIC.generateMTMFile(mOutputPath, templateFile, t, mAlgorithm, mTask, mPVTask, mQueue, "Single");
			}
		}
		
		// generateLibrary
		generateLibraryCode();
		
		generateCommCode();
		
		if(!mTarget.toUpperCase().equals("IROBOTCREATE"))	copyTaskFilesToTargetFolder();
		
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
	
	public void generateLibraryCode(){
		// Libraries are mapped on the same target proc
		if(mLibrary != null)
		{
			for(Library library: mLibrary.values()){
				int procId = library.getProc();
				Processor proc = mProcessor.get(procId);
				if(proc.getPoolName().contains("IRobotCreate")){
					boolean hasRemoteConn = false;
					for(int i=0; i<mAlgorithm.getLibraryConnections().getTaskLibraryConnection().size(); i++){
						TaskLibraryConnectionType taskLibCon = mAlgorithm.getLibraryConnections().getTaskLibraryConnection().get(i);
						if(!mTask.containsKey(taskLibCon.getMasterTask())){
							hasRemoteConn = true;
							break;
						}
					}
					CommonLibraries.Library.generateLibraryCode(mOutputPath, library, mAlgorithm);
					if(hasRemoteConn){
						CommonLibraries.Library.generateLibraryWrapperCode(mOutputPath, library, mAlgorithm);
						mLibraryWrapperList.add(library);
					}
				}
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
							Util.copyFile(mOutputPath + "/"+ library.getHeader(), mOutputPath + "/../" + library.getHeader());
							CommonLibraries.Library.generateLibraryStubCode(mOutputPath, library, mAlgorithm, false);
							mLibraryStubList.add(library);
						}
					}
				}
			}
		}
		
		if(mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0 ){
			String fileOut = mOutputPath + "lib_channels.h";
			String templateFile = mTranslatorPath + "templates/common/library/lib_channels.h";
			CommonLibraries.Library.generateLibraryChannelHeader(fileOut, templateFile, mLibraryStubList, mLibraryWrapperList);
	
			fileOut = mOutputPath + "libchannel_def.h";
			templateFile = mTranslatorPath + "templates/common/library/libchannel_def.h.template";
			CommonLibraries.Library.generateLibraryChannelDef(fileOut, templateFile, mLibraryStubList, mLibraryWrapperList);
		}
		if(mLibraryWrapperList.size() > 0 ){
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
	
	public int calculateRemoteConnection(){
		int ret = 0;
		
		// Need to add for normal channel
		
		ArrayList<TaskLibraryConnectionType> taskLibConn = new ArrayList<TaskLibraryConnectionType>();
		for(Library library: mLibrary.values()){
			for(int i=0; i<mAlgorithm.getLibraryConnections().getTaskLibraryConnection().size(); i++){
				TaskLibraryConnectionType tlc = mAlgorithm.getLibraryConnections().getTaskLibraryConnection().get(i);
				if(tlc.getSlaveLibrary().equals(library.getName()))	taskLibConn.add(tlc);
			}
		}
		
		for(Processor proc: mProcessor.values()){
			if(proc.getPoolName().contains("IRobotCreate"))	continue;
			else{
				for(TaskLibraryConnectionType tlc: taskLibConn){
					for(MappingTaskType mtt: mMapping.getTask()){
						for(MappingProcessorIdType mpit: mtt.getProcessor()){
							if(mpit.getPool().equals(proc.getPoolName()) && mtt.getName().equals(tlc.getMasterTask())){
								ret = ret + 1;
								break;
							}
						}
					}
				}
			}
		}
		
		for(Library library: mLibrary.values()){
			int procId = library.getProc();
			Processor proc = mProcessor.get(procId);
			if(proc.getPoolName().contains("IRobotCreate")){
				boolean hasRemoteConn = false;
				for(int i=0; i<mAlgorithm.getLibraryConnections().getTaskLibraryConnection().size(); i++){
					TaskLibraryConnectionType tlc = mAlgorithm.getLibraryConnections().getTaskLibraryConnection().get(i);
					if(tlc.getSlaveLibrary().equals(library.getName()))	taskLibConn.add(tlc);
				}
			}
		}
		
		
		return ret;
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
				if(!mLibraryWrapperList.contains(library)){
					libraryInitWrapup += "extern void l_"+library.getName()+"_init(void);\n";
					libraryInitWrapup += "extern void l_"+library.getName()+"_wrapup(void);\n\n";
				}
			}
		}
	
		libraryInitWrapup += "static void init_libs() {\n";
		if(mLibrary != null){
			for(Library library: mLibrary.values()){
				if(!mLibraryWrapperList.contains(library))
					libraryInitWrapup += "\tl_" + library.getName() + "_init();\n";
			}
		}
		libraryInitWrapup += "}\n\n";
		
		libraryInitWrapup += "static void wrapup_libs() {\n";
		if(mLibrary != null){
			for(Library library: mLibrary.values()){
				if(!mLibraryWrapperList.contains(library))
					libraryInitWrapup += "\tl_" + library.getName() + "_wrapup();\n";
			}
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
		
		// OS_DEPENDENT_INCLUDE_HEADERS //
		String os_dep_includeHeader = "";
		os_dep_includeHeader = "#include \"includes.h\"\n";
		Util.copyFile(mOutputPath+"includes.h", mTranslatorPath + "templates/common/common_template/includes.h.linux");
		code = code.replace("##OS_DEPENDENT_INCLUDE_HEADERS", os_dep_includeHeader);
		/////////////////////////////////
		
		// TARGET_DEPENDENT_HEADER_INCLUDE //
		String target_dep_includeHeader = "#include <termios.h>\n#include <unistd.h>\n#include <errno.h>\n#include <sys/file.h>";
		code = code.replace("##TARGET_DEPENDENT_INCLUDE_HEADERS", target_dep_includeHeader);
		/////////////////////////////////

		// TARGET_DEPENDENT_IMPLEMENTATION //
		String targetDependentImpl = "";
		templateFile = mTranslatorPath + "templates/target/IRobot/irobot_conn.template";
		targetDependentImpl += CommonLibraries.Util.getCodeFromTemplate(templateFile, "##IROBOT_INIT");
		if(mLibraryWrapperList.size() > 0){
			int num = calculateRemoteConnection();
			targetDependentImpl += "#define NUM_BLUETOOTH_CONNECTION " + Integer.toString(num) + "\n";
			templateFile = mTranslatorPath + "templates/common/communication_wrapper/bluetooth/general_linux_master.template";
			targetDependentImpl += CommonLibraries.Util.getCodeFromTemplate(templateFile, "##BLUETOOTH_MASTER");
			
			//fill the mac address 
			String connectionAddressDefine = "";
			//System.out.println("test >> " + (Communication)mCommunication.);
			for(Communication com : mCommunication)
			{
				if(com.getType() == TYPE_BLUETOOTH)
				{
					BluetoothComm btcom = com.getBluetoothComm();
					if(btcom != null)
					{
						int i = 0;
						if(btcom.getMasterProc().mBluetoothName.contains("IRobotCreate"))
						{
							for(BluetoothNode slave: btcom.getSlaveProc())
							{
								connectionAddressDefine += "\taddress[" + (i++) + "] = \"" + slave.getMac() + "\";	//"+ slave.mBluetoothName + "\n";
							}
						}
					}	
				}				
			}
								
			
			targetDependentImpl = targetDependentImpl.replace("##CONNECTION_ADDRESS_DEFINE", connectionAddressDefine);
			
			String connectionMapDefine = "";
			String numConnectionMapDefine = "";			
					
			//targetDependentImpl = targetDependentImpl.replace("##CONNECTION_MAP_DEFINE", connectionMapDefine);
			//targetDependentImpl = targetDependentImpl.replace("##NUM_CONNECTION_MAP_DEFINE", numConnectionMapDefine);			
		
		}
		code = code.replace("##TARGET_DEPENDENT_IMPLEMENTATION", targetDependentImpl);
		/////////////////////////////////////
		
		// TARGET_DEPENDENT_INIT_CALL //
		String targetDependentInit = "";
		targetDependentInit += "\tinit_irobot();\n";
	
		if(mLibraryWrapperList.size() > 0){
			targetDependentInit += "\tbluetoothMasterRun();\n";
		}
		code = code.replace("##TARGET_DEPENDENT_INIT_CALL", targetDependentInit);
		////////////////////////////////
				
		// TARGET_DEPENDENT_WRAPUP_CALL //
		String targetDependentWrapup = "return EXIT_SUCCESS;";
		code = code.replace("##TARGET_DEPENDENT_WRAPUP_CALL", targetDependentWrapup);
		//////////////////////////////////
		
		templateFile = mTranslatorPath + "templates/common/channel_manage/general_linux.template";
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
		String readWriteACPort = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##READ_WRITE_AC_PORT");
		code = code.replace("##READ_WRITE_AC_PORT", readWriteACPort);
		////////////////////////
		
		templateFile = mTranslatorPath + "templates/common/task_execution/timed_top_pn_th_bottom_df_func.template";
		// DATA_TASK_ROUTINE //
		String dataTaskRoutine = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##DATA_TASK_ROUTINE");
		code = code.replace("##DATA_TASK_ROUTINE", dataTaskRoutine);
		//////////////////////////
		
		// TIME_TASK_ROUTINE //
		String timeTaskRoutine = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TIME_TASK_ROUTINE");
		code = code.replace("##TIME_TASK_ROUTINE", timeTaskRoutine);
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
		templateFile = mTranslatorPath + "templates/common/time_code/general_linux.template";
		String getCurrentTimeBase = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##GET_CURRENT_TIME_BASE");
		code = code.replace("##GET_CURRENT_TIME_BASE", getCurrentTimeBase);
		///////////////////////////
		
		// CHANGE_TIME_UNIT //
		String changeTimeUnit = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##CHANGE_TIME_UNIT");
		code = code.replace("##CHANGE_TIME_UNIT", changeTimeUnit);
		///////////////////////////
			
		// DEBUG_CODE //
		String debugCode = "";
		code = code.replace("##DEBUG_CODE_IMPLEMENTATION", debugCode);
		///////////////////////////
		
		// MAIN_FUNCTION //
		String mainFunc = "int main(int argc, char *argv[])";
		code = code.replace("##MAIN_FUNCTION", mainFunc);
		//////////////
		
		// LIB_INCLUDE //
		String libInclude = "";
		if(mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0 ){
			libInclude += "#include \"LIB_port.h\"\n";
			libInclude += "#include \"lib_channels.h\"\n#include \"libchannel_def.h\"\n";
			libInclude += "#define num_libchannels (int)(sizeof(lib_channels)/sizeof(lib_channels[0]))\n";
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
		
		templateFile = mTranslatorPath + "templates/common/lib_channel_manage/general_linux.template";
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
			
		    mRootPath = mRootPath.replace("\\","/");
		    mRootPath = mRootPath.replace("C:", "/cygdrive/C");
		    mRootPath = mRootPath.replace("D:", "/cygdrive/D");
		    mRootPath = mRootPath.replace("E:", "/cygdrive/E");
		    mRootPath = mRootPath.replace("F:", "/cygdrive/F");
		    mRootPath = mRootPath.replace("G:", "/cygdrive/G");

		    outstream.write("CFLAGS=-Wall -O0 -g -DDISPLAY\n".getBytes());
	        outstream.write("#CFLAGS=-Wall -O2 -DDISPLAY\n".getBytes());
	        
	        outstream.write("LDFLAGS=-lasound -lpthread -lm -Xlinker --warn-common".getBytes());
		    if(mLibraryWrapperList.size() > 0)	 outstream.write((" -lbluetooth").getBytes());
		    
		    for(String ldflag: ldFlagList.values())
		        outstream.write((" " + ldflag).getBytes());
		    outstream.write("\n\n".getBytes());
		    
	
		    	    
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
		    	for(Library library: mLibrary.values())
		    		outstream.write((" " + library.getName() + ".o").getBytes());
		    
	    	for(Library library: mLibraryWrapperList){
	    		String wrapper = library.getName() + "_wrapper";
	    		outstream.write((" " + wrapper + ".o").getBytes());
	    	}
	    	
	    	for(Library library: mLibraryStubList){
	    		String stub = library.getName() + "_wrapper";
	    		outstream.write((" " + stub + ".o").getBytes());
	    	}
		    
		    outstream.write(" proc.o\n".getBytes());
		    outstream.write("\t$(LD) $^ -o proc $(LDFLAGS)\n\n".getBytes());
		    
		    outstream.write(("proc.o: proc" + srcExtension + " CIC_port.h ").getBytes());
		    
		    if(mAlgorithm.getHeaders() != null)
		    	for(String headerFile: mAlgorithm.getHeaders().getHeaderFile())
		    		outstream.write((" " + headerFile).getBytes());
		    
		    outstream.write("\n".getBytes());
		    outstream.write(("\t$(CC) $(CFLAGS) -c proc" + srcExtension + " -o proc.o\n\n").getBytes());
		    
		    for(Task task: mTask.values()){
		    	if(task.getCICFile().endsWith(".cic"))
		    		outstream.write((task.getName() + ".o: " + task.getName() + srcExtension + " " + task.getCICFile() + " CIC_port.h ").getBytes());
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
		    outstream.write("\n\n".getBytes());
		    
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}	
	}
	
}
