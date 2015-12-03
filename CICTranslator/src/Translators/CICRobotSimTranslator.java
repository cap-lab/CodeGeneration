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

public class CICRobotSimTranslator implements CICTargetCodeTranslator 
{	
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
	
	private Map<String, Task> mVTask;
	private Map<String, Task> mPVTask;
	
	private CICAlgorithmType mAlgorithm;
	private CICControlType mControl;
	private CICScheduleType mSchedule;
	private CICGPUSetupType mGpusetup;
	private CICMappingType mMapping;

	private String strategy;

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
		mProcessor = processor;
		
		mVTask = vtask;
		mPVTask = pvtask;
		
		mAlgorithm = algorithm;
		mControl = control;
		mSchedule = schedule;
		mGpusetup = gpusetup;
		mMapping = mapping;
			
		Util.copyFile(mOutputPath+"target_task_model.h", mTranslatorPath + "templates/common/task_model/pthread.template");
		Util.copyFile(mOutputPath+"target_system_model.h", mTranslatorPath + "templates/common/system_model/general_linux.template");
		
		Util.copyFile(mOutputPath+"gl2ps.h", mTranslatorPath + "templates/target/RobotSim/simulator_files/gl2ps.h");
		Util.copyFile(mOutputPath+"gl2ps.c", mTranslatorPath + "templates/target/RobotSim/simulator_files/gl2ps.c");
		Util.copyFile(mOutputPath+"matplotpp.h", mTranslatorPath + "templates/target/RobotSim/simulator_files/matplotpp.h");
		Util.copyFile(mOutputPath+"matplotpp.cc", mTranslatorPath + "templates/target/RobotSim/simulator_files/matplotpp.cc");
		Util.copyFile(mOutputPath+"robot_kin.h", mTranslatorPath + "templates/target/RobotSim/simulator_files/robot_kin.h");
		Util.copyFile(mOutputPath+"robot_kin.cpp", mTranslatorPath + "templates/target/RobotSim/simulator_files/robot_kin.cpp");
		
		// generate proc.cpp
		String fileOut = null;	
		String templateFile = "";
		
		// generate cic_tasks.h
		fileOut = mOutputPath + "task_def.h";
		templateFile = mTranslatorPath + "templates/common/common_template/task_def.h.template";
		CommonLibraries.CIC.generateTaskDataStructure(fileOut, templateFile, mTask, mGlobalPeriod, mGlobalPeriodMetric, mThreadVer, mCodeGenType, mVTask, mPVTask);
		
		String srcExtension = "";
		srcExtension = ".cpp";
		
		// generate task_name.c (include task_name.cic)
		for(Task t: mTask.values()){
			if(t.getHasSubgraph().equalsIgnoreCase("Yes") && t.getHasMTM().equalsIgnoreCase("Yes")){
				fileOut = mOutputPath + t.getName() + srcExtension;
				templateFile = mTranslatorPath + "templates/common/common_template/task_mtm_code_template.c";
				CommonLibraries.CIC.generateTaskMTMCode(fileOut, templateFile, t, mAlgorithm);
			}
			else{
				fileOut = mOutputPath + t.getName() + srcExtension;
				templateFile = mTranslatorPath + "templates/common/common_template/task_code_template.c";
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
		
		if(mLibrary != null){
			for(Library l: mLibrary.values()) 
				CommonLibraries.Library.generateLibraryCode(mOutputPath, l, mAlgorithm);
		}
		
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
		os_dep_includeHeader = "#include \"includes.h\"\n";
		Util.copyFile(mOutputPath+"includes.h", mTranslatorPath + "templates/common/common_template/includes.h.linux");
		code = code.replace("##OS_DEPENDENT_INCLUDE_HEADERS", os_dep_includeHeader);
		/////////////////////////////////
		
		// TARGET_DEPENDENT_HEADER_INCLUDE //
		String target_dep_includeHeader = "#include \"robot_kin.h\"\n#include \"matplotpp.h\"\n#include <eigen3/Eigen/Dense>\n";
		code = code.replace("##TARGET_DEPENDENT_INCLUDE_HEADERS", target_dep_includeHeader);
		/////////////////////////////////

		// TARGET_DEPENDENT_IMPLEMENTATION //
		templateFile = mTranslatorPath + "templates/target/RobotSim/target_dependent.template";
		int numRobots = 0;
		for(Processor proc: mProcessor.values()){
			if(proc.getProcName().contains("RobotSimulation"))	numRobots++;
		}
		String targetDependentImpl = "int No_agent = " + numRobots + ";\n\n";
		targetDependentImpl += CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TARGET_DEPENDENT");
		code = code.replace("##TARGET_DEPENDENT_IMPLEMENTATION", targetDependentImpl);
		/////////////////////////////////////
		
		// TARGET_DEPENDENT_INIT_CALL //
		String targetDependentInit = "";
		targetDependentInit = "target_dependent_init(argc, argv);\n";		
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
		
		// EXECUTE_TASKS //
		templateFile = mTranslatorPath + "templates/target/RobotSim/task_execution.template";
		String executeTasks = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##EXECUTE_TASKS");
		code = code.replace("##EXECUTE_TASKS", executeTasks);
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
		code = code.replace("##LIB_INCLUDES", libInclude);
		//////////////
		
		// CONN_INCLUDES //
		String conInclude = "";
		code = code.replace("##CONN_INCLUDES", conInclude);
		//////////////
		
		// INIT_WRAPUP_LIB_CHANNELS //
		String initWrapupLibChannels = "";
		code = code.replace("##INIT_WRAPUP_LIBRARY_CHANNELS", initWrapupLibChannels);
		//////////////
		
		// INIT_WRAPUP_LIB_CHANNELS //
		String readWriteLibPort = "";
		code = code.replace("##READ_WRITE_LIB_PORT", readWriteLibPort);
		//////////////
		
		// LIB_INIT //
		String libInit = "init_libs();\n";
		code = code.replace("##LIB_INIT", libInit);
		//////////////
		
		// LIB_WRAPUP //
		String libWrapup = "wrapup_libs();";
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

			String srcExtension = ".cpp";
			outstream.write("CC=g++\n".getBytes());
			outstream.write("LD=g++\n".getBytes());
			
		    mRootPath = mRootPath.replace("\\","/");
		    mRootPath = mRootPath.replace("C:", "/cygdrive/C");
		    mRootPath = mRootPath.replace("D:", "/cygdrive/D");
		    mRootPath = mRootPath.replace("E:", "/cygdrive/E");
		    mRootPath = mRootPath.replace("F:", "/cygdrive/F");
		    mRootPath = mRootPath.replace("G:", "/cygdrive/G");
		    
		    outstream.write("CFLAGS=-std=c++11 -O0 -Wno-write-strings\n".getBytes());
		    outstream.write("LDFLAGS= -lpthread -lm -lX11 -lglut -lGL -lGLU\n".getBytes());
		    	    
		    for(String ldflag: ldFlagList.values())
		        outstream.write((" " + ldflag).getBytes());
		    outstream.write("\n\n".getBytes());
		    
		    outstream.write("all: proc\n\n".getBytes());
		    
		    outstream.write("proc:".getBytes());
		    
		    outstream.write((" gl2ps.o robot_kin.o matplotpp.o").getBytes());
		    
		    for(Task task: mTask.values())
		    	outstream.write((" " + task.getName() + ".o").getBytes());
		    
		    for(String extraSource: extraSourceList.values())
		    	outstream.write((" " + extraSource + ".o").getBytes());

		    for(String extraLibSource: extraLibSourceList.values())
		    	outstream.write((" " + extraLibSource + ".o").getBytes());
		    
		    if(mAlgorithm.getLibraries() != null)
		    	for(Library library: mLibrary.values())
		    		outstream.write((" " + library.getName() + ".o").getBytes());
		    
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
		    
		    for(String extraSource: extraLibSourceList.keySet()){
		    	outstream.write((extraLibSourceList.get(extraSource) + ".o: " + extraSource).getBytes());
		    	for(String extraHeader: extraLibHeaderList.values())
		    		outstream.write((" " + extraHeader).getBytes());
		    	outstream.write("\n".getBytes());
		    	outstream.write(("\t$(CC) $(CFLAGS) -c " + extraSource + " -o " + extraSourceList.get(extraSource) + ".o\n").getBytes());
		    }
		    
		    outstream.write("robot_kin.o: robot_kin.cpp\n".getBytes());
		    outstream.write("\t$(CC) $(CFLAGS) -c robot_kin.cpp -o robot_kin.o\n\n".getBytes());
		    outstream.write("matplotpp.o: matplotpp.cc\n".getBytes());
		    outstream.write("\t$(CC) $(CFLAGS) -c matplotpp.cc -o matplotpp.o\n\n".getBytes());
		    outstream.write("gl2ps.o: gl2ps.c\n".getBytes());
		    outstream.write("\t$(CC) $(CFLAGS) -c gl2ps.c -o gl2ps.o\n\n".getBytes());
		    
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

	@Override
	public int generateCodeWithComm(String mTarget, String mTranslatorPath,
			String mOutputPath, String mRootPath,
			Map<Integer, Processor> mProcessor,
			List<Communication> mCommunication,
			Map<String, Task> mTask, Map<Integer, Queue> mQueue,
			Map<String, Library> mLibrary, Map<String, Library> mGlobalLibrary,
			int mGlobalPeriod, String mGlbalPeriodMetric, String mCICXMLFile,
			String language, String threadVer, CICAlgorithmType mAlgorithm,
			CICControlType mControl, CICScheduleType mSchedule,
			CICGPUSetupType mGpusetup, CICMappingType mMapping,
			Map<Integer, List<Task>> mConnectedTaskGraph,
			Map<Integer, List<List<Task>>> mConnectedSDFTaskSet,
			Map<String, Task> mVTask, Map<String, Task> mPVTask, String mCodeGenType) throws FileNotFoundException {
		// TODO Auto-generated method stub
		return 0;
	}
	
}
