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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import CommonLibraries.Util;
import InnerDataStructures.Communication;
import InnerDataStructures.Library;
import InnerDataStructures.Processor;
import InnerDataStructures.Queue;
import InnerDataStructures.Task;

public class CICHSimUcosTranslator implements CICTargetCodeTranslator {
	private String mTarget;
	private String mTranslatorPath;
	private String mOutputPath;
	private String mRootPath;
	private String mCICXMLFile;
	private int mFuncSimPeriod;
	private String mFuncSimPeriodMetric;
	private String mThreadVer;
	private String mRuntimeExecutionPolicy;
	private String mCodeGenerationStyle;
	private String mLanguage;

	private Map<String, Task> mTask;
	private Map<Integer, Queue> mQueue;
	private Map<String, Library> mLibrary;
	private Map<String, Library> mGlobalLibrary;
	private Map<Integer, Processor> mProcessor;

	private Map<String, Task> mVTask;
	private Map<String, Task> mPVTask;

	private CICAlgorithmType mAlgorithm;
	private CICControlType mControl;
	private CICScheduleType mSchedule;
	private CICGPUSetupType mGpusetup;
	private CICMappingType mMapping;

	private ArrayList<Library> mLibraryStubList;
	private ArrayList<Library> mLibraryWrapperList;

	private String strategy;

	@Override
	public int generateCode(String target, String translatorPath, String outputPath, String rootPath,
			Map<Integer, Processor> processor, Map<String, Task> task, Map<Integer, Queue> queue,
			Map<String, Library> library, Map<String, Library> globalLibrary, int funcSimPeriod,
			String funcSimPeriodMetric, String cicxmlfile, String language, CICAlgorithmType algorithm,
			CICControlType control, CICScheduleType schedule, CICGPUSetupType gpusetup, CICMappingType mapping,
			Map<Integer, List<Task>> connectedtaskgraph, Map<Integer, List<List<Task>>> connectedsdftaskset,
			Map<String, Task> vtask, Map<String, Task> pvtask, String runtimeExecutionPolicy, String codeGenerationStyle)
					throws FileNotFoundException {
		mTarget = target;
		mTranslatorPath = translatorPath;
		mOutputPath = outputPath;
		mRootPath = rootPath;
		mCICXMLFile = cicxmlfile;
		mFuncSimPeriod = funcSimPeriod;
		mFuncSimPeriodMetric = funcSimPeriodMetric;
		mThreadVer = "Multi";	//need to check
		mRuntimeExecutionPolicy = runtimeExecutionPolicy;
		mCodeGenerationStyle = codeGenerationStyle;
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

		mLibraryStubList = new ArrayList<Library>();
		mLibraryWrapperList = new ArrayList<Library>();

		// create a folder for a processor and copy task file
		File f = new File(mOutputPath);
		if (!f.exists())
			f.mkdir();

		File df = new File(mOutputPath + "/convertedSDF3xml/");
		File sf = new File(mOutputPath + "../convertedSDF3xml/");
		if (!df.exists())
			df.mkdir();
		try {
			Util.copyAllFiles(df, sf);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		Util.copyExtensionFiles(mOutputPath, mOutputPath + "../", ".h");
		Util.copyExtensionFiles(mOutputPath, mOutputPath + "../", ".c");
		Util.copyExtensionFiles(mOutputPath, mOutputPath + "../", ".cic");
		Util.copyExtensionFiles(mOutputPath, mOutputPath + "../", ".cicl");
		Util.copyExtensionFiles(mOutputPath, mOutputPath + "../", ".cicl.h");
		Util.copyExtensionFiles(mOutputPath, mOutputPath + "../", ".mtm");

		Util.copyFile(mOutputPath + "target_task_model.h",
				mTranslatorPath + "templates/common/task_model/ucos_2.template");
		Util.copyFile(mOutputPath + "target_system_model.h",
				mTranslatorPath + "templates/common/system_model/ucos_2_hsim.template");

		// generate proc.c or proc.cpp
		String fileOut = null;
		String templateFile = "";

		// generate cic_tasks.h
		fileOut = mOutputPath + "task_def.h";
		templateFile = mTranslatorPath + "templates/common/common_template/task_def.h.template";
		CommonLibraries.CIC.generateTaskDataStructure(fileOut, templateFile, mTask, mFuncSimPeriod, mFuncSimPeriodMetric,
				mRuntimeExecutionPolicy, mCodeGenerationStyle, mVTask, mPVTask);

		// Need to copy cic_channels.h & generate channel_def.h

		String srcExtension = "";
		if (mLanguage.equals("c++"))
			srcExtension = ".cpp";
		else
			srcExtension = ".c";

		// generate task_name.c (include task_name.cic)
		for (Task t : mTask.values()) {
			if (t.getHasSubgraph().equalsIgnoreCase("Yes") && t.getHasMTM().equalsIgnoreCase("Yes")) {
				fileOut = mOutputPath + t.getName() + srcExtension;
				templateFile = mTranslatorPath + "templates/common/common_template/task_mtm_code_template.c";
				CommonLibraries.CIC.generateTaskMTMCode(fileOut, templateFile, t, mAlgorithm);
			} else {
				fileOut = mOutputPath + t.getName() + srcExtension;
				if (mLanguage.equals("c++"))
					templateFile = mTranslatorPath + "templates/common/common_template/task_code_template.cpp";
				else
					templateFile = mTranslatorPath + "templates/common/common_template/task_code_template.c";
				CommonLibraries.CIC.generateTaskCode(fileOut, templateFile, t, mAlgorithm, mControl);
			}
		}

		// generate mtm files from xml files
		for (Task t : mTask.values()) {
			if (t.getHasMTM().equalsIgnoreCase("Yes")) {
				templateFile = mTranslatorPath + "templates/common/common_template/task_code_template.mtm";
				CommonLibraries.CIC.generateMTMFile(mOutputPath, templateFile, t, mAlgorithm, mTask, mPVTask, mQueue,
						"Single", null);
			}
		}

		if (mLibrary != null)
			generateLibraryCode();

		fileOut = mOutputPath + "proc" + srcExtension;
		if (mThreadVer.equals("s"))
			templateFile = mTranslatorPath + "templates/common/common_template/proc.c.template"; // NEED
																									// TO
																									// FIX
		else
			templateFile = mTranslatorPath + "templates/common/common_template/proc.c.template";

		generateProcCode(fileOut, templateFile);

		// generate Makefile
		fileOut = mOutputPath + "Makefile";
		generateMakefile(fileOut);
		return 0;
	}

	public void generateLibraryCode() {
		// Libraries are mapped on the same target proc
		for (Library library : mLibrary.values()) {
			int procId = library.getProc();
			if (Integer.toString(procId).equals(mTarget)) {
				boolean hasRemoteConn = false;
				for (int i = 0; i < mAlgorithm.getLibraryConnections().getTaskLibraryConnection().size(); i++) {
					TaskLibraryConnectionType taskLibCon = mAlgorithm.getLibraryConnections().getTaskLibraryConnection()
							.get(i);
					Task t = mTask.get(taskLibCon.getMasterTask());
					if (t.getProc().get("Default").get("Default").get(0) != procId) { // Need
																						// to
																						// fix
						hasRemoteConn = true;
						break;
					}
				}
				CommonLibraries.Library.generateLibraryCode(mOutputPath, library, mAlgorithm);
				if (hasRemoteConn) {
					CommonLibraries.Library.generateLibraryWrapperCode(mOutputPath, library, mAlgorithm);
					mLibraryWrapperList.add(library);
				}
			}
		}

		// Libraries are mapped on other target procs
		// for(Task t: mTask.values()){
		for (Library l : mLibrary.values()) {
			int procId = l.getProc();
			if (!Integer.toString(procId).equals(mTarget)) {
				for (Task t : mTask.values()) {
					String taskName = t.getName();
					String libPortName = "";
					String libName = "";
					Library library = null;
					if (t.getLibraryPortList().size() != 0) {
						List<LibraryMasterPortType> libportList = t.getLibraryPortList();
						for (int i = 0; i < libportList.size(); i++) {
							LibraryMasterPortType libPort = libportList.get(i);
							libPortName = libPort.getName();
							break;
						}
						if (libPortName == "") {
							System.out.println("Library task does not exist!");
							System.exit(-1);
						} else {
							for (int i = 0; i < mAlgorithm.getLibraryConnections().getTaskLibraryConnection()
									.size(); i++) {
								TaskLibraryConnectionType taskLibCon = mAlgorithm.getLibraryConnections()
										.getTaskLibraryConnection().get(i);
								if (taskLibCon.getMasterTask().equals(taskName)
										&& taskLibCon.getMasterPort().equals(libPortName)) {
									libName = taskLibCon.getSlaveLibrary();
									break;
								}
							}
							for (Library lib : mGlobalLibrary.values()) {
								if (lib.getName().equals(libName)) {
									library = lib;
									break;
								}
							}
							if (library != null && Integer.toString(library.getProc()) != mTarget) {
								Util.copyFile(mOutputPath + "/" + library.getHeader(),
										mOutputPath + "/../" + library.getHeader());
								CommonLibraries.Library.generateLibraryStubCode(mOutputPath, library, mAlgorithm, true);
								mLibraryStubList.add(library);
								break;
							}
						}
					}
				}
			}
		}

		if (mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0) {
			String fileOut = mOutputPath + "lib_channels.h";
			String templateFile = mTranslatorPath + "templates/common/library/lib_channels.h";
			CommonLibraries.Library.generateLibraryChannelHeader(fileOut, templateFile, mLibraryStubList,
					mLibraryWrapperList);

			fileOut = mOutputPath + "libchannel_def.h";
			templateFile = mTranslatorPath + "templates/common/library/libchannel_def.h.template";
			/*
			 * ArrayList<Library> libraryWrapperOnProc = new
			 * ArrayList<Library>(); for(int i=0; i<mLibraryWrapperList.size();
			 * i++){ Library l = mLibraryWrapperList.get(i);
			 * if(Integer.toString(l.getProc()).equals(mTarget))
			 * libraryWrapperOnProc.add(l); }
			 * CommonLibraries.Library.generateLibraryChannelDef(fileOut,
			 * templateFile, mLibraryStubList, libraryWrapperOnProc);
			 */
			CommonLibraries.Library.generateLibraryChannelDef(fileOut, templateFile, mLibraryStubList,
					mLibraryWrapperList);
		}
		if (mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0) {
			String fileOut = mOutputPath + "lib_wrappers.h";
			String templateFile = mTranslatorPath + "templates/common/library/lib_wrapper.h";
			CommonLibraries.Util.copyFile(fileOut, templateFile);

			fileOut = mOutputPath + "libwrapper_def.h";
			templateFile = mTranslatorPath + "templates/common/library/libwrapper_def.h.template";
			CommonLibraries.Library.generateLibraryWrapperDef(fileOut, templateFile, mLibraryWrapperList,
					mLibraryStubList);
		}
	}

	public void generateProcCode(String mDestFile, String mTemplateFile) {
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

	public String translateProcCode(String mContent) {
		String code = mContent;
		String templateFile = "";

		// LIB_INIT_WRAPUP //
		String libraryInitWrapup = "";
		if (mLibrary != null) {
			for (Library library : mLibrary.values()) {
				libraryInitWrapup += "extern void l_" + library.getName() + "_init(void);\n";
				libraryInitWrapup += "extern void l_" + library.getName() + "_wrapup(void);\n\n";
			}
		}

		libraryInitWrapup += "static void init_libs() {\n";
		if (mLibrary != null) {
			for (Library library : mLibrary.values())
				libraryInitWrapup += "\tl_" + library.getName() + "_init();\n";
		}
		libraryInitWrapup += "}\n\n";

		libraryInitWrapup += "static void wrapup_libs() {\n";
		if (mLibrary != null) {
			for (Library library : mLibrary.values())
				libraryInitWrapup += "\tl_" + library.getName() + "_wrapup();\n";
		}
		libraryInitWrapup += "}\n\n";
		code = code.replace("##LIB_INIT_WRAPUP", libraryInitWrapup);
		/////////////////////

		// EXTERNAL_GLOBAL_HEADERS //
		String externalGlobalHeaders = "";
		if (mAlgorithm.getHeaders() != null) {
			for (String header : mAlgorithm.getHeaders().getHeaderFile())
				externalGlobalHeaders += "#include\"" + header + "\"\n";
		}
		code = code.replace("##EXTERNAL_GLOBAL_HEADERS", externalGlobalHeaders);
		////////////////////////////

		// SCHEDULE_CODE //
		String outPath = mOutputPath + "/convertedSDF3xml/";
		String staticScheduleCode = CommonLibraries.Schedule.generateSingleProcessorStaticScheduleCode(outPath, mTask,
				mVTask);
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
		String targetDependentImpl = "";
		targetDependentImpl += "#define PROC_NAME \"proc." + mTarget + "\"\n";
		targetDependentImpl += "static int num_overall_tasks= "
				+ Integer.toString(mTask.size() + mLibraryWrapperList.size()) + ";\n";
		targetDependentImpl += "static int my_proc_id = -1;\n";
		targetDependentImpl += "volatile int *task_wakeup_list;\n";
		targetDependentImpl += "static OS_FLAG_GRP main_flag_intr;\n";
		targetDependentImpl += "static OS_FLAG_GRP *flag_intr["
				+ Integer.toString(mTask.size() + mLibraryWrapperList.size()) + "];\n\n";
		targetDependentImpl += "static OS_STK runner_stack[1024];\n";
		targetDependentImpl += "static  void  runner(void  *p_arg);\n";
		targetDependentImpl = targetDependentCodeGeneration(targetDependentImpl);

		templateFile = mTranslatorPath + "templates/target/HSim/interrupt.template";
		targetDependentImpl += CommonLibraries.Util.getCodeFromTemplate(templateFile, "##INTERRUPT");
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

		templateFile = mTranslatorPath + "templates/common/channel_manage/hsim_ucos.template";
		// INIT_WRAPUP_CHANNELS //
		String initWrapupChannels = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##INIT_WRAPUP_CHANNELS");
		code = code.replace("##INIT_WRAPUP_CHANNELS", initWrapupChannels);
		//////////////////////////

		// INIT_WRAPUP_CHANNELS //
		String initWrapupTaskChannels = CommonLibraries.Util.getCodeFromTemplate(templateFile,
				"##INIT_WRAPUP_TASK_CHANNELS");
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

		templateFile = mTranslatorPath + "templates/target/HSim/task_execution.template";
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
		String setProc = "";
		code = code.replace("##SET_PROC", setProc);
		//////////////

		// GET_CURRENT_TIME_BASE //
		templateFile = mTranslatorPath + "templates/common/time_code/ucos_2.template";
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
		templateFile = mTranslatorPath + "templates/target/HSim/target_dependent.template";
		String mainFunc = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##MAIN_FUNCTION");
		code = code.replace("##MAIN_FUNCTION", mainFunc);
		//////////////

		// LIB_INCLUDE //
		String libInclude = "";
		if (mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0) {
			libInclude += "#include \"lib_channels.h\"\n#include \"libchannel_def.h\"\n";
			libInclude += "#define num_libchannels (int)(sizeof(lib_channels)/sizeof(lib_channels[0]))\n";
			libInclude += "#include \"lib_wrappers.h\"\n#include \"libwrapper_def.h\"\n";
			libInclude += "#define LIB_CONN 1\n";
		}
		if (mLibraryWrapperList.size() > 0)
			libInclude += "#define LIB_WRAPPER 1\n";

		code = code.replace("##LIB_INCLUDES", libInclude);
		//////////////

		// CONN_INCLUDES //
		String conInclude = "";
		code = code.replace("##CONN_INCLUDES", conInclude);
		//////////////

		templateFile = mTranslatorPath + "templates/common/lib_channel_manage/ucos_2.template";
		// INIT_WRAPUP_LIB_CHANNELS //
		String initWrapupLibChannels = "";
		if (mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0)
			initWrapupLibChannels = CommonLibraries.Util.getCodeFromTemplate(templateFile,
					"##INIT_WRAPUP_LIBRARY_CHANNELS");
		code = code.replace("##INIT_WRAPUP_LIBRARY_CHANNELS", initWrapupLibChannels);
		//////////////

		// READ_WRITE_LIB_PORT //
		String readWriteLibPort = "";
		if (mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0)
			readWriteLibPort = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##READ_WRITE_LIB_PORT");
		code = code.replace("##READ_WRITE_LIB_PORT", readWriteLibPort);
		//////////////

		// LIB_INIT //
		String libInit = "";
		if (mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0)
			libInit = "\tinit_lib_channel();\n\tinit_libs();\n";
		else
			libInit = "\tinit_libs();\n";
		libInit += "task_wakeup_list = (int *)(unsigned int *)(SHARED_BASE + c_gap);\n";
		code = code.replace("##LIB_INIT", libInit);
		//////////////

		// LIB_WRAPUP //
		String libWrapup = "";
		if (mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0)
			libWrapup = "\twrapup_libs();\n\twrapup_lib_channel();\n";
		else
			libWrapup = "\twrapup_libs();\n";
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

	public String targetDependentCodeGeneration(String mContent) {
		String code = mContent;

		code += "static int *waiting_reader_task_list[num_channels];\n";
		code += "static int *waiting_writer_task_list[num_channels];\n";

		String readerCode = "static int waiting_reader_task_list_size[num_channels] = {";
		String writerCode = "static int waiting_writer_task_list_size[num_channels] = {";
		for (Queue q : mQueue.values()) {
			String reader = Integer.toString(mTask.get(q.getDst()).getProc().size());
			String writer = Integer.toString(mTask.get(q.getSrc()).getProc().size());
			readerCode += reader + ", ";
			writerCode += writer + ", ";
		}
		readerCode += "};\n";
		writerCode += "};\n";

		code += readerCode;
		code += writerCode;

		code += "static int *waiting_reader_library_list[num_libchannels];\n";
		code += "static int *waiting_writer_library_list[num_libchannels];\n";
		code += "static int waiting_reader_library_list_size[num_libchannels] = {1, };\n";
		code += "static int waiting_writer_library_list_size[num_libchannels] = {1, };\n";

		String templateFile = mTranslatorPath + "templates/target/HSim/target_dependent.template";
		String targetCode = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TARGET_DEPENDENT");

		code += targetCode;

		return code;
	}

	public void generateMakefile(String mDestFile) {
		Map<String, String> extraSourceList = new HashMap<String, String>();
		Map<String, String> extraHeaderList = new HashMap<String, String>();
		Map<String, String> extraLibSourceList = new HashMap<String, String>();
		Map<String, String> extraLibHeaderList = new HashMap<String, String>();
		Map<String, String> ldFlagList = new HashMap<String, String>();

		try {
			FileOutputStream outstream = new FileOutputStream(mDestFile);

			for (Task task : mTask.values()) {
				for (String extraSource : task.getExtraSource())
					extraSourceList.put(extraSource, extraSource.substring(0, extraSource.length() - 2));
				for (String extraHeader : task.getExtraHeader())
					extraHeaderList.put(extraHeader, extraHeader.substring(0, extraHeader.length() - 2));
			}

			if (mAlgorithm.getLibraries() != null) {
				for (Library library : mLibrary.values()) {
					for (String extraSource : library.getExtraSource())
						extraLibSourceList.put(extraSource, extraSource.substring(0, extraSource.length() - 2));
					for (String extraHeader : library.getExtraHeader())
						extraLibHeaderList.put(extraHeader, extraHeader.substring(0, extraHeader.length() - 2));
				}
			}

			for (Task task : mTask.values())
				if (!task.getLDflag().isEmpty())
					ldFlagList.put(task.getLDflag(), task.getLDflag());

			String srcExtension = null;
			if (mLanguage.equals("c++")) {
				srcExtension = ".cpp";
				outstream.write("CC=g++\n".getBytes());
				outstream.write("LD=g++\n".getBytes());
			} else {
				srcExtension = ".c";
				outstream.write("CC=armcc\n".getBytes());
				outstream.write("LD=armlink\n".getBytes());
			}

			outstream.write("CFLAGS=--arm --cpu ARM926EJ-S -Ospace -c -g -O0 -I./\n".getBytes());
			outstream.write("LDFLAGS= --map --list $@.map --ro-base 0x0 --first \"vectors.o(Vect)\" --entry 0x0 -o $@\n"
					.getBytes());

			for (String ldflag : ldFlagList.values())
				outstream.write((" " + ldflag).getBytes());
			outstream.write("\n\n".getBytes());

			outstream.write("all: proc.axf\n\n".getBytes());

			outstream.write("proc.axf:".getBytes());
			for (Task task : mTask.values())
				outstream.write((" " + task.getName() + ".o").getBytes());

			for (String extraSource : extraSourceList.values())
				outstream.write((" " + extraSource + ".o").getBytes());

			for (String extraLibSource : extraLibSourceList.values())
				outstream.write((" " + extraLibSource + ".o").getBytes());

			if (mAlgorithm.getLibraries() != null) {
				for (Library library : mLibrary.values())
					outstream.write((" " + library.getName() + ".o").getBytes());
				for (int i = 0; i < mLibraryWrapperList.size(); i++)
					outstream.write((" " + mLibraryWrapperList.get(i).getName() + "_wrapper.o").getBytes());
			}

			outstream.write(" proc.o ".getBytes());
			outstream.write(" ../ucos_2/vectors.o ".getBytes());
			outstream.write(" ../ucos_2/HSimUCOS.a ".getBytes());
			outstream.write(" ../esim/esim_lib.o\n".getBytes());
			outstream.write("\t$(LD) $^ -o proc $(LDFLAGS)\n\n".getBytes());

			outstream.write(("proc.o: proc" + srcExtension + " CIC_port.h ").getBytes());

			if (mAlgorithm.getHeaders() != null)
				for (String headerFile : mAlgorithm.getHeaders().getHeaderFile())
					outstream.write((" " + headerFile).getBytes());

			outstream.write("\n".getBytes());
			outstream.write(("\t$(CC) $(CFLAGS) -c proc" + srcExtension + " -o proc.o\n\n").getBytes());

			for (Task task : mTask.values()) {
				if (task.getCICFile().endsWith(".cic"))
					outstream.write((task.getName() + ".o: " + task.getName() + srcExtension + " " + task.getCICFile()
							+ " CIC_port.h ").getBytes());
				else if (task.getCICFile().endsWith(".xml"))
					outstream.write((task.getName() + ".o: " + task.getName() + srcExtension + " " + " CIC_port.h ")
							.getBytes());
				for (String header : task.getExtraHeader())
					outstream.write((" " + header).getBytes());
				outstream.write("\n".getBytes());
				outstream.write(("\t$(CC) $(CFLAGS) " + task.getCflag() + " -c " + task.getName() + srcExtension
						+ " -o " + task.getName() + ".o\n\n").getBytes());

			}

			for (String extraSource : extraSourceList.keySet()) {
				outstream.write((extraSourceList.get(extraSource) + ".o: " + extraSourceList.get(extraSource) + ".c")
						.getBytes());
				for (String extraHeader : extraHeaderList.keySet())
					outstream.write((" " + extraHeader).getBytes());
				outstream.write("\n".getBytes());
				outstream.write(("\t$(CC) $(CFLAGS) -c " + extraSourceList.get(extraSource) + ".c -o "
						+ extraSourceList.get(extraSource) + ".o\n\n").getBytes());
			}

			if (mAlgorithm.getLibraries() != null) {
				for (Library library : mLibrary.values()) {
					outstream.write((library.getName() + ".o: " + library.getName() + ".c").getBytes());
					for (String extraHeader : library.getExtraHeader())
						outstream.write((" " + extraHeader).getBytes());
					outstream.write("\n".getBytes());
					if (!library.getCflag().isEmpty())
						outstream.write(("\t$(CC) $(CFLAGS) " + library.getCflag()).getBytes());
					else
						outstream.write(("\t$(CC) $(CFLAGS)").getBytes());
					outstream.write((" -c " + library.getName() + ".c -o " + library.getName() + ".o\n").getBytes());
				}
				outstream.write("\n".getBytes());
				for (int i = 0; i < mLibraryWrapperList.size(); i++) {
					Library library = mLibraryWrapperList.get(i);
					outstream.write((library.getName() + "_wrapper.o: " + library.getName() + "_wrapper.c").getBytes());
					for (String extraHeader : library.getExtraHeader())
						outstream.write((" " + extraHeader).getBytes());
					outstream.write("\n".getBytes());
					if (!library.getCflag().isEmpty())
						outstream.write(("\t$(CC) $(CFLAGS) " + library.getCflag()).getBytes());
					else
						outstream.write(("\t$(CC) $(CFLAGS)").getBytes());
					outstream.write((" -c " + library.getName() + "_wrapper.c -o " + library.getName() + "_wrapper.o\n")
							.getBytes());
				}
				outstream.write("\n".getBytes());
			} else
				outstream.write("\n".getBytes());

			for (String extraSource : extraLibSourceList.keySet()) {
				outstream.write((extraLibSourceList.get(extraSource) + ".o: " + extraSource).getBytes());
				for (String extraHeader : extraLibHeaderList.values())
					outstream.write((" " + extraHeader).getBytes());
				outstream.write("\n".getBytes());
				outstream.write(
						("\t$(CC) $(CFLAGS) -c " + extraSource + " -o " + extraSourceList.get(extraSource) + ".o\n")
								.getBytes());
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

	public static void generateHSimCode(String mDestFile, String mTemplateFile, Map<Integer, Processor> mProcessor) {
		File fileOut = new File(mDestFile);
		File templateFile = new File(mTemplateFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);

			outstream.write(translateHSimCode(content, mProcessor).getBytes());
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public static String translateHSimCode(String mContent, Map<Integer, Processor> mProcessor) {
		String code = mContent;
		String armProcessorSC = "";
		String scIntrCtrl = "";
		String procSetParent = "";
		String scSignalIrqFiq = "";
		String procBusPort = "";
		String procClock = "";
		String procIrqFiq = "";
		String intcDefinition = "";
		String procDefinition = "";
		String procInitialize = "";
		String intcInitialize = "";
		String destroy = "";

		int index = 0;
		for (Processor proc : mProcessor.values()) {
			String sIndex = "", sZero = "";
			if (index != 0)
				sIndex = Integer.toString(index);
			if (index <= 9 && index > 0)
				sZero = "0";
			else if (index == 0)
				sZero = "0000";

			armProcessorSC += "\tARMProcessorSC\t\tproc" + sIndex + "(\"ARM" + sIndex + "\");\n";
			scIntrCtrl += "\tSCIntrCtrl\t\tintc" + sIndex + "(\"Intc" + sIndex + "\");\n";
			procSetParent += "\tproc" + sIndex + ".setParent(pKernel);\n";

			scSignalIrqFiq += "\tsc_signal<bool> signalIRQ" + sIndex + ";\n";
			scSignalIrqFiq += "\tsc_signal<bool> signalFIQ" + sIndex + ";\n";

			procBusPort += "\tproc" + sIndex + ".bus_port(ahbBus);\n";
			procClock += "\tproc" + sIndex + ".clock(clock.clock);\n";

			procIrqFiq += "\tproc" + sIndex + ".fiq(signalFIQ" + sIndex + ");\n";
			procIrqFiq += "\tproc" + sIndex + ".irq(signalIRQ" + sIndex + ");\n";

			intcDefinition += "\t// INTC" + sIndex + "\n";
			intcDefinition += "\tintc" + sIndex + ".clock(clock.clock);\n";
			intcDefinition += "\tintc" + sIndex + ".int_REQn(signal);\n";
			intcDefinition += "\tintc" + sIndex + ".int_IRQn(signalIRQ" + sIndex + ");\n";
			intcDefinition += "\tintc" + sIndex + ".int_FIQn(signalFIQ" + sIndex + ");\n\n";
			intcDefinition += "\tintc" + sIndex + ".setParameter(\"log\", \"true\");\n";
			intcDefinition += "\tpPort = (AbstractSlavePort*)intc" + sIndex + ".findPortWithName(\"slave\");\n";
			intcDefinition += "\tpPort->base = 0x10A" + sZero + Integer.toHexString(index * 0x1000) + ";\n";
			intcDefinition += "\tpPort->size = 0x1000;\n";
			intcDefinition += "\tapbBus.apb_slave(intc" + sIndex + ");\n";

			procDefinition += "\tproc" + sIndex + ".setParameter(\"image\", \"./proc." + index + "/proc.axf\");\n";
			procDefinition += "\tproc" + sIndex + ".setParameter(\"trace\", \"true\");\n";
			procDefinition += "\tproc" + sIndex + ".setParameter(\"log\", \"true\");\n";
			procDefinition += "\tproc" + sIndex + ".setParameter(\"priority\", \"" + index + "\");\n";
			procDefinition += "\tproc" + sIndex + ".addMemorymap(\"MEMORY\", 0x10000000, 0x800000, \"SHARED\");\n";
			procDefinition += "\tproc" + sIndex + ".addMemorymap(\"LOCK\", 0x10900000, 0x100000, \"SHARED\");\n";
			procDefinition += "\tproc" + sIndex + ".addMemorymap(\"APB\", 0x10A00000, 0x100000, \"SHARED\");\n";
			procDefinition += "\tproc" + sIndex + ".setup();\n";

			procInitialize += "\tproc" + sIndex + ".onInitialize();\n";
			intcInitialize += "\tintc" + sIndex + ".onInitialize();\n";
			destroy += "\tproc" + sIndex + ".onDestroy();\n";
			destroy += "\tintc" + sIndex + ".onDestroy();\n";

			index++;
		}

		code = code.replace("##ARMPROCESSOR_SC", armProcessorSC);
		code = code.replace("##SC_INTR_CTRL", scIntrCtrl);
		code = code.replace("##PROC_SET_PARENT", procSetParent);
		code = code.replace("##SC_SIGNAL_IRQ_FIQ", scSignalIrqFiq);
		code = code.replace("##PROC_BUS_PORT", procBusPort);
		code = code.replace("##PROC_CLOCK", procClock);
		code = code.replace("##PROC_IRQ_FIQ", procIrqFiq);
		code = code.replace("##INTC_DEFINITION", intcDefinition);
		code = code.replace("##PROC_DEFINITION", procDefinition);
		code = code.replace("##PROC_INITIALIZE", procInitialize);
		code = code.replace("##INTC_INITIALIZE", intcInitialize);
		code = code.replace("##DESTROY", destroy);

		return code;
	}

	public static void generateGlobalMakefile(String mDestFile, String mTemplateFile,
			Map<Integer, Processor> mProcessor) {
		File fileOut = new File(mDestFile);
		File templateFile = new File(mTemplateFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);

			outstream.write(translateGlobalMakefile(content, mProcessor).getBytes());
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public static String translateGlobalMakefile(String mContent, Map<Integer, Processor> mProcessor) {
		String code = mContent;
		String subdirs = "SUBDIRS = esim ucos_2 ";

		int index = 0;
		for (Processor proc : mProcessor.values()) {
			subdirs += "proc." + index + " ";
			index++;
		}

		code = code.replace("##SUBDIRS", subdirs);

		return code;
	}

	public static void copyOSandLibrary(String mOutputPath, String mTranslatorPath) {

		try {
			File f = new File(mOutputPath + "ucos_2");
			if (!f.exists())
				f.mkdir();
			File s = new File(mTranslatorPath + "templates/target/hsim/ucos_2");
			if (!s.exists())
				s.mkdir();
			CommonLibraries.Util.copyAllFiles(f, s);

			f.delete();
			s.delete();

			f = new File(mOutputPath + "esim");
			if (!f.exists())
				f.mkdir();
			s = new File(mTranslatorPath + "templates/target/hsim/esim");
			if (!s.exists())
				s.mkdir();
			CommonLibraries.Util.copyAllFiles(f, s);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public static void wrapup(String mOutputPath) {
		Util.deleteFiles(mOutputPath, ".cic");
		Util.deleteFiles(mOutputPath, ".cicl");
		Util.deleteFiles(mOutputPath, ".cicl.h");
		Util.deleteFiles(mOutputPath, ".h");
		Util.deleteFiles(mOutputPath, ".c");
		Util.deleteFiles(mOutputPath, ".mtm");
		Util.deleteFiles(mOutputPath, "expo_org.xml");
		Util.deleteFiles(mOutputPath, "open_list.xml");
		Util.deleteFiles(mOutputPath, "task.xml");
	}

	@Override
	public int generateCodeWithComm(String mTarget, String mTranslatorPath, String mOutputPath, String mRootPath,
			Map<Integer, Processor> mProcessor, List<Communication> mCommunication, Map<String, Task> mTask,
			Map<Integer, Queue> mQueue, Map<String, Library> mLibrary, Map<String, Library> mGlobalLibrary,
			int mGlobalPeriod, String mGlbalPeriodMetric, String mCICXMLFile, String language,
			CICAlgorithmType mAlgorithm, CICControlType mControl, CICScheduleType mSchedule, CICGPUSetupType mGpusetup,
			CICMappingType mMapping, Map<Integer, List<Task>> mConnectedTaskGraph,
			Map<Integer, List<List<Task>>> mConnectedSDFTaskSet, Map<String, Task> mVTask, Map<String, Task> mPVTask,
			String mRuntimeExecutionPolicy, String codeGenerationStyle) throws FileNotFoundException {
		// TODO Auto-generated method stub
		return 0;
	}

}