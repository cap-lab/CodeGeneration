package Translators;

import java.io.*;
import java.math.*;
import java.util.*;

import CommonLibraries.*;
import InnerDataStructures.*;
import InnerDataStructures.Library;
import InnerDataStructures.Queue;
import hopes.cic.xml.*;

public class CICCellTranslator implements CICTargetCodeTranslator {
	private String mTarget;
	private String mTranslatorPath;
	private String mOutputPath;
	private String mRootPath;
	private String mCICXMLFile;
	private int mGlobalPeriod;
	private String mGlobalPeriodMetric;
	private String mThreadVer;
	private String mRuntimeExecutionPolicy;
	private String mCodeGenerationStyle;
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
	private CICMappingType mMapping;

	private boolean mTaskUseSPE;
	private boolean mLibraryUseSPE;

	private List<Task> mTaskOnSPE;
	private List<Library> mLibraryOnSPE;

	private List<TaskLibraryConnectionType> mTaskLibConnection; // List of
																// taskLibraryconnection
																// (eliminated
																// connection of
																// same library)
	private List<LibraryLibraryConnectionType> mLibLibConnection; // List of
																	// libraryLibraryconnection
																	// (eliminated
																	// connection
																	// of same
																	// library)
	private List<String> mStubList; // List of stub
	private List<String> mStubFileList; // List of callee(stub) on PPE (not
										// include wrapper)
	private List<String> mL_FileList; // list of caller on PPE
	private List<String> mPPEToPPEList; // List of ppe to ppe connection (not
										// need channel & port)

	@Override
	public int generateCode(String target, String translatorPath, String outputPath, String rootPath,
			Map<Integer, Processor> processor, Map<String, Task> task, Map<Integer, Queue> queue,
			Map<String, Library> library, Map<String, Library> globalLibrary, int globalPeriod,
			String globalPeriodMetric, String cicxmlfile, String language, CICAlgorithmType algorithm,
			CICControlType control, CICScheduleType schedule, CICGPUSetupType gpusetup, CICMappingType mapping,
			Map<Integer, List<Task>> connectedtaskgraph, Map<Integer, List<List<Task>>> connectedsdftaskset,
			Map<String, Task> vtask, Map<String, Task> pvtask, String runtimeExecutionPolicy, String codeGenerationStyle)
					throws FileNotFoundException {
		mTarget = target;
		mTranslatorPath = translatorPath;
		mOutputPath = outputPath;
		mRootPath = rootPath;
		mCICXMLFile = cicxmlfile;
		mGlobalPeriod = globalPeriod;
		mGlobalPeriodMetric = globalPeriodMetric;
		mThreadVer = "Multi";
		mRuntimeExecutionPolicy = runtimeExecutionPolicy;
		mCodeGenerationStyle = codeGenerationStyle;
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
		mMapping = mapping;

		mTaskUseSPE = false;
		mLibraryUseSPE = false;

		mTaskOnSPE = new ArrayList<Task>();
		mLibraryOnSPE = new ArrayList<Library>();

		// Make Output Directory
		File f = new File(mOutputPath);

		f.mkdir();

		mOutputPath = mOutputPath + "/";
		mTranslatorPath = mTranslatorPath + "/";

		String fileOut = null;
		String templateFile = null;

		// Check Use SPE (or return and retranslate Pthread)
		int ret = CheckUseSPEs();
		if (ret == 1)
			return -1;

		//////////////////////////////////// Task
		//////////////////////////////////// ///////////////////////////////////////////

		// generate cic_tasks.h
		fileOut = mOutputPath + "cic_tasks.h";
		templateFile = mTranslatorPath + "templates/common/cic/cic_tasks.h.template";
		CommonLibraries.CIC.generateTaskDataStructure(fileOut, templateFile, mTask, mGlobalPeriod, mGlobalPeriodMetric,
				mRuntimeExecutionPolicy, mCodeGenerationStyle, mVTask, mPVTask);

		// generate cic_channels.h
		fileOut = mOutputPath + "cic_channels.h";
		templateFile = mTranslatorPath + "templates/common/cic/cic_channels.h.template";
		CommonLibraries.CIC.generateChannelHeader(fileOut, templateFile, mQueue, mThreadVer);

		// generate cic_portmap.h
		fileOut = mOutputPath + "cic_portmap.h";
		templateFile = mTranslatorPath + "templates/common/cic/cic_portmap.h.template";
		CommonLibraries.CIC.generatePortmapHeader(fileOut, templateFile, mTask, mQueue);

		// generate task_name.info.h
		generateCICInfoFile();

		////////////////////////////////////////////////////////////////////////////////////

		//////////////////////////////////// Library
		//////////////////////////////////// ///////////////////////////////////////
		mTaskLibConnection = new ArrayList<TaskLibraryConnectionType>();// List
																		// of
																		// taskLibraryconnection
																		// (eliminated
																		// connection
																		// of
																		// same
																		// library)
		mLibLibConnection = new ArrayList<LibraryLibraryConnectionType>(); // List
																			// of
																			// libraryLibararyconnection
																			// (eliminated
																			// connection
																			// of
																			// same
																			// library)
		mStubList = new ArrayList<String>(); // List of stub
		mStubFileList = new ArrayList<String>(); // List of callee(stub) on PPE
													// (not include wrapper)
		mL_FileList = new ArrayList<String>(); // list of caller on PPE
		mPPEToPPEList = new ArrayList<String>(); // List of ppe to ppe
													// connection (not need
													// channel & port)

		LibraryRelationGeneration();

		// generate lib_channels.h
		fileOut = mOutputPath + "lib_channels.h";
		templateFile = mTranslatorPath + "templates/common/library/lib_channels.h.template";
		generateLibraryChannelHeader(fileOut, templateFile);

		// generate lib_portmap.h
		fileOut = mOutputPath + "lib_portmap.h";
		templateFile = mTranslatorPath + "templates/common/library/lib_portmap.h.template";
		generateLibraryPortmapHeader(fileOut, templateFile);

		// generate lib_stubs.h
		fileOut = mOutputPath + "lib_stubs.h";
		templateFile = mTranslatorPath + "templates/common/library/lib_stubs.h.template";
		generateLibraryStubHeader(fileOut, templateFile);

		// generate lib_refs.h
		fileOut = mOutputPath + "lib_refs.h";
		templateFile = mTranslatorPath + "templates/common/library/lib_refs.h.template";
		generateLibraryRefHeader(fileOut, templateFile);

		// generate lib_name_data_structures.h
		generateLibraryDataStructureHeader();

		// generate callee file
		generateLibraryCalleeFile();

		// generate caller file
		generateLibraryCallerFile();

		// generate library_name.info.h
		generateLibraryInfoFile();

		////////////////////////////////////////////////////////////////////////////////////

		//////////////////////////////////// Control
		//////////////////////////////////// ///////////////////////////////////////

		// generate con_taskmap.h
		fileOut = mOutputPath + "con_taskmap.h";
		templateFile = mTranslatorPath + "templates/common/control/con_taskmap.h.template";
		generateControlTaskMapHeader(fileOut, templateFile);

		// generate con_channels.h
		fileOut = mOutputPath + "con_channels.h";
		templateFile = mTranslatorPath + "templates/common/control/con_channels.h.template";
		generateControlChannelHeader(fileOut, templateFile);

		// generate con_portmap.h
		fileOut = mOutputPath + "con_portmap.h";
		templateFile = mTranslatorPath + "templates/common/control/con_portmap.h.template";
		generateControlPortMapHeader(fileOut, templateFile);

		// generate param_list.h
		fileOut = mOutputPath + "param_list.h";
		templateFile = mTranslatorPath + "templates/common/control/param_list.h.template";
		generateControlParamListHeader(fileOut, templateFile);

		// generate control_info.h
		fileOut = mOutputPath + "control_info.h";
		templateFile = mTranslatorPath + "templates/common/control/control_info.h.template";
		generateControlInfoHeader(fileOut, templateFile);

		// generate slave_task_name.info.h
		generateControlSlaveInfoFile();

		////////////////////////////////////////////////////////////////////////////////////

		// generate task_name.c (include task_name.cic)
		for (Task t : mTask.values()) {
			fileOut = mOutputPath + t.getName() + ".c";
			generateTaskCode(fileOut, t);
		}

		fileOut = mOutputPath;
		if (mAlgorithm.getLibraries() != null) {
			// generate library_name.c & library_name.h
			for (Library l : mLibrary.values()) {
				CommonLibraries.Library.generateLibraryCode(fileOut, l, mAlgorithm);
			}
		}

		// copy *.cic files, library files and external files
		try {
			Util.copyAllFiles(new File(mOutputPath), new File(mTranslatorPath + "templates/target/cell/cic"));
			Util.copyAllFiles(new File(mOutputPath), new File(mTranslatorPath + "templates/target/cell/library"));
			Util.copyAllFiles(new File(mOutputPath), new File(mTranslatorPath + "templates/target/cell/control"));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		Util.copyExtensionFiles(mOutputPath, "./", ".h");
		Util.copyExtensionFiles(mOutputPath, "./", ".c");
		Util.copyExtensionFiles(mOutputPath, "./", ".cic");
		Util.copyExtensionFiles(mOutputPath, "./", ".cicl");
		Util.copyExtensionFiles(mOutputPath, "./", ".cicl.h");

		// generate Makefile
		fileOut = mOutputPath + "Makefile";
		generateMakefile(fileOut);

		return 0;
	}

	public int CheckUseSPEs() {
		for (Task task : mTask.values()) {
			for (Map<String, List<Integer>> procMap : task.getProc().values()) {
				for (List<Integer> procList : procMap.values()) {
					for (int proc : procList) {
						if (mProcessor.get(proc).getProcName().toUpperCase().equals("SPE")) {
							mTaskUseSPE = true;
							mTaskOnSPE.add(task);
							// break;
						}
					}
				}
			}
		}

		if (mAlgorithm.getLibraries() != null) {
			for (Library library : mLibrary.values()) {
				if (mProcessor.get(library.getProc()).getProcName().toUpperCase().equals("SPE")) {
					mLibraryUseSPE = true;
					mLibraryOnSPE.add(library);
				}
			}
		}

		if (mTaskUseSPE == false && mLibraryUseSPE == false) {
			System.out.println("Translate Pthread ver");
			return 1;
		} else {
			System.out.println("Translate Cell ver");
			return 0;
		}

	}

	public void generateCICInfoFile() {
		for (Task task : mTask.values()) {
			if (!task.getIsSlaveTask()) {
				if (mTaskOnSPE.contains(task)) {
					String fileOut = mOutputPath + task.getName() + ".info.c";
					File f = new File(fileOut);
					FileOutputStream outstream;
					try {
						outstream = new FileOutputStream(fileOut);
						outstream.write(translateCICInfoHeader(task).getBytes());
						outstream.close();
					} catch (FileNotFoundException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					} catch (IOException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
			}
		}
	}

	public String translateCICInfoHeader(Task task) {
		String content = new String();

		content += "\n";
		content += "#define TASKID " + task.getIndex() + "\n";
		content += "#define TASKNAME \"" + task.getName() + "\"\n";
		content += "#define FUNC_INIT " + task.getName() + "_init\n";
		content += "#define FUNC_GO " + task.getName() + "_go\n";
		content += "#define FUNC_WRAPUP " + task.getName() + "_wrapup\n";
		content += "\n";
		content += "#include <mars/task.h>\n#include \"CIC_wrapper.h\"\n\n";

		if (mAlgorithm.getHeaders() != null)
			for (String headerFile : mAlgorithm.getHeaders().getHeaderFile())
				content += "#include \"" + headerFile + "\"\n";

		content += "#ifdef __PPU__\n\n";
		content += "#define NUM_SPE_TO_USE       (" + Integer.toString(task.getProc().size()) + ")\n";

		int globalPeriod = 0;
		if (mGlobalPeriodMetric.equalsIgnoreCase("h"))
			globalPeriod = mGlobalPeriod * 3600 * 1000 * 1000;
		else if (mGlobalPeriodMetric.equalsIgnoreCase("m"))
			globalPeriod = mGlobalPeriod * 60 * 1000 * 1000;
		else if (mGlobalPeriodMetric.equalsIgnoreCase("s"))
			globalPeriod = mGlobalPeriod * 1000 * 1000;
		else if (mGlobalPeriodMetric.equalsIgnoreCase("ms"))
			globalPeriod = mGlobalPeriod * 1000;
		else if (mGlobalPeriodMetric.equalsIgnoreCase("us"))
			globalPeriod = mGlobalPeriod * 1;
		else {
			System.out.println("[makeTasks] Not supported metric of period");
			System.exit(-1);
		}

		int runCount = 0;
		if (task.getRunCondition().equals("TIME_DRIVEN"))
			runCount = globalPeriod / Integer.parseInt(task.getPeriod());
		else
			runCount = 0;

		content += "#define CIC_RUNCOUNT     (" + Integer.toString(runCount) + "*" + task.getRunRate() + ")\n";
		content += "#define SPE_PROG_NAME     CIC_wrapper_spe_" + task.getName() + "_prog\n";

		content += "static cic_channel_info_entry __attribute__((aligned(32))) cic_channel_info_ppe[][NUM_SPE_TO_USE] = {\n";
		int index = 0;
		while (index < mQueue.size()) {
			Queue queue = mQueue.get(index);

			if (queue.getSrc().equals(task.getName()) || queue.getDst().equals(task.getName())) {
				String portName = new String();
				String portId = new String();
				String direction = new String();
				if (queue.getSrc().equals(task.getName())) {
					portName = queue.getSrcPortName();
					portId = queue.getSrcPortId();
					direction = "MARS_TASK_QUEUE_MPU_TO_HOST";
				} else if (queue.getDst().equals(task.getName())) {
					portName = queue.getDstPortName();
					portId = queue.getDstPortId();
					direction = "MARS_TASK_QUEUE_HOST_TO_MPU";
				}
				content += "\t{\n";
				for (int i = 0; i < task.getProc().size(); i++) {
					if (queue.getSampleType().isEmpty()) {
						content += "\t\t{0x0, " + queue.getIndex() + ", " + direction + ", \"" + portName + "\", "
								+ portId + ", " + queue.getSampleSize() + ", " + queue.getSize() + " / "
								+ queue.getSampleSize() + "},\n";
					} else {
						content += "\t\t{0x0, " + queue.getIndex() + ", " + direction + ", \"" + portName + "\", "
								+ portId + ", sizeof(" + queue.getSampleType() + "), " + queue.getSize() + " / "
								+ queue.getSampleSize() + "},\n";
					}

				}
				content += "\t},\n";
			}
			index++;
		}
		content += "};\n\n";

		content += "#elif defined(__SPU__)\n\n";
		content += "cic_task_function_t cic_task_function = { FUNC_INIT, FUNC_GO, FUNC_WRAPUP };\n\n";
		content += "static cic_channel_info_entry __attribute__((aligned(32))) cic_channel_info_spe[] = {\n";
		index = 0;
		while (index < mQueue.size()) {
			Queue queue = mQueue.get(index);

			if (queue.getSrc().equals(task.getName()) || queue.getDst().equals(task.getName())) {
				String portName = new String();
				String portId = new String();
				String direction = new String();
				if (queue.getSrc().equals(task.getName())) {
					portName = queue.getSrcPortName();
					portId = queue.getSrcPortId();
					direction = "MARS_TASK_QUEUE_MPU_TO_HOST";
				} else if (queue.getDst().equals(task.getName())) {
					portName = queue.getDstPortName();
					portId = queue.getDstPortId();
					direction = "MARS_TASK_QUEUE_HOST_TO_MPU";
				}
				if (queue.getSampleType().isEmpty()) {
					content += "\t\t{0x0, " + queue.getIndex() + ", " + direction + ", \"" + portName + "\", " + portId
							+ ", " + queue.getSampleSize() + ", " + queue.getSize() + " / " + queue.getSampleSize()
							+ "},\n";
				} else {
					content += "\t\t{0x0, " + queue.getIndex() + ", " + direction + ", \"" + portName + "\", " + portId
							+ ", sizeof(" + queue.getSampleType() + "), " + queue.getSize() + " / "
							+ queue.getSampleSize() + "},\n";
				}
			}
			index++;
		}
		content += "};\n\n";
		content += "#else\n\n    #error\n\n#endif\n\n";

		return content;
	}

	public void LibraryRelationGeneration() {
		for (TaskLibraryConnectionType taskLibCon : mAlgorithm.getLibraryConnections().getTaskLibraryConnection()) {
			int taskLibConFlag = 0;
			for (TaskLibraryConnectionType t : mTaskLibConnection) {
				if (t.getSlaveLibrary().equals(taskLibCon.getSlaveLibrary())) {
					taskLibConFlag = 1;
					break;
				}
			}
			if (taskLibConFlag == 0)
				mTaskLibConnection.add(taskLibCon);
		}

		for (LibraryLibraryConnectionType libLibCon : mAlgorithm.getLibraryConnections()
				.getLibraryLibraryConnection()) {
			int libLibConFlag = 0;
			for (LibraryLibraryConnectionType t : mLibLibConnection) {
				if (t.getSlaveLibrary().equals(libLibCon.getSlaveLibrary())) {
					libLibConFlag = 1;
					break;
				}
			}
			if (libLibConFlag == 0)
				mLibLibConnection.add(libLibCon);
		}

		for (TaskLibraryConnectionType taskLibCon : mTaskLibConnection) {
			int flag_1 = 0, flag_2 = 0, flag_3 = 0;
			for (Task task : mTaskOnSPE)
				if (taskLibCon.getMasterTask().equals(task.getName()))
					flag_1 = 1;
			for (Library library : mLibraryOnSPE)
				if (taskLibCon.getSlaveLibrary().equals(library.getName()))
					flag_2 = 1;

			if (flag_1 == 0 && flag_2 == 0)
				mPPEToPPEList.add(taskLibCon.getSlaveLibrary());
			else if (flag_1 == 0 && flag_2 == 1) {
				mL_FileList.add(taskLibCon.getSlaveLibrary());
				for (String stub : mStubList)
					if (stub.equals(taskLibCon.getSlaveLibrary()))
						flag_3 = 1;
				if (flag_3 == 0)
					mStubList.add(taskLibCon.getSlaveLibrary());
			} else {
				for (String stub : mStubList)
					if (stub.equals(taskLibCon.getSlaveLibrary()))
						flag_3 = 1;
				if (flag_3 == 0)
					mStubList.add(taskLibCon.getSlaveLibrary());
			}
		}

		for (LibraryLibraryConnectionType libLibCon : mLibLibConnection) {
			int flag_1 = 0, flag_2 = 0, flag_3 = 0;
			for (Library library : mLibraryOnSPE)
				if (libLibCon.getMasterLibrary().equals(library.getName()))
					flag_1 = 1;
			for (Library library : mLibraryOnSPE)
				if (libLibCon.getSlaveLibrary().equals(library.getName()))
					flag_2 = 1;

			if (flag_1 == 0 && flag_2 == 0)
				mPPEToPPEList.add(libLibCon.getSlaveLibrary());
			else if (flag_1 == 0 && flag_2 == 1) {
				mL_FileList.add(libLibCon.getSlaveLibrary());
				for (String stub : mStubList)
					if (stub.equals(libLibCon.getSlaveLibrary()))
						flag_3 = 1;
				if (flag_3 == 0)
					mStubList.add(libLibCon.getSlaveLibrary());
			} else {
				for (String stub : mStubList)
					if (stub.equals(libLibCon.getSlaveLibrary()))
						flag_3 = 1;
				if (flag_3 == 0)
					mStubList.add(libLibCon.getSlaveLibrary());
			}
		}

		if (mAlgorithm.getLibraries() != null) {
			for (Library library : mLibrary.values()) {
				int flag = 0;
				if (mLibraryOnSPE.contains(library))
					continue;
				else {
					for (TaskLibraryConnectionType taskLibCon : mTaskLibConnection)
						if (taskLibCon.getSlaveLibrary().equals(library.getName()))
							for (Task task : mTaskOnSPE)
								if (task.getName().equals(taskLibCon.getMasterTask()))
									flag = 1;
					for (LibraryLibraryConnectionType libLibCon : mLibLibConnection)
						if (libLibCon.getSlaveLibrary().equals(library.getName()))
							for (Library lib : mLibraryOnSPE)
								if (lib.getName().equals(libLibCon.getMasterLibrary()))
									flag = 1;
					if (flag == 1)
						mStubFileList.add(library.getName());
				}

			}
		}
	}

	public void generateLibraryChannelHeader(String mDestFile, String mTemplateFile) {
		File fileOut = new File(mDestFile);
		File templateFile = new File(mTemplateFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);

			outstream.write(translateLibraryChannelHeader(content).getBytes());
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public String translateLibraryChannelHeader(String mContent) {
		String code = mContent;
		String headerIncludeCode = "";
		String channelEntriesCode = "";

		if (mAlgorithm.getLibraries() != null)
			for (LibraryType library : mAlgorithm.getLibraries().getLibrary())
				headerIncludeCode += "#include \"" + library.getName() + "_data_structure.h\"\n";

		if (mAlgorithm.getLibraries() != null) {
			for (Library library : mLibrary.values()) {
				int flag = 0;
				for (String ppe_ppe : mPPEToPPEList)
					if (ppe_ppe == library.getName())
						flag = 1;
				if (flag == 1)
					continue;
				else {
					channelEntriesCode += "\t{\n\t\t" + library.getIndex() + ", 0, \n"
							+ "\t\t{CHANNEL_TYPE_ARRAY_CHANNEL, NULL, NULL, NULL, sizeof(" + library.getName()
							+ "_func_data), -1, NULL, NULL, NULL, PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, sizeof("
							+ library.getName() + "_func_data), false, false }, \n";
					channelEntriesCode += "\t\t{CHANNEL_TYPE_ARRAY_CHANNEL, NULL, NULL, NULL, sizeof("
							+ library.getName()
							+ "_ret_data), -1, NULL, NULL, NULL, PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, sizeof("
							+ library.getName() + "_ret_data), false, false }, \n";
					channelEntriesCode += "\t\tPTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER\n";
					channelEntriesCode += "\t},\n";
				}
			}
		}

		code = code.replace("##HEADER_INCLUDE", headerIncludeCode);
		code = code.replace("##CHANNEL_ENTRIES", channelEntriesCode);

		return code;
	}

	public void generateLibraryPortmapHeader(String mDestFile, String mTemplateFile) {
		File fileOut = new File(mDestFile);
		File templateFile = new File(mTemplateFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);

			outstream.write(translateLibraryPortmapHeader(content).getBytes());
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public String translateLibraryPortmapHeader(String mContent) {
		String code = mContent;
		String portmapEntriesCode = "";

		if (mAlgorithm.getLibraries() != null) {
			for (Library library : mLibrary.values()) {
				int flag = 0;
				for (String ppe_ppe : mPPEToPPEList)
					if (ppe_ppe == library.getName())
						flag = 1;
				if (flag == 1)
					continue;
				else {
					portmapEntriesCode += "\tENTRY(" + library.getIndex() + "/*" + library.getName() + "*/, "
							+ library.getIndex() + ", " + library.getIndex() + ", 'r'),\n";
					portmapEntriesCode += "\tENTRY(" + library.getIndex() + "/*" + library.getName() + "*/, "
							+ library.getIndex() + ", " + library.getIndex() + ", 'w'),\n";
				}
			}
		}
		code = code.replace("##PORTMAP_ENTRIES", portmapEntriesCode);

		return code;
	}

	public void generateLibraryStubHeader(String mDestFile, String mTemplateFile) {
		File fileOut = new File(mDestFile);
		File templateFile = new File(mTemplateFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);

			outstream.write(translateLibraryStubHeader(content).getBytes());
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public String translateLibraryStubHeader(String mContent) {
		String code = mContent;
		String externalPrototypeCode = "";
		String stubEntriesCode = "";

		for (String stub : mStubList) {
			externalPrototypeCode += "extern void " + stub + "_stub_init    (void);\n";
			externalPrototypeCode += "extern int " + stub + "_stub_go       (void);\n";
			externalPrototypeCode += "extern void " + stub + "_stub_wrapup  (void);\n\n";
		}

		int index = 0;
		for (String stub : mStubList) {
			stubEntriesCode += "ENTRY(" + Integer.toString(index) + ", \"" + stub + "\", " + stub + "_stub_init, "
					+ stub + "_stub_go, " + stub + "_stub_wrapup, 1),\n";
			index++;
		}

		code = code.replace("##EXTERNAL_FUNCTION_PROTOTYPES", externalPrototypeCode);
		code = code.replace("##STUB_ENTRIES", stubEntriesCode);

		return code;
	}

	public void generateLibraryRefHeader(String mDestFile, String mTemplateFile) {
		File fileOut = new File(mDestFile);
		File templateFile = new File(mTemplateFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);

			outstream.write(translateLibraryRefHeader(content).getBytes());
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public String translateLibraryRefHeader(String mContent) {
		String code = mContent;
		String externalPrototypeCode = "";
		String refEntriesCode = "";
		List<String> refList = new ArrayList<String>();

		if (mAlgorithm.getLibraries() != null) {
			for (Library library : mLibrary.values()) {
				int flag = 0;
				for (String stub : mStubList)
					if (stub == library.getName())
						flag = 1;
				if (flag == 0) {
					externalPrototypeCode += "extern void l_" + library.getName() + "_init    (void);\n";
					externalPrototypeCode += "extern void l_" + library.getName() + "_wrapup  (void);\n\n";
					refList.add(library.getName());
				}
			}
		}

		int index = 0;
		for (String ref : refList) {
			refEntriesCode += "ENTRY(" + Integer.toString(index) + ", \"" + ref + "\", l_" + ref + "_init, l_" + ref
					+ "_wrapup, 1),\n";
			index++;
		}

		code = code.replace("##EXTERNAL_FUNCTION_PROTOTYPES", externalPrototypeCode);
		code = code.replace("##REF_ENTRIES", refEntriesCode);

		return code;
	}

	public void generateLibraryCalleeFile() {
		if (mAlgorithm.getLibraries() != null) {
			for (Library library : mLibrary.values()) {
				File fileOut = new File(mOutputPath + "stub_" + library.getName() + ".c");
				String content = "";
				try {
					FileOutputStream outstream = new FileOutputStream(fileOut);
					outstream.write(translateLibraryCalleeFile(library).getBytes());
					outstream.close();
				} catch (FileNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
	}

	public String translateLibraryCalleeFile(Library library) {
		int index = 0;
		String content = new String();

		content += "\n#include \"" + library.getName() + ".h\"\n";
		content += "\n#include \"LIB_port.h\"\n\n";
		content += "#define LIBCALL(x, ...) LIBCALL_##x(__VA_ARGS__)\n";

		String t_libCon = new String();
		for (TaskLibraryConnectionType libCon : mAlgorithm.getLibraryConnections().getTaskLibraryConnection()) {
			if (libCon.getSlaveLibrary().equals(library.getName())) {
				t_libCon = libCon.getMasterPort();
				content += "#define LIBCALL_" + libCon.getMasterPort() + "(f, ...) l_" + library.getName()
						+ "_##f(__VA_ARGS__)\n\n";
				break;
			}
		}

		for (LibraryLibraryConnectionType libCon : mAlgorithm.getLibraryConnections().getLibraryLibraryConnection()) {
			if (libCon.getSlaveLibrary().equals(library.getName())) {
				t_libCon = libCon.getMasterPort();
				content += "#define LIBCALL_" + libCon.getMasterPort() + "(f, ...) l_" + library.getName()
						+ "_##f(__VA_ARGS__)\n\n";
				break;
			}
		}

		content += "static int channel_id;\n";
		content += "void " + library.getName() + "_stub_init()\n{\n";
		content += "\tchannel_id = init_lib_port(" + library.getIndex() + ");\n";
		content += "\tLIBCALL(" + t_libCon + ", init);\n}\n\n";

		content += "void " + library.getName() + "_stub_go()\n{\n";
		content += "\tint func_index;\n\tint index = 0;\n";
		content += "\t" + library.getName() + "_func_data receive_data;\n";
		content += "\t" + library.getName() + "_ret_data send_data;\n";

		for (Function func : library.getFuncList()) {
			if (func.getReturnType().equalsIgnoreCase("void"))
				content += "\tint ret_" + func.getFunctionName() + " = 0;\n";
			else
				content += "\t" + func.getReturnType() + " ret_" + func.getFunctionName() + ";\n";
		}

		content += "\twhile(1)\n\t{\n\t\tindex = LIB_AC_CHECK(channel_id, 0);\n";
		content += "\t\tLIB_RECEIVE(channel_id, 0, &receive_data, sizeof(" + library.getName()
				+ "_func_data), index);\n";
		content += "\n\t\tfunc_index = receive_data.func_num;\n\n";
		content += "\t\tswitch(func_index) {\n";

		for (Function func : library.getFuncList()) {
			content += "\t\t\tcase " + func.getIndex() + " : \n";
			if (func.getReturnType().equalsIgnoreCase("void"))
				content += "\t\t\t\tLIBCALL(" + t_libCon + ", " + func.getFunctionName();
			else
				content += "\t\t\t\tret_" + func.getFunctionName() + "= LIBCALL(" + t_libCon + ", "
						+ func.getFunctionName();

			for (Argument arg : func.getArgList())
				content += ", receive_data.func." + func.getFunctionName() + "." + arg.getVariableName();
			content += ");\n";
			content += "\t\t\t\tsend_data.ret.ret_" + func.getFunctionName() + " = ret_" + func.getFunctionName()
					+ ";\n";
			content += "\t\t\t\tLIB_SEND(channel_id, 1, &send_data, sizeof(" + library.getName()
					+ "_ret_data), index);\n";
			content += "\t\t\t\tbreak; \n\n";
		}
		content += "\t\t}\n";
		content += "\t}\n}\n\n";

		content += "void " + library.getName() + "_stub_wrapup()\n{\n";
		content += "\tLIBCALL(" + t_libCon + ", wrapup);\n}\n\n";

		return content;
	}

	public void generateLibraryCallerFile() {
		if (mAlgorithm.getLibraries() != null) {
			for (Library library : mLibrary.values()) {
				File fileOut = new File(mOutputPath + "l_" + library.getName() + ".c");
				String content = "";
				try {
					FileOutputStream outstream = new FileOutputStream(fileOut);
					outstream.write(translateLibraryCallerFile(library).getBytes());
					outstream.close();
				} catch (FileNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}

	}

	public String translateLibraryCallerFile(Library library) {
		int index = 0;
		String content = new String();

		int flag = 0;
		int taskLibraryFlag = 0;
		for (TaskLibraryConnectionType taskLib : mAlgorithm.getLibraryConnections().getTaskLibraryConnection()) {
			if (taskLib.getSlaveLibrary().equals(library.getName())) {
				taskLibraryFlag = 1;
				for (Task task : mTaskOnSPE)
					if (taskLib.getMasterTask().equals(task.getName()))
						flag = 1;
			}
		}
		for (LibraryLibraryConnectionType libLib : mAlgorithm.getLibraryConnections().getLibraryLibraryConnection()) {
			if (libLib.getSlaveLibrary().equals(library.getName())) {
				taskLibraryFlag = 2;
				for (Library subLib : mLibraryOnSPE)
					if (libLib.getMasterLibrary().equals(subLib.getName()))
						flag = 1;
			}
		}

		content += "#include \"LIB_port.h\"\n#include <pthread.h>\n";
		content += "#define LIBFUNC(rtype, f, ...) rtype l_" + library.getName() + "_##f(__VA_ARGS__)\n\n";
		content += "LIBFUNC(void, init, void)\n{\n\t// initialize\n}\nLIBFUNC(void, wrapup, void)\n{\n\t// wrapup\n}\n";

		for (Function func : library.getFuncList()) {
			content += "LIBFUNC(" + func.getReturnType() + ", " + func.getFunctionName();
			int count = 0;
			for (Argument arg : func.getArgList()) {
				content += ", " + arg.getType() + " " + arg.getVariableName();
				count++;
			}
			if (count == 0)
				content += ", void";

			content += ")\n{\n";
			content += "\tint index = 0;\n";
			content += "\t" + library.getName() + "_func_data send_data;\n";
			content += "\t" + library.getName() + "_ret_data receive_data;\n\t";

			if (!func.getReturnType().equals("void"))
				content += func.getReturnType() + " ret;\n\n";

			content += "\tint channel_id = init_lib_port(" + library.getIndex() + ");\n\n";

			if (taskLibraryFlag == 1) {
				content += "\tsend_data.task_library = 1;\n";
				content += "\tsend_data.task_id = get_mytask_id();\n";
			} else if (taskLibraryFlag == 2) {
				content += "\tsend_data.task_library = 2;\n";
				content += "\tsend_data.task_id = " + library.getIndex() + ";\n";
			}
			content += "\tsend_data.func_num = " + func.getIndex() + ";\n";

			for (Argument arg : func.getArgList())
				content += "\tsend_data.func." + func.getFunctionName() + "." + arg.getVariableName() + " = "
						+ arg.getVariableName() + ";\n";

			content += "\n\t// write port\n";
			if (flag == 1)
				content += "//";
			content += "\tlock_lib_channel(channel_id);\n";

			content += "\tLIB_SEND(channel_id, 0, &send_data, sizeof(" + library.getName() + "_func_data), index);\n";
			content += "\t// read port\n";
			content += "\tindex = LIB_AC_CHECK(channel_id, 1);\n";
			content += "\tLIB_RECEIVE(channel_id, 1, &receive_data, sizeof(" + library.getName()
					+ "_ret_data), index);\n";

			if (flag == 1)
				content += "//";
			content += "\tunlock_lib_channel(channel_id);\n";

			if (!func.getReturnType().equals("void")) {
				content += "\tret = receive_data.ret.ret_" + func.getFunctionName() + ";\n\n";
				content += "\treturn ret;\n";
			}

			content += "}\n\n";
		}
		content += "#undef LIBFUNC\n\n";

		return content;
	}

	public void generateLibraryDataStructureHeader() {
		if (mAlgorithm.getLibraries() != null) {
			for (Library library : mLibrary.values()) {
				File fileOut = new File(mOutputPath + library.getName() + "_data_structure.h");
				String content = "";
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
		}
	}

	public String translateLibraryDataStructureHeader(Library library) {
		int index = 0;
		String content = new String();

		content += "#ifndef _" + library.getName().toUpperCase() + "_DATA_STRUCTURE_H_\n";
		content += "#define _" + library.getName().toUpperCase() + "_DATA_STRUCTURE_H_\n\n";
		content += "#include <stdint.h>\n\n";

		for (String extraHeader : library.getExtraHeader())
			content += "#include \"" + extraHeader + "\"\n\n";

		content += "// Added by jhw at 10.01.21 for library\n" + "typedef struct {\n" + "\tint task_library;\n"
				+ "\tint task_id;\n" + "\tint func_num;\n" + "\tunion {\n";

		for (Function func : library.getFuncList()) {
			content += "\t\tstruct {\n";
			for (Argument arg : func.getArgList()) {
				content += "\t\t\t" + arg.getType() + " " + arg.getVariableName() + ";\n";
				index++;
			}
			if (index == 0)
				content += "\t\t\tint temp;\n";
			content += "\t\t} " + func.getFunctionName() + ";\n";
		}
		content += "\t} func;\n";
		content += "} " + library.getName() + "_func_data;\n\n";

		index = 0;
		content += "typedef struct {\n" + "\tint func_num;\n" + "\tunion {\n";
		for (Function func : library.getFuncList()) {
			if (func.getReturnType().equals("void"))
				content += "\t\tint ret_" + func.getFunctionName() + ";\n";
			else
				content += "\t\t" + func.getReturnType() + " ret_" + func.getFunctionName() + ";\n";
			index++;
		}
		if (index == 0)
			content += "\t\t\tint temp;\n";

		content += "\t} ret;\n";
		content += "} " + library.getName() + "_ret_data;\n\n";
		content += "\n#endif\n";

		return content;
	}

	public void generateLibraryInfoFile() {
		if (mAlgorithm.getLibraries() != null) {
			for (Library library : mLibrary.values()) {
				if (mLibraryOnSPE.contains(library)) {
					String fileOut = mOutputPath + library.getName() + ".info.c";
					File f = new File(fileOut);
					FileOutputStream outstream;
					try {
						outstream = new FileOutputStream(fileOut);
						outstream.write(translateLibraryInfoHeader(library).getBytes());
						outstream.close();
					} catch (FileNotFoundException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					} catch (IOException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
			}
		}
	}

	public String translateLibraryInfoHeader(Library library) {
		String content = new String();

		content += "\n";
		content += "#define LIBNAME \"" + library.getName() + "\"\n";
		content += "#define LIB_INIT " + library.getName() + "_stub_init\n";
		content += "#define LIB_GO " + library.getName() + "_stub_go\n";
		content += "#define LIB_WRAPUP " + library.getName() + "_stub_wrapup\n";
		content += "\n";
		content += "#include <mars/task.h>\n#include \"LIB_wrapper.h\"\n\n";

		for (LibraryLibraryConnectionType libLib : mAlgorithm.getLibraryConnections().getLibraryLibraryConnection())
			if (libLib.getMasterLibrary().equals(library.getName()))
				content += "#include \"" + libLib.getSlaveLibrary() + "_data_structure.h\"\n";

		if (mAlgorithm.getHeaders() != null)
			for (String headerFile : mAlgorithm.getHeaders().getHeaderFile())
				content += "#include \"" + headerFile + "\"\n";

		content += "#ifdef __PPU__\n\n";
		content += "#define NUM_SPE_TO_USE       (1)\n";
		content += "#define L_SPE_PROG_NAME     LIB_wrapper_spe_" + library.getName() + "_prog\n";

		content += "static uint64_t __attribute__((aligned(32))) lib_channel_event_flag_ppe[NUM_SPE_TO_USE] = {0x0};\n\n";
		content += "static lib_channel_info_entry __attribute__((aligned(32))) lib_channel_info_ppe[][NUM_SPE_TO_USE] = {\n";

		{
			content += "\t{\n";
			content += "\t\t{0x0, " + library.getIndex() + ", MARS_TASK_QUEUE_HOST_TO_MPU, 0, sizeof("
					+ library.getName() + "_func_data), 1},\n";
			content += "\t},\n";
			content += "\t{\n";
			content += "\t\t{0x0, " + library.getIndex() + ", MARS_TASK_QUEUE_MPU_TO_HOST, 1, sizeof("
					+ library.getName() + "_ret_data), 1},\n";
			content += "\t},\n";
		}

		Library libLibLibrary = null;
		for (LibraryLibraryConnectionType libLib : mAlgorithm.getLibraryConnections().getLibraryLibraryConnection()) {
			if (libLib.getMasterLibrary().equals(library.getName())) {
				for (Library lib : mLibrary.values())
					if (libLib.getSlaveLibrary().equals(lib.getName()))
						libLibLibrary = lib;
				content += "\t{\n";
				content += "\t\t{0x0, " + libLibLibrary.getIndex() + ", MARS_TASK_QUEUE_MPU_TO_HOST, 0, sizeof("
						+ libLibLibrary.getName() + "_func_data), 1},\n";
				content += "\t},\n";
				content += "\t{\n";
				content += "\t\t{0x0, " + libLibLibrary.getIndex() + ", MARS_TASK_QUEUE_HOST_TO_MPU, 1, sizeof("
						+ libLibLibrary.getName() + "_ret_data), 1},\n";
				content += "\t},\n";
			}
		}

		content += "};\n\n";
		content += "#elif defined(__SPU__)\n\n";
		content += "lib_stub_function_t lib_stub_function = { LIB_INIT, LIB_GO, LIB_WRAPUP };\n\n";
		content += "static uint64_t __attribute__((aligned(32))) lib_channel_event_flag_spe = {0x0};\n\n";
		content += "static lib_channel_info_entry __attribute__((aligned(32))) lib_channel_info_spe[] = {\n";

		content += "\t{ 0x0, " + library.getIndex() + ", MARS_TASK_QUEUE_HOST_TO_MPU, 0, sizeof(" + library.getName()
				+ "_func_data), 1},\n";
		content += "\t{ 0x0, " + library.getIndex() + ", MARS_TASK_QUEUE_MPU_TO_HOST, 1, sizeof(" + library.getName()
				+ "_ret_data), 1},\n";

		for (LibraryLibraryConnectionType libLib : mAlgorithm.getLibraryConnections().getLibraryLibraryConnection()) {
			if (libLib.getMasterLibrary().equals(library.getName())) {
				for (Library lib : mLibrary.values())
					if (libLib.getSlaveLibrary().equals(lib.getName()))
						libLibLibrary = lib;
				content += "\t{ 0x0, " + library.getIndex() + ", MARS_TASK_QUEUE_MPU_TO_HOST, 0, sizeof("
						+ library.getName() + "_func_data), 1},\n";
				content += "\t{ 0x0, " + library.getIndex() + ", MARS_TASK_QUEUE_HOST_TO_MPU, 1, sizeof("
						+ library.getName() + "_ret_data), 1},\n";
			}
		}

		content += "};\n\n";
		content += "#else\n\n    #error\n\n#endif\n\n";

		return content;
	}

	public void generateControlTaskMapHeader(String mDestFile, String mTemplateFile) {
		File fileOut = new File(mDestFile);
		File templateFile = new File(mTemplateFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);

			outstream.write(translateControlTaskMapHeader(content).getBytes());
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public String translateControlTaskMapHeader(String mContent) {
		String code = mContent;
		String controlTaskMapCode = "";

		for (Task task : mTask.values()) {
			String controlTaskId = new String();
			if (task.getIsSlaveTask()) {
				for (String controlTask : task.getControllingTask()) {
					for (Task t : mTask.values()) {
						if (t.getName().equals(controlTask)) {
							controlTaskId = t.getIndex();
							break;
						}
					}
					controlTaskMapCode += "\tENTRY(" + task.getIndex() + "\"" + task.getName() + "\", " + controlTaskId
							+ ", \"" + controlTask + "\"),\n";
				}
			}
		}

		code = code.replace("##CONTROL_TASKMAP", controlTaskMapCode);

		return code;
	}

	public void generateControlPortMapHeader(String mDestFile, String mTemplateFile) {
		File fileOut = new File(mDestFile);
		File templateFile = new File(mTemplateFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);

			outstream.write(translateControlPortMapHeader(content).getBytes());
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public String translateControlPortMapHeader(String mContent) {
		String code = mContent;
		String controlPortMapCode = "";

		int index = 0;
		for (Task task : mTask.values()) {
			if (mTaskOnSPE.contains(task)) {
				if (task.getIsSlaveTask()) {
					controlPortMapCode += "\tENTRY(" + task.getIndex() + "/* " + task.getName() + "*/, \""
							+ task.getName() + "\", " + index + "),\n";
				}
			}
		}

		code = code.replace("##CONTROL_PORTMAP", controlPortMapCode);

		return code;
	}

	public void generateControlChannelHeader(String mDestFile, String mTemplateFile) {
		File fileOut = new File(mDestFile);
		File templateFile = new File(mTemplateFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);

			outstream.write(translateControlChannelHeader(content).getBytes());
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public String translateControlChannelHeader(String mContent) {
		String code = mContent;
		String controlChannelCode = "";

		int index = 0;
		for (Task task : mTask.values()) {
			if (mTaskOnSPE.contains(task)) {
				if (task.getIsSlaveTask()) {
					controlChannelCode += "\t{\n\t\t" + index + ", 0, \n";
					controlChannelCode += "\t\t{CHANNEL_TYPE_ARRAY_CHANNEL, NULL, NULL, NULL,  sizeof(CONTROL_SEND_PACKET), -1, NULL, NULL, NULL, PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, sizeof(CONTROL_SEND_PACKET), false, false, 0 },\n";
					controlChannelCode += "\t\t{CHANNEL_TYPE_ARRAY_CHANNEL, NULL, NULL, NULL,  sizeof(CONTROL_SEND_PACKET), -1, NULL, NULL, NULL, PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, sizeof(CONTROL_SEND_PACKET), false, false, 0 },\n";
					controlChannelCode += "\t\tPTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER\n";
					controlChannelCode += "\t},\n";
					index++;
				}
			}
		}

		code = code.replace("##CONTROL_CHANNELS", controlChannelCode);

		return code;
	}

	public void generateControlParamListHeader(String mDestFile, String mTemplateFile) {
		File fileOut = new File(mDestFile);
		File templateFile = new File(mTemplateFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);

			outstream.write(translateControlParamListHeader(content).getBytes());
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public String translateControlParamListHeader(String mContent) {
		String code = mContent;
		String controlParamListCode = "";

		int index = 0;
		for (Task task : mTask.values()) {
			index = 0;
			for (TaskParameterType param : task.getParameter()) {
				controlParamListCode += "\t{" + task.getIndex() + ", " + index + ", \"" + task.getName() + "\", \""
						+ param.getName() + "\", " + param.getValue() + "},\n";
				index++;
			}
		}

		code = code.replace("##PARAM_LIST", controlParamListCode);

		return code;
	}

	public void generateControlInfoHeader(String mDestFile, String mTemplateFile) {
		File fileOut = new File(mDestFile);
		File templateFile = new File(mTemplateFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);

			outstream.write(translateControlInfoHeader(content).getBytes());
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public String translateControlInfoHeader(String mContent) {
		String code = mContent;
		String controlGroupCountCode = "";
		String controlChannelCountCode = "";
		String controlChannelCode = "";

		if (mControl.getExclusiveControlTasksList() == null)
			controlGroupCountCode += "#define CONTROL_GROUP_COUNT 1";
		else
			controlGroupCountCode += "#define CONTROL_GROUP_COUNT "
					+ mControl.getExclusiveControlTasksList().getExclusiveControlTasks().size();

		if (mControl.getControlTasks() == null)
			controlChannelCountCode += "#define CONTROL_CHANNEL_COUNT 1";
		else
			controlChannelCountCode += "#define CONTROL_CHANNEL_COUNT "
					+ mControl.getControlTasks().getControlTask().size();

		if (mControl.getControlTasks() != null) {
			int groupIndex = 0;
			for (ControlTaskType controlTask : mControl.getControlTasks().getControlTask()) {
				int slaveProcId = 0;
				int masterProcId = 0;
				int taskId = 0;
				int groupFlag = 0;

				for (Task task : mTask.values()) {
					if (task.getName().equals(controlTask.getTask())) {
						taskId = Integer.parseInt(task.getIndex());
						int proc = task.getProc().get("Default").get("Default").get(0); // Need
																						// to
																						// fix
						masterProcId = proc;
						break;
					}
				}
				for (Task task : mTask.values()) {
					if (task.getName().equals(controlTask.getSlaveTask())) {
						int proc = task.getProc().get("Default").get("Default").get(0); // Need
																						// to
																						// fix
						slaveProcId = proc;
						break;
					}
				}

				int index = 0;
				int groupId = 0;

				for (ExclusiveControlTasksType exclusiveTasks : mControl.getExclusiveControlTasksList()
						.getExclusiveControlTasks()) {
					for (String cTask : exclusiveTasks.getControlTask()) {
						if (controlTask.getTask().equals(cTask)) {
							groupId = index;
							groupFlag = 1;
							break;
						}
					}
					index++;
					if (groupFlag == 0) {
						groupId = groupIndex;
						groupIndex++;
					}
				}

				controlChannelCode += "\t{" + taskId + ", " + controlTask.getPriority().intValue() + ", " + groupId
						+ ", 0, 0, {0, }, {0, } },\n";
			}
		}

		code = code.replace("##CONTROL_GROUP_COUNT", controlGroupCountCode);
		code = code.replace("##CONTROL_CHANNEL_COUNT", controlChannelCountCode);
		code = code.replace("##CONTROL_CHANNELS", controlChannelCode);

		return code;
	}

	public void generateControlSlaveInfoFile() {
		int chId = 0;
		for (Task task : mTask.values()) {
			if (task.getIsSlaveTask()) {
				if (mTaskOnSPE.contains(task)) {
					String fileOut = mOutputPath + task.getName() + ".info.c";
					File f = new File(fileOut);
					FileOutputStream outstream;
					try {
						outstream = new FileOutputStream(fileOut);
						outstream.write(translateControlSlaveInfoFile(task, chId).getBytes());
						outstream.close();
						chId++;
					} catch (FileNotFoundException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					} catch (IOException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
			}
		}
	}

	public String translateControlSlaveInfoFile(Task task, int chId) {
		String content = new String();

		content += "\n";
		content += "#define TASKID " + task.getIndex() + "\n";
		content += "#define CONTASKNAME \"" + task.getName() + "\"\n";
		content += "#define FUNC_INIT " + task.getName() + "_init\n";
		content += "#define FUNC_GO " + task.getName() + "_go\n";
		content += "#define FUNC_WRAPUP " + task.getName() + "_wrapup\n";
		content += "\n";
		content += "#include <mars/task.h>\n#include \"CON_wrapper.h\"\n#include \"control_info.h\"\n";

		if (mAlgorithm.getHeaders() != null)
			for (String headerFile : mAlgorithm.getHeaders().getHeaderFile())
				content += "#include \"" + headerFile + "\"\n";

		content += "#ifdef __PPU__\n\n";
		content += "#define NUM_SPE_TO_USE       (" + Integer.toString(task.getProc().size()) + ")\n";
		content += "#define C_SPE_PROG_NAME     CON_wrapper_spe_" + task.getName() + "_prog\n";

		content += "static uint64_t __attribute__((aligned(32))) con_channel_event_flag_ppe[NUM_SPE_TO_USE] = {0x0};\n\n";
		content += "static con_channel_info_entry __attribute__((aligned(32))) con_channel_info_ppe[][NUM_SPE_TO_USE] = {\n";

		{
			content += "\t{\n";
			content += "\t\t{0x0, " + chId + ", MARS_TASK_QUEUE_HOST_TO_MPU, 0, sizeof(CONTROL_SEND_PACKET), 1},\n";
			content += "\t},\n";
			content += "\t{\n";
			content += "\t\t{0x0, " + chId + ", MARS_TASK_QUEUE_MPU_TO_HOST, 1, sizeof(CONTROL_SEND_PACKET), 1},\n";
			content += "\t},\n";
			content += "};\n\n";
		}

		content += "static cic_channel_info_entry __attribute__((aligned(32))) cic_channel_info_ppe[][NUM_SPE_TO_USE] = {\n";

		int index = 0;
		while (index < mQueue.size()) {
			Queue queue = mQueue.get(index);

			if (queue.getSrc().equals(task.getName()) || queue.getDst().equals(task.getName())) {
				String portName = new String();
				String portId = new String();
				String direction = new String();
				if (queue.getSrc().equals(task.getName())) {
					portName = queue.getSrcPortName();
					portId = queue.getSrcPortId();
					direction = "MARS_TASK_QUEUE_MPU_TO_HOST";
				} else if (queue.getDst().equals(task.getName())) {
					portName = queue.getDstPortName();
					portId = queue.getDstPortId();
					direction = "MARS_TASK_QUEUE_HOST_TO_MPU";
				}

				content += "\t{\n";
				for (int i = 0; i < task.getProc().size(); i++) {
					if (queue.getSampleType().isEmpty()) {
						content += "\t\t{0x0, " + queue.getIndex() + ", " + direction + ", \"" + portName + "\", "
								+ portId + ", " + queue.getSampleSize() + ", " + queue.getSize() + " / "
								+ queue.getSampleSize() + "},\n";
					} else {
						content += "\t\t{0x0, " + queue.getIndex() + ", " + direction + ", \"" + portName + "\", "
								+ portId + ", sizeof(" + queue.getSampleType() + "), " + queue.getSize() + " / "
								+ queue.getSampleSize() + "},\n";
					}

				}
				content += "\t},\n";
			}
			index++;
		}
		content += "};\n\n";

		content += "#elif defined(__SPU__)\n\n";
		content += "con_task_function_t con_task_function = { FUNC_INIT, FUNC_GO, FUNC_WRAPUP };\n\n";
		content += "static uint64_t __attribute__((aligned(32))) con_channel_event_flag_spe = {0x0};\n\n";
		content += "static con_channel_info_entry __attribute__((aligned(32))) con_channel_info_spe[] = {\n";

		content += "\t{ 0x0, " + chId + ", MARS_TASK_QUEUE_HOST_TO_MPU, 0, sizeof(CONTROL_SEND_PACKET), 1},\n";
		content += "\t{ 0x0, " + chId + ", MARS_TASK_QUEUE_MPU_TO_HOST, 1, sizeof(CONTROL_SEND_PACKET), 1},\n";
		content += "};\n\n";

		content += "static cic_channel_info_entry __attribute__((aligned(32))) cic_channel_info_spe[] = {";
		index = 0;
		while (index < mQueue.size()) {
			Queue queue = mQueue.get(index);

			if (queue.getSrc().equals(task.getName()) || queue.getDst().equals(task.getName())) {
				String portName = new String();
				String portId = new String();
				String direction = new String();
				if (queue.getSrc().equals(task.getName())) {
					portName = queue.getSrcPortName();
					portId = queue.getSrcPortId();
					direction = "MARS_TASK_QUEUE_MPU_TO_HOST";
				} else if (queue.getDst().equals(task.getName())) {
					portName = queue.getDstPortName();
					portId = queue.getDstPortId();
					direction = "MARS_TASK_QUEUE_HOST_TO_MPU";
				}

				if (queue.getSampleType().isEmpty()) {
					content += "\t\t{0x0, " + queue.getIndex() + ", " + direction + ", \"" + portName + "\", " + portId
							+ ", " + queue.getSampleSize() + ", " + queue.getSize() + " / " + queue.getSampleSize()
							+ "},\n";
				} else {
					content += "\t\t{0x0, " + queue.getIndex() + ", " + direction + ", \"" + portName + "\", " + portId
							+ ", sizeof(" + queue.getSampleType() + "), " + queue.getSize() + " / "
							+ queue.getSampleSize() + "},\n";
				}
			}
			index++;
		}
		content += "};\n\n";
		content += "#else\n\n    #error\n\n#endif\n\n";

		return content;
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

			outstream.write("CC=ppu-gcc\n".getBytes());
			outstream.write("LD=ppu-gcc\n".getBytes());

			mRootPath = mRootPath.replace("\\", "/");
			mRootPath = mRootPath.replace("C:", "/cygdrive/C");
			mRootPath = mRootPath.replace("D:", "/cygdrive/D");
			mRootPath = mRootPath.replace("E:", "/cygdrive/E");
			mRootPath = mRootPath.replace("F:", "/cygdrive/F");
			mRootPath = mRootPath.replace("G:", "/cygdrive/G");

			outstream.write("#CFLAGS=-Wall -O0 -g -Wall -DDISPLAY -DTHREAD_STYLE -funroll-loops -m32\n".getBytes());
			outstream.write("CFLAGS=-Wall -O2 -DDISPLAY -DTHREAD_STYLE -funroll-loops -m32\n".getBytes());
			outstream
					.write("LDFLAGS=-m32 ./ppu/lib/libmars_task.a ./ppu/lib/libmars_base.a -lm -lX11 -lspe2 -lpthread -Xlinker --warn-common"
							.getBytes());
			for (String ldflag : ldFlagList.values())
				outstream.write((" " + ldflag).getBytes());
			outstream.write("\n\n".getBytes());

			outstream.write("CC_SPU=spu-gcc\n".getBytes());
			;
			outstream.write("#CFLAGS_SPU=-O3 -Wall -DDISPLAY  -DTHREAD_STYLE -funroll-loops\n".getBytes());
			outstream.write("CFLAGS_SPU=-O3 -Wall -DDISPLAY -DTHREAD_STYLE\n".getBytes());
			outstream
					.write("LDFLAGS_SPU=-Wl,--section-start,.init=0x4000 ./spu/lib/libmars_task.a ./spu/lib/libmars_base.a -Wl,-N -Wl,-gc-sections\n"
							.getBytes());
			outstream.write("EMBEDSPU=ppu-embedspu".getBytes());
			outstream.write("\n\n".getBytes());

			outstream.write("all: proc\n\n".getBytes());

			outstream.write("proc: proc.o".getBytes());

			for (Task task : mTask.values()) {
				if (mTaskOnSPE.contains(task)) {
					if (!task.getIsSlaveTask()) {
						outstream.write((" CIC_wrapper_ppe_" + task.getName() + ".o").getBytes());
						outstream.write((" CIC_wrapper_spe_" + task.getName() + ".task_eo").getBytes());
					} else if (task.getIsSlaveTask()) {
						outstream.write((" CON_wrapper_ppe_" + task.getName() + ".o").getBytes());
						outstream.write((" CON_wrapper_spe_" + task.getName() + ".task_eo").getBytes());
					}
				} else
					outstream.write((" " + task.getName() + ".o").getBytes());
			}

			if (mAlgorithm.getLibraries() != null) {
				for (Library library : mLibrary.values()) {
					if (mLibraryOnSPE.contains(library)) {
						outstream.write((" LIB_wrapper_ppe_" + library.getName() + ".o").getBytes());
						outstream.write((" LIB_wrapper_spe_" + library.getName() + ".task_eo").getBytes());
					} else {
						outstream.write((" " + library.getName() + ".o").getBytes());
						for (String extraSource : library.getExtraSource())
							outstream.write((" " + extraSource.substring(0, extraSource.length() - 2)).getBytes());
					}
				}
			}

			for (String extraSource : extraSourceList.values())
				outstream.write((" " + extraSource + ".o").getBytes());

			for (String l_file : mL_FileList)
				outstream.write((" l_" + l_file + ".o").getBytes());

			for (String stub_file : mStubFileList)
				outstream.write((" " + stub_file + ".o").getBytes());
			outstream.write("\n".getBytes());

			outstream.write("\t$(LD) $^ -o proc $(LDFLAGS)\n\n".getBytes());

			outstream
					.write(("proc.o: CIC_proc.c CIC_port.h cic_portmap.h cic_tasks.h cic_channels.h CON_port.h con_taskmap.h con_channels.h")
							.getBytes());

			if (mAlgorithm.getLibraries() != null) {
				outstream.write((" lib_portmap.h LIB_port.h lib_channels.h lib_stubs.h\n").getBytes());
				outstream.write(("\t$(CC) $(CFLAGS) -I./ppu/include -c CIC_proc.c -o proc.o\n").getBytes());
				outstream.write(("\n").getBytes());
			} else {
				outstream.write(("\n").getBytes());
				outstream.write(("\t$(CC) $(CFLAGS) -I./ppu/include -c CIC_proc.c -o proc.o\n").getBytes());
				outstream.write(("\n").getBytes());
			}

			for (Task task : mTask.values()) {
				if (mTaskOnSPE.contains(task)) {
					if (!task.getIsSlaveTask()) {
						outstream.write(("CIC_wrapper_ppe_" + task.getName() + ".o: " + task.getName()
								+ ".info.c CIC_wrapper_ppe.c CIC_port.h cic_portmap.h").getBytes());
						for (String header : task.getExtraHeader())
							outstream.write((" " + header).getBytes());
						outstream.write(("\n").getBytes());
						outstream.write(("\t$(CC) $(CFLAGS) " + task.getCflag() + " -I./ppu/include -o CIC_wrapper_ppe_"
								+ task.getName() + ".o -c CIC_wrapper_ppe.c" + " -include " + task.getName()
								+ ".info.c\n").getBytes());
						outstream.write(("\n").getBytes());
						outstream.write(("CIC_wrapper_spe_" + task.getName() + ".task_eo: CIC_port_spe.c "
								+ task.getName() + ".c " + task.getCICFile() + " " + task.getName() + ".info.c "
								+ "CIC_wrapper_spe.c CIC_port.h cic_portmap.h").getBytes());
						outstream.write(("\n").getBytes());
						outstream.write(
								("\t$(CC_SPU) $(CFLAGS_SPU) " + task.getCflag() + " -I./spu/include -o CIC_wrapper_spe_"
										+ task.getName() + ".task_eo" + " -c CIC_port_spe.c -include " + task.getName()
										+ ".c -include " + task.getName() + ".info.c -include CIC_wrapper_spe.c")
												.getBytes());
						for (String source : task.getExtraSource())
							outstream.write((" -include " + source).getBytes());
						outstream.write(("\n").getBytes());
						outstream
								.write(("\t$(CC_SPU) $(CFLAGS_SPU) -I./spu/include -o CIC_wrapper_spe_" + task.getName()
										+ ".task" + " CIC_wrapper_spe_" + task.getName() + ".task_eo $(LDFLAGS_SPU)\n")
												.getBytes());
						outstream.write(
								("\t$(EMBEDSPU) -m32 CIC_wrapper_spe_" + task.getName() + "_prog CIC_wrapper_spe_"
										+ task.getName() + ".task CIC_wrapper_spe_" + task.getName() + ".task_eo\n")
												.getBytes());
						outstream.write(("\n").getBytes());
					} else if (task.getIsSlaveTask()) {
						outstream.write(("CON_wrapper_ppe_" + task.getName() + ".o: " + task.getName()
								+ ".info.c CON_wrapper_ppe.c CON_port.h CIC_port.h cic_portmap.h").getBytes());
						for (String header : task.getExtraHeader())
							outstream.write((" " + header).getBytes());
						outstream.write(("\n").getBytes());
						outstream.write(("\t$(CC) $(CFLAGS) " + task.getCflag() + " -I./ppu/include -o CON_wrapper_ppe_"
								+ task.getName() + ".o -c CON_wrapper_ppe.c" + " -include " + task.getName()
								+ ".info.c\n").getBytes());
						outstream.write(("\n").getBytes());
						outstream.write(("CON_wrapper_spe_" + task.getName() + ".task_eo: CON_port_spe.c "
								+ task.getName() + ".c " + task.getCICFile() + " " + task.getName() + ".info.c "
								+ "CON_wrapper_spe.c CIC_port.h cic_portmap.h CON_port.h con_portmap.h").getBytes());
						for (String header : task.getExtraHeader())
							outstream.write((" " + header).getBytes());
						outstream.write(("\n").getBytes());
						outstream.write(
								("\t$(CC_SPU) $(CFLAGS_SPU) " + task.getCflag() + " -I./spu/include -o CON_wrapper_spe_"
										+ task.getName() + ".task_eo" + " -c CON_port_spe.c -include " + task.getName()
										+ ".c -include " + task.getName() + ".info.c -include CON_wrapper_spe.c")
												.getBytes());
						for (String source : task.getExtraSource())
							outstream.write((" -include " + source).getBytes());
						outstream.write(("\n").getBytes());
						outstream
								.write(("\t$(CC_SPU) $(CFLAGS_SPU) -I./spu/include -o CON_wrapper_spe_" + task.getName()
										+ ".task" + " CON_wrapper_spe_" + task.getName() + ".task_o $(LDFLAGS_SPU)\n")
												.getBytes());
						outstream.write(
								("\t$(EMBEDSPU) -m32 CON_wrapper_spe_" + task.getName() + "_prog CON_wrapper_spe_"
										+ task.getName() + ".task CON_wrapper_spe_" + task.getName() + ".task_eo\n")
												.getBytes());
						outstream.write(("\n").getBytes());
					}
				} else {
					outstream.write(
							(task.getName() + ".o: " + task.getName() + ".c " + task.getCICFile() + " CIC_port.h")
									.getBytes());
					for (String header : task.getExtraHeader())
						outstream.write((" " + header).getBytes());
					outstream.write(("\n").getBytes());
					outstream.write(("\t$(CC) $(CFLAGS) " + task.getCflag() + " -o " + task.getName() + ".o -c "
							+ task.getName() + ".c\n\n").getBytes());
				}
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
				Library libLibLibrary = null;
				for (Library library : mLibrary.values()) {
					List<Library> libLibList = new ArrayList<Library>();
					for (LibraryLibraryConnectionType libLib : mAlgorithm.getLibraryConnections()
							.getLibraryLibraryConnection()) {
						if (libLib.getMasterLibrary().equals(library.getName())) {
							for (Library lib : mLibrary.values())
								if (libLib.getSlaveLibrary().equals(lib.getName()))
									libLibLibrary = lib;
							libLibList.add(libLibLibrary);
						}
					}

					if (mLibraryOnSPE.contains(library)) {
						outstream.write(("LIB_wrapper_ppe_" + library.getName() + ".o: " + library.getName()
								+ ".info.c " + library.getName()
								+ "_data_structure.h LIB_wrapper_ppe.c LIB_port.h lib_portmap.h\n").getBytes());
						outstream.write(("\t$(CC) $(CFLAGS) -I./ppu/include -o LIB_wrapper_ppe_" + library.getName()
								+ ".o -c LIB_wrapper_ppe.c" + " -include " + library.getName()
								+ "_data_structure.h -include " + library.getName() + ".info.c\n").getBytes());
						outstream.write(("\n").getBytes());
						outstream.write(("LIB_wrapper_spe_" + library.getName() + ".task_eo: stub_" + library.getName()
								+ ".c " + library.getName() + ".info.c " + library.getName() + "_data_structure.h "
								+ library.getName() + ".c " + library.getName() + ".h " + library.getFile()
								+ " LIB_port_spe.c LIB_wrapper_spe.c LIB_port.h lib_portmap.h").getBytes());
						for (String header : library.getExtraHeader())
							outstream.write((" " + header).getBytes());
						for (String source : library.getExtraSource())
							outstream.write((" " + source).getBytes());
						outstream.write(("\n").getBytes());
						for (String source : library.getExtraSource())
							outstream.write(("\t$(CC_SPU) $(CFLAGS_SPU) -c " + library.getName() + " -o "
									+ library.getName() + ".o\n").getBytes());
						outstream.write(("\t$(CC_SPU) $(CFLAGS_SPU) -I./spu/include -o LIB_wrapper_spe_"
								+ library.getName() + ".task_o" + " -c LIB_port_spe.c -include " + library.getName()
								+ "_data_structure.h -include stub_" + library.getName() + ".c -include "
								+ library.getName() + ".info.c -include LIB_wrapper_spe.c -include " + library.getName()
								+ ".c").getBytes());

						for (Library libLib : libLibList)
							outstream.write((" -include l_" + libLib.getName()).getBytes());
						outstream.write(("\n").getBytes());

						outstream.write(
								("\t$(CC_SPU) $(CFLAGS_SPU) -I./spu/include -o LIB_wrapper_spe_" + library.getName()
										+ ".task" + " LIB_wrapper_spe_" + library.getName() + ".task_o").getBytes());
						for (String extraSource : library.getExtraSource())
							outstream.write((" " + extraSource.substring(0, extraSource.length() - 2)).getBytes());
						outstream.write((" $(LDFLAGS_SPU)\n").getBytes());
						outstream.write(("\t$(EMBEDSPU) -m32 LIB_wrapper_spe_" + library.getName()
								+ "_prog LIB_wrapper_spe_" + library.getName() + ".task LIB_wrapper_spe_"
								+ library.getName() + ".task_eo\n").getBytes());
						outstream.write(("\n").getBytes());
					} else {
						int flag = 0;
						for (String stubFile : mStubFileList)
							if (stubFile.equals(library.getName()))
								flag = 1;
						if (flag == 1) {
							outstream.write(("stub_" + library.getName() + ".o: stub_" + library.getName() + ".c "
									+ library.getName() + "_data_structure.h LIB_port.h\n").getBytes());
							outstream.write(("\t$(CC) $(CFLAGS) -o stub_" + library.getName() + ".o -c stub_"
									+ library.getName() + ".c -include " + library.getName() + "_data_structure.h\n\n")
											.getBytes());
						}
						outstream.write((library.getName() + ".o: " + library.getName() + ".c " + library.getFile()
								+ " " + library.getHeader() + "\n").getBytes());
						outstream.write(
								("\t$(CC) $(CFLAGS) -o " + library.getName() + ".o -c " + library.getName() + ".c\n\n")
										.getBytes());

						for (String extraSource : library.getExtraSource()) {
							outstream.write((extraSource.substring(0, extraSource.length() - 2) + ".o: ").getBytes());
							for (String header : library.getExtraHeader())
								outstream.write((" " + header).getBytes());
							for (String source : library.getExtraSource())
								outstream.write((" " + source).getBytes());
							outstream.write(("\n").getBytes());
							outstream.write(("\t$(CC) $(CFLAGS) -c " + extraSource + ".c -o "
									+ extraSource.substring(0, extraSource.length() - 2) + ".o\n\n").getBytes());
							outstream.write(("\n").getBytes());
						}
					}
				}
				for (String l_fileList : mL_FileList) {
					outstream.write(("l_" + l_fileList + ".o: " + "l_" + l_fileList + ".c " + "l_" + l_fileList + ".h "
							+ l_fileList + "_data_structure.h\n").getBytes());
					outstream.write(("\t$(CC) $(CFLAGS)").getBytes());
					outstream.write((" -c " + "l_" + l_fileList + ".c -o " + "l_" + l_fileList + ".o " + "-include "
							+ "l_" + l_fileList + ".h " + "-include " + l_fileList + "_data_structure.h\n\n")
									.getBytes());
				}
			}

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
			outstream.write("\trm -f proc *.o *.task *.task_o *.task_eo".getBytes());
			outstream.write("\n\n".getBytes());

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public void generateTaskCode(String fileOut, Task task) {
		File f = new File(fileOut);
		FileOutputStream outstream;
		try {
			outstream = new FileOutputStream(fileOut);
			outstream.write(translateTaskCode(task).getBytes());
			outstream.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public String translateTaskCode(Task task) {
		String content = new String();

		content += "\n#define SYS_REQ(x, ...) SYS_REQ_##x(__VA_ARGS__)\n\n";

		if (!task.getIsSlaveTask())
			content += "\n// for CIC port api\n#include \"CIC_port.h\"\n\n";
		else if (task.getIsSlaveTask())
			content += "\n// for CON port api\n#include \"CON_port.h\"\n\n";

		content += "#define TASK_ID " + task.getIndex() + "\n";
		content += "#define TASK_CODE_BEGIN\n";
		content += "#define TASK_CODE_END\n";
		content += "#define TASK_INIT void " + task.getName() + "_init(int __task_id)\n";
		content += "#define TASK_GO void " + task.getName() + "_go()\n";
		content += "#define TASK_WRAPUP void " + task.getName() + "_wrapup()\n";

		if (mAlgorithm.getLibraries() != null) {
			content += "#define LIBCALL(x, ...) LIBCALL_##x(__VA_ARGS__)\n";
			for (TaskLibraryConnectionType taskLibCon : mAlgorithm.getLibraryConnections().getTaskLibraryConnection()) {
				if (taskLibCon.getMasterTask().equals(task.getName())) {
					content += "\n#include \"" + taskLibCon.getSlaveLibrary() + ".h\"\n";
					content += "#define LIBCALL_" + taskLibCon.getMasterPort() + "(f, ...) l_"
							+ taskLibCon.getSlaveLibrary() + "_##f(__VA_ARGS__)\n";
				}
			}
		}

		content += "#define STATIC static\n";
		content += "\n#include \"" + task.getCICFile() + "\"\n";

		return content;
	}

	@Override
	public int generateCodeWithComm(String mTarget, String mTranslatorPath, String mOutputPath, String mRootPath,
			Map<Integer, Processor> mProcessor, List<Communication> mCommunication, Map<String, Task> mTask,
			Map<Integer, Queue> mQueue, Map<String, Library> mLibrary, Map<String, Library> mGlobalLibrary,
			int mGlobalPeriod, String mGlbalPeriodMetric, String mCICXMLFile, String language,
			CICAlgorithmType mAlgorithm, CICControlType mControl, CICScheduleType mSchedule, CICGPUSetupType mGpusetup,
			CICMappingType mMapping, Map<Integer, List<Task>> mConnectedTaskGraph,
			Map<Integer, List<List<Task>>> mConnectedSDFTaskSet, Map<String, Task> mVTask, Map<String, Task> mPVtask,
			String mRuntimeExecutionPolicy, String codeGenerationStyle) throws FileNotFoundException {
		// TODO Auto-generated method stub
		return 0;
	}

}
