package Translators;

import java.io.*;
import java.util.*;

import javax.swing.JOptionPane;

import CommonLibraries.*;
import InnerDataStructures.*;
import InnerDataStructures.Library;
import InnerDataStructures.Queue;
import hopes.cic.exception.CICXMLException;
import hopes.cic.xml.*;

public class CICMulticoreTranslator implements CICTargetCodeTranslator {
	private String mTarget;
	private String mTranslatorPath;
	private String mOutputPath;
	private String mRootPath;
	private String mCICXMLFile;
	private int mFuncSimPeriod;
	private String mFuncSimPeriodMetric;
	private String mGraphType;
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
	private CICGPUSetupType mGpusetup;
	private CICMappingType mMapping;

	@Override
	public int generateCode(String target, String translatorPath, String outputPath, String rootPath,
			Map<Integer, Processor> processor, Map<String, Task> task, Map<Integer, Queue> queue,
			Map<String, Library> library, Map<String, Library> globalLibrary, int funcSimPeriod,
			String funcSimPeriodMetric, String cicxmlfile, String language, CICAlgorithmType algorithm,
			CICControlType control, CICScheduleType schedule, CICGPUSetupType gpusetup, CICMappingType mapping,
			Map<Integer, List<Task>> connectedtaskgraph, Map<Integer, List<List<Task>>> connectedsdftaskset,
			Map<String, Task> vtask, Map<String, Task> pvtask, String graphType, String runtimeExecutionPolicy,
			String codeGenerationStyle) throws FileNotFoundException {
		mTarget = target;
		mTranslatorPath = translatorPath;
		mOutputPath = outputPath;
		mRootPath = rootPath;
		mCICXMLFile = cicxmlfile;
		mFuncSimPeriod = funcSimPeriod;
		mFuncSimPeriodMetric = funcSimPeriodMetric;
		mGraphType = graphType;
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
		mGpusetup = gpusetup;
		mMapping = mapping;

		Util.copyFile(mOutputPath + "target_task_model.h",
				mTranslatorPath + "templates/common/task_model/pthread.template");
		Util.copyFile(mOutputPath + "target_system_model.h",
				mTranslatorPath + "templates/common/system_model/general_linux.template");
		Util.copyFile(mOutputPath + "includes.h",
				mTranslatorPath + "templates/common/common_template/includes.h.linux");

		String fileOut = null;
		String templateFile = "";

		String srcExtension = "";
		if (mLanguage.equals("c++"))
			srcExtension = ".cpp";
		else
			srcExtension = ".c";

		// generate task_name.c (include task_name.cic)
		for (Task t : mTask.values()) {
			if (t.getHasSubgraph().equalsIgnoreCase("Yes") && t.getHasMTM() == true) {
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
			if (t.getHasMTM() == true) {
				if (mGraphType.equals("ProcessNetwork"))
				{
					templateFile = mTranslatorPath + "templates/common/mtm_template/thread_per_processor.template";
					CommonLibraries.CIC.generateMTMFile(mOutputPath, templateFile, t, mAlgorithm, mTask, mPVTask, mQueue,
							mRuntimeExecutionPolicy, mCodeGenerationStyle);
				}
				else 
				{
					if (mCodeGenerationStyle.equals(HopesInterface.CodeGenerationPolicy_Thread)) {
						templateFile = mTranslatorPath + "templates/common/mtm_template/thread_per_task.template";
						CommonLibraries.CIC.generateMTMFile(mOutputPath, templateFile, t, mAlgorithm, mTask, mPVTask,
								mQueue, mRuntimeExecutionPolicy, mCodeGenerationStyle);
					} else if (mCodeGenerationStyle.equals(HopesInterface.CodeGenerationPolicy_FunctionCall))
					{
						templateFile = mTranslatorPath + "templates/common/mtm_template/thread_per_processor.template";
						CommonLibraries.CIC.generateMTMFile(mOutputPath, templateFile, t, mAlgorithm, mTask, mPVTask,
								mQueue, mRuntimeExecutionPolicy, mCodeGenerationStyle);
					}
				}				
			}
		}
		
		//generate mtm files when this graph is SDF
		for (Task vt : mVTask.values()) {
			templateFile = mTranslatorPath + "templates/common/mtm_template/thread_per_processor.template";
			CommonLibraries.CIC.generateMTMFile(mOutputPath, templateFile, vt, mAlgorithm, mTask, mPVTask, mQueue,
					mRuntimeExecutionPolicy, mCodeGenerationStyle);
			
			fileOut = mOutputPath + vt.getName() + srcExtension;
			templateFile = mTranslatorPath + "templates/common/common_template/task_mtm_code_template.c";
			CommonLibraries.CIC.generateTaskMTMCode(fileOut, templateFile, vt, mAlgorithm);
		}

		// generate library files
		if (mLibrary != null) {
			for (Library l : mLibrary.values())
				CommonLibraries.Library.generateLibraryCode(mOutputPath, l, mAlgorithm);
		}

		// generate proc.c file
		fileOut = mOutputPath + "proc" + srcExtension;
		templateFile = mTranslatorPath + "templates/common/common_template/proc.c.template";

		generateProcCode(fileOut, templateFile);

		// generate Makefile
		fileOut = mOutputPath + "Makefile";
		generateMakefile(fileOut);

		return 0;
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

			if (mLanguage.equals("c++"))
				content = "#define CPP_CODE_GENERATION 1\\n\\n" + content;

			outstream.write(translateProcCode(content).getBytes());
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public String generateTaskToCoreMap() {
		String code = "";

		int max_proc_num = 0;
		int max_mode_num = 0;
		int max_parallel_num = 0;
		for (Task task : mTask.values()) {
			if (task.getProc().size() > max_proc_num)
				max_proc_num = task.getProc().size();
			for (String procNum : task.getProc().keySet()) {
				Map<String, List<Integer>> coreMap = task.getProc().get(procNum);
				if (coreMap.size() > max_mode_num)
					max_mode_num = coreMap.size();

				for (List<Integer> procList : coreMap.values()) {
					if (procList.size() > max_parallel_num)
						max_parallel_num = procList.size();
				}
			}
		}

		code += "#define MAX_SCHED_NUM " + max_proc_num + "\n";
		code += "#define MAX_MODE_NUM " + max_mode_num + "\n";
		code += "#define MAX_PARALLEL_NUM " + max_parallel_num + "\n\n";

		code += "CIC_TYPEDEF CIC_T_STRUCT{\n";
		code += "\tCIC_T_INT task_id;\n";
		code += "\tCIC_T_INT proc_num_list[MAX_SCHED_NUM];\n";
		code += "\tCIC_T_CHAR* mode_list[MAX_MODE_NUM];\n";
		code += "\tCIC_T_BOOL is_external_dp;\n";
		code += "\tCIC_T_BARRIER barrier;\n";
		code += "\tCIC_T_INT thread_num[MAX_SCHED_NUM];\n";
		code += "\tCIC_T_THREAD threads[MAX_PARALLEL_NUM];\n";
		code += "\tCIC_T_INT call_count[MAX_SCHED_NUM][MAX_MODE_NUM][MAX_PARALLEL_NUM];\n";
		code += "\tCIC_T_INT core_map[MAX_SCHED_NUM][MAX_MODE_NUM][MAX_PARALLEL_NUM];\n";
		code += "\tCIC_T_INT schedule_id;\n";
		code += "\tCIC_T_INT throughput_constraint[MAX_SCHED_NUM];\n";
		code += "}CIC_UT_TASK_TO_CORE_MAP;\n\n";

		code += "CIC_UT_TASK_TO_CORE_MAP task_to_core_map[] = {\n";

		int index=0;
		while (index < mTask.size()) {
			Task task = null;
			for (Task t : mTask.values()) {
				if (Integer.parseInt(t.getIndex()) == index) {
					task = t;
					break;
				}
			}
						
			code += "\t{" + task.getIndex() + ", {";
			for (String pn : task.getProc().keySet()) {
				String procNum = "";
				if (pn.equals("Default"))
					procNum = "0";
				else
					procNum = pn;
				code += procNum + ", ";
			}
			
			code += "}, {";
			for (String mode : task.getProc().get(task.getProc().keySet().iterator().next()).keySet()) {
				code += "\"" + mode + "\", ";
			}
			code += "}, ";
		
			int max_thread_num = 1;
			for (String procNum : task.getProc().keySet()) {
				Map<String, List<Integer>> coreMap = task.getProc().get(procNum);
				for (List<Integer> procList : coreMap.values()) {
					int size = procList.size();
					if(max_thread_num < size)	max_thread_num = size;
				}
			}
			
			String is_external_dp = "CIC_V_FALSE";
			if(max_thread_num > 1)	is_external_dp = "CIC_V_TRUE";
			
			code += is_external_dp + ", {0, }, {" + max_thread_num + ", }, {0, }, {";
			
			if(task.getCallCount() == null){
				code += "{{0, }, }, ";
			}
			else{
				for (String procNum : task.getCallCount().keySet()) {
					Map<String, List<Integer>> callMap = task.getCallCount().get(procNum);
					code += "{";
					for (List<Integer> callList : callMap.values()) {
						code += "{";
						if (callList.size() == 0) {
							code += 0 + ", ";
						} else {
							for (int proc : callList) {
								code += proc + ", ";
							}
						}
						code += "}, ";
					}
					code += "}, ";
				}
			}
			
			code += "}, {";
			
			for (String procNum : task.getProc().keySet()) {
				Map<String, List<Integer>> coreMap = task.getProc().get(procNum);
				code += "{";
				for (List<Integer> procList : coreMap.values()) {
					code += "{";
					if (procList.size() == 0) {
						code += -1 + ", ";
					} else {
						for (int proc : procList) {
							code += proc + ", ";
						}
					}
					code += "}, ";
				}
				code += "}, ";
			}

			code += "}, "; 
			
			//throughput_constraint
			String mode;
			if(mTask.get(task.getParentTask()) == null) //Flatten graph
				mode = "Default";
			else if(mTask.get(task.getParentTask()).getMTM().getModes() == null)
				mode = "Default";				
			else
				mode = mTask.get(task.getParentTask()).getMTM().getModes().get(0);
			String schedulePath = mOutputPath + "/convertedSDF3xml/";
			ArrayList<File> schedFileList = new ArrayList<File>();
			File file = new File(schedulePath);
			File[] fileList = file.listFiles();
			for (File f : fileList) {
				if (f.getName().contains(task.getParentTask() + "_" + mode + "_")
						&& f.getName().endsWith("_schedule.xml")) {
					schedFileList.add(f);
				}
			}
			if (schedFileList.size() <= 0) {
				code += "0, {0, ";
			}
			else
			{
				String temp = schedFileList.get(0).getName().replace(task.getParentTask() + "_" + mode + "_", "");
				temp = temp.replace("_schedule.xml", "");
				String temp2[] = temp.split("_");
				int sched_id = Integer.parseInt(temp2[0]);
				code += sched_id +", {";
				for (int f_i = 0; f_i < schedFileList.size(); f_i++) {
					temp = schedFileList.get(f_i).getName().replace(task.getParentTask() + "_" + mode + "_", "");
					temp = temp.replace("_schedule.xml", "");
					String temp3[] = temp.split("_");
					int constraint = Integer.parseInt(temp3[1]);
					code += constraint +", ";
				}
			}	
			
			code += "}},\n";
			index++;
		}
		
		int v_index = 0; 
		while (v_index < mVTask.size()) {
			Task task = null;
			for (Task t : mVTask.values()) {
				if (Integer.parseInt(t.getIndex()) == index) {
					task = t;
					break;
				}
			}
						
			code += "\t{" + task.getIndex() + ", {0, }, {\"Default\", }, ";
			
			code += "CIC_V_FALSE" + ", {0, }, {1, }, {0, }, {";
			
			if(task.getCallCount() == null){
				code += "{{0, }, }, ";
			}
			else{
				for (String procNum : task.getCallCount().keySet()) {
					Map<String, List<Integer>> callMap = task.getCallCount().get(procNum);
					code += "{";
					for (List<Integer> callList : callMap.values()) {
						code += "{";
						if (callList.size() == 0) {
							code += 0 + ", ";
						} else {
							for (int proc : callList) {
								code += proc + ", ";
							}
						}
						code += "}, ";
					}
					code += "}, ";
				}
			}
			
			code += "}, {{{-1, }, }, }, ";

			//throughput_constraint
			String mode = "Default";
			String schedulePath = mOutputPath + "/convertedSDF3xml/";
			ArrayList<File> schedFileList = new ArrayList<File>();
			File file = new File(schedulePath);
			File[] fileList = file.listFiles();
			for (File f : fileList) {
				if (f.getName().contains(task.getName() + "_" + mode + "_")
						&& f.getName().endsWith("_schedule.xml")) {
					schedFileList.add(f);
				}
			}
			if (schedFileList.size() <= 0) {
				code += "0, {0, ";
			}
			else
			{
				String temp = schedFileList.get(0).getName().replace(task.getName() + "_" + mode + "_", "");
				temp = temp.replace("_schedule.xml", "");
				String temp2[] = temp.split("_");
				int sched_id = Integer.parseInt(temp2[0]);
				code += sched_id +", {0, ";
				
			}	
			
			code += "}},\n";
			index++;
			v_index++;
		}
		code += "};\n";

		return code;
	}

	public String generateVirtualTaskToCoreMap() {
		String code = "";

		code += "\nCIC_TYPEDEF CIC_T_STRUCT{\n" + "\tCIC_T_INT virtual_task_id; \n" + "\tCIC_T_INT processor_id; \n"
				+ "}CIC_UT_VIRTUAL_TASK_TO_CORE_MAP;  \n";

		code += "\nCIC_UT_VIRTUAL_TASK_TO_CORE_MAP virtual_task_to_core_map[] = {\n";

		for (Task task : mPVTask.values()) {
			String[] result = task.getName().split("_");
			String processor_id = result[result.length - 1];
			code += "\t{" + task.getIndex()/* task_id */ + ", " + processor_id + "},\n";
		}
		code += "};\n\n";

		return code;

	}

	public String generateTaskToWCET() {
		String code = "";

		code += "\nCIC_TYPEDEF CIC_T_STRUCT{\n" + "\tCIC_T_INT task_id; \n" + "\tCIC_T_CHAR* mode_name; \n"
				+ "\tCIC_T_INT worst_case_execution_time; \n" + "}CIC_UT_TASK_TO_WCET; \n";

		code += "\nCIC_UT_TASK_TO_WCET task_to_wcet[] = {\n";

		int taskIndex = 0;
		for (Task task : mTask.values()) {
			// unit : us, cycle, ms -> we only support 'us' , 'ms', not cycle!
			if (task.getHasSubgraph().equalsIgnoreCase("No")) {
				Map<String, Integer> time_per_mode = new HashMap<String, Integer>();
				Map<String, String> timeunit_per_mode = new HashMap<String, String>();
				time_per_mode = task.getExecutionTimeValue();
				timeunit_per_mode = task.getExecutionTimeMetric();

				// change unit to 'us'
				Iterator<String> modes = time_per_mode.keySet().iterator();
				Iterator<String> modes2 = timeunit_per_mode.keySet().iterator();
				while (modes.hasNext() && modes2.hasNext()) {
					String mode = modes.next();
					String mode2 = modes2.next();

					int time = 0;
					if (mode.equals(mode2)) {
						if (timeunit_per_mode.get(mode2).equals("us"))
							time = time_per_mode.get(mode);
						else if (timeunit_per_mode.get(mode2).equals("ms"))
							time = time_per_mode.get(mode) * 1000;

						code += "\t{" + taskIndex + ", \"" + mode + "\", " + time + "},\n";
					}
				}
			}
			taskIndex++;
		}
		code += "};\n\n";

		return code;
	}

	public String generateTaskToPriority() {
		//now, we doesn't support SADF. 
		//To support SADF, we need to insert mode information, because there are task_priorities per mode => We need to fix it! 
		String code = "";

		code += "\nCIC_TYPEDEF CIC_T_STRUCT{\n" + "\tCIC_T_INT task_id; \n" + "\tCIC_T_INT schedule_id; \n" 
				+"\tCIC_T_INT processor_id; \n" + "\tCIC_T_CHAR* mode_name; \n" + "\tCIC_T_INT task_priority; \n" 
				+ "}CIC_UT_TASK_TO_PRIORITY;  \n";

		code += "\nCIC_UT_TASK_TO_PRIORITY task_to_priority[] = {\n";

		String outPath = mOutputPath + "/convertedSDF3xml/";
		code += generateTaskPriorityDefineCode(outPath, mTask);

		code += "};\n\n";

		return code;
	}

	public String generateTaskPriorityDefineCode(String outputPath, Map<String, Task> mTask) {
		//now, we doesn't support SADF. 
		//To support SADF, we need to insert mode information, because there are task_priorities per mode => We need to fix it! 

		String taskPriorityDefineCode = "";
		CICScheduleTypeLoader scheduleLoader = new CICScheduleTypeLoader();
		CICScheduleType mSchedule;

		List<Task> parentTaskList = new ArrayList<Task>();
		List<Task> taskList = new ArrayList<Task>();
		Task parentTask = null;
		Task task = null;

		for (Task ta : mTask.values()) {
			for (Task t : mTask.values()) {
				if (t.getName().equals(ta.getParentTask()) && !(t.equals(ta))) {
					if (!parentTaskList.contains(t) && !taskList.contains(ta)) {
						parentTask = t;
						task = ta;
						parentTaskList.add(parentTask);
						taskList.add(task);
					}
					break;
				}
			}
		}
		for (Task ta : mTask.values()) {
			for (Task vt : mVTask.values()) {
				if (vt.getName().equals(ta.getParentTask())) {
					if (!parentTaskList.contains(vt) && !taskList.contains(ta)) {
						parentTask = vt;
						task = ta;
						parentTaskList.add(parentTask);
						taskList.add(task);
					}
					break;
				}
			}
		}

		for (int index = 0; index < parentTaskList.size(); index++) {
			parentTask = parentTaskList.get(index);
			task = taskList.get(index);

			List<String> modeList = new ArrayList<String>();
			if (parentTask.getMTM() != null)
				modeList = parentTask.getMTM().getModes();
			else
				modeList.add("Default");

			for (String mode : modeList) {
				ArrayList<String> history = new ArrayList<String>();
				try {
					ArrayList<File> schedFileList = new ArrayList<File>();
					File file = new File(outputPath);
					File[] fileList = file.listFiles();
					for (File f : fileList) {
						if (f.getName().contains(task.getParentTask() + "_" + mode)
								&& f.getName().endsWith("_schedule.xml")) {
							schedFileList.add(f);
						}
					}
					if (schedFileList.size() <= 0) {
						JOptionPane.showMessageDialog(null, "You should execute 'Analysis' before build!");
						System.exit(-1);
					}

					for (int f_i = 0; f_i < schedFileList.size(); f_i++) {
						mSchedule = scheduleLoader.loadResource(schedFileList.get(f_i).getAbsolutePath());

						String temp = schedFileList.get(f_i).getName().replace(task.getParentTask() + "_" + mode + "_", "");
						temp = temp.replace("_schedule.xml", "");
						String temp2[] = temp.split("_");
						int sched_id = Integer.parseInt(temp2[0]);
						
						TaskGroupsType taskGroups = mSchedule.getTaskGroups();
						List<TaskGroupForScheduleType> taskGroupList = taskGroups.getTaskGroup();
						for (int i = 0; i < taskGroupList.size(); i++) {
							List<ScheduleGroupType> scheduleGroup = taskGroupList.get(i).getScheduleGroup();
							for (int j = 0; j < scheduleGroup.size(); j++) {
								int proc_id = 0;
								for(Processor proc: mProcessor.values()){
									if(proc.getPoolName().equals(scheduleGroup.get(j).getPoolName()) 
											&& proc.getLocalIndex() == scheduleGroup.get(j).getLocalId().intValue()){
										proc_id = proc.getIndex();
										break;
									}
								}
								
								int taskPriority = 10;
								List<ScheduleElementType> schedules = scheduleGroup.get(j).getScheduleElement();
								for (int k = 0; k < schedules.size(); k++) {
									ScheduleElementType schedule = schedules.get(k);
									String taskName = schedule.getTask().getName();
									String taskId = "0";

									for (Task t : mTask.values()) {
										if (t.getName().equals(taskName)) {
											taskId = t.getIndex();
											break;
										}
									}
									taskPriorityDefineCode += "{" + taskId + ", " + sched_id + ", " + proc_id + ", \"" + mode + "\", "
											+ taskPriority + "},\n";
									taskPriority++;
								}
							}

						}
					}
				} catch (CICXMLException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}

		return taskPriorityDefineCode;
	}

	public String generateScheduleOrderFromScheduleFile() {
		//now, we doesn't support SADF. 
		//To support SADF, we need to insert mode information, because there are task_priorities per mode => We need to fix it! 
		String code = "";
		String outPath = mOutputPath + "/convertedSDF3xml/";

		code += translateScheduleOrderFromScheduleFile(outPath, mTask);

		return code;
	}

	private String translateScheduleOrderFromScheduleFile(String outputPath, Map<String, Task> mTask) {
		//now, we doesn't support SADF. 
		//To support SADF, we need to insert mode information, because there are task_priorities per mode => We need to fix it! 

		String taskPriorityDefineCode = "";
		CICScheduleTypeLoader scheduleLoader = new CICScheduleTypeLoader();
		CICScheduleType mSchedule;

		List<Task> parentTaskList = new ArrayList<Task>();
		List<Task> taskList = new ArrayList<Task>();
		Task parentTask = null;
		Task task = null;

		for (Task ta : mTask.values()) {
			for (Task t : mTask.values()) {
				if (t.getName().equals(ta.getParentTask())) {
					if (!parentTaskList.contains(t) && !taskList.contains(ta)) {
						parentTask = t;
						task = ta;
						parentTaskList.add(parentTask);
						taskList.add(task);
					}
					break;
				}
			}
		}
		for (Task ta : mTask.values()) {
			for (Task vt : mVTask.values()) {
				if (vt.getName().equals(ta.getParentTask())) {
					if (!parentTaskList.contains(vt) && !taskList.contains(ta)) {
						parentTask = vt;
						task = ta;
						parentTaskList.add(parentTask);
						taskList.add(task);
					}
					break;
				}
			}
		}
		
		int max_schedule_num_per_processor = 0;

		for (int index = 0; index < parentTaskList.size(); index++) {
			parentTask = parentTaskList.get(index);
			task = taskList.get(index);

			List<String> modeList = new ArrayList<String>();
			if (parentTask.getMTM() != null)
				modeList = parentTask.getMTM().getModes();
			else
				modeList.add("Default"); 

			for (String mode : modeList) {
				try {
					ArrayList<File> schedFileList = new ArrayList<File>();
					File file = new File(outputPath);
					File[] fileList = file.listFiles();
					for (File f : fileList) {
						if (f.getName().contains(task.getParentTask() + "_" + mode)
								&& f.getName().endsWith("_schedule.xml")) {
							schedFileList.add(f);
						}
					}
					if (schedFileList.size() <= 0) {
						JOptionPane.showMessageDialog(null, "You should execute 'Analysis' before build!");
						System.exit(-1);
					}

					for (int f_i = 0; f_i < schedFileList.size(); f_i++) {
						// we assume that we save only the first schedule 
						mSchedule = scheduleLoader.loadResource(schedFileList.get(0).getAbsolutePath());

						TaskGroupsType taskGroups = mSchedule.getTaskGroups();
						List<TaskGroupForScheduleType> taskGroupList = taskGroups.getTaskGroup();
						for (int i = 0; i < taskGroupList.size(); i++) {
							List<ScheduleGroupType> scheduleGroup = taskGroupList.get(i).getScheduleGroup();
							for (int j = 0; j < scheduleGroup.size(); j++) {
								List<ScheduleElementType> schedules = scheduleGroup.get(j).getScheduleElement();

								// count the max_suchedule num per processor
								if (max_schedule_num_per_processor < schedules.size())
									max_schedule_num_per_processor = schedules.size();
							}
						}
					}
				} catch (CICXMLException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}

		taskPriorityDefineCode += "#define MAX_SCHED_NUM_PER_PROC " + max_schedule_num_per_processor;

		taskPriorityDefineCode += "\nCIC_TYPEDEF CIC_T_STRUCT{\n" + "\tCIC_T_INT processor_id; \n"
				+ "\tCIC_T_INT parent_task_id; \n" + "\tCIC_VOLATILE CIC_T_INT execution_index; \n"
				+ "\tCIC_T_INT max_schedule_length; \n" + "\tCIC_T_INT task_execution_order[MAX_SCHED_NUM_PER_PROC]; \n"
				+ "}CIC_UT_SCHEDULE_RESULT_PER_PROC;  \n";

		taskPriorityDefineCode += "CIC_UT_SCHEDULE_RESULT_PER_PROC schedules[] = {\n";

		for (int index = 0; index < parentTaskList.size(); index++) {
			parentTask = parentTaskList.get(index);
			task = taskList.get(index);

			List<String> modeList = new ArrayList<String>();
			if (parentTask.getMTM() != null)
				modeList = parentTask.getMTM().getModes();
			else
				modeList.add("Default"); 

			for (String mode : modeList) {
				try {
					ArrayList<File> schedFileList = new ArrayList<File>();
					File file = new File(outputPath);
					File[] fileList = file.listFiles();
					for (File f : fileList) {
						if (f.getName().contains(task.getParentTask() + "_" + mode)
								&& f.getName().endsWith("_schedule.xml")) {
							schedFileList.add(f);
						}
					}
					if (schedFileList.size() <= 0) {
						JOptionPane.showMessageDialog(null, "You should execute 'Analysis' before build!");
						System.exit(-1);
					}

					for (int f_i = 0; f_i < schedFileList.size(); f_i++) {
						// we assume that we save only the first schedule 
						mSchedule = scheduleLoader.loadResource(schedFileList.get(0).getAbsolutePath());

						TaskGroupsType taskGroups = mSchedule.getTaskGroups();
						List<TaskGroupForScheduleType> taskGroupList = taskGroups.getTaskGroup();
						for (int i = 0; i < taskGroupList.size(); i++) {
							List<ScheduleGroupType> scheduleGroup = taskGroupList.get(i).getScheduleGroup();
							for (int j = 0; j < scheduleGroup.size(); j++) {
								List<ScheduleElementType> schedules = scheduleGroup.get(j).getScheduleElement();

								taskPriorityDefineCode += "\t{" + scheduleGroup.get(j).getLocalId() + ", "
										+ parentTask.getIndex() + ", 0, " + schedules.size() + ", {";

								for (int k = 0; k < schedules.size(); k++) {
									ScheduleElementType schedule = schedules.get(k);
									String taskName = schedule.getTask().getName();
									String taskId = "0";

									for (Task t : mTask.values()) {
										if (t.getName().equals(taskName)) {
											taskId = t.getIndex();
											break;
										}
									}
									taskPriorityDefineCode += taskId + ", ";
									// processor id, execution_index(0), max, {execution order}
								}

								if (schedules.size() < max_schedule_num_per_processor) {
									for (int temp = 0; temp < (max_schedule_num_per_processor
											- schedules.size()); temp++) {
										taskPriorityDefineCode += "-1, ";
									}
								}
								taskPriorityDefineCode += "}},\n";

							}

						}
					}
				} catch (CICXMLException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}

		taskPriorityDefineCode += "};\n\n";

		return taskPriorityDefineCode;
	}

	private String getTaskExecutionTemplateFile(String graphType, String runtimeExecutionPolicy, String codeGenerationStyle) {
		String templateFile = "";

		//[CODE_REVIEW]: hshong(4/21):need to check func.sim
		/*
		if (mRuntimeExecutionPolicy.equals("Single"))
			templateFile = mTranslatorPath
					+ "templates/common/task_execution/multi_thread_hybrid_thread_per_application.template";
		else
		*/ 
		if (graphType.equals("ProcessNetwork"))
			templateFile = mTranslatorPath
				+ "templates/common/task_execution/multi_thread_process_network.template";
		else
		{
			if (runtimeExecutionPolicy.equals(HopesInterface.RuntimeExecutionPolicy_FullyStatic)) {
				// assume that we supports only function call version to fully-static policy
				templateFile = mTranslatorPath
						+ "templates/common/task_execution/multi_thread_hybrid_fully_static_function_call.template";
			} else if (runtimeExecutionPolicy.equals(HopesInterface.RuntimeExecutionPolicy_SelfTimed)) {
				// assume that we supports only function call version to self-timed policy
				templateFile = mTranslatorPath
						+ "templates/common/task_execution/multi_thread_hybrid_self_timed_function_call.template";
			} else if (runtimeExecutionPolicy.equals(HopesInterface.RuntimeExecutionPolicy_StaticAssign)) {
				// assume that we supports only thread version to static-assignement policy
				templateFile = mTranslatorPath
						+ "templates/common/task_execution/multi_thread_hybrid_static_assign_thread.template";
			} else if (runtimeExecutionPolicy.equals(HopesInterface.RuntimeExecutionPolicy_FullyDynamic)) {
				// assume that we supports only thread version to fully-dynamic policy
				templateFile = mTranslatorPath
						+ "templates/common/task_execution/multi_thread_hybrid_fully_dynamic_thread.template";
			}
		}		

		return templateFile;
	}

	public String generateTargetDependentCode() {
		String code = "";
		String templateFile = getTaskExecutionTemplateFile(mGraphType, mRuntimeExecutionPolicy, mCodeGenerationStyle);
		
		code += generateTaskToCoreMap();
		if (mRuntimeExecutionPolicy
				.equals(HopesInterface.RuntimeExecutionPolicy_FullyStatic)) {
			code += generateTaskToWCET();
			code += CommonLibraries.Util.getCodeFromTemplate(templateFile, "##WORST_CASE_EXECUTION_TIME");

			code += generateVirtualTaskToCoreMap();
			templateFile = mTranslatorPath + "templates/target/Multicore/target_dependent_code.template";
			code += CommonLibraries.Util.getCodeFromTemplate(templateFile, "##PROCESSOR_ID_FROM_VIRTUAL_TASK_ID");
			code += CommonLibraries.Util.getCodeFromTemplate(templateFile, "##GET_VIRTUAL_TASK_INDEX_FROM_THREAD");
		} else if (mRuntimeExecutionPolicy.equals(HopesInterface.RuntimeExecutionPolicy_SelfTimed)) {
			code += generateVirtualTaskToCoreMap();
			templateFile = mTranslatorPath + "templates/target/Multicore/target_dependent_code.template";
			code += CommonLibraries.Util.getCodeFromTemplate(templateFile, "##PROCESSOR_ID_FROM_VIRTUAL_TASK_ID");
			code += CommonLibraries.Util.getCodeFromTemplate(templateFile, "##GET_VIRTUAL_TASK_INDEX_FROM_THREAD"); 
		} else if (mRuntimeExecutionPolicy.equals(HopesInterface.RuntimeExecutionPolicy_StaticAssign)) {
			code += generateTaskToPriority();
			templateFile = mTranslatorPath + "templates/target/Multicore/target_dependent_code.template";
			code += CommonLibraries.Util.getCodeFromTemplate(templateFile, "##GET_TASK_PRIORITY_FROM_TASK_ID");
			code += CommonLibraries.Util.getCodeFromTemplate(templateFile, "##GET_VIRTUAL_TASK_INDEX_FROM_THREAD");
		} else if (mRuntimeExecutionPolicy.equals(HopesInterface.RuntimeExecutionPolicy_FullyDynamic)) {
			code += generateTaskToPriority();
			templateFile = mTranslatorPath + "templates/target/Multicore/target_dependent_code.template";
			code += CommonLibraries.Util.getCodeFromTemplate(templateFile, "##GET_TASK_PRIORITY_FROM_TASK_ID");
			code += CommonLibraries.Util.getCodeFromTemplate(templateFile, "##GET_VIRTUAL_TASK_INDEX_FROM_THREAD");
		} 

		templateFile = mTranslatorPath + "templates/target/Multicore/target_dependent_code.template";
		code += CommonLibraries.Util.getCodeFromTemplate(templateFile, "##GET_SCHEDULE_ID");
		code += CommonLibraries.Util.getCodeFromTemplate(templateFile, "##GET_PROCESSOR_ID");
		code += CommonLibraries.Util.getCodeFromTemplate(templateFile, "##GET_TASK_CALL_COUNT");

		return code;
	}
	
	public String generateStaticSchedulingCode()
	{
		String code = "";
		if (mRuntimeExecutionPolicy.equals("Single")) {
			String outPath = mOutputPath + "/convertedSDF3xml/";
			code = CommonLibraries.Schedule.generateSingleProcessorStaticScheduleCode(outPath, mTask,
					mVTask);
			code = code.replace("##SCHEDULE_CODE", code);
		} else if (mRuntimeExecutionPolicy
				.equals(HopesInterface.RuntimeExecutionPolicy_FullyStatic)) {
			String outPath = mOutputPath + "/convertedSDF3xml/";
			code = CommonLibraries.Schedule.generateMultiProcessorStaticScheduleCodeWithExecutionPolicy(
					outPath, mTask, mVTask, mPVTask, mRuntimeExecutionPolicy, mProcessor);
			code = code.replace("##SCHEDULE_CODE", code);
		} else if (mRuntimeExecutionPolicy.equals(HopesInterface.RuntimeExecutionPolicy_SelfTimed)) {
			String outPath = mOutputPath + "/convertedSDF3xml/";
			code = CommonLibraries.Schedule
					.generateMultiProcessorStaticScheduleCodeWithExecutionPolicy(outPath, mTask, mVTask, mPVTask,
							mRuntimeExecutionPolicy, mProcessor);
			code = code.replace("##SCHEDULE_CODE", code);
		} 
		return code;
	}

	public String translateProcCode(String mContent) {
		String code = mContent;
		String templateFile = "";

		// EXTERNAL_GLOBAL_HEADERS //
		String externalGlobalHeaders = "";
		if (mAlgorithm.getHeaders() != null) {
			for (String header : mAlgorithm.getHeaders().getHeaderFile())
				externalGlobalHeaders += "#include\"" + header + "\"\n";
		}
		code = code.replace("##EXTERNAL_GLOBAL_HEADERS", externalGlobalHeaders);
		////////////////////////////

		// LIB_INCLUDE //
		String libInclude = "";
		code = code.replace("##LIB_INCLUDES", libInclude);
		//////////////

		// CONN_INCLUDES //
		String conInclude = "";
		code = code.replace("##CONN_INCLUDES", conInclude);
		//////////////

		// EXTERN_TASK_FUNCTION_DECLARATION, TASK_ENTRIES,
		// EXTERN_MTM_FUNCTION_DECLARATION, MTM_ENTRIES,
		// VIRTUAL_TASK_ENTRIES //
		code = CommonLibraries.CIC.translateTaskDataStructure(code, mTask, mFuncSimPeriod, mFuncSimPeriodMetric,
				mRuntimeExecutionPolicy, mCodeGenerationStyle, mVTask, mPVTask);

		// CHANNEL_ENTRIES //
		code = CommonLibraries.CIC.translateChannelDataStructure(code, mQueue);

		// PORT_ENTRIES //
		code = CommonLibraries.CIC.translatePortmapDataStructure(code, mTask, mQueue);

		// CONNMAP_ENTRIES //
		String connMapEntries = "";
		code = code.replace("##CONNMAP_ENTRIES", connMapEntries);

		// CONTROL_GROUP_COUNT, CONTROL_CHANNEL_COUNT, PARAM_LIST_ENTRIES,
		// CONTROL_CHANNEL_LIST_ENTRIES //
		code = CommonLibraries.CIC.translateControlDataStructure(code, mTask, mControl);

		templateFile = getTaskExecutionTemplateFile(mGraphType, mRuntimeExecutionPolicy, mCodeGenerationStyle);

		// TASK_VARIABLE_DECLARATION // 
		String taskVariableDecl = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TASK_VARIABLE_DECLARATION");
		code = code.replace("##TASK_VARIABLE_DECLARATION", taskVariableDecl);
		//////////////////////////

		// DEBUG_CODE // 
		String debugCode = "";
		code = code.replace("##DEBUG_CODE_IMPLEMENTATION", debugCode);
		///////////////////////////

		// TARGET_DEPENDENT_IMPLEMENTATION //
		String targetDependentImpl = generateTargetDependentCode();
		code = code.replace("##TARGET_DEPENDENT_IMPLEMENTATION", targetDependentImpl);
		/////////////////////////////////////

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

		templateFile = mTranslatorPath + "templates/common/time_code/general_linux.template";
		// GET_CURRENT_TIME_BASE //
		String getCurrentTimeBase = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##GET_CURRENT_TIME_BASE");
		code = code.replace("##GET_CURRENT_TIME_BASE", getCurrentTimeBase);
		///////////////////////////

		// CHANGE_TIME_UNIT //  
		String changeTimeUnit = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##CHANGE_TIME_UNIT");
		code = code.replace("##CHANGE_TIME_UNIT", changeTimeUnit);
		///////////////////////////

		templateFile = mTranslatorPath + "templates/common/channel_manage/general_linux_multi_thread.template";

		// INIT_WRAPUP_TASK_CHANNELS //
		String initWrapupTaskChannels = CommonLibraries.Util.getCodeFromTemplate(templateFile,
				"##INIT_WRAPUP_TASK_CHANNELS");
		code = code.replace("##INIT_WRAPUP_TASK_CHANNELS", initWrapupTaskChannels);
		//////////////////////////

		// INIT_WRAPUP_LIBRARY_CHANNELS //
		String initWrapupLibChannels = "";
		code = code.replace("##INIT_WRAPUP_LIBRARY_CHANNELS", initWrapupLibChannels);
		////////////////////////////

		// INIT_WRAPUP_CHANNELS //
		String initWrapupChannels = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##INIT_WRAPUP_CHANNELS");
		code = code.replace("##INIT_WRAPUP_CHANNELS", initWrapupChannels);
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

		// READ_WRITE_LIB_PORT //
		String readWriteLibPort = "";
		code = code.replace("##READ_WRITE_LIB_PORT", readWriteLibPort);
		//////////////

		// EXTERN_LIBRARY_INIT_WRAPUP_FUNCTION_DECLARATION, LIB_INIT_FUNCTION
		// LIB_WRAPUP_FUNCTION //
		String externLibraryInitWrapupFunctionDecl = "";
		String libraryInit = "";
		String libraryWrapup = "";

		if (mLibrary != null) {
			for (Library library : mLibrary.values()) {
				externLibraryInitWrapupFunctionDecl += "CIC_EXTERN CIC_T_VOID l_" + library.getName()
						+ "_Init(CIC_T_VOID);\n";
				externLibraryInitWrapupFunctionDecl += "CIC_EXTERN CIC_T_VOID l_" + library.getName()
						+ "_Wrapup(CIC_T_VOID);\n\n";
			}

			libraryInit += "CIC_STATIC CIC_T_VOID InitLibraries(){\n";
			for (Library library : mLibrary.values())
				libraryInit += "\tl_" + library.getName() + "_init();\n";
			libraryInit += "}";

			libraryWrapup += "CIC_STATIC CIC_T_VOID WrapupLibraries(){\n";
			for (Library library : mLibrary.values())
				libraryWrapup += "\tl_" + library.getName() + "_wrapup();\n";
			libraryWrapup += "}";
		}

		code = code.replace("##EXTERN_LIBRARY_INIT_WRAPUP_FUNCTION_DECLARATION", externLibraryInitWrapupFunctionDecl);
		code = code.replace("##LIB_INIT_FUNCTION", libraryInit);
		code = code.replace("##LIB_WRAPUP_FUNCTION", libraryWrapup);
		/////////////////////

		// SCHEDULE_CODE //
		String staticScheduleCode = generateStaticSchedulingCode();
		code = code.replace("##SCHEDULE_CODE", staticScheduleCode);
		///////////////////
		
		templateFile = getTaskExecutionTemplateFile(mGraphType, mRuntimeExecutionPolicy, mCodeGenerationStyle);	
			
		// TASK_ROUTINE //
		String timeTaskRoutine = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TASK_ROUTINE");
		code = code.replace("##TASK_ROUTINE", timeTaskRoutine);
		//////////////////////////

		// SET_PROC //
		if (!mRuntimeExecutionPolicy.equals(HopesInterface.RuntimeExecutionPolicy_FullyDynamic))
		{
			String setProc = "\tschedule_id = GetScheduleIdFromTaskIndex(task_index);\n";
			setProc +=  "\tproc_id = GetProcessorId(task_id, schedule_id, mode_name, 0);\n";
			setProc += "\tcpu_set_t cpuset;\n"
					+ "\tCPU_ZERO(&cpuset);\n" + "\tCPU_SET(proc_id, &cpuset);\n"
					+ "\tpthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);\n";
			code = code.replace("##SET_PROC", setProc);
		}		
		//////////////
		
		if (!mGraphType.equals("ProcessNetwork"))
		{
			if (mRuntimeExecutionPolicy.equals(HopesInterface.RuntimeExecutionPolicy_FullyStatic)) {
				// SET_VIRTUAL_PROC //
				String setVirtualProc = "\tcpu_set_t cpuset;\n" + "\tCPU_ZERO(&cpuset);\n"
						+ "\tprocessor_id = GetProcessorIdFromVirtualTaskId(task_id);\n"
						+ "\tCPU_SET(processor_id, &cpuset);\n"
						+ "\tpthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);\n";
				code = code.replace("##SET_VIRTUAL_PROC", setVirtualProc);
				//////////////
			} else if (mRuntimeExecutionPolicy.equals(HopesInterface.RuntimeExecutionPolicy_SelfTimed)) {
				// SET_VIRTUAL_PROC //
				String setVirtualProc = "\tcpu_set_t cpuset;\n" + "\tCPU_ZERO(&cpuset);\n"
						+ "\tprocessor_id = GetProcessorIdFromVirtualTaskId(task_id);\n"
						+ "\tCPU_SET(processor_id, &cpuset);\n"
						+ "\tpthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);\n";
				code = code.replace("##SET_VIRTUAL_PROC", setVirtualProc);
				////////////// 
			} 
		}		

		// EXECUTE_TASKS //
		String executeTasks = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##EXECUTE_TASKS");
		code = code.replace("##EXECUTE_TASKS", executeTasks);
		//////////////////////////

		templateFile = mTranslatorPath + "templates/common/control/general_linux.template";
		// MTM_API // 
		boolean mtmFlag = false;
		for (Task t : mTask.values()) {
			if (t.getHasMTM() == true) {
				mtmFlag = true;
				break;
			}
		}
		if (mtmFlag) {
			String mtmApi = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##MTM_API");
			code = code.replace("##MTM_API", mtmApi);
		} else
			code = code.replace("##MTM_API", "");
		///////////////////////////

		templateFile = mTranslatorPath + "templates/common/control/general_linux.template";
		if (mControl.getControlTasks() != null) {
			// CONTROL_API //  
			String controlApi = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##CONTROL_API");
			code = code.replace("##CONTROL_API", controlApi);
			//////////////////////////

			templateFile = getTaskExecutionTemplateFile(mGraphType, mRuntimeExecutionPolicy, mCodeGenerationStyle);

			// CONTROL_END_TASK //
			String controlEndTask = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##CONTROL_END_TASK");
			code = code.replace("##CONTROL_END_TASK", controlEndTask);
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
			String controlSuspendTask = CommonLibraries.Util.getCodeFromTemplate(templateFile,
					"##CONTROL_SUSPEND_TASK");
			code = code.replace("##CONTROL_SUSPEND_TASK", controlSuspendTask);
			//////////////////////////
		} else {
			code = code.replace("##CONTROL_API", "");
			
			if(!mGraphType.equals("ProcessNetwork"))
			{
				templateFile = getTaskExecutionTemplateFile(mGraphType, mRuntimeExecutionPolicy, mCodeGenerationStyle);
				
				// CONTROL_END_TASK //
				String controlEndTask = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##CONTROL_END_TASK");
				code = code.replace("##CONTROL_END_TASK", controlEndTask);
			}
			else
			{
				code = code.replace("##CONTROL_END_TASK", "");
			}
			//////////////////////////
		}

		// INIT_SYSTEM_VARIABLES //
		String initSystemVariables = "";
		initSystemVariables += "\tCIC_F_MUTEX_INIT(&global_mutex);\n" + "\tCIC_F_COND_INIT(&global_cond);\n";

		code = code.replace("##INIT_SYSTEM_VARIABLES", initSystemVariables);
		//////////////////////////////////

		// WRAPUP_SYSTEM_VARIABLES //
		String wrapupSystemVariables = "";
		wrapupSystemVariables += "\tCIC_F_MUTEX_WRAPUP(&global_mutex);\n" + "\tCIC_F_COND_WRAPUP(&global_cond);\n";

		code = code.replace("##WRAPUP_SYSTEM_VARIABLES", wrapupSystemVariables);
		//////////////////////////////////

		// MAIN_FUNCTION //
		String mainFunc = "int main(int argc, char *argv[])";
		code = code.replace("##MAIN_FUNCTION", mainFunc);
		//////////////

		// TARGET_DEPENDENT_INIT_CALL //
		String targetDependentInit = "";
		targetDependentInit = "#if defined(WATCH_DEBUG) && (WATCH_DEBUG==1)\n" + "\tUpdateWatch();\n#endif\n"
				+ "#if defined(BREAK_DEBUG) && (BREAK_DEBUG==1)\n" + "\tUpdateBreak();\n#endif\n";
		
		code = code.replace("##TARGET_DEPENDENT_INIT_CALL", targetDependentInit);
		////////////////////////////////

		// TARGET_DEPENDENT_WRAPUP_CALL //
		String targetDependentWrapup = "";		

		targetDependentWrapup += "return EXIT_SUCCESS;";
		code = code.replace("##TARGET_DEPENDENT_WRAPUP_CALL", targetDependentWrapup);
		//////////////////////////////////

		// LIB_INIT, LIB_WRAPUP //
		String libInit = "";
		String libWrapup = "";
		if (mLibrary != null) {
			libInit = "InitLibraries();\n";
			libWrapup = "WrapupLibraries();\n";
		}
		code = code.replace("##LIB_INIT", libInit);
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
				outstream.write("CC=gcc\n".getBytes());
				outstream.write("LD=gcc\n".getBytes());
			}

			mRootPath = mRootPath.replace("\\", "/");
			mRootPath = mRootPath.replace("C:", "/cygdrive/C");
			mRootPath = mRootPath.replace("D:", "/cygdrive/D");
			mRootPath = mRootPath.replace("E:", "/cygdrive/E");
			mRootPath = mRootPath.replace("F:", "/cygdrive/F");
			mRootPath = mRootPath.replace("G:", "/cygdrive/G");

			outstream.write("#CFLAGS=-Wall -O0 -g -DDISPLAY -DTHREAD_STYLE\n".getBytes());
			outstream.write("CFLAGS=-Wall -O2 -DDISPLAY -DTHREAD_STYLE\n".getBytes());

			outstream.write("LDFLAGS=-lX11 -lpthread -lm -Xlinker --warn-common".getBytes());
			for (String ldflag : ldFlagList.values())
				outstream.write((" " + ldflag).getBytes());
			outstream.write("\n\n".getBytes());

			outstream.write("all: proc\n\n".getBytes());

			outstream.write("proc:".getBytes());
			
			for (Task vTask : mVTask.values())
				outstream.write((" " + vTask.getName() + ".o").getBytes());
			for (Task task : mTask.values())
				outstream.write((" " + task.getName() + ".o").getBytes());

			for (String extraSource : extraSourceList.values())
				outstream.write((" " + extraSource + ".o").getBytes());

			for (String extraLibSource : extraLibSourceList.values())
				outstream.write((" " + extraLibSource + ".o").getBytes());

			if (mAlgorithm.getLibraries() != null)
				for (Library library : mLibrary.values())
					outstream.write((" " + library.getName() + ".o").getBytes());

			outstream.write(" proc.o\n".getBytes());
			outstream.write("\t$(LD) $^ -o proc $(LDFLAGS)\n\n".getBytes());

			outstream.write(("proc.o: proc" + srcExtension + " ").getBytes());

			if (mAlgorithm.getHeaders() != null)
				for (String headerFile : mAlgorithm.getHeaders().getHeaderFile())
					outstream.write((" " + headerFile).getBytes());

			outstream.write("\n".getBytes());
			outstream.write(("\t$(CC) $(CFLAGS) -c proc" + srcExtension + " -o proc.o\n\n").getBytes());

			for (Task vTask : mVTask.values()) {
				outstream.write((vTask.getName() + ".o: " + vTask.getName() + srcExtension + " " + " ").getBytes());
				
				outstream.write("\n".getBytes());
				outstream.write(("\t$(CC) $(CFLAGS) " + vTask.getCflag() + " -c " + vTask.getName() + srcExtension
						+ " -o " + vTask.getName() + ".o\n\n").getBytes());

			}			
			for (Task task : mTask.values()) {
				if (task.getCICFile().endsWith(".cic"))
					outstream.write(
							(task.getName() + ".o: " + task.getName() + srcExtension + " " + task.getCICFile() + " ")
									.getBytes());
				else if (task.getCICFile().endsWith(".xml"))
					outstream.write((task.getName() + ".o: " + task.getName() + srcExtension + " " + " ").getBytes());
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
			}
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

	@Override
	public int generateCodeWithComm(String mTarget, String mTranslatorPath, String mOutputPath, String mRootPath,
			Map<Integer, Processor> mProcessor, List<Communication> mCommunication, Map<String, Task> mTask,
			Map<Integer, Queue> mQueue, Map<String, Library> mLibrary, Map<String, Library> mGlobalLibrary,
			int funcSimPeriod, String funcSimPeriodMetric, String mCICXMLFile, String language,
			CICAlgorithmType mAlgorithm, CICControlType mControl, CICScheduleType mSchedule, CICGPUSetupType mGpusetup,
			CICMappingType mMapping, Map<Integer, List<Task>> mConnectedTaskGraph,
			Map<Integer, List<List<Task>>> mConnectedSDFTaskSet, Map<String, Task> mVTask, Map<String, Task> mPVTask,
			String mGraphType, String mRuntimeExecutionPolicy, String codeGenerationStyle) throws FileNotFoundException {
		// TODO Auto-generated method stub
		return 0;
	}
}
