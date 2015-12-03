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


public class CICRobonovaTranslator implements CICTargetCodeTranslator 
{	
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
	
	private CICAlgorithmType mAlgorithm;
	private CICControlType mControl;
	private CICScheduleType mSchedule;
	private CICGPUSetupType mGpusetup;
	private CICMappingType mMapping;

	CICScheduleTypeLoader loaderSched;
	
	public static enum STRATEGY {
		APGAN,
		MINBUF,
		DLC,
		FLAT,
		PROCEDURE,
		TwoNode,
	}
	
	private STRATEGY strategy;
	private int loopId=0;
	
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
		mLanguage = language;
		
		mTask = task;
		mQueue = queue;
		mLibrary = library;
		mProcessor = processor;
		
		mAlgorithm = algorithm;
		mControl = control;
		mGpusetup = gpusetup;
		mMapping = mapping;

		// Make Output Directory
		File f = new File(mOutputPath);	
		
		f.mkdir();

		mOutputPath = mOutputPath + "\\";
		mTranslatorPath = mTranslatorPath + "\\";
		
		// Copy CIC_port.h file
		Util.copyFile(mOutputPath+"CIC_port.h", mTranslatorPath + "templates/common/cic/CIC_port.h");
		
		// Copy robonova files
		Util.copyFile(mOutputPath+"robot_protocol.h", mTranslatorPath + "templates/robonova/robot_protocol.h");
		Util.copyFile(mOutputPath+"robot_protocol.c", mTranslatorPath + "templates/robonova/robot_protocol.c");
		
		// generate proc.c or proc.cpp
		String fileOut = null;
		if(mLanguage.equals("c++")) {
			fileOut = mOutputPath+"proc.cpp";
		} else {
			fileOut = mOutputPath+"proc.c";
		}
		
		// generate proc.c
		String templateFile = "";
		
		// generate cic_tasks.h
		fileOut = mOutputPath + "cic_tasks.h";
		templateFile = mTranslatorPath + "templates/robonova/cic_tasks.h.template";
		generateTaskHeader(fileOut, templateFile, mTask, mGlobalPeriod, mGlobalPeriodMetric);
		
		// generate cic_channels.h
		fileOut = mOutputPath + "cic_channels.h";
		templateFile = mTranslatorPath + "templates/common/cic/cic_channels.h.template";
		CommonLibraries.CIC.generateChannelHeader(fileOut, templateFile, mQueue, mThreadVer);
		
		// generate cic_portmap.h
		fileOut = mOutputPath + "cic_portmap.h";
		templateFile = mTranslatorPath + "templates/common/cic/cic_portmap.h.template";
		CommonLibraries.CIC.generatePortmapHeader(fileOut, templateFile, mTask, mQueue);
		
		String srcExtension = "";
		if(mLanguage.equals("c++"))	srcExtension = ".cpp";
		else						srcExtension = ".c";
		
		// generate task_name.c (include task_name.cic)
		for(Task t: mTask.values()){
			if(t.getHasSubgraph().equalsIgnoreCase("Yes")){
				fileOut = mOutputPath + t.getName() + srcExtension;
				templateFile = mTranslatorPath + "templates/robonova/task_mtm_code_template.c";
				generateTaskMTMCode(fileOut, templateFile, t, mAlgorithm);
			}
			else{
				fileOut = mOutputPath + t.getName() + srcExtension;
				if(mLanguage.equals("c++"))	templateFile = mTranslatorPath + "templates/robonova/task_code_template.cpp";
				else						templateFile = mTranslatorPath + "templates/robonova/task_code_template.c";
				CommonLibraries.CIC.generateTaskCode(fileOut, templateFile, t, mAlgorithm, mControl);
			}
		}
		
		// Copy mtm.h file
		Util.copyFile(mOutputPath+"mtm.h", mTranslatorPath + "templates/common/cic/mtm.h");
		
		if(mLanguage.equals("c++")) {
			fileOut = mOutputPath+"proc.cpp";
		} else {
			fileOut = mOutputPath+"proc.c";
		}
		templateFile = mTranslatorPath + "templates/robonova/proc.c.template";
		generateProcCode(fileOut, templateFile);
		
		// generate mtm files from xml files	
		for(Task t: mTask.values()){
			if(t.getHasMTM().equalsIgnoreCase("Yes")){
				templateFile = mTranslatorPath + "templates/robonova/task_code_template.mtm";			
				generateMTMFile(mOutputPath, templateFile, t, mAlgorithm);
			}
		}

		fileOut = mOutputPath;
		if(mAlgorithm.getLibraries() != null){
			if(!mAlgorithm.getLibraries().getLibrary().isEmpty()){
				// generate library_name.c & library_name.h
				for(Library l: mLibrary.values()){			
					CommonLibraries.Library.generateLibraryCode(fileOut, l, mAlgorithm);
				}
			}
		}
				
		// copy *.cic files, library files and external files
	    Util.copyExtensionFiles(mOutputPath,"./", ".h");
	    Util.copyExtensionFiles(mOutputPath,"./", ".c");
	    Util.copyExtensionFiles(mOutputPath,"./", ".cic");
	    Util.copyExtensionFiles(mOutputPath,"./", ".cicl");
	    Util.copyExtensionFiles(mOutputPath,"./", ".cicl.h");
	    Util.copyExtensionFiles(mOutputPath,"./", ".mtm");
	    Util.copyExtensionFiles(mOutputPath,"./", ".xml");
	    
	    for(Task t: mTask.values()){
	    	if(t.getCICFile().endsWith(".xml")){
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
	    
	    // generate VS project file
	    fileOut = mOutputPath + "proc.vcproj";
		templateFile = mTranslatorPath + "templates/robonova/proc.vcproj.template";
	    generateVCProjectFile(fileOut, templateFile);
	    
	    // generate Makefile
	    fileOut = mOutputPath + "Makefile";
	    generateMakefile(fileOut);
	
	    return 0;
	}
	
	

	public static void generateTaskMTMCode(String mDestFile, String mTemplateFile, Task task, CICAlgorithmType mAlgorithm){
		File fileOut = new File(mDestFile);
		File templateFile = new File(mTemplateFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);
			
			outstream.write(translateTaskMTMCode(content, task, mAlgorithm).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	

	public static String translateTaskMTMCode(String mContent, Task task, CICAlgorithmType mAlgorithm){
		String code = mContent;
		String parameterDef="";
		
		for(TaskParameterType parameter: task.getParameter())
			parameterDef += "#define " + parameter.getName() + " " + parameter.getValue() + "\n";

		code = code.replace("##PARAMETER_DEFINITION", parameterDef);
		code = code.replace("##TASK_NAME", task.getName());
	    code = code.replace("##CIC_INCLUDE", "#include \"" + task.getName() + ".mtm\"\n");
		
		return code;
	}

	public void generateMTMFile(String outputPath, String mTemplateFile, Task task, CICAlgorithmType mAlgorithm){
		File fileOut = new File(outputPath + task.getName() + ".mtm");
		File templateFile = new File(mTemplateFile);
		
		try {			
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);
			
			outstream.write(translateMTMFile(outputPath, content, task, mAlgorithm).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public String translateMTMFile(String outputPath, String mContent, Task task, CICAlgorithmType mAlgorithm){
		String code = mContent;
		String modeMap = "";
		String intVarMap = "";
		String stringVarMap = "";
		String transition = "";
		String taskiterinfo = "";
		
		Map<String, int[]> taskRepMap = new HashMap<String, int[]> ();
		for(Task i_t: mTask.values()){
			if(i_t.getParentTask().equals(task.getName()) && i_t.getHasSubgraph().equalsIgnoreCase("No")){
				int[] rateList = new int[task.getMTM().getModes().size()];
				taskRepMap.put(i_t.getName(), rateList);
			}
		}
		
		int id=0;
		for(String mode: task.getMTM().getModes()){
			Map<String, Integer> tr = CommonLibraries.Schedule.generateIterationCount(task, mode, mTask, mQueue);
			for(Task i_t: mTask.values()){
				if(i_t.getParentTask().equals(task.getName()) && i_t.getHasSubgraph().equalsIgnoreCase("No")){
					int rate = 0;
					if(tr.get(i_t.getName()) != null)	rate = tr.get(i_t.getName());
					int[] rates = taskRepMap.get(i_t.getName());
					rates[id] = rate;
					taskRepMap.put(i_t.getName(), rates);
				}
			}
			id++;
		}
			
		for(String taskName: taskRepMap.keySet()){
			int[] rates = taskRepMap.get(taskName);
			taskiterinfo += "\t{\"" + taskName + "\", {";
			for(int j=0; j<rates.length; j++){
				taskiterinfo += rates[j] + ", ";
			}
			taskiterinfo += "}, 0},\n";
		}
		
		List<String> modeList = task.getMTM().getModes();
	    for(int i=0; i<modeList.size(); i++){
	        	String mode = modeList.get(i);
	        	modeMap += "\t{" + i + ", \"" + mode + "\"},\n";
        }
        
        transition += "\t";
        List<Variable> varList = task.getMTM().getVariables();
        for(int i=0; i<varList.size(); i++){
        	Variable var = varList.get(i);
        	String type = var.getType();
        	if(type.equals("Integer")){
        		intVarMap += "\t{" + i + ", \"" + var.getName() + "\", 0},\n";
        		transition += "int " + var.getName() + " = " + task.getName() + "_get_variable_int(\"" + var.getName() + "\");";
        	}
        	else if(type.equals("String")){
        		stringVarMap += "\t{" + i + ", " + var.getName() + ", \"\"},\n";
        		transition += "char* " + var.getName() + " = " + task.getName() + "_get_variable_string(\"" + var.getName() + "\");";
        	}
        }
        
        for(String mode: task.getMTM().getModes()){
        	Map<String, Integer> taskInterMap = CommonLibraries.Schedule.generateIterationCount(task, mode, mTask, mQueue);
        }
        for(TaskType t: mAlgorithm.getTasks().getTask()){
        	if(t.getParentTask().equals(task.getName())){
        		String ti_info = "";
        		ti_info += "\t\"" + t.getName() + "\", {";
        		
        	}
        }
        
        transition += "\n\tswitch(current_mode){\n";
        List<Transition> transList = task.getMTM().getTransitions();
        String t = "";
        for(int i=0; i<modeList.size(); i++){
        	String mode = modeList.get(i);
        	t += "\t\tcase " + i + ":\n";	        	
        	// transition list
        	for(int j=0; j<transList.size(); j++){
        		Transition trans = transList.get(j);

		        String src = trans.getSrcMode();
		        // condition list
		        if(src.equals(mode)){
	        		List<Condition> condList = trans.getConditionList();
	        		t += "\t\t\tif(";
	        		for(int k=0; k<condList.size(); k++){
	        			Condition cond = condList.get(k);
	        			String var = cond.getVariable();
	        			String comp = cond.getComparator();
	        			String val = cond.getValue();
	        			t += (var + " " + comp + " " + val);
	        			if(k != condList.size() - 1) t += " && "; 
	        		}
	        		t += "){\n";
		        	
		        	int dst_id = 0;
		        	String dst = trans.getDstMode();
		        	for(int l=0; l<modeList.size(); l++){
		        		String mName = modeList.get(l);
		        		if(dst.equals(mName)){
		        			dst_id = l;
		        			break;
		        		}
		        	}
			        t += "\t\t\t\tcurrent_mode = " + dst_id + ";\n\t\t\t}";     		
		        }
        	}
        	t += "\t\t\tbreak;\n";
        }
        
        transition += t;
        transition += "\t}";
             
		// Replace code
		code = code.replace("##MODEMAP", modeMap);
		code = code.replace("##INTVARMAP", intVarMap);
		code = code.replace("##STRINGVARMAP", stringVarMap);
		code = code.replace("##TRANSITION", transition);
		code = code.replace("##TASKITERINFO", taskiterinfo);	
		
		return code;
	}
	
	public static void generateTaskHeader(String mDestFile, String mTemplateFile, Map<String, Task> mTask, int globalPeriod, String globalPeriodMetric)
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
			
			outstream.write(translateTaskHeader(content, mTask, globalPeriod, globalPeriodMetric).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
		
	public static String translateTaskHeader(String mContent, Map<String, Task> mTask, int mGlobalPeriod, String mGlobalPeriodMetric)
	{
		String code = mContent;
		
		// C++ code generation
		String externalFunctions = "#ifdef CPP_CODE_GENERATION\n";
		for(Task task: mTask.values())
			externalFunctions += "extern class CICTask* " + task.getName()+";\n";
		
		// C code generation
		externalFunctions +=  "#else\n";
		for(Task task: mTask.values()){
			if(task.getHasSubgraph().equals("No")){
				externalFunctions += "extern void "+task.getName()+"_init(int);\n";
				externalFunctions += "extern int "+task.getName()+"_go(void);\n";
				externalFunctions += "extern void "+task.getName()+"_wrapup(void);\n";
			}
		}
		externalFunctions += "#endif\n";

		String taskEntriesCode="";
		int index = 0;
		while(index < mTask.size()) {
			Task task = null;
			for(Task t: mTask.values()){
				if(Integer.parseInt(t.getIndex()) == index){
					task = t;
					break;
				}
			}
			String taskDrivenType = null;
			String runState = null;
			if(task.getRunCondition().equals("DATA_DRIVEN")) {
				taskDrivenType = "DataDriven";
				runState = "Run";
			} else if(task.getRunCondition().equals("TIME_DRIVEN")) {
				taskDrivenType = "TimeDriven";
				runState = "Run";
			} else if(task.getRunCondition().equals("RUN_ONCE")) {
				taskDrivenType = "RunOnce";
				runState = "Wait";
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
				System.out.println("[makeTasks] Not supported metric of period");
				System.exit(-1);
			}
			
			String taskType = task.getTaskType();
			
			// Added by jhw for multi-mode 
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
			
			// check isSrcTask
			String isSrcTask = "true";
			int flag = 0;
			for(Queue taskQueue: task.getQueue()){
				if(taskQueue.getDst().equals(task.getName()) && Integer.parseInt(taskQueue.getInitData()) == 0){
					String srcTask = taskQueue.getSrc();
					for(Task t: mTask.values()){
						if(t.getName().equals(srcTask) && t.getName().equals(t.getParentTask())){
							flag = 1;
							break;
						}
					}
					
					if(flag == 0)	isSrcTask = "false";
					break;
				}
			}
			
			if(hasSubgraph.equals("false")){
				taskEntriesCode += "\t{" + task.getIndex() +", \""+ task.getName()+"\", " + taskType + ", "
					+ task.getName() + "_init, "+ task.getName() + "_go, "+ task.getName() + "_wrapup, "+ taskDrivenType+", "+runState+", "
					+ task.getPeriodMetric().toUpperCase() + ", " 
					+ task.getRunRate() + "/*rate*/, " 
					+ task.getPeriod() +"/*period*/, "
					+ globalPeriod +"/*global period*/, "
					+ Integer.toString(globalPeriod / Integer.parseInt(task.getPeriod()) * Integer.parseInt(task.getRunRate())) +", "
					+ ", " + isSrcTask + ", " + hasMTM + ", " + hasSubgraph + ", " + isChildTask + ", " + parentTaskId + ", "
	 				+ "PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER},\n";
			}
			else{
				taskEntriesCode += "\t{" + task.getIndex() +", \""+ task.getName()+"\", " + taskType + ", "
						+ "NULL, " + "NULL, " + "NULL, "+ taskDrivenType+", "+runState+", "
						+ task.getPeriodMetric().toUpperCase() + ", " 
						+ task.getRunRate() + "/*rate*/, " 
						+ task.getPeriod() +"/*period*/, "
						+ globalPeriod +"/*global period*/, "
						+ Integer.toString(globalPeriod / Integer.parseInt(task.getPeriod()) * Integer.parseInt(task.getRunRate())) +", "
						 + ", " + isSrcTask + ", " + hasMTM + ", " + hasSubgraph + ", " + isChildTask + ", " + parentTaskId + ", "
		 				+ "PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER},\n";
			}
			index++;
		}
		
		// Added by jhw for multi-mode 
		String mtmEntriesCode="";
		String externalMTMFunctions =  "";
		index = 0;
		for(Task t: mTask.values()){
			if(t.getHasMTM().equalsIgnoreCase("Yes")){
				mtmEntriesCode += "\tENTRY(" + t.getIndex() + ", " + t.getName() + ", PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER),\n";
				externalMTMFunctions += "extern char* " + t.getName() + "_get_current_mode_name();\n"
						+ "extern int " + t.getName() + "_get_current_mode_id();\n"
						+ "extern char* " + t.getName() + "_get_mode_name(int);\n"
						+ "extern int " + t.getName() + "_get_variable_int(char*);\n"
						+ "extern void " + t.getName() + "_set_variable_int(char*, int);\n"
						+ "extern char* " + t.getName() + "_get_variable_string(char*);\n"
						+ "extern void " + t.getName() + "_set_variable_string(char*, char*);\n"
						+ "extern void " + t.getName() + "_transition();\n"
						+ "extern int " + t.getName() + "_get_task_iter_count(char*);\n"
						+ "extern void " + t.getName() + "_set_task_curr_count(char*);\n"
						+ "extern void " + t.getName() + "_reset_tasks_curr_count();\n"
						+ "extern int " + t.getName() + "_check_tasks_curr_count();\n";
				index++;
			}
		}
		
		code = code.replace("##EXTERN_FUNCTION_DECLARATION", externalFunctions);
		code = code.replace("##TASK_ENTRIES", taskEntriesCode);
		// Added by jhw for multi-mode 
		code = code.replace("##EXTERN_MTM_FUNCTION_DECLARATION", externalMTMFunctions);
		code = code.replace("##MTM_ENTRIES", mtmEntriesCode);
		
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
			
			if(mLanguage.equals("c++"))
				content = "#define CPP_CODE_GENERATION 1\\n\\n"+content;
			
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
		
		String libraryInitWrapup="";
		if(mAlgorithm.getLibraries() != null){
			for(Library library: mLibrary.values()){
				libraryInitWrapup += "extern void l_"+library.getName()+"_init(void);\n";
				libraryInitWrapup += "extern void l_"+library.getName()+"_wrapup(void);\n\n";
			}
		}
		
		libraryInitWrapup += "static void init_libs() {\n";
		if(mAlgorithm.getLibraries() != null){
			for(Library library: mLibrary.values())
				libraryInitWrapup += "\tl_" + library.getName() + "_init();\n";
		}
		libraryInitWrapup += "}\n\n";
		
		libraryInitWrapup += "static void wrapup_libs() {\n";
		if(mAlgorithm.getLibraries() != null){
			for(Library library: mLibrary.values())
				libraryInitWrapup += "\tl_" + library.getName() + "_wrapup();\n";
		}
		libraryInitWrapup += "}\n\n";
		
		// Common code generation
		String externalGlobalHeaders = "";
		if(mAlgorithm.getHeaders()!=null){
			for(String header: mAlgorithm.getHeaders().getHeaderFile())
				externalGlobalHeaders += "#include\"" + header +"\"\n";
		}
		
		String controlParamListEntries = CommonLibraries.Control.generateControlParamListEntries(mControl, mTask);
		String controlChannelListEntries = CommonLibraries.Control.generateControlChannelListEntries(mControl, mTask);
		
		String globalExecutionTime = "#define GLOBAL_EXECUTION_TIME " + Integer.toString(mGlobalPeriod) + "\n";
		
		String controlGroupCount = "1";
		if(mControl.getExclusiveControlTasksList() != null)	controlGroupCount = Integer.toString(mControl.getExclusiveControlTasksList().getExclusiveControlTasks().size());
		controlGroupCount = "#define CONTROL_GROUP_COUNT " + controlGroupCount;
		
		String controlChannelCount = "1";
		if(mControl.getControlTasks() != null)	controlChannelCount = Integer.toString(mControl.getControlTasks().getControlTask().size());
		controlChannelCount = "#define CONTROL_CHANNEL_COUNT " + controlChannelCount;
		
		// End code generation part
		code = code.replace("##EXTERNAL_GLOBAL_HEADERS", externalGlobalHeaders);
		code = code.replace("##GLOBAL_EXECUTION_TIME", globalExecutionTime);
		code = code.replace("##LIB_INIT_WRAPUP",  libraryInitWrapup);	
		code = code.replace("##PARAM_LIST_ENTRIES", controlParamListEntries);
		code = code.replace("##CONTROL_CHANNEL_LIST_ENTRIES", controlChannelListEntries);
		code = code.replace("##CONTROL_GROUP_COUNT", controlGroupCount);
		code = code.replace("##CONTROL_CHANNEL_COUNT", controlChannelCount);
		
		return code;
	}


	public void generateVCProjectFile(String mDestFile, String mTemplateFile){
		File fileOut = new File(mDestFile);
		File templateFile = new File(mTemplateFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);
			
			outstream.write(translateVCProjectFile(content).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public String translateVCProjectFile(String mContent){
		String code = mContent;
		String srcExtension = null;
		String sourceFiles = null;
		String headerFiles = null;
		String rootDirectory = null;
		
		if(mLanguage.equals("c++"))	srcExtension = ".cpp";
		else						srcExtension = ".c";

		for(Task task: mTask.values()) {	
			sourceFiles = sourceFiles + "\t\t\t<File\n\t\t\t\tRelativePath = \".\\" +task.getName() + srcExtension 
							+"\"\n\t\t\t\t>\n" + "\t\t\t</File>\n";
			if(task.getCICFile().startsWith("/") || task.getCICFile().startsWith(":", 1))
				headerFiles = headerFiles + "\t\t\t<File\n\t\t\t\tRelativePath = \"" + task.getCICFile() 
							+"\"\n\t\t\t\t>\n" + "\t\t\t</File>\n";
			else
				headerFiles = headerFiles + "\t\t\t<File\n\t\t\t\tRelativePath = \".\\" + task.getCICFile()  
							+"\"\n\t\t\t\t>\n" + "\t\t\t</File>\n";
		}
		sourceFiles = sourceFiles + "\t\t\t<File\n\t\t\t\tRelativePath = \".\\proc"+ srcExtension 
							+"\"\n\t\t\t\t>\n" + "\t\t\t</File>\n";
		
		code = code.replace("##NAME", "proc");
		code = code.replace("##SOURCEFILES", sourceFiles);
		code = code.replace("##HEADERFILES", headerFiles);
		code = code.replace("##ROOTDIR", mRootPath);
		
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
				outstream.write("CC=arm-linux-g++\n".getBytes());
				outstream.write("LD=arm-linux-g++\n".getBytes());
			}
			else{
				srcExtension = ".c";
				outstream.write("CC=arm-linux-g++\n".getBytes());
				outstream.write("LD=arm-linux-g++\n".getBytes());
			}
			
		    mRootPath = mRootPath.replace("\\","/");
		    mRootPath = mRootPath.replace("C:", "/cygdrive/C");
		    mRootPath = mRootPath.replace("D:", "/cygdrive/D");
		    mRootPath = mRootPath.replace("E:", "/cygdrive/E");
		    mRootPath = mRootPath.replace("F:", "/cygdrive/F");
		    mRootPath = mRootPath.replace("G:", "/cygdrive/G");
		    
		    if(mThreadVer.equals("m")){
		    	outstream.write("#CFLAGS=-Wall -O0 -g -DDISPLAY -DTHREAD_STYLE -I$(OPENCV_INC_PATH)\n".getBytes());
		        outstream.write("CFLAGS=-Wall -O2 -DDISPLAY -DTHREAD_STYLE -I$(OPENCV_INC_PATH)\n".getBytes());
		    }
		    else if(mThreadVer.equals("s")){
		    	outstream.write("#CFLAGS=-Wall -O0 -g -DDISPLAY -I$(OPENCV_INC_PATH)\n".getBytes());
		        outstream.write("CFLAGS=-Wall -O2 -DDISPLAY -I$(OPENCV_INC_PATH)\n".getBytes());
		    }
		    
		    outstream.write("LDFLAGS=-lpthread -lm -lcv -lcxcore -lhighgui -Xlinker --warn-common -L$(OPENCV_LIB_PATH)".getBytes());
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
		    
		    outstream.write(" robot_protocol.o proc.o\n".getBytes());
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
		    
		    outstream.write(("\nrobot_protocol.o: robot_protocol.c robot_protocol.h\n\t$(CC) $(CFLAGS)  -c robot_protocol.c -o robot_protocol.o\n").getBytes());
		    
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
