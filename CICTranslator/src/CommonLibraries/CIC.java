package CommonLibraries;

import hopes.cic.exception.CICXMLException;
import hopes.cic.xml.*;

import java.io.*;
import java.util.*;

import javax.swing.JOptionPane;
import javax.xml.parsers.*;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import InnerDataStructures.*;
import InnerDataStructures.Library;
import InnerDataStructures.Queue;

public class CIC {

	public static void generateTaskCode(String mDestFile, String mTemplateFile, Task task, CICAlgorithmType mAlgorithm, CICControlType mControl){
		File fileOut = new File(mDestFile);
		File templateFile = new File(mTemplateFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);
			
			outstream.write(translateTaskCode(content, task, mAlgorithm, mControl).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public static String translateTaskCode(String mContent, Task task, CICAlgorithmType mAlgorithm, CICControlType mControl){
		String code = mContent;
		String parameterDef="";
		String libraryDef="";
		String sysportDef="";
		String cicInclude="";

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
		
		cicInclude += "#include \"" + task.getCICFile() + "\"\n";
		
		String includeHeaders = "#include \"target_system_model.h\"\n";
		includeHeaders += "#include \"cic_comm_apis.h\"\n";
		includeHeaders += "#include \"cic_basic_control_apis.h\"\n";
		
		boolean mtmFlag = false;
		for(TaskType t: mAlgorithm.getTasks().getTask()){
			if(t.getHasMTM().equals("Yes")){
				mtmFlag = true;
				break;
			}
		}
		if(mtmFlag)	includeHeaders += "#include \"cic_mtm_apis.h\"\n";
		
		if(mControl.getControlTasks() != null)
			includeHeaders += "#include \"cic_control_apis.h\"\n";
		
		code = code.replace("##INCLUDE_HEADERS", includeHeaders);
		code = code.replace("##LIBRARY_DEFINITION", libraryDef);
		code = code.replace("##TASK_INSTANCE_NAME", task.getName());	    
	    code = code.replace("##SYSPORT_DEFINITION", sysportDef);
	    code = code.replace("##CIC_INCLUDE", cicInclude);
	    code = code.replace("##TASK_NAME", task.getName());
		
		return code;
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

		code = code.replace("##TASK_NAME", task.getName());
	    code = code.replace("##CIC_INCLUDE", "#include \"" + task.getName() + ".mtm\"\n");
		
		return code;
	}
	
	public static void generateMTMFile(String outputPath, String mTemplateFile, Task task, CICAlgorithmType mAlgorithm, Map<String, Task> mTask, Map<String, Task> mPVTask, Map<Integer, Queue> mQueue, String version){
		File fileOut = new File(outputPath + task.getName() + ".mtm");
		File templateFile = new File(mTemplateFile);
		
		try {			
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);
			
			outstream.write(translateMTMFile(outputPath, content, task, mAlgorithm, mTask, mPVTask, mQueue, version).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static String translateMTMFile(String outputPath, String mContent, Task task, CICAlgorithmType mAlgorithm, Map<String, Task> mTask, Map<String, Task> mPVTask, Map<Integer, Queue> mQueue, String version){
		String code = mContent;
		String modeMap = "";
		String intVarMap = "";
		String stringVarMap = "";
		String transition = "";
		String transitionVarInit = "";
		String taskiterinfo = "";
		String numSubTasks = "";
		String srcTaskIndex = "";
		
		if(version.equals("Global")){
			String srcTaskName = "";
			int srcTaskId = 0;
			int subTaskCount = 0;
			
			for(Task t: mTask.values()){
				if(t.getParentTask().equals(task.getName()) && !t.getName().equals(task.getName())){
					subTaskCount++;
				}
			}
			
			for(Task t: mTask.values()){
				if(t.getParentTask().equals(task.getName()) && !t.getName().equals(task.getName())){
					boolean isSrcTask = true;
					for(Queue taskQueue: t.getQueue()){
						if(taskQueue.getDst().equals(t.getName()) && Integer.parseInt(taskQueue.getInitData()) == 0){
							isSrcTask = false;
							break;
						}
					}
					if(isSrcTask){
						srcTaskName = t.getName();
						break;
					}
				}
			}
			
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
			
			int count = 0;
			for(String taskName: taskRepMap.keySet()){
				int[] rates = taskRepMap.get(taskName);
				taskiterinfo += "\t{\"" + taskName + "\", {";
				for(int j=0; j<rates.length; j++){
					taskiterinfo += rates[j] + ", ";
				}
				taskiterinfo += "}, 0},\n";
				if(srcTaskName.equals(taskName)){
					srcTaskId = count;
				}
				count++;
			}
			
			numSubTasks += "#define NUM_SUB_TASKS " + subTaskCount;
			srcTaskIndex += "CIC_STATIC CIC_T_INT src_task_index = " + srcTaskId + ";";
		}
		else if(version.equals("Partitioned")){
			int subTaskCount = 0;
			for(Task pt: mPVTask.values()){
				if(pt.getParentTask().equals(task.getName())){
					taskiterinfo += "\t{\"" + pt.getName() + "\", {0, }, 0},\n";
					subTaskCount++;
				}
			}
			numSubTasks += "#define NUM_SUB_TASKS " + subTaskCount;
		}
		
		if(taskiterinfo.equals("")){
			taskiterinfo += "\t{\"__temp__\", {0, 0}, 0},\n";
        }
		
		List<String> modeList = task.getMTM().getModes();
	    for(int i=0; i<modeList.size(); i++){
        	String mode = modeList.get(i);
        	modeMap += "\t{" + i + ", \"" + mode + "\"},\n";
        }
        
	    transitionVarInit += "\t";
        List<Variable> varList = task.getMTM().getVariables();
        for(int i=0; i<varList.size(); i++){
        	Variable var = varList.get(i);
        	String type = var.getType();
        	if(type.equals("Integer")){
        		intVarMap += "\t{" + i + ", \"" + var.getName() + "\", 0},\n";
        		transitionVarInit += "CIC_T_INT " + var.getName() + " = " + task.getName() + "_GetVariableInt(\"" + var.getName() + "\");";
        	}
        	else if(type.equals("String")){
        		stringVarMap += "\t{" + i + ", " + var.getName() + ", \"\"},\n";
        		transitionVarInit += "CIC_T_CHAR* " + var.getName() + " = " + task.getName() + "_GetVariableString(\"" + var.getName() + "\");";
        	}
        }
        
        if(intVarMap.equals("")){
        	intVarMap += "\t{-1, \"__temp__\", -1},\n";
        }
        if(stringVarMap.equals("")){
        	stringVarMap += "\t{-1, \"__temp__\", NULL},\n";
        }

        transition += "\tswitch(current_mode){\n";
        List<Transition> transList = task.getMTM().getTransitions();
        String t = "";
        for(int i=0; i<modeList.size(); i++){
        	String mode = modeList.get(i);
        	t += "\t\tcase " + i + ":\n";	        	
        	// transition list
        	if(transList.size() == 0){
		        t += "\t\t\tsrc_continue_count++;\n";
        	}
        	else{
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
				        t += "\t\t\t\tnext_mode = " + dst_id + ";\n";     	
				        t += "\t\t\t\tis_transition = CIC_V_TRUE;\n\t\t\t}\n";
				        t += "\t\t\telse{\n";
				        t += "\t\t\t\tsrc_continue_count++;\n";
				        t += "\t\t\t}\n";
			        }
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
		code = code.replace("##TRANS_VAR_INIT", transitionVarInit);
		code = code.replace("##TASKITERINFO", taskiterinfo);
		code = code.replace("##NUMSUBTASKS", numSubTasks);
		code = code.replace("##SRCTASKINDEX", srcTaskIndex);
		
		return code;
	}
	

	
	public static void generateTaskDataStructure(String mDestFile, String mTemplateFile, Map<String, Task> mTask, int globalPeriod, String globalPeriodMetric, String mThreadVer, String mCodeGenType, Map<String, Task> mVTask, Map<String, Task> mPVTask)
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
			
			outstream.write(translateTaskDataStructure(content, mTask, globalPeriod, globalPeriodMetric, mThreadVer, mCodeGenType, mVTask, mPVTask).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
		
	public static String translateTaskDataStructure(String code, Map<String, Task> mTask, int mGlobalPeriod, String mGlobalPeriodMetric, String mThreadVer, String mCodeGenType, Map<String, Task> mVTask, Map<String, Task> mPVTask)
	{
		String taskEntriesCode="";
		String virtualTaskEntriesCode="";
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
			String state = null;
			String runState = null;
			if(task.getRunCondition().equals("DATA_DRIVEN")) {
				taskDrivenType = "DATA_DRIVEN";
				state = "STATE_RUN";
				runState = "RUNNING";
			} else if(task.getRunCondition().equals("TIME_DRIVEN")) {
				taskDrivenType = "TIME_DRIVEN";
				state = "STATE_RUN";
				runState = "RUNNING";
			} else if(task.getRunCondition().equals("CONTROL_DRIVEN")) {
				taskDrivenType = "CONTROL_DRIVEN";
				state = "STATE_STOP";
				runState = "RUNNING";
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
			else													globalPeriod = mGlobalPeriod * 1;

			
			String taskType = task.getTaskType().toUpperCase();
			
			// Added by jhw for multi-mode 
			String hasMTM = "";
			if(task.getHasMTM().equalsIgnoreCase("Yes"))	hasMTM = "CIC_V_TRUE";
			else											hasMTM = "CIC_V_FALSE";
			
			String hasSubgraph = "";
			if(task.getHasSubgraph().equalsIgnoreCase("Yes"))	hasSubgraph = "CIC_V_TRUE";
			else												hasSubgraph = "CIC_V_FALSE";
			
			String isChildTask = "CIC_V_FALSE";
			int parentTaskId = -1;
			
			if(hasSubgraph.equals("CIC_V_FALSE")){
				if(!task.getName().equals(task.getParentTask())){
					for(Task t: mTask.values()){
						if(task.getParentTask().equals(t.getName())){
							isChildTask = "CIC_V_TRUE";
							parentTaskId = Integer.parseInt(t.getIndex());
							break;
						}
					}
					for(Task t: mVTask.values()){
						if(task.getParentTask().equals(t.getName())){
							isChildTask = "CIC_V_TRUE";
							parentTaskId = Integer.parseInt(t.getIndex());
							break;
						}
					}
				}
				else{
					isChildTask = "CIC_V_FALSE";
					parentTaskId = index;
				}
			}
			else{
				isChildTask = "CIC_V_FALSE";
				parentTaskId = index;
			}
			
			if(parentTaskId == -1)	parentTaskId = Integer.parseInt(task.getIndex());
			
			// check isSrcTask
			String isSrcTask = "CIC_V_FALSE";
			if(task.getIsSrcTask())	isSrcTask = "CIC_V_TRUE";
			
			taskEntriesCode += "\tENTRY(" + task.getIndex() +", \""+ task.getName()+"\", " + taskType + ", "
				+ task.getName() + ", "+ taskDrivenType+", "+state+", " + runState+", "
				+ task.getPeriodMetric().toUpperCase() + ", " 
				+ task.getRunRate() + "/*rate*/, " 
				+ task.getPeriod() +"/*period*/, "
				+ globalPeriod +"/*global period*/, "
				+ Integer.toString(globalPeriod / Integer.parseInt(task.getPeriod()) /* * Integer.parseInt(task.getRunRate())*/) +", "
				+ hasMTM + ", " + hasSubgraph + ", " + isSrcTask + ", " + isChildTask + ", " + parentTaskId 
 				+ ", CIC_V_MUTEX_INIT_INLINE, CIC_V_COND_INIT_INLINE),\n";
			
			index++;
		}
		
		String parentHistory = "";
		while(index < mTask.size() + mVTask.size() + mPVTask.size()) {
			Task task = null;
			if(index < mTask.size() + mVTask.size()){
				for(Task t: mVTask.values()){
					if(Integer.parseInt(t.getIndex()) == index){
						task = t;
						break;
					}
				}
			}
			else{
				for(Task t: mPVTask.values()){
					if(Integer.parseInt(t.getIndex()) == index){
						task = t;
						break;
					}
				}
			}
			String taskDrivenType = null;
			String state = null;
			String runState = null;

			if(task.getRunCondition().equals("DATA_DRIVEN")) {
				taskDrivenType = "DATA_DRIVEN";
				state = "STATE_RUN";
				runState = "RUNNING";
			} else if(task.getRunCondition().equals("TIME_DRIVEN")) {
				taskDrivenType = "TIME_DRIVEN";
				state = "STATE_RUN";
				runState = "RUNNING";
			} else if(task.getRunCondition().equals("CONTROL_DRIVEN")) {
				taskDrivenType = "CONTROL_DRIVEN";
				state = "STATE_STOP";
				runState = "RUNNING";
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
			else													globalPeriod = mGlobalPeriod * 1;
			
			String taskType = "VIRTUAL";
			
			// Added by jhw for multi-mode 
			String hasMTM = "";
			if(task.getHasMTM().equalsIgnoreCase("Yes"))	hasMTM = "CIC_V_TRUE";
			else											hasMTM = "CIC_V_FALSE";
			
			String hasSubgraph = "";
			if(task.getHasSubgraph().equalsIgnoreCase("Yes"))	hasSubgraph = "CIC_V_TRUE";
			else												hasSubgraph = "CIC_V_FALSE";
			
			String isChildTask = "CIC_V_FALSE";
			int parentTaskId = -1;
			
			for(Task t: mTask.values()){
				if(task.getParentTask().equals(t.getName())){
					isChildTask = "CIC_V_FALSE";
					parentTaskId = Integer.parseInt(t.getIndex());
					break;
				}
			}
			
			/*
			if(hasSubgraph.equals("CIC_V_FALSE")){
				if(!task.getName().equals(task.getParentTask())){
					for(Task t: mVTask.values()){
						if(task.getParentTask().equals(t.getName())){
							isChildTask = "CIC_V_TRUE";
							parentTaskId = Integer.parseInt(t.getIndex());
							break;
						}
					}
				}
				else{
					isChildTask = "CIC_V_FALSE";
					parentTaskId = index;
				}
			}
			else{
				isChildTask = "CIC_V_FALSE";
				parentTaskId = index;
			}
			*/
			
			// to call Transition() once...
			String isSrcTask = "CIC_V_FALSE";
			if(task.getIsSrcTask())	isSrcTask = "CIC_V_TRUE";
			
			virtualTaskEntriesCode += "\tENTRY(" + task.getIndex() +", \""+ task.getName()+"\", " + taskType + ", "
				+ task.getName() + ", "+ taskDrivenType+", "+state+", "+runState+", "
				+ task.getPeriodMetric().toUpperCase() + ", " 
				+ task.getRunRate() + "/*rate*/, " 
				+ task.getPeriod() +"/*period*/, "
				+ globalPeriod +"/*global period*/, "
				+ Integer.toString(globalPeriod / Integer.parseInt(task.getPeriod()) / Integer.parseInt(task.getRunRate())) +", "
				+ hasMTM + ", " + hasSubgraph + ", " + isSrcTask + ", " + isChildTask + ", " + parentTaskId
 				+ ", CIC_V_MUTEX_INIT_INLINE, CIC_V_COND_INIT_INLINE),\n";
			
			index++;
		}
				
		// EXTERN_FUNCTION_DECLARATION //
		String externalFunctions = "";
	
		if(mCodeGenType.equals("Single")){
			for(Task task: mTask.values()){
				externalFunctions += "CIC_EXTERN CIC_T_VOID "+task.getName()+"_Init(CIC_T_INT);\n";
				externalFunctions += "CIC_EXTERN CIC_T_VOID "+task.getName()+"_Go(CIC_T_VOID);\n";
				externalFunctions += "CIC_EXTERN CIC_T_VOID "+task.getName()+"_Wrapup(CIC_T_VOID);\n";
			}
			for(Task task: mVTask.values()){
				externalFunctions += "CIC_STATIC CIC_T_VOID "+task.getName()+"_Init(CIC_T_INT);\n";
				externalFunctions += "CIC_STATIC CIC_T_VOID "+task.getName()+"_Go(CIC_T_VOID);\n";
				externalFunctions += "CIC_STATIC CIC_T_VOID "+task.getName()+"_Wrapup(CIC_T_VOID);\n";
			}
			for(Task task: mPVTask.values()){
				externalFunctions += "CIC_STATIC CIC_T_VOID "+task.getName()+"_Init(CIC_T_INT);\n";
				externalFunctions += "CIC_STATIC CIC_T_VOID "+task.getName()+"_Go(CIC_T_VOID);\n";
				externalFunctions += "CIC_STATIC CIC_T_VOID "+task.getName()+"_Wrapup(CIC_T_VOID);\n";
			}
		}
		else if(mCodeGenType.equals("Global")){
			for(Task task: mTask.values()){
				if(task.getHasSubgraph().equals("No")){
					externalFunctions += "CIC_EXTERN CIC_T_VOID "+task.getName()+"_Init(CIC_T_INT);\n";
					externalFunctions += "CIC_EXTERN CIC_T_VOID "+task.getName()+"_Go(CIC_T_VOID);\n";
					externalFunctions += "CIC_EXTERN CIC_T_VOID "+task.getName()+"_Wrapup(CIC_T_VOID);\n";
				}
				else{
					externalFunctions += "CIC_T_VOID "+task.getName()+"_Init(CIC_T_INT param) {};\n";
					externalFunctions += "CIC_T_VOID "+task.getName()+"_Go(CIC_T_VOID) {};\n";
					externalFunctions += "CIC_T_VOID "+task.getName()+"_Wrapup(CIC_T_VOID) {};\n";
				}
			}
		}
		else if(mCodeGenType.equals("Partitioned")){
			for(Task task: mTask.values()){
				if(task.getHasSubgraph().equals("No")){
					externalFunctions += "CIC_EXTERN CIC_T_VOID "+task.getName()+"_Init(CIC_T_INT);\n";
					externalFunctions += "CIC_EXTERN CIC_T_VOID "+task.getName()+"_Go(CIC_T_VOID);\n";
					externalFunctions += "CIC_EXTERN CIC_T_VOID "+task.getName()+"_Wrapup(CIC_T_VOID);\n";
				}
				else{
					externalFunctions += "CIC_T_VOID "+task.getName()+"_Init(CIC_T_INT param) {};\n";
					externalFunctions += "CIC_T_VOID "+task.getName()+"_Go(CIC_T_VOID) {};\n";
					externalFunctions += "CIC_T_VOID "+task.getName()+"_Wrapup(CIC_T_VOID) {};\n";
				}
			}
			for(Task task: mVTask.values()){
				externalFunctions += "CIC_STATIC CIC_T_VOID "+task.getName()+"_Init(CIC_T_INT);\n";
				externalFunctions += "CIC_STATIC CIC_T_VOID "+task.getName()+"_Go(CIC_T_VOID);\n";
				externalFunctions += "CIC_STATIC CIC_T_VOID "+task.getName()+"_Wrapup(CIC_T_VOID);\n";
			}
			for(Task task: mPVTask.values()){
				externalFunctions += "CIC_STATIC CIC_T_VOID "+task.getName()+"_Init(CIC_T_INT);\n";
				externalFunctions += "CIC_STATIC CIC_T_VOID "+task.getName()+"_Go(CIC_T_VOID);\n";
				externalFunctions += "CIC_STATIC CIC_T_VOID "+task.getName()+"_Wrapup(CIC_T_VOID);\n";
			}
		}
		
		/////////////////////////////////
		
		// Added by jhw for multi-mode 
		String mtmEntriesCode="";
		String externalMTMFunctions =  "";
		index = 0;
		for(Task t: mTask.values()){
			if(t.getHasMTM().equalsIgnoreCase("Yes")){
				String isSDF = "CIC_V_FALSE";
				if(t.getMTM().getModes().size() == 1){
					isSDF = "CIC_V_TRUE";
				}
				mtmEntriesCode += "\tENTRY(" + t.getIndex() + ", " + isSDF + ", " + t.getName() + ", CIC_V_MUTEX_INIT_INLINE, CIC_V_COND_INIT_INLINE),\n";
				externalMTMFunctions += "CIC_EXTERN CIC_T_VOID " + t.getName() + "_Initialize(CIC_T_VOID);\n"
						+ "CIC_EXTERN CIC_T_CHAR* " + t.getName() + "_GetCurrentModeName(CIC_T_VOID);\n"
						+ "CIC_EXTERN CIC_T_INT " + t.getName() + "_GetVariableInt(CIC_T_CHAR*);\n"
						+ "CIC_EXTERN CIC_T_VOID " + t.getName() + "_SetVariableInt(CIC_T_CHAR*, CIC_T_INT);\n"
						+ "CIC_EXTERN CIC_T_CHAR* " + t.getName() + "_GetVariableString(CIC_T_CHAR*);\n"
						+ "CIC_EXTERN CIC_T_VOID " + t.getName() + "_SetVariableString(CIC_T_CHAR*, CIC_T_CHAR*);\n"
						+ "CIC_EXTERN CIC_T_BOOL " + t.getName() + "_Transition(CIC_T_VOID);\n"
						+ "CIC_EXTERN CIC_T_INT " + t.getName() + "_UpdateCurrentMode(CIC_T_CHAR*);\n"
						+ "CIC_EXTERN CIC_T_INT " + t.getName() + "_GetTaskIterCount(CIC_T_CHAR*);\n"
						+ "CIC_EXTERN CIC_T_INT " + t.getName() + "_GetTaskRepeatCount(CIC_T_CHAR*, CIC_T_INT);\n";
				index++;
			}
		}
		
		if(index == 0)	mtmEntriesCode += "\t{0, CIC_V_FALSE, CIC_V_NULL, CIC_V_NULL, CIC_V_NULL, CIC_V_NULL, CIC_V_NULL, CIC_V_NULL, CIC_V_NULL, CIC_V_NULL, CIC_V_NULL, CIC_V_NULL, CIC_V_MUTEX_INIT_INLINE, CIC_V_COND_INIT_INLINE}\n";
		
		code = code.replace("##EXTERN_TASK_FUNCTION_DECLARATION", externalFunctions);
		code = code.replace("##TASK_ENTRIES", taskEntriesCode);
		code = code.replace("##VIRTUAL_TASK_ENTRIES", virtualTaskEntriesCode);
		
		// Added by jhw for multi-mode 
		code = code.replace("##EXTERN_MTM_FUNCTION_DECLARATION", externalMTMFunctions);
		code = code.replace("##MTM_ENTRIES", mtmEntriesCode);
		
		return code;
	}
	
	public static void generateChannelHeader(String mDestFile, String mTemplateFile, Map<Integer, Queue> mQueue, String mThreadVer)
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
			
			outstream.write(translateChannelDataStructure(content, mQueue, mThreadVer).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static int getNextChannelIndex(int index, Map<Integer, Queue> mQueue, HashMap<String, ArrayList<Queue>> history){
		int nextChannelIndex = -1;
		Queue target_queue = mQueue.get(index);
		for(Queue queue: mQueue.values()){
			int next_index = Integer.parseInt(queue.getIndex());
			if(next_index != index){
				if(target_queue.getSrc().equals(queue.getSrc()) && target_queue.getSrcPortId().equals(queue.getSrcPortId())){
					if(history.containsKey(queue.getSrc()+ "_" + queue.getSrcPortId())){
						if(!history.get(queue.getSrc()+ "_" +queue.getSrcPortId()).contains(queue)){
							nextChannelIndex = next_index;
							break;
						}
					}
					else{
						nextChannelIndex = next_index;
						break;
					}
				}
			}
		}

		return nextChannelIndex;
	}
	
	public static String translateChannelDataStructure(String mContent, Map<Integer, Queue> mQueue, String mThreadVer)
	{
		String code = mContent;
		String channelEntriesCode="";
		HashMap<String, ArrayList<Queue>> history = new HashMap<String, ArrayList<Queue>>();

		for(Queue queue: mQueue.values()){
			int nextChannelIndex = getNextChannelIndex(Integer.parseInt(queue.getIndex()), mQueue, history);
			if(nextChannelIndex != -1){
				if(history.containsKey(queue.getSrc() + "_" + queue.getSrcPortId())){
					history.get(queue.getSrc() + "_" + queue.getSrcPortId()).add(queue);
				}
				else{
					ArrayList<Queue> tmp = new ArrayList<Queue>();
					tmp.add(queue);
					history.put(queue.getSrc() + "_" + queue.getSrcPortId(), tmp);
				}
			}
	
			String queueSize = queue.getSize() + "*" + queue.getSampleSize();
			String initData = queue.getInitData() + "*" + queue.getSampleSize();
			String sampleSize = queue.getSampleSize();
			String sampleType = queue.getSampleType();
			if(sampleType=="")	sampleType = "CIC_T_UCHAR";
			
			if(queue.getSampleType() != ""){
				queueSize += "* CIC_SIZEOF(" + queue.getSampleType() + ")";
				sampleSize += "* CIC_SIZEOF(" + queue.getSampleType() + ")";
			}
			
			String type = "";
			if(queue.getTypeName().equals("CHANNEL_TYPE_NORMAL"))				type = "CIC_UT_CHANNEL_TYPE_NORMAL";
			else if(queue.getTypeName().equals("CHANNEL_TYPE_BUFFER"))			type = "CIC_UT_CHANNEL_TYPE_BUFFER";
			else if(queue.getTypeName().equals("CHANNEL_TYPE_ARRAY_CHANNEL"))	type = "CIC_UT_CHANNEL_TYPE_ARRAY_CHANNEL";
	
			channelEntriesCode += "\t{"+ queue.getIndex() +", " + nextChannelIndex +", "
					+ type + ", CIC_V_NULL, CIC_V_NULL, CIC_V_NULL, "
					+ queueSize + ", " + "-1, CIC_V_NULL, CIC_V_NULL, CIC_V_NULL,"+ initData + ", "
					+ sampleSize + ", \"" + sampleType + "\", CIC_V_FALSE, CIC_V_FALSE, " + queue.getSrcPortId()+ ", " 
					+ queue.getDstPortId()+", CIC_V_FALSE, CIC_V_FALSE, CIC_V_FALSE, ";
			if(mThreadVer.equals("m"))	channelEntriesCode += "CIC_V_MUTEX_INIT_INLINE, CIC_V_COND_INIT_INLINE },\n";
			else						channelEntriesCode += "},\n";
		}
		
		code = code.replace("##CHANNEL_ENTRIES", channelEntriesCode);
		
		return code;
	}
	
	public static void generatePortmapHeader(String mDestFile, String mTemplateFile, Map<String, Task> mTask, Map<Integer, Queue> mQueue)
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
			
			outstream.write(translatePortmapDataStructure(content, mTask, mQueue).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static String translatePortmapDataStructure(String mContent, Map<String, Task> mTask, Map<Integer, Queue> mQueue)
	{
		String code = mContent;
		String portmapEntriesCode="";

		for(Queue queue: mQueue.values()){
			if(mTask.get(queue.getSrc()) != null)
				portmapEntriesCode += "\t{" + mTask.get(queue.getSrc()).getIndex() +","
					+ queue.getSrcPortId() +","
					+ "\"" + queue.getSrcPortName() +"\","
					+ queue.getIndex()+ ",'w'},\n";
			if(mTask.get(queue.getDst()) != null)
				portmapEntriesCode += "\t{" + mTask.get(queue.getDst()).getIndex() +","
				+ queue.getDstPortId() +","
				+ "\"" + queue.getDstPortName() +"\","
				+ queue.getIndex()+ ",'r'},\n";
		}
		
		code = code.replace("##PORT_ENTRIES", portmapEntriesCode);
		
		return code;
	}
	
	public static void generateControlHeader(String mDestFile, String mTemplateFile, Map<String, Task> mTask, CICControlType mControl)
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
			
			outstream.write(translateControlDataStructure(content, mTask, mControl).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
		
	public static String translateControlDataStructure(String mContent, Map<String, Task> mTask, CICControlType mControl)
	{
		String code = mContent;
		String controlParamListEntries = "";
		String controlChannelListEntries = "";
		String controlGroupCount = "";
		String controlChannelCount = "";
		
		if(mControl.getControlTasks() != null){
			controlParamListEntries = CommonLibraries.Control.generateControlParamListEntries(mControl, mTask);
			controlChannelListEntries = CommonLibraries.Control.generateControlChannelListEntries(mControl, mTask);
			
			controlGroupCount = "1";
			//if(mControl.getExclusiveControlTasksList() != null)	controlGroupCount = Integer.toString(mControl.getExclusiveControlTasksList().getExclusiveControlTasks().size());
			controlGroupCount = "#define CIC_UV_CONTROL_GROUP_COUNT " + controlGroupCount;
			
			controlChannelCount = "1";
			//if(mControl.getControlTasks() != null)	controlChannelCount = Integer.toString(mControl.getControlTasks().getControlTask().size());
			controlChannelCount = "#define CIC_UV_CONTROL_CHANNEL_COUNT " + controlChannelCount;
		}
		
		code = code.replace("##PARAM_LIST_ENTRIES", controlParamListEntries);
		code = code.replace("##CONTROL_CHANNEL_LIST_ENTRIES", controlChannelListEntries);
		code = code.replace("##CONTROL_GROUP_COUNT", controlGroupCount);
		code = code.replace("##CONTROL_CHANNEL_COUNT", controlChannelCount);
		
		return code;
	}
	
	
	public static void generateCommonCode(String platform, String mOutputPath, String mTranslatorPath, Map<String, Task> mTask, Map<Integer, Queue> mQueue, Map<String, Library> mLibrary, String mThreadVer, CICAlgorithmType mAlgorithm, CICControlType mControl)
	{	
		// Copy cic_comm_apis.h file
		Util.copyFile(mOutputPath+"cic_comm_apis.h", mTranslatorPath + "templates/common/common_template/cic_comm_apis.h");
		
		// Copy cic_control_apis.h file
		Util.copyFile(mOutputPath+"cic_control_apis.h", mTranslatorPath + "templates/common/common_template/cic_control_apis.h");
		
		// Copy cic_control.h
		Util.copyFile(mOutputPath+"cic_basic_control_apis.h", mTranslatorPath + "templates/common/common_template/cic_basic_control_apis.h");
		
		// Copy cic_mtm_apis.h file
		Util.copyFile(mOutputPath+"cic_mtm_apis.h", mTranslatorPath + "templates/common/common_template/cic_mtm_apis.h");
		
		// Copy cic_error.h file
		Util.copyFile(mOutputPath+"cic_error.h", mTranslatorPath + "templates/common/common_template/cic_error.h");
		
		// Copy cic_tasks.h
		Util.copyFile(mOutputPath+"cic_tasks.h", mTranslatorPath + "templates/common/common_template/cic_tasks.h");
		
		// Copy cic_channels.h
		Util.copyFile(mOutputPath+"cic_channels.h", mTranslatorPath + "templates/common/common_template/cic_channels.h");
		
		// Copy cic_portmap.h
		Util.copyFile(mOutputPath+"cic_portmap.h", mTranslatorPath + "templates/common/common_template/cic_portmap.h");
		
		// Copy cic_control.h
		Util.copyFile(mOutputPath+"cic_control.h", mTranslatorPath + "templates/common/common_template/cic_control.h");
		
		// Copy mtm.h file
		Util.copyFile(mOutputPath+"mtm.h", mTranslatorPath + "templates/common/common_template/mtm.h");
		
		// Copy lib_apis.h file
	    Util.copyFile(mOutputPath+"lib_apis.h", mTranslatorPath + "templates/common/common_template/lib_apis.h");
	    
	    // Copy cic_conmap.h
	 	Util.copyFile(mOutputPath+"cic_conmap.h", mTranslatorPath + "templates/common/common_template/cic_conmap.h");

	}
}
