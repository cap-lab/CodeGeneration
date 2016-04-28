package Translators;

import hopes.cic.exception.CICXMLException;
import hopes.cic.xml.*;

import java.io.*;
import java.util.*;
import java.util.regex.*;

import javax.swing.JOptionPane;

import InnerDataStructures.*;
import InnerDataStructures.Communication.BluetoothComm;
import InnerDataStructures.Communication.VRepSharedMemComm;
import InnerDataStructures.Communication.WIFIComm;
import InnerDataStructures.Queue;


public class BuildInnerDataStructures {
	
///////////////////////////////////////////////////// Processor Generation /////////////////////////////////////////////////////
		
	public Map<Integer, Processor> makeProcessors(CICArchitectureType mArchitecture){
		int index = 0;
		Processor processor;
		String os = null, sched = null, processorname = null, poolname = null, arch = null;
		Map<Integer, Processor> processors = new HashMap();
		
		for(ArchitectureElementType element: mArchitecture.getElements().getElement()){
			ArchitectureElementTypeType elementType = null;
			for(ArchitectureElementTypeType t: mArchitecture.getElementTypes().getElementType()){
				if(t.getName().equals(element.getType())){
					elementType = t;
					break;
				}
			}

			if(elementType.getCategory().value().equals("processor")){
				os = elementType.getOS();
				sched = elementType.getScheduler().name();
				processorname = elementType.getModel();
				poolname = element.getName();
				// elmentType.archiType이 optional이어서 GPU가 아닌 경우에는, 해당 element가 없을 수도 있으므로, 아래와 같이 선택적으로 받는다.
				if(elementType.getModel().equalsIgnoreCase("GPU"))	arch = elementType.getArchiType();
				else												arch = "default";
			}
			
			if(elementType.getCategory().value().equals("processor")){
				if(element.getPoolSize().intValue() > 1){
					int t_index = index;
					for(int i=t_index; i<element.getPoolSize().intValue() + t_index; i++){
						processor = new Processor(i, processorname, poolname, i-t_index, os, sched, elementType.getName(), arch);
						processors.put(i, processor);
						index++;
					}
				}
				else{
					processor = new Processor(index, processorname, poolname, 0, os, sched, elementType.getName(), arch);
					processors.put(index, processor);
					index++;
				}
			}
		}
		
		return processors;
	}
	
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	
///////////////////////////////////////////////////// Communication Generation /////////////////////////////////////////////////////
	
	public List<Communication> makeCommunications(CICArchitectureType mArchitecture) 
	{
		Communication communication = null;
		List<Communication> communications = new ArrayList<Communication>();

		for (ArchitectureConnectType connection : mArchitecture.getConnections().getConnection()) 
		{	
			if (connection.getType().equals(ArchitectureConnectionCategoryType.BLUETOOTH))
			{
				for(ArchitectureConnectionBTType BT : connection.getBluetoothconnection())
				{
					communication = new Communication(connection.getType());
					BluetoothComm btcom = communication.getBluetoothComm();
					btcom.setMasterProc(BT.getMaster().getName(), BT.getMaster().getFriendlyName(), BT.getMaster().getMAC());
					for(ArchitectureConnectionBTSlaveType slave: BT.getSlave())
					{
						btcom.addSlaveProc(slave.getName(), slave.getFriendlyName(), slave.getMAC());
					}
					communications.add(communication);
				}					
			}
			else if(connection.getType().equals(ArchitectureConnectionCategoryType.I_2_C_BUS))
			{
				// to do 
			}
			else if(connection.getType().equals(ArchitectureConnectionCategoryType.WIFI))
			{
				for(ArchitectureConnectionWIFIType wifi: connection.getWIFIconnection())
				{
					communication = new Communication(connection.getType());
					WIFIComm wfcom = communication.getWifiComm();
					
					wfcom.setServerProc(wifi.getServer().getName(), wifi.getServer().getIp());
					for(ArchitectureConnectionWIFIclientType client : wifi.getClient())
					{
						wfcom.addClientProc(client.getName());
					}
					
					communications.add(communication);
				}	
			}
			else if(connection.getType().equals(ArchitectureConnectionCategoryType.V_REP_SHARED_BUS))
			{
				for(ArchitectureConnectionVRepSBType shm: connection.getVRepSharedBusconnection())
				{
					communication = new Communication(connection.getType());
					VRepSharedMemComm shmcom = communication.getVRepSharedMemComm();
					
					shmcom.setMasterProc(shm.getMaster().getName());
					shmcom.setKey(shm.getKey().intValue());
					for(ArchitectureConnectionVRepSBMemberType slave : shm.getSlave())
					{
						shmcom.addSlaveProc(slave.getName());
					}
					
					communications.add(communication);
				}	
			}
			
		}		
		
		return communications;
	}

	// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////// Connected Task Graph Generation ////////////////////////////////////////////////////
	public List<List<Task>> findSDFTaskSet(Map<String, Task> mTask, List<Task> connected_task_graph){
		List<List<Task>> taskSet = new ArrayList<List<Task>>();
		Map<String, Task> taskGraph = new HashMap<String, Task>();

		for(Task t: connected_task_graph)	taskGraph.put(t.getName(), t);
		
		while(true){			
			List<Task> currSet = null;	
			int end_flag = 1;
			String currTaskName = "";
			if(taskGraph.size() == 0)		break;
			for(Task currTask: taskGraph.values()){
				if(currTask.getPortList().size() > 0){
					for(Map<String, Integer> port: currTask.getPortList().values()){
						if(port.size() > 0){
							currSet = new ArrayList<Task>();
							currSet.add(currTask);
							currTaskName = currTask.getName();
							end_flag = 0;
							break;
						}
					}
				}
			}
			if(end_flag == 1) break;
			taskGraph.remove(currTaskName);

			int currPointer = 0;
			while(true){
				Task currSetTask = currSet.get(currPointer);
				for(int i = 0; i < currSetTask.getQueue().size(); i++){
					Queue currQueue = currSetTask.getQueue().get(i);
					if(currQueue.getSrc().equals(currSetTask.getName()) && !currQueue.getDst().equals(currSetTask.getName()) 
							&& taskGraph.get(currQueue.getDst()) != null && !currSet.contains(taskGraph.get(currQueue.getDst()))){
						if(currSetTask.getPortList().get(currQueue.getSrcPortName()).size() > 0){
							currSet.add(taskGraph.get(currQueue.getDst()));
							taskGraph.remove(currQueue.getDst());
						}
					}
					else if(currQueue.getDst().equals(currSetTask.getName()) && !currQueue.getSrc().equals(currSetTask.getName()) 
							&& taskGraph.get(currQueue.getSrc()) != null && !currSet.contains(taskGraph.get(currQueue.getSrc()))){
						if(currSetTask.getPortList().get(currQueue.getDstPortName()).size() > 0){
							currSet.add(taskGraph.get(currQueue.getSrc()));
							taskGraph.remove(currQueue.getSrc());
						}
					}
				}
				currPointer = currPointer + 1;
				if(currPointer == currSet.size()){
					if(currSet.size() > 1)	taskSet.add(currSet);
					break;
				}
			}
		}

		return taskSet;
	}
	
	public Map<Integer, List<Task>> findConnectedTaskGraph(Map<String, Task> mTask){
		int total_set_count = 0;
		Map<String, Task> history = new HashMap<String, Task>();
	    Map<Integer, List<Task>> connected_task_graph = new HashMap<Integer, List<Task>>();
		
	    for(Task t: mTask.values())	if(t.getName().equals(t.getParentTask())) history.put(t.getName(), t);
	    	
		while(true){
			if(history.size() == 0)		break;
			Task currTask = history.values().iterator().next();
			history.remove(currTask.getName());
			List<Task> currSet = new ArrayList<Task>();
			currSet.add(currTask);

			int currPointer = 0;
			while(true){
				Task currSetTask = currSet.get(currPointer);
				for(int i = 0; i < currSetTask.getQueue().size(); i++){
					Queue currQueue = currSetTask.getQueue().get(i);
					if(currQueue.getSrc().equals(currSetTask.getName()) && !currQueue.getDst().equals(currSetTask.getName()) && history.get(currQueue.getDst()) != null && !currSet.contains(history.get(currQueue.getDst()))){					
						currSet.add(history.get(currQueue.getDst()));
						history.remove(currQueue.getDst());
					}
					else if(currQueue.getDst().equals(currSetTask.getName()) && !currQueue.getSrc().equals(currSetTask.getName()) && history.get(currQueue.getSrc()) != null && !currSet.contains(history.get(currQueue.getSrc()))){				
						currSet.add(history.get(currQueue.getSrc()));
						history.remove(currQueue.getSrc());
					}
				}
				currPointer = currPointer + 1;
				if(currPointer == currSet.size()){
					connected_task_graph.put(total_set_count, currSet);
					total_set_count = total_set_count + 1;
					break;
				}
			}
		}
		return connected_task_graph;
	}
	
	public Map<String, Task> removeParentTask(Map<String, Task> mTask){
		Map<String, Task> modifiedTask = new HashMap<String, Task>();
		
		int index = 0;
		for(Task t: mTask.values()){
			if(t.getHasSubgraph().equalsIgnoreCase("No")){
				t.setIndex(index);
				modifiedTask.put(t.getName(), t);
				index++;
			}
		}
		
		return modifiedTask;
	}
	
	public Map<String, Task> addProcessorVirtualTask(Map<String, Task> mTask, Map<Integer, Queue> mQueue, 
			Map<Integer, Processor> mProcessor, Map<Integer, List<Task>> mConnectedTaskGraph, Map<Integer, 
			List<List<Task>>> mConnectedSDFTaskSet, String TaskGraphProperty, Map<String, Task> mVTask, String mOutputPath){
		Map<String, Task> mNewTask = new HashMap<String, Task>();
		ArrayList<Task> parentTaskList = new ArrayList<Task>();
		String schedFilePath = mOutputPath + "/convertedSDF3xml/";
		
		int index = mTask.size() + mVTask.size();

		// load schedule.xml for each mode of all tasks which have sub task graph
		for(Task task: mTask.values()){
			if(task.getName().equals(task.getParentTask()) && task.getHasSubgraph().equals("Yes")){
				parentTaskList.add(task);
			}
		}
		
		for(Task task: mVTask.values()){
			parentTaskList.add(task);
		}
		
		// get processor list that one more tasks are mapped (for top-level parent task)
		for(Task task: parentTaskList){
			ArrayList<Integer> usedProcList = new ArrayList<Integer>();
			ArrayList<String> modeList = new ArrayList<String>();
			if(task.getHasMTM().equals("Yes")){
				for(String mode: task.getMTM().getModes()){
					modeList.add(mode);
				}
			}
			else	modeList.add("Default");
			
			// for each mode
			for(String mode: modeList){
				ArrayList<File> schedFileList = new ArrayList<File>();
				File file = new File(schedFilePath);
				File[] fileList = file.listFiles();
				for(File f: fileList){
					if(f.getName().contains(task.getName() + "_" + mode) && f.getName().endsWith("_schedule.xml")){
						schedFileList.add(f);
					}
				}
				
				// for each processor number
				for(File schedFile: schedFileList){
					CICScheduleTypeLoader loaderSched;
					CICScheduleType schedule = null;
					loaderSched = new CICScheduleTypeLoader();
					try {
						schedule = loaderSched.loadResource(schedFile.getAbsolutePath());
					} catch (CICXMLException e) {
						// Auto-generated catch block
						e.printStackTrace();
					}
					for(ScheduleGroupType schedGroup: schedule.getTaskGroups().getTaskGroup().get(0).getScheduleGroup()){
						int procId = 0;
						for(Processor proc: mProcessor.values()){
							if(proc.getPoolName().equals(schedGroup.getPoolName()) && proc.getLocalIndex() == schedGroup.getLocalId().intValue()){
								procId = proc.getIndex();
								break;
							}
						}
						if(!usedProcList.contains(procId))	usedProcList.add(procId);
					}
				}
			}
			for(int procId: usedProcList){
				Task virtualTask = new Task(index, task.getName() + "_proc_" + procId, task.getName() + "_proc_" + procId, Integer.parseInt(task.getRunRate()), task.getPeriodMetric(), task.getRunCondition(), Integer.parseInt(task.getPeriod()));
				Map<String, Map<String, List<Integer>>> plmapmap = new HashMap<String, Map<String, List<Integer>>>();
				Map<String, List<Integer>> plmap = new HashMap<String, List<Integer>>();
				List<Integer> pl = new ArrayList<Integer>();
				pl.add(procId);
				plmap.put("Default", pl);
				plmapmap.put("Default", plmap);
				virtualTask.setProc(plmapmap);
				virtualTask.setParentTask(task.getName());
				mNewTask.put(virtualTask.getName(), virtualTask);
				index++;
			}
		}
		
		return mNewTask;
	}
	
	public Map<String, Task> makeVirtualTask(Map<String, Task> mTask, Map<Integer, Queue> mQueue, Map<Integer, List<List<Task>>> mConnectedSDFTaskSet, String TaskGraphProperty, int mFuncSimPeriod, String mFuncSimPeriodMetric){
		Map<String, Task> mNewTask = new HashMap<String, Task>();
		
		int index = mTask.size();

		// Make virtual tasks for sdf graphs in a top-level task graph
		for(List<List<Task>> taskList: mConnectedSDFTaskSet.values()){
			for(int i=0; i<taskList.size(); i++){
				Boolean isSrc = true;
				int period = 0;
				String periodMetric = "";
				int runRate = 0;
				String drivenType = "DATA_DRIVEN";
				
				List<Task> subTaskList = taskList.get(i);
				for(int j=0; j<subTaskList.size(); j++){
					isSrc = true;
					for(Queue taskQueue: subTaskList.get(j).getQueue()){
						if(taskQueue.getDst().equals(subTaskList.get(j).getName()) && Integer.parseInt(taskQueue.getInitData()) == 0){
							isSrc = false;
							period = Integer.parseInt(subTaskList.get(j).getPeriod());
							periodMetric = subTaskList.get(j).getPeriodMetric();
							runRate = Integer.parseInt(subTaskList.get(j).getRunRate());
							break;
						}
					}
					if(TaskGraphProperty.equals("DataFlow") && isSrc == true){
						drivenType = "TIME_DRIVEN";
						period = 1;
						periodMetric = mFuncSimPeriodMetric;
						runRate = 1;
					}
					else if(TaskGraphProperty.equals("Hybrid") && isSrc == true && subTaskList.get(j).getRunCondition().equals("TIME_DRIVEN")){
						drivenType = "TIME_DRIVEN";
						Map<String, Task> mConnectedTasks = new HashMap<String, Task>();
						Map<Integer, Queue> mConnectedQueues = new HashMap<Integer, Queue>();
						for(Task i_t: subTaskList){
							mConnectedTasks.put(i_t.getName(), i_t);
						}
						int c_id = 0;
						for(Queue i_q: mQueue.values()){
							for(Task i_t: mConnectedTasks.values()){
								if(i_q.getDst().equals(i_t.getName())){
									mConnectedQueues.put(c_id, i_q);
									c_id++;
									break;
								}
							}
						}
						Map<String, Integer> tr = CommonLibraries.Schedule.generateIterationCount(null, "Default", mConnectedTasks, mQueue);
						period = Integer.parseInt(subTaskList.get(j).getPeriod());
						periodMetric = subTaskList.get(j).getPeriodMetric();
						runRate = tr.get(subTaskList.get(j).getName());
						break;
					}	
				}
				
				Task virtualTask = new Task(index, "SDF_" + index, "SDF_" + index, runRate, periodMetric, drivenType, period);
				//System.out.println(virtualTask.getName());
				mNewTask.put(virtualTask.getName(), virtualTask);
				index = index + 1;
			}
		}
		
		modifyTaskStructure_parentTask(mTask, mConnectedSDFTaskSet);
	
		return mNewTask;
	}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
//////////////////////////////////////////////////////// Task Generation /////////////////////////////////////////////////////////
	private void fillMappingFromManualMapping(Map<String, Task> tasks, Map<Integer, Processor> processors, CICMappingType mapping){
		for(Task task: tasks.values()){
			List<MappingProcessorIdType> taskMaps = new ArrayList();
			List<Integer> procList = new ArrayList();
			Map<String, List<Integer>> procListMap = new HashMap<String, List<Integer>>();
			Map<String, Map<String, List<Integer>>> procListMapMap = new HashMap<String, Map<String, List<Integer>>>();
			Processor proc_result = null;
			
			for(MappingTaskType mtask: mapping.getTask()){
				if(mtask.getName().equalsIgnoreCase(task.getName())){
					taskMaps = (ArrayList<MappingProcessorIdType>) mtask.getProcessor();
					break;
				}
			}
			
			for(MappingProcessorIdType taskMap: taskMaps){
				for(Processor tproc: processors.values()){
					if(tproc.getPoolName().equals(taskMap.getPool()) && tproc.getLocalIndex() == taskMap.getLocalId().intValue()){
						procList.add(tproc.getIndex());
						break;
					}
				}
			}
			procListMap.put("Default", procList);
			procListMapMap.put("Default", procListMap);
			task.setProc(procListMapMap);
		}
	}
	
	private boolean fillMappingFromHierarchicalGraph(Map<String, Task> tasks, Map<Integer, Processor> processors, String outputPath)
	{
		boolean done = false;
		CICScheduleType schedule = null;
		CICScheduleTypeLoader scheduleLoader = new CICScheduleTypeLoader();
		
		//hierarchical graph 
		for(Task task: tasks.values()){
			if(task.getHasMTM().equals("Yes") && task.getHasSubgraph().equals("Yes")){
				// initialize
				//Map structure: <parent task name, <mode, <num_proc(schedule_id), processor id>>>
				Map<String, Map<String, Map<String, List<Integer>>>> taskMapForModeForNumProc = new HashMap<String, Map<String, Map<String, List<Integer>>>>();
				for(Task t: tasks.values()){
					if(t.getParentTask().equals(task.getName()) && t.getHasSubgraph().equals("No") ){
						Map<String, Map<String, List<Integer>>> taskMapForNumProc = new HashMap<String, Map<String, List<Integer>>> ();
						taskMapForModeForNumProc.put(t.getName(), taskMapForNumProc);
					}
				}
				
				// extract the processor information from schedule.xml
				for(String mode: task.getMTM().getModes()){					
					ArrayList<File> schedFileList = new ArrayList<File>();
					File file = new File(outputPath + "/convertedSDF3xml");
					File[] fileList = file.listFiles();
					for(File f: fileList){
						if(f.getName().contains(task.getParentTask() + "_" + mode) && f.getName().endsWith("_schedule.xml")){
							schedFileList.add(f);
						}
					}
					if(schedFileList.size() <= 0){
						JOptionPane.showMessageDialog(null, "You should execute 'Analysis' before build!");
						System.exit(-1);
					}
					
					// initialize
					for(Task t: tasks.values()){					
						if(t.getParentTask().equals(task.getName()) && t.getHasSubgraph().equals("No") ){
							Map<String, List<Integer>> taskProcMap = new HashMap<String, List<Integer>>();
							taskMapForModeForNumProc.get(t.getName()).put(mode, taskProcMap);
						}
					}

					for(int f_i=0; f_i < schedFileList.size(); f_i++){
						try {
							schedule = scheduleLoader.loadResource(schedFileList.get(f_i).getAbsolutePath());
						} catch (CICXMLException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
						String temp = schedFileList.get(f_i).getName().replace(task.getParentTask() + "_" + mode + "_", "");
						temp = temp.replace("_schedule.xml", "");
						int num_proc = Integer.parseInt(temp);
						// initialize
						for(Task t: tasks.values()){
							if(t.getParentTask().equals(task.getName()) && t.getHasSubgraph().equals("No") ){
								List<Integer> procList = new ArrayList<Integer>();
								taskMapForModeForNumProc.get(t.getName()).get(mode).put(Integer.toString(num_proc), procList);
							}
						}
							
						TaskGroupsType taskGroups = schedule.getTaskGroups();
						List<TaskGroupForScheduleType> taskGroupList = taskGroups.getTaskGroup();
						for(int i=0; i<taskGroupList.size();i++){
							List<ScheduleGroupType> schedGroup = taskGroupList.get(i).getScheduleGroup();
							for(int j=0; j<schedGroup.size(); j++){
								int proc_id = 0;
								for(Processor proc: processors.values()){
									if(proc.getPoolName().equals(schedGroup.get(j).getPoolName()) 
											&& proc.getLocalIndex() == schedGroup.get(j).getLocalId().intValue()){
										proc_id = proc.getIndex();
										break;
									}
								}
								List<ScheduleElementType> scheds = schedGroup.get(j).getScheduleElement();
								for(int k=0; k<scheds.size(); k++){
									ScheduleElementType sched = scheds.get(k);
									String taskName = sched.getTask().getName();
									if(!taskMapForModeForNumProc.get(taskName).get(mode).get(Integer.toString(num_proc)).contains(proc_id))
										taskMapForModeForNumProc.get(taskName).get(mode).get(Integer.toString(num_proc)).add(proc_id);
								}
							}
						}
						
						//System.out.println(taskMapForModeForNumProc);
					}
				}
				
				for(Task t: tasks.values()){
					if(t.getParentTask().equals(task.getName()) && t.getHasSubgraph().equals("No") ){
						Map<String, Map<String, List<Integer>>> before = taskMapForModeForNumProc.get(t.getName());
						Map<String, Map<String, List<Integer>>> after = new HashMap<String, Map<String, List<Integer>>>();
						
						List<String> modeList = task.getMTM().getModes();
						List<String> procNumList = new ArrayList<String> ();
						
						Set<String> pnl = before.get(modeList.get(0)).keySet();
						for(String procNum: pnl){
							procNumList.add(procNum);
						}
						
						for(String procNum: procNumList){
							Map<String, List<Integer>> modeMapList = new HashMap<String, List<Integer>>();
							for(String mode: modeList){
								List<Integer> procList = before.get(mode).get(procNum);
								modeMapList.put(mode, procList);
							}
							after.put(procNum, modeMapList);
						}
						t.setProc(after);
						done = true;
						//System.out.println("-- " + t.getName() + ": " + t.getProc());
					}
				}
			}
		}
		return done;
	}
	
	private void fillMappingFromFlatGraph(Map<String, Task> tasks, Map<String, Task> vTasks, Map<Integer, Processor> processors, String outputPath)
	{
		CICScheduleType schedule = null;
		CICScheduleTypeLoader scheduleLoader = new CICScheduleTypeLoader();
		
		List<Task> targetVtask = new ArrayList<Task>();
		for(Task vTask: vTasks.values())
		{
			for(Task task: tasks.values())
			{
				if(task.getParentTask().equals(vTask.getName()))
				{
					targetVtask.add(vTask);
					break;
				}
			}
		}
				
		for(Task vTask: targetVtask){	
			// initialize
			//Map structure: <parent task name, <mode, <num_proc(schedule_id), processor id>>>
			Map<String, Map<String, Map<String, List<Integer>>>> taskMapForModeForNumProc = new HashMap<String, Map<String, Map<String, List<Integer>>>>();
			for(Task t: tasks.values()){
				if(t.getParentTask().equals(vTask.getName()) && t.getHasSubgraph().equals("No") ){
					Map<String, Map<String, List<Integer>>> taskMapForNumProc = new HashMap<String, Map<String, List<Integer>>> ();
					taskMapForModeForNumProc.put(t.getName(), taskMapForNumProc);
				}
			}
			
			// extract the processor information from schedule.xml
			List<String> modeList = new ArrayList<String>();
			if (vTask.getMTM() != null)
				modeList = vTask.getMTM().getModes();
			else
				modeList.add("Default");
			
			for(String mode: modeList){					
				ArrayList<File> schedFileList = new ArrayList<File>();
				File file = new File(outputPath + "/convertedSDF3xml");
				File[] fileList = file.listFiles();
				for(File f: fileList){
					if(f.getName().contains(vTask.getParentTask() + "_" + mode) && f.getName().endsWith("_schedule.xml")){
						schedFileList.add(f);
					}
				}
				if(schedFileList.size() <= 0){
					JOptionPane.showMessageDialog(null, "You should execute 'Analysis' before build!");
					System.exit(-1);
				}
				
				// initialize
				for(Task t: tasks.values()){					
					if(t.getParentTask().equals(vTask.getName()) && t.getHasSubgraph().equals("No") ){
						Map<String, List<Integer>> taskProcMap = new HashMap<String, List<Integer>>();
						taskMapForModeForNumProc.get(t.getName()).put(mode, taskProcMap);
					}
				}

				for(int f_i=0; f_i < schedFileList.size(); f_i++){
					try {
						schedule = scheduleLoader.loadResource(schedFileList.get(f_i).getAbsolutePath());
					} catch (CICXMLException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
					String temp = schedFileList.get(f_i).getName().replace(vTask.getParentTask() + "_" + mode + "_", "");
					temp = temp.replace("_schedule.xml", "");
					int num_proc = Integer.parseInt(temp);
					// initialize
					for(Task t: tasks.values()){
						if(t.getParentTask().equals(vTask.getName()) && t.getHasSubgraph().equals("No") ){
							List<Integer> procList = new ArrayList<Integer>();
							taskMapForModeForNumProc.get(t.getName()).get(mode).put(Integer.toString(num_proc), procList);
						}
					}
						
					TaskGroupsType taskGroups = schedule.getTaskGroups();
					List<TaskGroupForScheduleType> taskGroupList = taskGroups.getTaskGroup();
					for(int i=0; i<taskGroupList.size();i++){
						List<ScheduleGroupType> schedGroup = taskGroupList.get(i).getScheduleGroup();
						for(int j=0; j<schedGroup.size(); j++){
							int proc_id = 0;
							for(Processor proc: processors.values()){
								if(proc.getPoolName().equals(schedGroup.get(j).getPoolName()) 
										&& proc.getLocalIndex() == schedGroup.get(j).getLocalId().intValue()){
									proc_id = proc.getIndex();
									break;
								}
							}
							List<ScheduleElementType> scheds = schedGroup.get(j).getScheduleElement();
							for(int k=0; k<scheds.size(); k++){
								ScheduleElementType sched = scheds.get(k);
								String taskName = sched.getTask().getName();
								if(!taskMapForModeForNumProc.get(taskName).get(mode).get(Integer.toString(num_proc)).contains(proc_id))
									taskMapForModeForNumProc.get(taskName).get(mode).get(Integer.toString(num_proc)).add(proc_id);
							}
						}
					}
					
//						System.out.println(taskMapForModeForNumProc);
				}
			}
			
			for(Task t: tasks.values()){
				if(t.getParentTask().equals(vTask.getName()) && t.getHasSubgraph().equals("No") ){
					Map<String, Map<String, List<Integer>>> before = taskMapForModeForNumProc.get(t.getName());
					Map<String, Map<String, List<Integer>>> after = new HashMap<String, Map<String, List<Integer>>>();
					
					List<String> procNumList = new ArrayList<String> ();
					
					Set<String> pnl = before.get(modeList.get(0)).keySet();
					for(String procNum: pnl){
						procNumList.add(procNum);
					}
					
					for(String procNum: procNumList){
						Map<String, List<Integer>> modeMapList = new HashMap<String, List<Integer>>();
						for(String mode: modeList){
							List<Integer> procList = before.get(mode).get(procNum);
							modeMapList.put(mode, procList);
						}
						after.put(procNum, modeMapList);
					}
					t.setProc(after);
//						System.out.println("-- " + t.getName() + ": " + t.getProc());
				}
			}
		}
	}

	
	private void fillMappingFromAutomaticMapping(Map<String, Task> tasks, Map<String, Task> vTasks, Map<Integer, Processor> processors, String outputPath){
		boolean result = fillMappingFromHierarchicalGraph(tasks, processors, outputPath);
		if(!result && vTasks.size() > 0)
			fillMappingFromFlatGraph(tasks, vTasks, processors, outputPath);
		
	}
	
	public void fillMappingForTask(Map<String, Task> tasks, Map<String, Task> vTasks, Map<Integer, Processor> processors, CICMappingType mapping, String outputPath, String graphType){
		
		fillMappingFromManualMapping(tasks, processors, mapping);
		if(!graphType.equals("ProcessNetwork")){
			fillMappingFromAutomaticMapping(tasks, vTasks, processors, outputPath);
		}	
	}
		
	
	public Map<String, Task> makeTasks(String mOutputPath, CICAlgorithmType mAlgorithm){
		int index=0;
		Map<String, Task> tasks = new HashMap<String, Task>();
		Task task_result = null;

		for(TaskType task: mAlgorithm.getTasks().getTask()){	
			String name = task.getName();
			
			String dataParallelType;
			TaskDataParallelType dataParallel = task.getDataParallel();
			if(dataParallel == null)									dataParallelType = "NONE";
			else{
				dataParallelType = dataParallel.getType().value();
				if(dataParallelType.equalsIgnoreCase("none"))			dataParallelType = "NONE";
				else if(dataParallelType.equalsIgnoreCase("loop"))		dataParallelType = "OPENMP";
				else if(dataParallelType.equalsIgnoreCase("wavefront"))	dataParallelType = "WAVEFRONT";
			}
			
			int width = 0;
			int height = 0;
			
			List<VectorType> dependencyList = new ArrayList();
			List<Integer> feedbackList = new ArrayList();
			
			// 현재는 Wavefront parallelism을 사용하는 경우가 없다.  
			// 예전에 Cell이나 HSim에서 Static하게 스케줄할 때 사용되었는데, 지금은 Array channel을 통해 dynamic하게 하므로 사용되지 않고 있다.
			if(dataParallelType == "WAVEFRONT"){
				if(!dataParallel.getVolume().getValue().isEmpty()){
					width = dataParallel.getVolume().getValue().get(0).intValue();
					height = dataParallel.getVolume().getValue().get(1).intValue();
					
					for(int j=0; j<dataParallel.getDependencyVector().getVector().size(); j++){
						dependencyList.add(dataParallel.getDependencyVector().getVector().get(j));
					}
					
					int port_index=0;
					for(int j=0; j<task.getPort().size(); j++){
						if(task.getPort().get(j).isIsFeedback())	feedbackList.add(port_index);
						port_index++;
					}
				}
			}
			
			// cic file of Task
			String cicfile = task.getFile().toString();
			
			String cflag;
			if(task.getCflags() == null)	cflag = "";
			else							cflag = task.getCflags();

			String ldflag;
			if(task.getLdflags() == null)	ldflag = "";
			else							ldflag = task.getLdflags();
			
			String runCondition;
			if(task.getRunCondition() != null)	runCondition = task.getRunCondition().name();
			else								runCondition = "run-once";
			
			String hasMTM = task.getHasMTM();
			String hasSubgraph = task.getHasSubGraph();
			String parentTask = task.getParentTask();
			String taskType = task.getTaskType();
			
			// Each port has mode and rate information(to support multi-rate)
			Map<String, Map<String, Integer>> inPortList = new HashMap<String, Map<String, Integer>>();
			Map<String, Map<String, Integer>> outPortList = new HashMap<String, Map<String, Integer>>();
			for(TaskPortType port: task.getPort()){
				String portName = port.getName();
				Map<String, Integer> portElem = new HashMap<String, Integer>();
				for(TaskRateType taskRate: port.getRate()){
					String mode = taskRate.getMode();
					int rate = taskRate.getRate().intValue();
					portElem.put(mode, rate);
				}
				if(port.getDirection().equals(PortDirectionType.INPUT))			inPortList.put(portName, portElem);
				else if(port.getDirection().equals(PortDirectionType.OUTPUT))	outPortList.put(portName, portElem);
			}
			
			boolean isSrcTask = false;
			if(inPortList.size() == 0)	isSrcTask = true;
			
	        MTM mtmInfo = null;
			if(hasMTM.equalsIgnoreCase("Yes")){
		       	List<String> modes = new ArrayList<String> ();
		       	List<Variable> variables = new ArrayList<Variable> ();
		       	List<Transition> transitions = new ArrayList<Transition> ();
		       	
		       	for(MTMModeType mode: task.getMtm().getModeList().getMode()){
		       		modes.add(mode.getName());
		       	}
		       	if(task.getMtm().getVariableList() != null){
			       	for(MTMVariableType var: task.getMtm().getVariableList().getVariable()){
			       		String vartype = var.getType();
			       		String varname = var.getName();
			       		Variable variable = new Variable(vartype, varname);
			       		variables.add(variable);
			       	}
		       	}
		       	if(task.getMtm().getTransitionList() != null){
			       	for(MTMTransitionType trans: task.getMtm().getTransitionList().getTransition()){
			       		String srcmode = trans.getSrcMode();
			       		String dstmode = trans.getDstMode();
			       		String transname = trans.getName();
			       		List<Condition> conditions = new ArrayList<Condition> ();
			       		for(MTMConditionType cond: trans.getConditionList().getCondition()){
			       			String condvar = cond.getVariable();
			       			String condval = cond.getValue();
			       			String condcomp = cond.getComparator();
			       			Condition condition = new Condition(condvar, condval, condcomp);
			       			conditions.add(condition);
			       		}
			       		Transition transition = new Transition(transname, srcmode, dstmode, conditions);
			       		transitions.add(transition);
			       	}
		       	}
		       	
		       	mtmInfo = new MTM(modes, variables, transitions);	
			}
			else	mtmInfo = new MTM();

			// create new Task
			task_result = new Task(index, name, cicfile, null, cflag, ldflag, dataParallelType
					, width, height, dependencyList, feedbackList, runCondition, task.getExtraHeader()
					, task.getExtraSource(), task.getLibraryMasterPort(), task.getParameter(), hasSubgraph
					, hasMTM, mtmInfo, parentTask, taskType, inPortList, outPortList, isSrcTask);
			index++;
						
			tasks.put(name, task_result);
		}
		
		ModeType mode = mAlgorithm.getModes().getMode().get(0);
		for(ModeTaskType mtask: mode.getTask()){
			if(mtask.getPeriod().getValue().intValue() != 0){
				int period = mtask.getPeriod().getValue().intValue();
				String periodMetric = mtask.getPeriod().getMetric().toString();
				/*
				if(mtask.getPeriod().getMetric().toString().equalsIgnoreCase("h"))			period = period * 3600 * 1000 * 1000;
				else if(mtask.getPeriod().getMetric().toString().equalsIgnoreCase("m"))		period = period * 60 * 1000 * 1000;
				else if(mtask.getPeriod().getMetric().toString().equalsIgnoreCase("s"))		period = period * 1000 * 1000;
				else if(mtask.getPeriod().getMetric().toString().equalsIgnoreCase("ms"))	period = period * 1000;
				else if(mtask.getPeriod().getMetric().toString().equalsIgnoreCase("us"))	period = period * 1;
				else{
					System.out.println("[makeTasks] Not supported metric of period");
					System.exit(-1);
				}
				*/
				tasks.get(mtask.getName()).setPeriod(period);
				tasks.get(mtask.getName()).setPeriodMetric(periodMetric);
			}
			else{
				tasks.get(mtask.getName()).setPeriod(1);
				tasks.get(mtask.getName()).setPeriodMetric("us");
			}
			
			// Deadline
			if(mtask.getDeadline() != null){
				int deadline = mtask.getDeadline().getValue().intValue();
				tasks.get(mtask.getName()).setDeadline(deadline);
			}
			//Priority
			if(mtask.getPriority() != null){
				int priority = mtask.getPriority().intValue();
				tasks.get(mtask.getName()).setPriority(priority);
			}
			// RunRate
			int runRate = 0;
			if(mtask.getRunRate() != null)	{
				runRate = mtask.getRunRate().intValue();
			}
			else							runRate = 1;	//Using xml, runRate is null -> runRate = 1
			tasks.get(mtask.getName()).setRunRate(runRate);			
		}
		
		return tasks;
	}
	
	public void modifyTaskStructure_runRate(Map<String, Task> mTask, Map<String, Task> mVTask, Map<Integer, Queue> mQueue)
	{
		//[CODE_REVIEW]: hshong(4/21): need to check: functional sim & multi-thread when the graph is flat.
		//Assume update runrate when the graph is SDF
		Task parentTask = null;
		String mode = null;
		for(Task t: mTask.values()){
			if(t.getHasMTM().equalsIgnoreCase("Yes")){ //first, find the parent task 
				if(t.getMTM() != null && t.getMTM().getModes().size() == 1){ //When the graph is SDF, mode has only one mode
					parentTask = t; 
					mode = t.getMTM().getModes().get(0);
					
					Map<String, Integer> tr = CommonLibraries.Schedule.generateIterationCount(parentTask, mode, mTask, mQueue);
					for(Task i_t: mTask.values())
					{
						if(!i_t.getParentTask().equals(i_t.getName())){
							int rate = 0;
							if(tr.get(i_t.getName()) != null)
								rate = tr.get(i_t.getName());
							if(rate != 0)
								i_t.setRunRate(rate); 
						}
								
					}
				}					
			}
			else {
				if(mVTask.size() > 0){ //case of flat graph
					Map<String, Integer> tr = CommonLibraries.Schedule.generateIterationCount(null, "Default", mTask, mQueue);
					for(Task i_t: mTask.values())
					{
						if(!i_t.getParentTask().equals(i_t.getName())){
							int rate = 0;
							if(tr.get(i_t.getName()) != null)
								rate = tr.get(i_t.getName());
							if(rate != 0)
								i_t.setRunRate(rate); 
						}
								
					}
				}
			}
		}
		
	}
	
	
	public void modifyTaskStructure_parentTask(Map<String, Task> mTask, Map<Integer, List<List<Task>>> mConnectedSDFTaskSet)
	{
		//change parent task to virtual parent task
		int index = mTask.size();
		for(List<List<Task>> taskListOut: mConnectedSDFTaskSet.values()){
			for(int i=0; i<taskListOut.size(); i++){
				List<Task> taskListIn = taskListOut.get(i);
				for(int j=0; j<taskListIn.size(); j++){
					for(Task t: mTask.values()){
						if(taskListIn.get(j).getName().equals(t.getName())){
							t.setParentTask("SDF_" + index);
							break;
						}
					}
				}
				index = index + 1;
			}
		}
	}
			

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	
	
////////////////////////////////////////////////////// Library Generation //////////////////////////////////////////////////////
	
	public List<Argument> fillArgumentList(String libraryName, String functionName, String argumentList){
		List<Argument> argumentListResult = new ArrayList<Argument>();
		
		String[] arguments = argumentList.split(",");
		int count = 0;
		
		for(String argument: arguments){
			String type = null;
			String variableName = null;
			
			argument = argument.replaceAll("\\s+", " ");
			argument = argument.replaceAll("(^\\s|$\\s)", "");
			String[] splittedArguments = argument.split("\\s");
			
			if(splittedArguments[0].equals("struct")){
					type = splittedArguments[0] + " " + splittedArguments[1];
					variableName = splittedArguments[2];
			}
			else if(splittedArguments[0].equals("void"))	continue;
			else{
				type = splittedArguments[0];
				variableName = splittedArguments[1];
			}
			
			Argument argumentResult = new Argument(count, libraryName, functionName, type, variableName);
			argumentListResult.add(argumentResult);
			count++;
		}
		
		return argumentListResult;
	}

	public List<Function> fillFunctionList(LibraryType library, String cicfilename, String outputpath) throws IOException{
		List<Function> functionList = new ArrayList<Function>();
		
		Pattern functionTag = Pattern.compile("LIBFUNC\\s*" +
				"\\(\\s*" + 
				"([a-zA-z]\\w*\\s*)" + 
				"(,\\s*[a-zA-z_]\\w*\\s*,\\s*)" + 
				"(" +
				 "(\\s*void\\s*)|" +
				  "(" +
				   "(\\s*struct\\s+[a-zA-Z_]\\w*\\s+[a-zA-Z_]\\w*\\s*,?)|" +
				   "(\\s*[a-zA-Z_]\\w*\\s+[a-zA-Z_]\\w*\\s*,?)" +
				  ")+" +
				")" +
				"\\)\\s*"
		);

		File f = new File(outputpath + "//" + library.getHeader());
		BufferedReader headerFile = null;

		if(!f.exists())	return null;
		else 			headerFile = new BufferedReader(new FileReader(outputpath + "//" + library.getHeader()));

		String libraryName = library.getName();
		int count = 0;
		String line = null;
		
		while((line = headerFile.readLine()) != null){
			Matcher functionStrings = functionTag.matcher(line);
			while(functionStrings.find()){
				String functionName = functionStrings.group(2).replaceAll("(\\s|,)", "");
				if(!functionName.equals("init") && !functionName.equals("wrapup")){
					String returnType = functionStrings.group(1).replaceAll("(\\s|,)", "");
					List<Argument> argumentList = fillArgumentList(libraryName, functionName, functionStrings.group(3));
					Function functionResult = new Function(count, libraryName, functionName, returnType, argumentList);
					functionList.add(functionResult);
					count++;
				}
			}
		}
		
		headerFile.close();
		
		return functionList;
	}
	
	// Need to fix about stub list
	public Map<String, Library> fillLibraryMapping(CICAlgorithmType mAlgorithm, CICMappingType mMapping, Map<Integer, Processor> processors, Map<String, Task> tasks, String cicfilename, String outputpath){
		Map<String, Library> libraries = new HashMap<String, Library>();
		int count = 0;
		int channelId = 0;
		int stubId = 0;

		for(LibraryType library: mAlgorithm.getLibraries().getLibrary()){
			String name = library.getName();
			String type = library.getType();
			String hasInternalState = null;
			if(library.getHasInternalStates() == null)	hasInternalState = null;
			else										hasInternalState = library.getHasInternalStates().value();
			String header = library.getHeader();
			String file = library.getFile();
			
			try {
				List<Function> functionList = fillFunctionList(library, cicfilename, outputpath);
				
				int proc = 0;
				for(MappingLibraryType mlibrary: mMapping.getLibrary()){
					if(mlibrary.getName().equals(name)){
						for(Processor tproc: processors.values()){
							if(tproc.getPoolName().equals(mlibrary.getProcessor().getPool()) 
									&& tproc.getLocalIndex() == mlibrary.getProcessor().getLocalId().intValue()){
								proc = tproc.getIndex();
								break;
							}
						}
						break;
					}
				}
				
				List<Integer> diffMappedProc = new ArrayList<Integer> ();
				for(TaskLibraryConnectionType taskLibCon: mAlgorithm.getLibraryConnections().getTaskLibraryConnection()){
					if(taskLibCon.getSlaveLibrary().equals(name)){
						for(int tproc: tasks.get(taskLibCon.getMasterTask()).getProc().get("Default").get("Default")){	// Need to fix
							if(tproc != proc)	diffMappedProc.add(tproc);
						}
					}
				}
				
				for(LibraryLibraryConnectionType libLibCon: mAlgorithm.getLibraryConnections().getLibraryLibraryConnection()){
					if(libLibCon.getSlaveLibrary().equals(name)){
						for(MappingLibraryType mlibrary: mMapping.getLibrary()){
							if(mlibrary.getName().equals(libLibCon.getMasterLibrary())){
								for(Processor tproc: processors.values()){
									if(tproc.getPoolName().equals(mlibrary.getProcessor().getPool()) 
											&& tproc.getLocalIndex() == mlibrary.getProcessor().getLocalId().intValue()){
										if(proc != tproc.getIndex())	diffMappedProc.add(tproc.getIndex());
										break;
									}
								}
								break;
							}
						}
					}
				}
				
				List<LibraryStub> stubList = new ArrayList<LibraryStub> ();
				int stubCount = 0;
				for(int diffProc: diffMappedProc){
					String stubName = "stub_lib_" + name + "_for_proc_" + Integer.toString(diffProc);
					int rcvChId = channelId;
					int sndChId = channelId + 1;
					int targetProc = diffProc;
					int myProc = proc;
					stubList.add(new LibraryStub(stubName, stubId, sndChId, rcvChId, myProc, targetProc, functionList));
					stubCount++;
					channelId += 2;
					stubId ++;
					
					processors.get(diffProc).increaseTaskStartPrio(stubCount);
				}
				
				libraries.put(name, new Library(count, name, type, header, file, proc, functionList, diffMappedProc, stubList, 
						library.getLibraryMasterPort(), library.getExtraSource(), library.getExtraHeader(), library.getCflags(), library.getLdflags()));
				count++;
				
				
				
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		return libraries;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////// QUEUE Generation ///////////////////////////////////////////////////////
	
	public Map<Integer, Queue> makeQueues(CICAlgorithmType mAlgorithm, Map<String, Task> tasks){
		int index = 0;
		Map<Integer, Queue> queues = new HashMap<Integer, Queue>();
		
		if(mAlgorithm.getChannels() != null){
			for(ChannelType channel: mAlgorithm.getChannels().getChannel()){
				int flag=0;
				int srcPortId=0, dstPortId=0;
				int srcRate=1, dstRate=1;
				String srcPortName = null, dstPortName = null;
				String src = null, dst = null;
				
				String type = null;
				if(channel.getType().value().equals("fifo"))				type = "CHANNEL_TYPE_NORMAL";
				else if(channel.getType().value().equals("array"))			type = "CHANNEL_TYPE_ARRAY_CHANNEL";
				else if(channel.getType().value().equals("overwritable"))	type = "CHANNEL_TYPE_BUFFER";
				else														type = "";
				
				int size = channel.getSize().intValue();
				
				String sampleType = null;
				if(channel.getSampleType().isEmpty())	sampleType = "";
				else									sampleType = channel.getSampleType();
				
				int initData = channel.getInitialDataSize().intValue();
				int sampleSize = channel.getSampleSize().intValue();
				
				for(TaskType task: mAlgorithm.getTasks().getTask()){
					//if(!task.getName().equals(channel.getSrc().get(0).getPort()))	continue;
					
					if(task.getName().equals(channel.getSrc().get(0).getTask())){
						int t_index=0;
						for(TaskPortType port: task.getPort()){
							if(port.getName().equals(channel.getSrc().get(0).getPort())){
								src = channel.getSrc().get(0).getTask();
								srcPortId = t_index;
								srcPortName = port.getName();
								//[CODE_REVIEW]: hshong(4/21):sadf 지원하도록 확장
								if(port.getRate().size() > 0)
									srcRate = port.getRate().get(0).getRate().intValue(); //현재는 sdf만 지원하기 때문
								else //process network
									srcRate = 1;
								//sampleSize = port.getSampleSize().intValue();
								flag=1;
								break;
							}
							t_index++;
						}
						if(flag==1)	break;
					}
				}
				
				flag=0;
				for(TaskType task: mAlgorithm.getTasks().getTask()){
					//if(!task.getName().equals(channel.getDst().get(0).getPort()))	continue;
	
					if(task.getName().equals(channel.getDst().get(0).getTask())){
						int t_index=0;
						for(TaskPortType port: task.getPort()){
							if(port.getName().equals(channel.getDst().get(0).getPort())){
								dst = channel.getDst().get(0).getTask();
								dstPortId = t_index;
								dstPortName = port.getName();
								//[CODE_REVIEW]: hshong(4/21):sadf 지원하도록 확장
								if(port.getRate().size() > 0)
									dstRate = port.getRate().get(0).getRate().intValue(); //현재는 sdf만 지원하기 때문
								else //process network
									dstRate = 1;
								flag=1;
								break;
							}
							t_index++;
						}
						if(flag==1)	break;
					}
				}
				
				flag=0;
				Queue queue_result = new Queue(index, src, srcPortId, srcPortName, srcRate, dst, dstPortId, dstPortName, dstRate, size, initData, sampleSize, type, channel.getType(), sampleType); 
				queues.put(index, queue_result);
				index++;
			}
		}
		
		for(Queue queue: queues.values()){
			if(!tasks.get(queue.getSrc()).getNext().contains(queue.getDst())){
				tasks.get(queue.getSrc()).getNext().add(queue.getDst());
				tasks.get(queue.getDst()).getPrev().add(queue.getSrc());
			}
			tasks.get(queue.getSrc()).getQueue().add(queue);
			tasks.get(queue.getDst()).getQueue().add(queue);
		}
		
		return queues;
	}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	
/////////////////////////////////////////////////////// Etc. Generation ///////////////////////////////////////////////////////

	public void fillExecutionTimeInfo(CICProfileType mProfile, Map<String, Task> tasks){
		for(Task task: tasks.values()){	
			if(task.getHasSubgraph().equalsIgnoreCase("No")){
				Map<String, Integer> taskprofilevalinfo = new HashMap<String, Integer>();
				Map<String, String > taskprofileunitinfo = new HashMap<String, String>();
				
				for(ProfileTaskType profiledTask: mProfile.getTask()){
					String taskname = profiledTask.getName();
					if(taskname.equals(task.getName())){
						for(ProfileTaskModeType profilemode: profiledTask.getMode()){
							for(ProfileType profileProcessor: profilemode.getProfile()){
								taskprofilevalinfo.put(profilemode.getName(), profileProcessor.getValue().intValue());
								taskprofileunitinfo.put(profilemode.getName(), profileProcessor.getUnit());
							}
						}
						tasks.get(task.getName()).setExecutionTimeValue(taskprofilevalinfo);
						tasks.get(task.getName()).setExecutionTimeMetric(taskprofileunitinfo);
						break;
					}
				}
				
//				Map<String, Map<Integer, Integer>> taskprofilevalinfo = new HashMap<String, Map<Integer, Integer>>();
//				Map<String, Map<Integer, String>> taskprofileunitinfo = new HashMap<String, Map<Integer, String>>();				
//				for(ProfileTaskType profiledTask: mProfile.getTask()){
//					String taskname = profiledTask.getName();
//					if(taskname.equals(task.getName())){
//						for(ProfileTaskModeType profilemode: profiledTask.getMode()){
//							Map<Integer, Integer> procExec = new HashMap<Integer, Integer>();
//							Map<Integer, String> procMetric = new HashMap<Integer, String>();
//							for(ProfileType profileProcessor: profilemode.getProfile()){
//								int procId = 0;
//								for(Processor proc: processors.values()){
//									if(proc.getProcType().equals(profileProcessor.getProcessorType())){
//										procId = proc.getIndex();
//										break;
//									}
//								}
//								procExec.put(procId, profileProcessor.getValue().intValue());
//								procMetric.put(procId, profileProcessor.getUnit());
//							}
//							taskprofilevalinfo.put(profilemode.getName(), procExec);
//							taskprofileunitinfo.put(profilemode.getName(), procMetric);
//						}
//						tasks.get(task.getName()).setExecutionTimeValue(taskprofilevalinfo);
//						tasks.get(task.getName()).setExecutionTimeMetric(taskprofileunitinfo);
//						break;
//					}
//					else	continue;
//				}
			}
		}
	}
	
	public void checkSlaveTask(Map<String, Task> tasks, CICControlType mControl){
		if(mControl.getControlTasks() == null)	return;
		else{
			for(Task task: tasks.values()){
				for(ControlTaskType controlTask: mControl.getControlTasks().getControlTask()){
					for(String slaveTask: controlTask.getSlaveTask()){
						if(task.getName().equals(slaveTask)){
							task.setIsSlaveTask(true);
							task.getControllingTask().add(controlTask.getTask());
						}
					}
				}
			}
		}
	}

	public int setControlQueueIndex(Map<String, Task> tasks, Map<Integer, Processor> processors){
		int controlQueueIndex=0;
		
		for(Task task: tasks.values()){
			if(task.getHasSubgraph().equalsIgnoreCase("No")){
				if(task.getIsSlaveTask()){
					int proc = task.getProc().get("Default").get("Default").get(0);	// Need to fix
					if(processors.get(proc).getControlQueueIndex() == -1){
						processors.get(proc).setControlQueueIndex(controlQueueIndex);
						controlQueueIndex++;
					}
				}
			}
		}
				
		return controlQueueIndex;
	}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
}