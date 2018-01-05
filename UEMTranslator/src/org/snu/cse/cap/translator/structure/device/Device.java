package org.snu.cse.cap.translator.structure.device;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.snu.cse.cap.translator.structure.InvalidDataInMetadataFileException;
import org.snu.cse.cap.translator.structure.TaskGraph;
import org.snu.cse.cap.translator.Constants;
import org.snu.cse.cap.translator.structure.ExecutionPolicy;
import org.snu.cse.cap.translator.structure.channel.Channel;
import org.snu.cse.cap.translator.structure.channel.Port;
import org.snu.cse.cap.translator.structure.device.connection.InvalidDeviceConnectionException;
import org.snu.cse.cap.translator.structure.library.Library;
import org.snu.cse.cap.translator.structure.library.LibraryConnection;
import org.snu.cse.cap.translator.structure.mapping.CompositeTaskMappedProcessor;
import org.snu.cse.cap.translator.structure.mapping.CompositeTaskMappingInfo;
import org.snu.cse.cap.translator.structure.mapping.CompositeTaskSchedule;
import org.snu.cse.cap.translator.structure.mapping.GeneralTaskMappingInfo;
import org.snu.cse.cap.translator.structure.mapping.InvalidScheduleFileNameException;
import org.snu.cse.cap.translator.structure.mapping.MappedProcessor;
import org.snu.cse.cap.translator.structure.mapping.MappingInfo;
import org.snu.cse.cap.translator.structure.mapping.ScheduleFileFilter;
import org.snu.cse.cap.translator.structure.mapping.ScheduleFileNameOffset;
import org.snu.cse.cap.translator.structure.mapping.ScheduleItem;
import org.snu.cse.cap.translator.structure.mapping.ScheduleItemType;
import org.snu.cse.cap.translator.structure.mapping.ScheduleLoop;
import org.snu.cse.cap.translator.structure.mapping.ScheduleTask;
import org.snu.cse.cap.translator.structure.task.Task;
import org.snu.cse.cap.translator.structure.task.TaskMode;
import org.snu.cse.cap.translator.structure.task.TaskModeTransition;
import org.snu.cse.cap.translator.structure.task.TaskShapeType;
import org.snu.cse.cap.translator.structure.task.TaskMode.ChildTaskTraverseCallback;

import hopes.cic.exception.CICXMLException;
import hopes.cic.xml.CICMappingType;
import hopes.cic.xml.CICScheduleType;
import hopes.cic.xml.CICScheduleTypeLoader;
import hopes.cic.xml.MappingDeviceType;
import hopes.cic.xml.MappingProcessorIdType;
import hopes.cic.xml.MappingTaskType;
import hopes.cic.xml.ScheduleElementType;
import hopes.cic.xml.ScheduleGroupType;
import hopes.cic.xml.TaskGroupForScheduleType;

public class Device {
	private String name;
	private ArrayList<Processor> processorList;
	private HashMap<String, Connection> connectionList;
	private ArchitectureType architecture;
	private SoftwarePlatformType platform;
	private RuntimeType runtime;
	
	// in-device metadata information
	private ArrayList<Channel> channelList;
	private HashMap<String, Task> taskMap; // Task name : Task class
	private HashMap<String, TaskGraph> taskGraphMap; // Task graph name : TaskGraph class
	private HashMap<String, GeneralTaskMappingInfo> generalMappingInfo; // Task name : GeneralTaskMappingInfo class
	private HashMap<String, CompositeTaskMappingInfo> staticScheduleMappingInfo; // Parent task Name : CompositeTaskMappingInfo class
	private ArrayList<Port> portList;
	private HashMap<String, Library> libraryMap;
	private HashMap<String, Integer> portKeyToIndex;  //Key: taskName/portName/direction, ex) MB_Y/inMB_Y/input
	
	public Device(String name, String architecture, String platform, String runtime) 
	{
		this.name = name;
		this.architecture = ArchitectureType.fromValue(architecture);
		this.platform = SoftwarePlatformType.fromValue(platform);
		this.runtime = RuntimeType.fromValue(runtime);
		this.processorList = new ArrayList<Processor>();
		this.connectionList = new HashMap<String, Connection>();
		
	
		this.channelList = new ArrayList<Channel>();
		this.taskMap = new HashMap<String, Task>();
		this.taskGraphMap = new HashMap<String, TaskGraph>();
		this.generalMappingInfo = new HashMap<String, GeneralTaskMappingInfo>();
		this.staticScheduleMappingInfo = new HashMap<String, CompositeTaskMappingInfo>();
		this.libraryMap = new HashMap<String, Library>();
		this.portList = new ArrayList<Port>();
		this.portKeyToIndex = new HashMap<String, Integer>();
	}
	
	private class TaskFuncIdChecker 
	{
		private int curTaskFuncId;
		private boolean isUsed;
		
		public TaskFuncIdChecker()
		{
			this.curTaskFuncId = 0;
			this.isUsed = false;
		}

		public int getCurTaskFuncId() {
			return curTaskFuncId;
		}

		public void increaseCurTaskFuncId() {
			this.curTaskFuncId++;
		}

		public boolean isUsed() {
			return isUsed;
		}

		public void setUsed(boolean isUsed) {
			this.isUsed = isUsed;
		}
	}
	
	public String getName() {
		return name;
	}
	
	// recursive function
	private int recursiveScheduleLoopInsert(ArrayList<ScheduleItem> scheduleItemList, List<ScheduleElementType> scheduleElementList, 
										int depth, int maxDepth, HashMap<String, Task> globalTaskMap)
	{
		ScheduleLoop scheduleInloop;
		ScheduleTask scheduleTask;
		int nextDepth = 0;
		
		if(maxDepth < depth)
			maxDepth = depth;
		
		for(ScheduleElementType scheduleElement: scheduleElementList)
		{
			if(scheduleElement.getLoop() != null)
			{
				scheduleInloop = new ScheduleLoop(scheduleElement.getLoop().getRepetition().intValue(), depth);
				if(scheduleInloop.getRepetition() > 1)
					nextDepth = depth + 1;
				else
					nextDepth = depth;
				maxDepth = recursiveScheduleLoopInsert(scheduleInloop.getScheduleItemList(), scheduleElement.getLoop().getScheduleElement(), 
														nextDepth, maxDepth, globalTaskMap);
				scheduleItemList.add(scheduleInloop);
			}
			else if(scheduleElement.getTask() != null) 
			{
				scheduleTask = new ScheduleTask(scheduleElement.getTask().getName(), scheduleElement.getTask().getRepetition().intValue(), depth);
				scheduleItemList.add(scheduleTask);
				putTaskHierarchicallyToTaskMap(scheduleTask.getTaskName(), globalTaskMap);
			}
			else
			{
				// do nothing
			}			
		}
		
		return maxDepth;
	}
	
	private CompositeTaskSchedule fillCompositeTaskSchedule(CompositeTaskSchedule taskSchedule, ScheduleGroupType scheduleGroup, 
															HashMap<String, Task> globalTaskMap) 
	{ 	
		int maxDepth = 0;
		
		maxDepth = recursiveScheduleLoopInsert(taskSchedule.getScheduleList(), scheduleGroup.getScheduleElement(), 0, maxDepth, globalTaskMap);
		taskSchedule.setMaxLoopVariableNum(maxDepth);
		
		return taskSchedule;
	}
	
	private int getProcessorIdByName(String processorName) throws InvalidDataInMetadataFileException,NoProcessorFoundException {
		int processorId = Constants.INVALID_ID_VALUE;

		for(Processor processor: this.getProcessorList())
		{
			if(processorName.equals(processor.getName()))
			{
				processorId = processor.getId();
				break;
			}
		}
		
		if(processorId == Constants.INVALID_ID_VALUE)
			throw new NoProcessorFoundException("There is no processor name called " + processorName);
		
		return processorId; 
	}
	
	private int getModeIdByName(String taskName, String modeName, HashMap<String, Task> globalTaskMap) throws InvalidDataInMetadataFileException
	{
		int modeId;
		Task task;
		
		task = globalTaskMap.get(taskName);
		if(task == null) // check it is root task graph
		{
			// if the task name consists of "SDF_" + total task number, it means top-level graph is target task graph
			if(taskName.equals(Constants.TOP_TASKGRAPH_NAME))
			{
				modeId = 0;
			}
			else
			{
				throw new InvalidDataInMetadataFileException("there is no task name called" + taskName);
			}
		}
		else if(task.getModeTransition() == null)
		{
			modeId = 0;
		}
		else
		{
			modeId = task.getModeTransition().getModeIdFromName(modeName);
		}
		
		return modeId;
	}
	
	private int getThroughputConstraintFromScheduleFileName(String[] splitedFileName)
	{
		int throughputConstraint;
		
		if(splitedFileName.length == ScheduleFileNameOffset.values().length)
		{
			throughputConstraint = Integer.parseInt(splitedFileName[ScheduleFileNameOffset.THROUGHPUT_CONSTRAINT.getValue()]);
		}
		else // throughput constraint is missing (splitedFileName.length == ScheduleFileNameOffset.values().length - 1)
		{
			throughputConstraint = 0;
		}
		
		return throughputConstraint;
	}
	
	private CompositeTaskMappingInfo getCompositeMappingInfo(String taskName, int taskId) throws InvalidDataInMetadataFileException
	{
		CompositeTaskMappingInfo compositeMappingInfo;
		
		if(this.staticScheduleMappingInfo.containsKey(taskName) == false)
		{//modeId 
			compositeMappingInfo = new CompositeTaskMappingInfo(taskName, taskId);
			this.staticScheduleMappingInfo.put(taskName, compositeMappingInfo);
		}
		else
		{
			compositeMappingInfo = (CompositeTaskMappingInfo) this.staticScheduleMappingInfo.get(taskName);
			if(compositeMappingInfo.getMappedTaskType() != TaskShapeType.COMPOSITE)
			{
				throw new InvalidDataInMetadataFileException();
			}
		}
		
		return compositeMappingInfo;
	}
	
	private boolean isProcessorNameLocatedInDevice(String processorName) {
		boolean processorNameFound = false;
		
		for(Processor proc: getProcessorList())
		{
			if(processorName.equals(proc.getName()))
			{
				processorNameFound = true;
				break;
			}
		}
					
		return processorNameFound;
	}
	
	private void putTaskHierarchicallyToTaskMap(String taskName, HashMap<String, Task> globalTaskMap)
	{
		Task currentTask;
		TaskGraph taskGraph;
		int inGraphIndex = 0;
		currentTask = globalTaskMap.get(taskName);
		while(this.taskMap.containsKey(currentTask.getName()) == false)
		{
			this.taskMap.put(currentTask.getName(), currentTask);
			
			if(this.taskGraphMap.containsKey(currentTask.getParentTaskGraphName()))
			{
				taskGraph = this.taskGraphMap.get(currentTask.getParentTaskGraphName());
			}
			else
			{
				taskGraph = new TaskGraph(currentTask.getParentTaskGraphName());
				this.taskGraphMap.put(currentTask.getParentTaskGraphName(), taskGraph);
			}

			inGraphIndex = taskGraph.getNumOfTasks();
			currentTask.setInGraphIndex(inGraphIndex);
			
			taskGraph.putTask(currentTask);
			
			if(currentTask.getParentTaskGraphName().equals(Constants.TOP_TASKGRAPH_NAME) == true)
				break;
			
			currentTask = globalTaskMap.get(currentTask.getParentTaskGraphName());
		}
	}
	
	private void makeMultipleCompositeTaskMapping(String[] splitedFileName, File scheduleFile, HashMap<String, Task> globalTaskMap)
										throws CICXMLException, InvalidDataInMetadataFileException 
	{
		int numOfUsableCPU;
		CICScheduleTypeLoader scheduleLoader = new CICScheduleTypeLoader();
		CICScheduleType scheduleDOM;
		String taskName;
		String modeName;
		int modeId = Constants.INVALID_ID_VALUE;
		CompositeTaskMappingInfo compositeMappingInfo;
		int procId = Constants.INVALID_ID_VALUE;
		int sequenceId = 0;
		Task task;
		int throughputConstraint;
		String processorName = "";
		CompositeTaskMappedProcessor mappedProcessor;
		
		scheduleDOM = scheduleLoader.loadResource(scheduleFile.getAbsolutePath());
		
		numOfUsableCPU = Integer.parseInt(splitedFileName[ScheduleFileNameOffset.NUM_OF_USABLE_CPU.getValue()]);
		taskName = splitedFileName[ScheduleFileNameOffset.TASK_NAME.getValue()];
		if(taskName.equals("SDF_" + globalTaskMap.size())) // it means top-level graph
		{
			taskName = Constants.TOP_TASKGRAPH_NAME;
		}
		modeName = splitedFileName[ScheduleFileNameOffset.MODE_NAME.getValue()];
		modeId = getModeIdByName(taskName, modeName, globalTaskMap);
		throughputConstraint = getThroughputConstraintFromScheduleFileName(splitedFileName);
		task = globalTaskMap.get(taskName);
		
		for(TaskGroupForScheduleType taskGroup: scheduleDOM.getTaskGroups().getTaskGroup())
		{
			for(ScheduleGroupType scheduleGroup : taskGroup.getScheduleGroup())
			{
				try {
					procId = getProcessorIdByName(scheduleGroup.getPoolName());
				} catch (NoProcessorFoundException e) {
					// skip processor
					continue;
				}
				// TODO: currently the last-picked processor name is used for searching the device name
				// get processor name from schedule information
				processorName = scheduleGroup.getPoolName();

				if(isProcessorNameLocatedInDevice(processorName) == true)
				{
					if(taskName.equals(Constants.TOP_TASKGRAPH_NAME))
					{
						compositeMappingInfo = getCompositeMappingInfo(taskName, Constants.INVALID_ID_VALUE);
					}
					else
						compositeMappingInfo = getCompositeMappingInfo(taskName, task.getId());
					
					CompositeTaskSchedule taskSchedule = new CompositeTaskSchedule(numOfUsableCPU, throughputConstraint);
					
					fillCompositeTaskSchedule(taskSchedule, scheduleGroup, globalTaskMap) ;
					
					mappedProcessor = compositeMappingInfo.getMappedProcessorInfo(modeId, procId, scheduleGroup.getLocalId().intValue());
					
					if(mappedProcessor == null)
					{
						mappedProcessor = new CompositeTaskMappedProcessor(procId, 
								scheduleGroup.getLocalId().intValue(), modeId, sequenceId);						
						compositeMappingInfo.putProcessor(mappedProcessor);
					}
					
					mappedProcessor.putCompositeTaskSchedule(taskSchedule);
					compositeMappingInfo.setMappedDeviceName(this.name);
					
					sequenceId++;
				}
			}
		}
	}
	
	private void setCompositeTaskMappingInfo(HashMap<String, Task> globalTaskMap, String scheduleFolderPath) throws FileNotFoundException, InvalidScheduleFileNameException, InvalidDataInMetadataFileException {
		ScheduleFileFilter scheduleXMLFilefilter = new ScheduleFileFilter(); 
		String[] splitedFileName = null;
		File scheduleFolder = new File(scheduleFolderPath);
		
		if(scheduleFolder.exists() == false || scheduleFolder.isDirectory() == false)
		{
			throw new FileNotFoundException();
		}
		
		for(File file : scheduleFolder.listFiles(scheduleXMLFilefilter)) 
		{
			splitedFileName = file.getName().split(Constants.SCHEDULE_FILE_SPLITER);
			
			if(splitedFileName.length != ScheduleFileNameOffset.values().length && 
				splitedFileName.length != ScheduleFileNameOffset.values().length - 1)
			{
				throw new InvalidScheduleFileNameException();
			}
			
			try {
				makeMultipleCompositeTaskMapping(splitedFileName, file, globalTaskMap);
			} catch (CICXMLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	
	private TaskShapeType getTaskType(String taskName)
	{
		return this.taskMap.get(taskName).getType();
	}
	
	private boolean checkTaskIsIncludedInCompositeTask(String taskName, HashMap<String, Task> globalTaskMap)
	{
		boolean isInsideCompositeTask = false;
		Task task;
		
		task = globalTaskMap.get(taskName);
		
		do
		{
			if(this.staticScheduleMappingInfo.containsKey(task.getParentTaskGraphName()) == true)
			{
				isInsideCompositeTask = true;
				break;
			}
			
			// set parent task to next task
			task = globalTaskMap.get(task.getParentTaskGraphName());
		} while(task != null);
		
		return isInsideCompositeTask;
	}
	

	private void setGeneralTaskMappingInfo(CICMappingType mapping_metadata, HashMap<String, Task> globalTaskMap) throws InvalidDataInMetadataFileException, NoProcessorFoundException
	{		
		for(MappingTaskType mappedTask: mapping_metadata.getTask())
		{
			for(MappingDeviceType device: mappedTask.getDevice())
			{
				if(device.getName().equals(this.name) && 
					checkTaskIsIncludedInCompositeTask(mappedTask.getName(), globalTaskMap) == false)
				{
					Task task = globalTaskMap.get(mappedTask.getName());
					
					putTaskHierarchicallyToTaskMap(task.getName(), globalTaskMap);
					
					GeneralTaskMappingInfo mappingInfo = new GeneralTaskMappingInfo(mappedTask.getName(), getTaskType(mappedTask.getName()), 
							task.getParentTaskGraphName(), task.getInGraphIndex());
					
					mappingInfo.setMappedDeviceName(device.getName());
					
					for(MappingProcessorIdType proc: device.getProcessor())
					{
						MappedProcessor processor = new MappedProcessor(getProcessorIdByName(proc.getPool()), proc.getLocalId().intValue());
						mappingInfo.putProcessor(processor);
					}
					
					if(this.generalMappingInfo.containsKey(mappedTask.getName()) == false)
					{
						this.generalMappingInfo.put(mappedTask.getName(), mappingInfo);				
					}
					else // if same task is already in the mappingInfo, ignore the later one
					{
						// ignore the mapping (because the duplicated key means it already registered 
					}
					break;
				}
			}
		}
	}
	
	private void recursiveSetSubgraphTaskToStaticScheduled(TaskGraph taskGraph)
	{		
		TaskGraph subTaskGraph;
		for(Task subTask: taskGraph.getTaskList())
		{
			subTask.setStaticScheduled(true);
			if(subTask.getChildTaskGraphName() != null) 
			{
				subTaskGraph = this.taskGraphMap.get(subTask.getChildTaskGraphName());
				recursiveSetSubgraphTaskToStaticScheduled(subTaskGraph);
			}
		}
	}
	
	private void setTaskExtraInformationFromMappingInfo()
	{
		Task task;
		TaskGraph taskGraph;
		for(MappingInfo mappingInfo : this.staticScheduleMappingInfo.values())
		{
			CompositeTaskMappingInfo compositeMappingInfo = (CompositeTaskMappingInfo) mappingInfo;
			task = this.taskMap.get(compositeMappingInfo.getParentTaskName());
			if(task != null)
			{
				task.setStaticScheduled(true);
				taskGraph = this.taskGraphMap.get(task.getChildTaskGraphName());
				recursiveSetSubgraphTaskToStaticScheduled(taskGraph);
			}
		}
	}
	
	private void recursivePutTask(ArrayList<ScheduleItem> scheduleItemList, TaskModeTransition targetTaskModeTransition, 
			CompositeTaskMappedProcessor compositeMappedProc) 
	{		
		for(ScheduleItem item: scheduleItemList)
		{
			switch(item.getItemType())
			{
			case LOOP:
				ScheduleLoop scheduleLoop = (ScheduleLoop) item; 
				recursivePutTask(scheduleLoop.getScheduleItemList(), targetTaskModeTransition, compositeMappedProc);
				break;
			case TASK:
				ScheduleTask task = (ScheduleTask) item;
				Task  taskObject = this.taskMap.get(task.getTaskName());
				targetTaskModeTransition.putRelatedChildTask(compositeMappedProc.getProcessorId(), compositeMappedProc.getProcessorLocalId(), 
						compositeMappedProc.getModeId(), taskObject);
				
				break;
			}
		}
	}

	private void putRelatedChildTaskInCompositeTask(TaskModeTransition targetTaskModeTransition, 
			CompositeTaskMappedProcessor compositeMappedProc)
	{
		for(CompositeTaskSchedule schedule: compositeMappedProc.getCompositeTaskScheduleList())
		{		
			recursivePutTask(schedule.getScheduleList(), targetTaskModeTransition, compositeMappedProc);
		}
	}
	
	private void setRelatedChildTasksOfMTMTask()
	{
		CompositeTaskMappingInfo compositeMappingInfo;
		CompositeTaskMappedProcessor compositeMappedProc;
		for(Task task: this.taskMap.values())
		{
			if(task.getModeTransition() != null && task.getChildTaskGraphName() != null && task.isStaticScheduled() == true)
			{
				compositeMappingInfo = this.staticScheduleMappingInfo.get(task.getName());
				for(MappedProcessor mappedProcessor: compositeMappingInfo.getMappedProcessorList())
				{
					compositeMappedProc = (CompositeTaskMappedProcessor) mappedProcessor;
					compositeMappedProc.getModeId();
					putRelatedChildTaskInCompositeTask(task.getModeTransition(), compositeMappedProc);
				}
			}
		}
	}
	
	private void setChildTaskProc(HashMap<String, TaskMode> modeMap)
	{
		ChildTaskTraverseCallback<HashMap<String, Integer>> childTaskCallback;
		HashMap<String, Integer> relatedTaskMap = new HashMap<String, Integer>();
		
		childTaskCallback = new ChildTaskTraverseCallback<HashMap<String, Integer>>() {
			@Override
			public void traverseCallback(String taskName, int procId, int procLocalId, HashMap<String, Integer> taskSet) {
				Integer intValue;
				
				if(taskSet.containsKey(taskName) == false)
				{
					taskSet.put(taskName, new Integer(1));					
				}
				else // key exists
				{
					intValue = taskSet.get(taskName);
					intValue++;
					taskSet.put(taskName, intValue);
				}
			}
		};
		
		for(TaskMode mode: modeMap.values())
		{
			mode.traverseRelatedChildTask(childTaskCallback, relatedTaskMap);
			
			for(String taskName: relatedTaskMap.keySet())
			{
				Task task = this.taskMap.get(taskName);
				int newMappedProcNum = relatedTaskMap.get(taskName).intValue();
				if(task.getTaskFuncNum() < newMappedProcNum)
				{
					task.setTaskFuncNum(newMappedProcNum);
				}
			}
		}
	}
	
	// set taskFuncNum which is same to the number processors mapped to each task
	private void setNumOfProcsOfTasks()
	{
		Task task;
		for(CompositeTaskMappingInfo compositeMappingInfo: this.staticScheduleMappingInfo.values())
		{
			task = this.taskMap.get(compositeMappingInfo.getParentTaskName());
			if(task != null)
			{
				setChildTaskProc(task.getModeTransition().getModeMap());
			}
		}
		
		for(GeneralTaskMappingInfo generalMappingInfo: this.generalMappingInfo.values())
		{
			task = this.taskMap.get(generalMappingInfo.getTaskName());
			task.setTaskFuncNum(generalMappingInfo.getMappedProcessorList().size());			
		}
	}
	
	private void recursiveScheduleLoopTraverse(ArrayList<ScheduleItem> scheduleItemList, HashMap<String, TaskFuncIdChecker> taskFuncIdMap, int modeId)
	{
		for(ScheduleItem scheduleItem : scheduleItemList)
		{
			if(scheduleItem.getItemType() == ScheduleItemType.LOOP)
			{
				ScheduleLoop scheduleInnerLoop = (ScheduleLoop) scheduleItem;
				recursiveScheduleLoopTraverse(scheduleInnerLoop.getScheduleItemList(), taskFuncIdMap, modeId);
			}
			else
			{
				ScheduleTask scheduleTask = (ScheduleTask) scheduleItem;
				String taskFuncIdKey = modeId + scheduleTask.getTaskName();
				
				if(taskFuncIdMap.containsKey(taskFuncIdKey))
				{
					scheduleTask.setTaskFuncId(taskFuncIdMap.get(taskFuncIdKey).getCurTaskFuncId());
					taskFuncIdMap.get(taskFuncIdKey).setUsed(true);
				}
				else
				{
					taskFuncIdMap.put(taskFuncIdKey, new TaskFuncIdChecker());
				}
			}
		}
	}
	
	private void setScheduleListIndexAndTaskFuncId()
	{
		int index = 0;
		
		for(CompositeTaskMappingInfo mappingInfo : this.staticScheduleMappingInfo.values())
		{
			HashMap<String, TaskFuncIdChecker> taskFuncIdMap = new HashMap<String, TaskFuncIdChecker>();
 
			for(MappedProcessor mappedProcessor: mappingInfo.getMappedProcessorList())
			{
				CompositeTaskMappedProcessor compositeMappedProcessor = (CompositeTaskMappedProcessor) mappedProcessor;
				compositeMappedProcessor.setInArrayIndex(index);
				
				for(CompositeTaskSchedule taskScheule: compositeMappedProcessor.getCompositeTaskScheduleList())
				{
					recursiveScheduleLoopTraverse(taskScheule.getScheduleList(), taskFuncIdMap, compositeMappedProcessor.getModeId());
					
					for(TaskFuncIdChecker funcIdChecker: taskFuncIdMap.values())
					{
						if(funcIdChecker.isUsed() == true)
						{
							funcIdChecker.increaseCurTaskFuncId();
						}
						funcIdChecker.setUsed(false);
					}
				}
				
				index++;
			}
		}
	}
	
	private void setParentTaskOfTaskGraph()
	{
		Task task;
		
		for(TaskGraph taskGraph: this.taskGraphMap.values())
		{
			if(taskGraph.getName().equals(Constants.TOP_TASKGRAPH_NAME))
			{
				// Top-level task graph, no parent task
			}
			else
			{
				task = this.taskMap.get(taskGraph.getName());
				taskGraph.setParentTask(task);
			}
		}
	}
	
	public void putInDeviceTaskInformation(HashMap<String, Task> globalTaskMap, 
									String scheduleFolderPath, CICMappingType mapping_metadata, ExecutionPolicy executionPolicy)
	throws FileNotFoundException, InvalidScheduleFileNameException, InvalidDataInMetadataFileException, NoProcessorFoundException
	{
		switch(executionPolicy)
		{
		// TODO: fully static is not supported now
		case FULLY_STATIC: // Need schedule with time information (needed file: mapping, profile, schedule)
		case SELF_TIMED: // Need schedule (needed file: mapping, schedule)
			setCompositeTaskMappingInfo(globalTaskMap, scheduleFolderPath);
			setGeneralTaskMappingInfo( mapping_metadata, globalTaskMap);
			setParentTaskOfTaskGraph();
			setTaskExtraInformationFromMappingInfo();
			setRelatedChildTasksOfMTMTask();
			setNumOfProcsOfTasks();
			setScheduleListIndexAndTaskFuncId();
			break;
		case STATIC_ASSIGNMENT: // Need mapping only (needed file: mapping)
			setGeneralTaskMappingInfo( mapping_metadata, globalTaskMap);
			setNumOfProcsOfTasks();
			break;
		// TODO: fully dynamic is not supported now
		case FULLY_DYNAMIC: // Need mapped device information (needed file: mapping)
			setGeneralTaskMappingInfo( mapping_metadata, globalTaskMap);
			break;
		}
	}
	

	
	private boolean recursiveIsLibraryUsedInDevice(ArrayList<LibraryConnection> libraryConnection, HashMap<String, Library> globalLibraryMap)
	{
		boolean isUsed = false;
		
		for(LibraryConnection connection: libraryConnection)
		{
			if(connection.isMasterLibrary() == true)
			{
				Library library = globalLibraryMap.get(connection.getMasterName());
				isUsed = recursiveIsLibraryUsedInDevice(library.getLibraryConnectionList(), globalLibraryMap);			}
			else
			{
				if(this.taskMap.containsKey(connection.getMasterName()) == true)
				{
					isUsed = true;
				}
			}
			
			if(isUsed == true)
			{
				break;
			}
		}
		
		return isUsed;
	}
	
	public void putInDeviceLibraryInformation(HashMap<String, Library> globalLibraryMap)
	{
		for(Library library: globalLibraryMap.values())
		{
			if(recursiveIsLibraryUsedInDevice(library.getLibraryConnectionList(), globalLibraryMap) == true)
			{
				this.libraryMap.put(library.getName(), library);
			}
		}
		
		linkibraryToTasks();
	}
	
	private void linkibraryToTasks()
	{
		for(Library library: this.libraryMap.values())
		{
			for(LibraryConnection connection: library.getLibraryConnectionList())
			{
				if(connection.isMasterLibrary() == false)
				{
					Task task = this.taskMap.get(connection.getMasterName());
					if(task.getLinkedLibraryHeaderList().contains(library.getHeader()) == false)
					{
						task.getLinkedLibraryHeaderList().add(library.getHeader());					
					}
				}
			}
			
		}
	}

	public void putProcessingElement(int id, String name, ProcessorCategory type, int poolSize) 
	{
		Processor processor = new Processor(id, name, type, poolSize);
			
		this.processorList.add(processor);
	}
	
	public void putConnection(Connection connection) 
	{
		this.connectionList.put(connection.getName(), connection);
	}
	
	public Connection getConnection(String connectionName) throws InvalidDeviceConnectionException 
	{
		Connection connection;
		
		if(this.connectionList.containsKey(connectionName))
		{
			connection = this.connectionList.get(connectionName);	
		}
		else
		{
			throw new InvalidDeviceConnectionException();
		}
		
		return connection;
	}

	public ArchitectureType getArchitecture() {
		return architecture;
	}
	
	public SoftwarePlatformType getPlatform() {
		return platform;
	}
	
	public RuntimeType getRuntime() {
		return runtime;
	}
	
	public void setArchitecture(ArchitectureType architecture) {
		this.architecture = architecture;
	}
	
	public void setPlatform(SoftwarePlatformType platform) {
		this.platform = platform;
	}
	
	public void setRuntime(RuntimeType runtime) {
		this.runtime = runtime;
	}

	public ArrayList<Processor> getProcessorList() {
		return processorList;
	}

	public void setProcessorList(ArrayList<Processor> processorList) {
		this.processorList = processorList;
	}

	public HashMap<String, Task> getTaskMap() {
		return taskMap;
	}

	public HashMap<String, GeneralTaskMappingInfo> getGeneralMappingInfo() {
		return generalMappingInfo;
	}

	public HashMap<String, CompositeTaskMappingInfo> getStaticScheduleMappingInfo() {
		return staticScheduleMappingInfo;
	}

	public ArrayList<Channel> getChannelList() {
		return channelList;
	}

	public HashMap<String, TaskGraph> getTaskGraphMap() {
		return taskGraphMap;
	}

	public HashMap<String, Library> getLibraryMap() {
		return libraryMap;
	}

	public ArrayList<Port> getPortList() {
		return portList;
	}

	public HashMap<String, Integer> getPortKeyToIndex() {
		return portKeyToIndex;
	}
}
