package org.snu.cse.cap.translator.structure.device;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.snu.cse.cap.translator.structure.InvalidDataInMetadataFileException;
import org.snu.cse.cap.translator.structure.TaskGraph;
import org.snu.cse.cap.translator.structure.channel.Channel;
import org.snu.cse.cap.translator.structure.channel.Port;
import org.snu.cse.cap.translator.structure.device.connection.InvalidDeviceConnectionException;
import org.snu.cse.cap.translator.structure.mapping.CompositeTaskMappedProcessor;
import org.snu.cse.cap.translator.structure.mapping.CompositeTaskMappingInfo;
import org.snu.cse.cap.translator.structure.mapping.CompositeTaskSchedule;
import org.snu.cse.cap.translator.structure.mapping.GeneralTaskMappingInfo;
import org.snu.cse.cap.translator.structure.mapping.InvalidScheduleFileNameException;
import org.snu.cse.cap.translator.structure.mapping.MappedProcessor;
import org.snu.cse.cap.translator.structure.mapping.ScheduleFileFilter;
import org.snu.cse.cap.translator.structure.mapping.ScheduleFileNameOffset;
import org.snu.cse.cap.translator.structure.mapping.ScheduleItem;
import org.snu.cse.cap.translator.structure.mapping.ScheduleLoop;
import org.snu.cse.cap.translator.structure.mapping.ScheduleTask;
import org.snu.cse.cap.translator.structure.task.Task;
import org.snu.cse.cap.translator.structure.task.TaskShapeType;

import Translators.Constants;
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

enum ArchitectureType {
	X86("x86"),
	X86_64("x86_64"),
	ARM("arm"),
	ARM64("arm64"),
	GENERIC("generic"),
	;
	
	private final String value;
	
	private ArchitectureType(final String value) {
		this.value = value;
	}
	
	@Override
	public String toString() {
		return value;
	}
	
	public static ArchitectureType fromValue(String value) {
		 for (ArchitectureType c : ArchitectureType.values()) {
			 if (c.value.equals(value)) {
				 return c;
			 }
		 }
		 throw new IllegalArgumentException(value.toString());
	}
}

enum SoftwarePlatformType {
	ARDUINO("arduino"),
	WINDOWS("windows"),
	LINUX("linux"),
	UCOS3("ucos-3"),
	;
	
	private final String value;
	
	private SoftwarePlatformType(final String value) {
		this.value = value;
	}
	
	@Override
	public String toString() {
		return value;
	}
	
	public static SoftwarePlatformType fromValue(String value) {
		 for (SoftwarePlatformType c : SoftwarePlatformType.values()) {
			 if (c.value.equals(value)) {
				 return c;
			 }
		 }
		 throw new IllegalArgumentException(value.toString());
	}
}

enum RuntimeType {
	NATIVE("native"),
	SOPHY("sophy"),
	HSIM("hsim"),
	;

	private final String value;
	
	private RuntimeType(final String value) {
		this.value = value;
	}
	
	@Override
	public String toString() {
		return value;
	}
	
	public static RuntimeType fromValue(String value) {
		 for (RuntimeType c : RuntimeType.values()) {
			 if (c.value.equals(value)) {
				 return c;
			 }
		 }
		 throw new IllegalArgumentException(value.toString());
	}
}

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
	private HashMap<String, Port> portInfo; // Key: taskName/portName/direction, ex) MB_Y/inMB_Y/input
	
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
		this.portInfo = new HashMap<String, Port>();
	}
	
	public String getName() {
		return name;
	}
	
	// recursive function
	private int recursiveScheduleLoopInsert(ArrayList<ScheduleItem> scheduleItemList, List<ScheduleElementType> scheduleElementList, int depth, int maxDepth)
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
				maxDepth = recursiveScheduleLoopInsert(scheduleInloop.getScheduleItemList(), scheduleElement.getLoop().getScheduleElement(), nextDepth, maxDepth);
				scheduleItemList.add(scheduleInloop);
			}
			else if(scheduleElement.getTask() != null) 
			{
				scheduleTask = new ScheduleTask(scheduleElement.getTask().getName(), scheduleElement.getTask().getRepetition().intValue(), depth);
				scheduleItemList.add(scheduleTask);
			}
			else
			{
				// do nothing
			}			
		}
		
		return maxDepth;
	}
	
	private CompositeTaskSchedule fillCompositeTaskSchedule(CompositeTaskSchedule taskSchedule, ScheduleGroupType scheduleGroup) 
	{ 	
		int maxDepth = 0;
		
		maxDepth = recursiveScheduleLoopInsert(taskSchedule.getScheduleList(), scheduleGroup.getScheduleElement(), 0, maxDepth);
		taskSchedule.setMaxLoopVariableNum(maxDepth);
		
		return taskSchedule;
	}
	
	private int getProcessorIdByName(String processorName) throws InvalidDataInMetadataFileException {
		int processorId = Constants.INVALID_ID_VALUE;

		for(Processor processor: this.getProcessorList())
		{
			if(processorName.equals(this.name))
			{
				processorId = processor.getId();
				break;
			}
		}
		
		if(processorId == Constants.INVALID_ID_VALUE)
			throw new InvalidDataInMetadataFileException("There is no processor name called " + processorName);
		
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
	
	private void putTaskHierarchicallyToTaskMap(Task task, HashMap<String, Task> globalTaskMap)
	{
		Task currentTask;
		currentTask = task;
		while(this.taskMap.containsKey(currentTask.getName()) == false && 
			currentTask.getParentTaskGraphName().equals(Constants.TOP_TASKGRAPH_NAME) == false)
		{
			this.taskMap.put(currentTask.getName(), currentTask);
			currentTask = globalTaskMap.get(currentTask.getParentTaskGraphName());
		}
		
	}
	
	private void makeMultipleCompositeTaskMapping(String[] splitedFileName, File scheduleFile, HashMap<String, Task> globalTaskMap)
										throws CICXMLException, InvalidDataInMetadataFileException 
	{
		int scheduleId;
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
		
		scheduleDOM = scheduleLoader.loadResource(scheduleFile.getAbsolutePath());
		
		scheduleId = Integer.parseInt(splitedFileName[ScheduleFileNameOffset.SCHEDULE_ID.getValue()]);
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
				procId = getProcessorIdByName(scheduleGroup.getPoolName());
				// TODO: currently the last-picked processor name is used for searching the device name
				// get processor name from schedule information
				processorName = scheduleGroup.getPoolName();

				if(isProcessorNameLocatedInDevice(processorName) == true)
				{
					CompositeTaskMappedProcessor mappedProcessor = new CompositeTaskMappedProcessor(procId, 
							scheduleGroup.getLocalId().intValue(), modeId, sequenceId);
					CompositeTaskSchedule taskSchedule = new CompositeTaskSchedule(scheduleId, throughputConstraint);
					
					if(taskName.equals(Constants.TOP_TASKGRAPH_NAME))
						compositeMappingInfo = getCompositeMappingInfo(taskName, Constants.INVALID_ID_VALUE);
					else
						compositeMappingInfo = getCompositeMappingInfo(taskName, task.getId());
					
					fillCompositeTaskSchedule(taskSchedule, scheduleGroup) ;				
					mappedProcessor.putCompositeTaskSchedule(taskSchedule);
					compositeMappingInfo.putProcessor(mappedProcessor);
					compositeMappingInfo.setMappedDeviceName(this.name);
					sequenceId++;
					putTaskHierarchicallyToTaskMap(task, globalTaskMap);
				}
			}
		}
	}
	
	private void makeCompositeTaskMappingInfo(HashMap<String, Task> globalTaskMap, String scheduleFolderPath) throws FileNotFoundException, InvalidScheduleFileNameException, InvalidDataInMetadataFileException {
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
	
	private boolean checkTaskIsIncludedInCompositeTask(String taskName)
	{
		boolean isInsideCompositeTask = false;
		Task task;
		TaskGraph parentTaskGraph;
		
		task = this.taskMap.get(taskName);
		
		do
		{
			if(this.staticScheduleMappingInfo.containsKey(task.getParentTaskGraphName()) == true)
			{
				isInsideCompositeTask = true;
				break;
			}
			
			parentTaskGraph = this.taskGraphMap.get(task.getParentTaskGraphName());
			task = parentTaskGraph.getParentTask();
		} while(task != null);
		
		return isInsideCompositeTask;
	}
	
	// TODO
	private void makeGeneralTaskMappingInfo(CICMappingType mapping_metadata) throws InvalidDataInMetadataFileException
	{		
		for(MappingTaskType mappedTask: mapping_metadata.getTask())
		{
			if(checkTaskIsIncludedInCompositeTask(mappedTask.getName()) == false)
			{
				Task task = this.taskMap.get(mappedTask.getName());
				GeneralTaskMappingInfo mappingInfo = new GeneralTaskMappingInfo(mappedTask.getName(), getTaskType(mappedTask.getName()), 
														task.getParentTaskGraphName(), task.getInGraphIndex());
				for(MappingDeviceType device: mappedTask.getDevice())
				{
					// TODO: multiple task mapping on different devices is not supported now
					mappingInfo.setMappedDeviceName(device.getName()); 
					
					for(MappingProcessorIdType proc: device.getProcessor())
					{
						MappedProcessor processor = new MappedProcessor(getProcessorIdByName(proc.getPool()), proc.getLocalId().intValue());
						mappingInfo.putProcessor(processor);
					}
				}
				
				if(this.generalMappingInfo.containsKey(mappedTask.getName()) == false)
				{
					this.generalMappingInfo.put(mappedTask.getName(), mappingInfo);				
				}
				else // if same task is already in the mappingInfo, ignore the later one
				{
					// ignore the mapping (because the duplicated key means it already registered 
				}
			}
		}
	}
	
	public void putInDeviceTaskInformation(HashMap<String, Task> globalTaskMap, 
									String scheduleFolderPath, CICMappingType mapping_metadata)
	{
		try {
			makeCompositeTaskMappingInfo(globalTaskMap, scheduleFolderPath);
		} catch (FileNotFoundException | InvalidScheduleFileNameException | InvalidDataInMetadataFileException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
//		for(GeneralTaskMappingInfo mappingInfo: globalMappingInfo.values())
//		{
//			// mapped to current device
//			if(mappingInfo.getMappedDeviceName().equals(this.name))
//			{
//				TaskGraph taskGraph;
//				
//				//while()
//				Task task =	globalTaskMap.get(mappingInfo.getTaskName());
//				this.taskMap.put(task.getName(), task);
//				
//				if(this.taskGraphMap.containsKey(task.getParentTaskGraphName()))
//				{
//					taskGraph = this.taskGraphMap.get(task.getParentTaskGraphName());
//				}
//				else
//				{
//					taskGraph = new TaskGraph(task.getParentTaskGraphName());
//					this.taskGraphMap.put(task.getParentTaskGraphName(), taskGraph);
//				}
//				
//				taskGraph.putTask(task);
//			}
//		}
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
}
