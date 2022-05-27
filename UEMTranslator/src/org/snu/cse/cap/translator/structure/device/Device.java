package org.snu.cse.cap.translator.structure.device;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import org.snu.cse.cap.translator.Constants;
import org.snu.cse.cap.translator.structure.ExecutionPolicy;
import org.snu.cse.cap.translator.structure.InvalidDataInMetadataFileException;
import org.snu.cse.cap.translator.structure.TaskGraph;
import org.snu.cse.cap.translator.structure.TaskGraphController;
import org.snu.cse.cap.translator.structure.TaskGraphType;
import org.snu.cse.cap.translator.structure.communication.channel.Channel;
import org.snu.cse.cap.translator.structure.communication.channel.ChannelPort;
import org.snu.cse.cap.translator.structure.communication.multicast.MulticastGroup;
import org.snu.cse.cap.translator.structure.communication.multicast.MulticastPort;
import org.snu.cse.cap.translator.structure.device.connection.Connection;
import org.snu.cse.cap.translator.structure.device.connection.ConstrainedSerialConnection;
import org.snu.cse.cap.translator.structure.device.connection.EncryptionInfo;
import org.snu.cse.cap.translator.structure.device.connection.IPConnection;
import org.snu.cse.cap.translator.structure.device.connection.InvalidDeviceConnectionException;
import org.snu.cse.cap.translator.structure.device.connection.SSLKeyInfo;
import org.snu.cse.cap.translator.structure.device.connection.SerialConnection;
import org.snu.cse.cap.translator.structure.device.connection.TCPConnection;
import org.snu.cse.cap.translator.structure.device.connection.UDPConnection;
import org.snu.cse.cap.translator.structure.device.connection.UnconstrainedSerialConnection;
import org.snu.cse.cap.translator.structure.device.connection.SSLTCPConnection;
import org.snu.cse.cap.translator.structure.gpu.TaskGPUSetupInfo;
import org.snu.cse.cap.translator.structure.library.Library;
import org.snu.cse.cap.translator.structure.library.LibraryConnection;
import org.snu.cse.cap.translator.structure.mapping.CompositeTaskMappedProcessor;
import org.snu.cse.cap.translator.structure.mapping.CompositeTaskMappingInfo;
import org.snu.cse.cap.translator.structure.mapping.CompositeTaskSchedule;
import org.snu.cse.cap.translator.structure.mapping.GeneralMappedProcessor;
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
import org.snu.cse.cap.translator.structure.module.Module;
import org.snu.cse.cap.translator.structure.task.Task;
import org.snu.cse.cap.translator.structure.task.TaskLoopType;
import org.snu.cse.cap.translator.structure.task.TaskMode;
import org.snu.cse.cap.translator.structure.task.TaskMode.ChildTaskTraverseCallback;
import org.snu.cse.cap.translator.structure.task.TaskModeTransition;
import org.snu.cse.cap.translator.structure.task.TaskShapeType;

import hopes.cic.exception.CICXMLException;
import hopes.cic.xml.CICGPUSetupType;
import hopes.cic.xml.CICMappingType;
import hopes.cic.xml.CICScheduleType;
import hopes.cic.xml.CICScheduleTypeLoader;
import hopes.cic.xml.GPUTaskType;
import hopes.cic.xml.MappingDeviceType;
import hopes.cic.xml.MappingSetType;
import hopes.cic.xml.MappingProcessorIdType;
import hopes.cic.xml.MappingTaskType;
import hopes.cic.xml.ScheduleElementType;
import hopes.cic.xml.ScheduleGroupType;
import hopes.cic.xml.TaskGroupForScheduleType;

public class Device {
	private String name;
	private int id;
	private ArrayList<Processor> processorList;
	private HashMap<String, Connection> connectionList;

	private ArchitectureType architecture;
	private SoftwarePlatformType platform;
	private RuntimeType runtime;
	private SchedulingMethod scheduler;

	// in-device metadata information
	private ArrayList<Channel> channelList;
	private HashMap<String, MulticastGroup> multicastGroupList; // Group name : MulticastGroup class
	private HashMap<String, Task> taskMap; // Task name : Task class
	private HashMap<String, TaskGraph> taskGraphMap; // Task graph name : TaskGraph class
	private HashMap<String, GeneralTaskMappingInfo> generalMappingInfo; // Task name : GeneralTaskMappingInfo class
	private HashMap<String, TaskGPUSetupInfo> gpuSetupInfo; // Task name : TaskGPUSetupInfo class
	private HashMap<String, CompositeTaskMappingInfo> staticScheduleMappingInfo; // Parent task Name : CompositeTaskMappingInfo class
	private ArrayList<ChannelPort> portList;
	private HashMap<String, Library> libraryMap;
	
	private ArrayList<Module> moduleList;
	private ArrayList<EnvironmentVariable> environmentVariableList;
	
	private HashMap<String, Integer> portKeyToIndex;  //Key: taskName/portName/direction, ex) MB_Y/inMB_Y/input
	private ArrayList<TCPConnection> tcpServerList;
	private ArrayList<TCPConnection> tcpClientList;
	private HashMap<String, UDPConnection> udpList;
	private ArrayList<UnconstrainedSerialConnection> bluetoothMasterList;
	private ArrayList<UnconstrainedSerialConnection> bluetoothUnconstrainedSlaveList;
	private ArrayList<ConstrainedSerialConnection> serialConstrainedMasterList;
	private ArrayList<ConstrainedSerialConnection> serialConstrainedSlaveList;
	private ArrayList<UnconstrainedSerialConnection> serialUnconstrainedMasterList;
	private ArrayList<UnconstrainedSerialConnection> serialUnconstrainedSlaveList;
	private ArrayList<SSLTCPConnection> secureTcpServerList;
	private ArrayList<SSLTCPConnection> secureTcpClientList;
	private ArrayList<SSLKeyInfo> sslKeyInfoList;
	private ArrayList<EncryptionInfo> encryptionList;
	
	private HashSet<DeviceCommunicationType> supportedConnectionTypeList;
	private HashSet<DeviceEncryptionType> supportedEncryptionTypeSet;

	public Device(String name, int id, String architecture, String platform, String runtime, String scheduler) 
	{
		this.name = name;
		this.id = id;
		this.architecture = ArchitectureType.fromValue(architecture);
		this.platform = SoftwarePlatformType.fromValue(platform);
		this.runtime = RuntimeType.fromValue(runtime);
		this.scheduler = SchedulingMethod.fromValue(scheduler);
		this.processorList = new ArrayList<Processor>();
		this.environmentVariableList = new ArrayList<EnvironmentVariable>();
		this.connectionList = new HashMap<String, Connection>();
	
		this.channelList = new ArrayList<Channel>();
		this.taskMap = new HashMap<String, Task>();
		this.taskGraphMap = new HashMap<String, TaskGraph>();
		this.generalMappingInfo = new HashMap<String, GeneralTaskMappingInfo>();
		this.gpuSetupInfo = new HashMap<String, TaskGPUSetupInfo>();
		this.staticScheduleMappingInfo = new HashMap<String, CompositeTaskMappingInfo>();
		this.libraryMap = new HashMap<String, Library>();
		this.portList = new ArrayList<ChannelPort>();
		this.multicastGroupList = new HashMap<String, MulticastGroup>();
		
		this.moduleList = new ArrayList<Module>();
		
		this.portKeyToIndex = new HashMap<String, Integer>();
		this.tcpServerList = new ArrayList<TCPConnection>();
		this.tcpClientList = new ArrayList<TCPConnection>();
		this.udpList = new HashMap<String, UDPConnection>();
		this.bluetoothMasterList = new ArrayList<UnconstrainedSerialConnection>();
		this.bluetoothUnconstrainedSlaveList = new ArrayList<UnconstrainedSerialConnection>();
		this.serialConstrainedMasterList = new ArrayList<ConstrainedSerialConnection>();
		this.serialConstrainedSlaveList = new ArrayList<ConstrainedSerialConnection>();
		this.serialUnconstrainedMasterList = new ArrayList<UnconstrainedSerialConnection>();
		this.serialUnconstrainedSlaveList = new ArrayList<UnconstrainedSerialConnection>();
		this.secureTcpServerList = new ArrayList<SSLTCPConnection>();
		this.secureTcpClientList = new ArrayList<SSLTCPConnection>();
		this.sslKeyInfoList = new ArrayList<SSLKeyInfo>();
		this.encryptionList = new ArrayList<EncryptionInfo>();
		

		this.supportedConnectionTypeList = new HashSet<DeviceCommunicationType>();
		this.supportedEncryptionTypeSet = new HashSet<DeviceEncryptionType>();
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
	
	public int getId() {
		return this.id;
	}
	
	public boolean isGPUMapped()
	{
		if (this.gpuSetupInfo.size() == 0)
		{
			return false;
		}
		return true;
	}
	
	public boolean useCommunication()
	{
		if (this.connectionList.size() == 0 && this.supportedConnectionTypeList.size() == 0)
		{
			return false;
		}
		return true;
	}

	public boolean useEncryption() 
	{
		if (this.encryptionList.size() == 0) 
		{
			return false;
		}

		return true;
	}


	// recursive function
	private int recursiveScheduleLoopInsert(List<ScheduleItem> scheduleItemList,
			List<ScheduleElementType> scheduleElementList, int depth, int maxDepth, Map<String, Task> globalTaskMap)
	{
		ScheduleLoop scheduleInloop;
		ScheduleTask scheduleTask;
		int nextDepth = 0;
		
		if(maxDepth < depth)
			maxDepth = depth;
		
		for(ScheduleElementType scheduleElement: scheduleElementList)
		{
			if(scheduleElement.getLoop() != null) {
				scheduleInloop = new ScheduleLoop(scheduleElement.getLoop().getRepetition().intValue(), depth);
				if(scheduleInloop.getRepetition() > 1)
					nextDepth = depth + 1;
				else
					nextDepth = depth;
				maxDepth = recursiveScheduleLoopInsert(scheduleInloop.getScheduleItemList(), scheduleElement.getLoop().getScheduleElement(), 
														nextDepth, maxDepth, globalTaskMap);
				scheduleItemList.add(scheduleInloop);
			}
			else if(scheduleElement.getTask() != null) 	{
				scheduleTask = new ScheduleTask(scheduleElement.getTask().getName(), scheduleElement.getTask().getRepetition().intValue(), depth);
				scheduleItemList.add(scheduleTask);
				putTaskHierarchicallyToTaskMap(scheduleTask.getTaskName(), globalTaskMap);
			}
			else {
				// do nothing
			}			
		}
		
		return maxDepth;
	}
	
	private CompositeTaskSchedule fillCompositeTaskSchedule(CompositeTaskSchedule taskSchedule, ScheduleGroupType scheduleGroup, 
			Map<String, Task> globalTaskMap) 
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
	
	private int getModeIdByName(String taskName, String modeName, Map<String, Task> globalTaskMap)
			throws InvalidDataInMetadataFileException
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
	
	private CompositeTaskMappingInfo getCompositeMappingInfo(String taskName, int taskId, int priority)
			throws InvalidDataInMetadataFileException
	{
		CompositeTaskMappingInfo compositeMappingInfo;

		if(this.staticScheduleMappingInfo.containsKey(taskName) == false)
		{//modeId 
			compositeMappingInfo = new CompositeTaskMappingInfo(taskName, taskId, priority);

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
	
	private boolean containsStaticScheduleTask(ArrayList<Task> taskList)
	{
		boolean staticScheduledTaskFound = false;
		
		for(Task task : taskList)
		{
			if(task.isStaticScheduled() == true)
			{
				staticScheduledTaskFound = true;
				break;
			}
		}
		
		return staticScheduledTaskFound;
	}
	
	private boolean containsControlTask(ArrayList<Task> taskList)
	{
		boolean controlTaskFound = false;
		
		for(Task task : taskList)
		{
			if(task.getType() == TaskShapeType.CONTROL)
			{
				controlTaskFound = true;
				break;
			}
		}
		
		return controlTaskFound;
	}
	
	private void setTaskGraphControllerType()
	{
		for(TaskGraph taskGraph: this.taskGraphMap.values())
		{
			switch(taskGraph.getTaskGraphType())
			{
			case DATAFLOW:
				if(taskGraph.getParentTask() != null) {
					if(taskGraph.getParentTask().getModeTransition() != null && 
						taskGraph.getParentTask().getModeTransition().getModeMap().size() > 1) {
						if(containsStaticScheduleTask(taskGraph.getTaskList()) == true) {
							taskGraph.setControllerType(TaskGraphController.STATIC_MODE_TRANSITION);
						}
						else {
							taskGraph.setControllerType(TaskGraphController.DYNAMIC_MODE_TRANSITION);
						}
					}
					else if(taskGraph.getParentTask().getLoopStruct() != null) {
						if(containsStaticScheduleTask(taskGraph.getTaskList()) == true) {
							if(taskGraph.getParentTask().getLoopStruct().getLoopType() == TaskLoopType.DATA) {
								taskGraph.setControllerType(TaskGraphController.STATIC_DATA_LOOP);
							}
							else {
								taskGraph.setControllerType(TaskGraphController.STATIC_CONVERGENT_LOOP);
							}
						}
						else {
							if(taskGraph.getParentTask().getLoopStruct().getLoopType() == TaskLoopType.DATA) {
								taskGraph.setControllerType(TaskGraphController.DYNAMIC_DATA_LOOP);
							}
							else {
								taskGraph.setControllerType(TaskGraphController.DYNAMIC_CONVERGENT_LOOP);
							}
						}
					}
				}
				break;
			case PROCESS_NETWORK:
				if(containsControlTask(taskGraph.getTaskList()) == true) {
					taskGraph.setControllerType(TaskGraphController.CONTROL_TASK_INCLUDED);
				}
				break;
			default:
				// skip HYBRID (TODO: remove HYBRID?)
				break;
			}
		}
	}
	
	private void putTaskHierarchicallyToTaskMap(String taskName, Map<String, Task> globalTaskMap)
	{
		Task currentTask;
		Task parentTask;
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
				parentTask = globalTaskMap.get(currentTask.getParentTaskGraphName());
				
				if(parentTask != null)
				{
					taskGraph = new TaskGraph(currentTask.getParentTaskGraphName(), parentTask.getTaskGraphProperty());	
				}
				else
				{
					taskGraph = new TaskGraph(currentTask.getParentTaskGraphName());
				}
				
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
	
	private void makeMultipleCompositeTaskMapping(String[] splitedFileName,
			File scheduleFile,
			Map<String, Task> globalTaskMap)
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
						compositeMappingInfo = getCompositeMappingInfo(taskName, Constants.INVALID_ID_VALUE,
								Constants.INVALID_ID_VALUE);
					}
					else
						compositeMappingInfo = getCompositeMappingInfo(taskName, task.getId(), task.getPriority());
					
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
	
	private void setCompositeTaskMappingInfo(Map<String, Task> globalTaskMap,
			String scheduleFolderPath)
			throws FileNotFoundException, InvalidScheduleFileNameException, InvalidDataInMetadataFileException {
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
	
	private boolean checkTaskIsIncludedInCompositeTask(String taskName, Map<String, Task> globalTaskMap)
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
	

	private void setGeneralTaskMappingInfo(CICMappingType mapping_metadata, Map<String, Task> globalTaskMap)
			throws InvalidDataInMetadataFileException, NoProcessorFoundException
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
							task.getParentTaskGraphName(), task.getInGraphIndex(), task.getPriority());
					
					mappingInfo.setMappedDeviceName(device.getName());

					for (MappingSetType setType : device.getMappingSet()) {
						for (MappingProcessorIdType proc : setType.getProcessor()) {
							MappedProcessor processor = new GeneralMappedProcessor(getProcessorIdByName(proc.getPool()),
									proc.getLocalId().intValue(), setType.getName());
							mappingInfo.putProcessor(processor);

							if (setType.getName().equals(Constants.DEFAULT_STRING_NAME)) {
								mappingInfo.putDefaultGeneralMappedProcessor((GeneralMappedProcessor) processor);
							}
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
					break;
				}
			}
		}
	}
	
	private void setupGPUInfoPerTask(CICGPUSetupType gpusetup_metadata, Map<String, Task> globalTaskMap)
			throws InvalidDataInMetadataFileException, NoProcessorFoundException
	{
		for(GPUTaskType mappedTask: gpusetup_metadata.getTasks().getTask())
		{
			if(this.taskMap.containsKey(mappedTask.getName()) == true /*&& checkTaskIsIncludedInCompositeTask(mappedTask.getName(), globalTaskMap) == false*/)
			{
				TaskGPUSetupInfo gpuSetupInfo = new TaskGPUSetupInfo(mappedTask.getName(), getTaskType(mappedTask.getName()), mappedTask.getClustering(), mappedTask.getPipelining(), mappedTask.getMaxStream().intValue());
				
				gpuSetupInfo.setBlockSizeWidth(mappedTask.getGlobalWorkSize().getWidth());
				gpuSetupInfo.setBlockSizeHeight(mappedTask.getGlobalWorkSize().getHeight());
				gpuSetupInfo.setBlockSizeDepth(mappedTask.getGlobalWorkSize().getDepth());

				gpuSetupInfo.setThreadSizeWidth(mappedTask.getLocalWorkSize().getWidth());
				gpuSetupInfo.setThreadSizeHeight(mappedTask.getLocalWorkSize().getHeight());
				gpuSetupInfo.setThreadSizeDepth(mappedTask.getLocalWorkSize().getDepth());
				
				if(this.gpuSetupInfo.containsKey(mappedTask.getName()) == false)
				{
					this.gpuSetupInfo.put(mappedTask.getName(), gpuSetupInfo);				
				}
				else // if same task is already in the gpumappingInfo, ignore the later one
				{
					// ignore the mapping (because the duplicated key means it already registered 
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
	
	public void setChannelPortIndex()
	{
		for(Channel channel : this.channelList)
		{
			channel.setPortIndexByPortList(this.portList);	
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
			else if(compositeMappingInfo.getParentTaskName().equals(Constants.TOP_TASKGRAPH_NAME))
			{
				// full graph is data flow
				taskGraph = this.taskGraphMap.get(compositeMappingInfo.getParentTaskName());
				recursiveSetSubgraphTaskToStaticScheduled(taskGraph);
			}
		}
	}
	
	private void recursivePutTask(CompositeTaskSchedule schedule, ArrayList<ScheduleItem> scheduleItemList, TaskModeTransition targetTaskModeTransition, 
			CompositeTaskMappedProcessor compositeMappedProc) 
	{		
		for(ScheduleItem item: scheduleItemList)
		{
			switch(item.getItemType())
			{
			case LOOP:
				ScheduleLoop scheduleLoop = (ScheduleLoop) item; 
				recursivePutTask(schedule, scheduleLoop.getScheduleItemList(), targetTaskModeTransition, compositeMappedProc);
				break;
			case TASK:
				ScheduleTask task = (ScheduleTask) item;
				Task  taskObject = this.taskMap.get(task.getTaskName());
				targetTaskModeTransition.putRelatedChildTask(compositeMappedProc.getProcessorId(), compositeMappedProc.getProcessorLocalId(), 
						compositeMappedProc.getModeId(), schedule.getThroughputConstraint(), taskObject);

				break;
			}
		}
	}

	private void putRelatedChildTaskInCompositeTask(TaskModeTransition targetTaskModeTransition, 
			CompositeTaskMappedProcessor compositeMappedProc)
	{
		for(CompositeTaskSchedule schedule: compositeMappedProc.getCompositeTaskScheduleList())
		{		
			recursivePutTask(schedule, schedule.getScheduleList(), targetTaskModeTransition, compositeMappedProc);
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
					taskSet.put(taskName, Integer.valueOf(1));
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
			for(Integer throughputConstraint: mode.getTaskProcMapWithThroughput().keySet())
			{
				mode.traverseRelatedChildTask(throughputConstraint, childTaskCallback, relatedTaskMap);
				
				for(String taskName: relatedTaskMap.keySet())
				{
					Task task = this.taskMap.get(taskName);
					int newMappedProcNum = relatedTaskMap.get(taskName).intValue();
					if(task.getTaskFuncNum() < newMappedProcNum)
					{
						task.setTaskFuncNum(newMappedProcNum);
						System.out.println("task ("+task.getName() +") num: " + newMappedProcNum);
					}
				}
				relatedTaskMap.clear();			
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
			task.setTaskFuncNum(generalMappingInfo.getDefaultGeneralMappedProcessorList().size());
		}
	}
	
	private void recursiveScheduleLoopTraverse(ArrayList<ScheduleItem> scheduleItemList, HashMap<String, TaskFuncIdChecker> taskFuncIdMap, int modeId, int throughputConstraint)
	{
		for(ScheduleItem scheduleItem : scheduleItemList)
		{
			if(scheduleItem.getItemType() == ScheduleItemType.LOOP)
			{
				ScheduleLoop scheduleInnerLoop = (ScheduleLoop) scheduleItem;
				recursiveScheduleLoopTraverse(scheduleInnerLoop.getScheduleItemList(), taskFuncIdMap, modeId, throughputConstraint);
			}
			else
			{
				ScheduleTask scheduleTask = (ScheduleTask) scheduleItem;
				String taskFuncIdKey = modeId + Constants.TASK_NAME_FUNC_ID_SEPARATOR + throughputConstraint + scheduleTask.getTaskName();
				
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
					recursiveScheduleLoopTraverse(taskScheule.getScheduleList(), taskFuncIdMap, compositeMappedProcessor.getModeId(), taskScheule.getThroughputConstraint());
					
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
	
	private boolean matchTaskIdInPort(ChannelPort inputPort, String taskName)
	{
		boolean matched = false;
		ChannelPort port;
		
		port = inputPort;
		
		while(port != null)
		{
			if(port.getTaskName().equals(taskName))
			{
				matched = true;
				break;
			}
			
			port = port.getSubgraphPort();
		}
		
		return matched;
	}
	
	private boolean isChannelLocatedInSameTaskGraph(Channel channel)
	{
		boolean sameGraph = false;
		ChannelPort inputPort = channel.getInputPort();
		ChannelPort outputPort = channel.getOutputPort();
		
		while(inputPort != null && outputPort != null)
		{
			if(inputPort.getTaskId() != outputPort.getTaskId())
			{
				if(inputPort.getSubgraphPort() == null && outputPort.getSubgraphPort() == null)
				{
					sameGraph = true;
				}
				break;
			}
			
			inputPort = inputPort.getSubgraphPort();
			outputPort = outputPort.getSubgraphPort();
		}
		
		return sameGraph;
	}
	
	private boolean checkIsSourceTask(Task task)
	{
		boolean isSourceTask = true;
		
		for(Channel channel: this.channelList)
		{
			if(matchTaskIdInPort(channel.getInputPort(), task.getName()) == true)
			{
				if(isChannelLocatedInSameTaskGraph(channel) == true)
				{
					isSourceTask = false;
					break;
				}
			}
		}
		
		return isSourceTask;
	}
	
	private void recursiveFindAndInsertSourceTask(CompositeTaskSchedule taskSchedule, ArrayList<ScheduleItem> scheduleItemList, HashMap<String, Task> srcTaskMap)
	{
		for(ScheduleItem scheduleItem : scheduleItemList)
		{
			if(scheduleItem.getItemType() == ScheduleItemType.LOOP)
			{
				ScheduleLoop scheduleInnerLoop = (ScheduleLoop) scheduleItem;
				recursiveFindAndInsertSourceTask(taskSchedule, scheduleInnerLoop.getScheduleItemList(), srcTaskMap);
			}
			else
			{
				ScheduleTask scheduleTask = (ScheduleTask) scheduleItem;
				
				Task task = this.taskMap.get(scheduleTask.getTaskName());
				
				if(checkIsSourceTask(task) == true && srcTaskMap.containsKey(scheduleTask.getTaskName()) == false)
				{
					srcTaskMap.put(scheduleTask.getTaskName(), task);
					taskSchedule.setHasSourceTask(true);
				}
			}
		}
	}
	
	public void setSrcTaskOfMTM()
	{
		CompositeTaskMappingInfo compositeMappingInfo;
		CompositeTaskMappedProcessor compositeMappedProc;
		for(Task task: this.taskMap.values())
		{
			if(task.getModeTransition() != null && task.getModeTransition().getModeMap().size() > 1 && 
				task.getChildTaskGraphName() != null && task.isStaticScheduled() == true)
			{
				compositeMappingInfo = this.staticScheduleMappingInfo.get(task.getName());
				for(MappedProcessor mappedProcessor: compositeMappingInfo.getMappedProcessorList())
				{
					compositeMappedProc = (CompositeTaskMappedProcessor) mappedProcessor;
					
					for(CompositeTaskSchedule taskScheule: compositeMappedProc.getCompositeTaskScheduleList())
					{
						recursiveFindAndInsertSourceTask(taskScheule, taskScheule.getScheduleList(), compositeMappedProc.getSrcTaskMap());
					}
				}
			}
		}

		
	}
	
	public void putInDeviceTaskInformation(Map<String, Task> globalTaskMap,
									String scheduleFolderPath, CICMappingType mapping_metadata, ExecutionPolicy executionPolicy, 
									TaskGraphType topTaskGraphType, CICGPUSetupType gpusetup_metadata)
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
			setParentTaskOfTaskGraph();
			setTaskGraphControllerType();
			setNumOfProcsOfTasks();
			break;
		// TODO: fully dynamic is not supported now
		case FULLY_DYNAMIC: // Need mapped device information (needed file: mapping)
			setGeneralTaskMappingInfo( mapping_metadata, globalTaskMap);
			setParentTaskOfTaskGraph();
			break;
		}
		
		setTaskGraphControllerType();
		
		if(gpusetup_metadata != null){
			setupGPUInfoPerTask(gpusetup_metadata,globalTaskMap);
		}
		// set top-level task graph property which is located at CICAlgorithm element's property attribute
		TaskGraph taskGraph = this.taskGraphMap.get(Constants.TOP_TASKGRAPH_NAME);
		if(taskGraph != null)
			taskGraph.setTaskGraphType(topTaskGraphType);
	}

	private boolean recursiveIsLibraryUsedInDevice(List<LibraryConnection> libraryConnection,
			Map<String, Library> globalLibraryMap)
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
	
	public void putInDeviceLibraryInformation(Map<String, Library> globalLibraryMap)
	{
		for(Library library: globalLibraryMap.values())
		{
			if(recursiveIsLibraryUsedInDevice(library.getLibraryConnectionList(), globalLibraryMap) == true)
			{
				this.libraryMap.put(library.getName(), library);
			}
		}
		
		linkLibraryToLibraryAndTasks();
	}
	
	private void linkLibraryToLibraryAndTasks()
	{
		for(Library library: this.libraryMap.values())
		{
			for(LibraryConnection connection: library.getLibraryConnectionList())
			{
				if(connection.isMasterLibrary() == false)
				{					
					Task task = this.taskMap.get(connection.getMasterName());
					
					if(task.getMasterPortToLibraryMap().containsKey(connection.getPortName()) == false)
					{
						task.getMasterPortToLibraryMap().put(connection.getPortName(), library);
					}
				}
				else
				{
					Library masterLibrary = this.libraryMap.get(connection.getMasterName());
					
					if(masterLibrary.getMasterPortToLibraryMap().containsKey(connection.getPortName()) == false)
					{
						masterLibrary.getMasterPortToLibraryMap().put(connection.getPortName(), library);
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

	public void putSupportedConnectionType(DeviceCommunicationType connectionType)
	{
		this.supportedConnectionTypeList.add(connectionType);
	}
	
	public void putSupportedEncryptionType(DeviceEncryptionType encryptionType)
	{
		this.supportedEncryptionTypeSet.add(encryptionType);
	}
	
	public HashSet<DeviceCommunicationType> getRequiredCommunicationSet()
	{
		return this.supportedConnectionTypeList;
	}

	public HashSet<DeviceEncryptionType> getRequiredEncryptionSet()
	{
		return this.supportedEncryptionTypeSet;
	}

	public void putEncryptionList(EncryptionInfo encryption)
	{
		this.encryptionList.add(encryption);
	}

	public int getEncryptionIndex(String encryptionType, String userKey) {
		int index = -1;

		for (EncryptionInfo e : this.encryptionList) {
			if (e.getEncryptionType().contentEquals(encryptionType) && e.getUserKey().contentEquals(userKey)) {
				index = this.encryptionList.indexOf(e);
				break;
			}
		}

		return index;
	}

	public void putUDPConnection(String connectionId, UDPConnection udpConnection) 
	{
		this.udpList.put(connectionId, udpConnection);
	}
	
	public UDPConnection getUDPConnection(String connectionId)
	{
		return this.udpList.get(connectionId);
	}
	public ArrayList<UDPConnection> getUDPConnectionList()
	{
		return new ArrayList<UDPConnection>(this.udpList.values());
	}
	
	public boolean checkUDPConnectionIsCreated(String connectionId) {
		return this.udpList.containsKey(connectionId);
	}
	
	private int putKeyInfo(SSLTCPConnection connection) {
		int index = -1;

		for (SSLKeyInfo keyInfo : sslKeyInfoList) {
			if (keyInfo.equals(connection.getSSLKeyInfo())) {
				index = sslKeyInfoList.indexOf(keyInfo);
			}
		}

		if (index == -1) {
			sslKeyInfoList.add(connection.getSSLKeyInfo());
			index = sslKeyInfoList.indexOf(connection.getSSLKeyInfo());
		}
		
		return index;
	}

	public void putConnection(Connection connection) 
	{
		this.connectionList.put(connection.getName(), connection);
		switch(connection.getProtocol())
		{
		case TCP:
			if(connection.getRole().equalsIgnoreCase(IPConnection.ROLE_SERVER) == true) {
				this.tcpServerList.add((TCPConnection) connection);
			}
			else {
				this.tcpClientList.add((TCPConnection) connection);	
			}
			break;
		case SERIAL:
			switch(this.platform)
			{
			case ARDUINO:
				if (connection.getRole().equalsIgnoreCase(SerialConnection.ROLE_MASTER) == true) {
					this.serialConstrainedMasterList.add((ConstrainedSerialConnection) connection);
				} else {
					this.serialConstrainedSlaveList.add((ConstrainedSerialConnection) connection);
				}
				break;
			case LINUX:
				switch(connection.getNetwork())
				{
				case BLUETOOTH:
					if(connection.getRole().equalsIgnoreCase(SerialConnection.ROLE_MASTER) == true) {
						this.bluetoothMasterList.add((UnconstrainedSerialConnection) connection);	
					}
					else {
						this.bluetoothUnconstrainedSlaveList.add((UnconstrainedSerialConnection) connection);
					}					
					break;
				case USB:
				case WIRE:
					if(connection.getRole().equalsIgnoreCase(SerialConnection.ROLE_MASTER) == true) {
						this.serialUnconstrainedMasterList.add((UnconstrainedSerialConnection) connection);	
					}
					else {
						this.serialUnconstrainedSlaveList.add((UnconstrainedSerialConnection) connection);	
					}			
					break;
				case ETHERNET_WI_FI:
				default:
					break;
				}

				break;
			case UCOS3:
			case WINDOWS:
			default:
				break;
			}
			break;
		case SECURE_TCP:
			int index = putKeyInfo((SSLTCPConnection) connection);
			((SSLTCPConnection) connection).setKeyInfoIndex(index);

			if (connection.getRole().equalsIgnoreCase(IPConnection.ROLE_SERVER) == true) {
				this.secureTcpServerList.add((SSLTCPConnection) connection);
			} else {
				this.secureTcpClientList.add((SSLTCPConnection) connection);
			}
			break;
		default:
			break;
		}
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
	
	public SchedulingMethod getScheduler() {
		return scheduler;
	}

	public void setScheduler(SchedulingMethod scheduler) {
		this.scheduler = scheduler;
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

	public ArrayList<ChannelPort> getPortList() {
		return portList;
	}

	public HashMap<String, MulticastGroup> getMulticastGroupMap() {
		return multicastGroupList;
	}
	
	public void putMulticastGroup(MulticastGroup multicastGroup) {
		this.multicastGroupList.put(multicastGroup.getGroupName(), multicastGroup);
	}
	
	public void putMulticastPort(MulticastPort multicastPort) {
		MulticastGroup multicastGroup;
		
		if(this.multicastGroupList.containsKey(multicastPort.getGroupName()))
		{
			multicastGroup = this.multicastGroupList.get(multicastPort.getGroupName());
		}
		else
		{
			throw new IllegalArgumentException();
		}
		
		multicastGroup.putPort(multicastPort.getDirection(), multicastPort);
		
	}
	
	public HashMap<String, Integer> getPortKeyToIndex() {
		return portKeyToIndex;
	}


	public HashMap<String, TaskGPUSetupInfo> getGpuSetupInfo() {
		return gpuSetupInfo;
	}

	public HashMap<String, Connection> getConnectionList() {
		return connectionList;
	}

	public ArrayList<TCPConnection> getTcpServerList() {
		return tcpServerList;
	}

	public ArrayList<TCPConnection> getTcpClientList() {
		return tcpClientList;
	}
	
	public ArrayList<Module> getModuleList() {
		return moduleList;
	}

	public ArrayList<EnvironmentVariable> getEnvironmentVariableList() {
		return environmentVariableList;
	}
	
	public ArrayList<UnconstrainedSerialConnection> getBluetoothMasterList() {
		return bluetoothMasterList;
	}

	public ArrayList<UnconstrainedSerialConnection> getBluetoothUnconstrainedSlaveList() {
		return bluetoothUnconstrainedSlaveList;
	}

	public ArrayList<ConstrainedSerialConnection> getSerialConstrainedMasterList() {
		return serialConstrainedMasterList;
	}

	public ArrayList<ConstrainedSerialConnection> getSerialConstrainedSlaveList() {
		return serialConstrainedSlaveList;
	}

	public ArrayList<UnconstrainedSerialConnection> getSerialUnconstrainedMasterList() {
		return serialUnconstrainedMasterList;
	}

	public ArrayList<UnconstrainedSerialConnection> getSerialUnconstrainedSlaveList() {
		return serialUnconstrainedSlaveList;
	}
	
	public ArrayList<SSLTCPConnection> getSecureTcpServerList() {
		return secureTcpServerList;
	}

	public ArrayList<SSLTCPConnection> getSecureTcpClientList() {
		return secureTcpClientList;
	}

	public ArrayList<SSLKeyInfo> getKeyInfoList() {
		return sslKeyInfoList;
	}

	public ArrayList<EncryptionInfo> getEncryptionList() {
		return encryptionList;
	}

}
