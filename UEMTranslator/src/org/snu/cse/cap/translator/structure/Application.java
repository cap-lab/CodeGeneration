package org.snu.cse.cap.translator.structure;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Stack;

import org.snu.cse.cap.translator.structure.channel.Channel;
import org.snu.cse.cap.translator.structure.channel.ChannelArrayType;
import org.snu.cse.cap.translator.structure.channel.CommunicationType;
import org.snu.cse.cap.translator.structure.channel.LoopPortType;
import org.snu.cse.cap.translator.structure.channel.Port;
import org.snu.cse.cap.translator.structure.channel.PortSampleRate;
import org.snu.cse.cap.translator.structure.device.BluetoothConnection;
import org.snu.cse.cap.translator.structure.device.Connection;
import org.snu.cse.cap.translator.structure.device.Device;
import org.snu.cse.cap.translator.structure.device.HWCategory;
import org.snu.cse.cap.translator.structure.device.HWElementType;
import org.snu.cse.cap.translator.structure.device.Processor;
import org.snu.cse.cap.translator.structure.device.ProcessorElementType;
import org.snu.cse.cap.translator.structure.device.TCPConnection;
import org.snu.cse.cap.translator.structure.device.connection.DeviceConnection;
import org.snu.cse.cap.translator.structure.device.connection.InvalidDeviceConnectionException;
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
import org.snu.cse.cap.translator.structure.task.TaskLoopType;
import org.snu.cse.cap.translator.structure.task.TaskMode;
import org.snu.cse.cap.translator.structure.task.TaskMode.ChildTaskTraverseCallback;
import org.snu.cse.cap.translator.structure.task.TaskModeTransition;
import org.snu.cse.cap.translator.structure.task.TaskShapeType;

import Translators.Constants;
import hopes.cic.exception.CICXMLException;
import hopes.cic.xml.ArchitectureConnectType;
import hopes.cic.xml.ArchitectureConnectionSlaveType;
import hopes.cic.xml.ArchitectureDeviceType;
import hopes.cic.xml.ArchitectureElementType;
import hopes.cic.xml.ArchitectureElementTypeType;
import hopes.cic.xml.BluetoothConnectionType;
import hopes.cic.xml.CICAlgorithmType;
import hopes.cic.xml.CICArchitectureType;
import hopes.cic.xml.CICConfigurationType;
import hopes.cic.xml.CICMappingType;
import hopes.cic.xml.CICProfileType;
import hopes.cic.xml.CICScheduleType;
import hopes.cic.xml.CICScheduleTypeLoader;
import hopes.cic.xml.ChannelPortType;
import hopes.cic.xml.ChannelType;
import hopes.cic.xml.DeviceConnectionListType;
import hopes.cic.xml.MappingDeviceType;
import hopes.cic.xml.MappingProcessorIdType;
import hopes.cic.xml.MappingTaskType;
import hopes.cic.xml.ModeTaskType;
import hopes.cic.xml.ModeType;
import hopes.cic.xml.PortMapType;
import hopes.cic.xml.ScheduleElementType;
import hopes.cic.xml.ScheduleGroupType;
import hopes.cic.xml.TCPConnectionType;
import hopes.cic.xml.TaskGroupForScheduleType;
import hopes.cic.xml.TaskPortType;
import hopes.cic.xml.TaskRateType;
import hopes.cic.xml.TaskType;

enum ExecutionPolicy {
	FULLY_STATIC("Fully-Static-Execution-Policy"),
	SELF_TIMED("Self-timed-Execution-Policy"),
	STATIC_ASSIGNMENT("Static-Assignment-Execution-Policy"),
	FULLY_DYNAMIC("Fully-Dynamic-Execution-Policy"),
	;

	private final String value;
	
	private ExecutionPolicy(final String value) {
		this.value = value;
	}
	
	@Override
	public String toString() {
		return value;
	}
	
	public static ExecutionPolicy fromValue(String value) {
		 for (ExecutionPolicy c : ExecutionPolicy.values()) {
			 if (c.value.equals(value)) {
				 return c;
			 }
		 }
		 throw new IllegalArgumentException(value.toString());
	}
}

public class Application {
	// Overall metadata information
	private ArrayList<Channel> channelList;
	private HashMap<String, Task> taskMap; // Task name : Task class
	private HashMap<String, TaskGraph> taskGraphMap; // Task graph name : TaskGraph class
	private HashMap<String, GeneralTaskMappingInfo> generalMappingInfo; // Task name : GeneralTaskMappingInfo class
	private HashMap<String, CompositeTaskMappingInfo> staticScheduleMappingInfo; // Parent task Name : CompositeTaskMappingInfo class
	private HashMap<String, Port> portInfo; // Key: taskName/portName/direction, ex) MB_Y/inMB_Y/input
	private HashMap<String, Device> deviceInfo; // device name: Device class
	private HashMap<String, DeviceConnection> deviceConnectionList;
	private HashMap<String, HWElementType> elementTypeHash; // element type name : HWElementType class
	private TaskGraphType applicationGraphProperty;	
	
	public Application()
	{
		this.channelList = new ArrayList<Channel>();
		this.taskMap = new HashMap<String, Task>();
		this.taskGraphMap = new HashMap<String, TaskGraph>();
		this.generalMappingInfo = new HashMap<String, GeneralTaskMappingInfo>();
		this.staticScheduleMappingInfo = new HashMap<String, CompositeTaskMappingInfo>();
		this.portInfo = new HashMap<String, Port>();
		this.deviceInfo = new HashMap<String, Device>();
		this.elementTypeHash = new HashMap<String, HWElementType>();
		this.applicationGraphProperty = null;
		this.deviceConnectionList = new HashMap<String, DeviceConnection>();
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
	
	private void putPortInfoFromTask(TaskType task_metadata, int taskId, String taskName) {
		for(TaskPortType portType: task_metadata.getPort())
		{
			Port port = new Port(taskId, taskName, portType.getName(), portType.getSampleSize().intValue(), portType.getType().value());
			
			this.portInfo.put(taskName + Constants.NAME_SPLITER + portType.getName() + Constants.NAME_SPLITER + portType.getDirection().value(), port);
			
			if(portType.getRate() != null)
			{
				for(TaskRateType taskRate: portType.getRate())
				{ 
					PortSampleRate sampleRate = new PortSampleRate(taskRate.getMode(), taskRate.getRate().intValue());
					port.putSampleRate(sampleRate);
				}
			}
			else
			{
				// variable sample rate, do nothing
			}
		}
	}
	
	private void fillBasicTaskMapAndGraphInfo(CICAlgorithmType algorithm_metadata)
	{
		int taskId = 0;
		Task task;
		int inGraphIndex = 0;
		
		for(TaskType task_metadata: algorithm_metadata.getTasks().getTask())
		{
			TaskGraph taskGraph;
			task = new Task(taskId, task_metadata);
						
			this.taskMap.put(task.getName(), task);
			
			if(this.taskGraphMap.containsKey(task.getParentTaskGraphName()) == false)
			{
				taskGraph = new TaskGraph(task.getParentTaskGraphName());				
				this.taskGraphMap.put(task.getParentTaskGraphName(), taskGraph);
			}
			else // == true
			{
				taskGraph = this.taskGraphMap.get(task.getParentTaskGraphName());
			}
			
			inGraphIndex = taskGraph.getNumOfTasks();
			task.setInGraphIndex(inGraphIndex);
			
			taskGraph.putTask(task);
			putPortInfoFromTask(task_metadata, taskId, task.getName());

			taskId++;
		}
		
		if(algorithm_metadata.getPortMaps() != null) 
		{
			setPortMapInformation(algorithm_metadata);	
		}
	}
	
	// subgraphPort, upperGraphPort, maxAvailableNum
	private void setPortMapInformation(CICAlgorithmType algorithm_metadata)
	{
		for(PortMapType portMapType: algorithm_metadata.getPortMaps().getPortMap())
		{
			Task task = this.taskMap.get(portMapType.getTask());
			Port port = this.portInfo.get(portMapType.getTask() + Constants.NAME_SPLITER + portMapType.getPort() + 
										Constants.NAME_SPLITER + portMapType.getDirection());
			
			Port childPort = this.portInfo.get(portMapType.getChildTask() + Constants.NAME_SPLITER + portMapType.getChildTaskPort() + 
					Constants.NAME_SPLITER + portMapType.getDirection());
			
			port.setSubgraphPort(childPort);
			childPort.setUpperGraphPort(port);
			
			port.setLoopPortType(LoopPortType.fromValue(portMapType.getType().value()));
			
			for(PortSampleRate portRate: port.getPortSampleRateList())
			{
				// maximum available number will be more than 1
				if(task.getLoopStruct().getLoopType() == TaskLoopType.CONVERGENT || 
					LoopPortType.fromValue(portMapType.getType().value()) == LoopPortType.BROADCASTING)
				{ 
					portRate.setMaxAvailableNum(task.getLoopStruct().getLoopCount());
				}
			}
		}
	}
	
	// taskMap, taskGraphList
	public void makeTaskInformation(CICAlgorithmType algorithm_metadata)
	{
		Task task;
		
		this.applicationGraphProperty = TaskGraphType.fromValue(algorithm_metadata.getProperty());
		
		fillBasicTaskMapAndGraphInfo(algorithm_metadata);
		
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

		// It only uses single modes - mode information in XML 
		ModeType mode = algorithm_metadata.getModes().getMode().get(0);
		
		for (ModeTaskType modeTask: mode.getTask())
		{
			task = this.taskMap.get(modeTask.getName());
			
			task.setExtraInformationFromModeInfo(modeTask);
		}
	}
	
	private void makeHardwareElementInformation(CICArchitectureType architecture_metadata) 
	{
		for(ArchitectureElementTypeType elementType : architecture_metadata.getElementTypes().getElementType())
		{
			if(HWCategory.PROCESSOR.getValue().equals(elementType.getCategory().value()) == true)
			{
				ProcessorElementType elementInfo = new ProcessorElementType(elementType.getName(), elementType.getModel(), 
																			elementType.getSubcategory());
				this.elementTypeHash.put(elementType.getName(), elementInfo);
			}
			else
			{
				// do nothing, ignore information
			}
		}
	}
	
	private void putConnectionsOnDevice(Device device, DeviceConnectionListType connectionList)
	{
		if(connectionList.getBluetoothConnection() != null)
		{
			for(BluetoothConnectionType connectionType: connectionList.getBluetoothConnection())
			{
				BluetoothConnection connection = new BluetoothConnection(connectionType.getName(), connectionType.getRole().toString(), 
						connectionType.getFriendlyName(), connectionType.getMAC());
				device.putConnection(connection);
			}
		}
			
		if(connectionList.getTCPConnection() != null) 
		{
			for(TCPConnectionType connectionType: connectionList.getTCPConnection())
			{
				TCPConnection connection = new TCPConnection(connectionType.getName(), connectionType.getRole().toString(), 
						connectionType.getIp(), connectionType.getPort().intValue());
				device.putConnection(connection);
			}
		}
	}
	
	private Connection findConnection(String deviceName, String connectionName) throws InvalidDeviceConnectionException
	{
		Device device;
		Connection connection;
		
		device = this.deviceInfo.get(deviceName);
		connection = device.getConnection(connectionName);
		
		return connection;
	}
	
	private void makeDeviceConnectionInformation(CICArchitectureType architecture_metadata)
	{
		if(architecture_metadata.getConnections() != null)
		{
			for(ArchitectureConnectType connectType: architecture_metadata.getConnections().getConnection())
			{
				DeviceConnection deviceConnection;
				Connection master;
				if(this.deviceConnectionList.containsKey(connectType.getMaster()))
				{
					deviceConnection = this.deviceConnectionList.get(connectType.getMaster());
				}
				else
				{
					deviceConnection = new DeviceConnection(connectType.getMaster());
					this.deviceConnectionList.put(connectType.getMaster(), deviceConnection);
				}
				
				try {
					master = findConnection(connectType.getMaster(), connectType.getConnection());

					for(ArchitectureConnectionSlaveType slaveType: connectType.getSlave())
					{
						Connection slave;
	
						slave = findConnection(slaveType.getDevice(), slaveType.getConnection());
						deviceConnection.putMasterToSlaveConnection(master, slave);
						deviceConnection.putSlaveToMasterConnection(slave, master);
					}
				} catch (InvalidDeviceConnectionException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
	}

	public void makeDeviceInformation(CICArchitectureType architecture_metadata)
	{	
		makeHardwareElementInformation(architecture_metadata);
		int id = 0;
		
		if(architecture_metadata.getDevices() != null)
		{
			for(ArchitectureDeviceType device_metadata: architecture_metadata.getDevices().getDevice())
			{
				Device device = new Device(device_metadata.getName(), device_metadata.getArchitecture(), 
											device_metadata.getPlatform(), device_metadata.getRuntime());

				for(ArchitectureElementType elementType: device_metadata.getElements().getElement())
				{
					// only handles the elements which use defined types
					if(this.elementTypeHash.containsKey(elementType.getType()) == true)
					{
						ProcessorElementType elementInfo = (ProcessorElementType) this.elementTypeHash.get(elementType.getType());
						device.putProcessingElement(id, elementType.getName(), elementInfo.getSubcategory(), elementType.getPoolSize().intValue());
						id++;
					}
				}
				
				if(device_metadata.getConnections() != null)
				{
					putConnectionsOnDevice(device, device_metadata.getConnections());
				}

				this.deviceInfo.put(device_metadata.getName(), device);
			}
			
			makeDeviceConnectionInformation(architecture_metadata);
		}
	}
	
	private MappingInfo findMappingInfoByTaskName(String taskName) throws InvalidDataInMetadataFileException
	{
		Task task;
		MappingInfo mappingInfo;
		
		mappingInfo = this.generalMappingInfo.get(taskName);
		
		if(mappingInfo == null)
		{
			task = this.taskMap.get(taskName);
			mappingInfo = this.staticScheduleMappingInfo.get(task.getParentTaskGraphName());
			
			while(mappingInfo == null && task != null)
			{
				task = this.taskMap.get(task.getParentTaskGraphName());
				mappingInfo = this.staticScheduleMappingInfo.get(task.getParentTaskGraphName());
			}
		}
		
		if(mappingInfo == null)
		{
			throw new InvalidDataInMetadataFileException();
		}
		
		return mappingInfo;
	}
	
	private void setChannelCommunicationType(Channel channel, ChannelPortType channelSrcPort, ChannelPortType channelDstPort) 
	{

		MappingInfo srcTaskMappingInfo;
		MappingInfo dstTaskMappingInfo;	
		
		try {
			srcTaskMappingInfo = findMappingInfoByTaskName(channelSrcPort.getTask());
			dstTaskMappingInfo = findMappingInfoByTaskName(channelDstPort.getTask());
			
			// Two tasks are connected on different devices
			if(srcTaskMappingInfo.getMappedDeviceName().equals(dstTaskMappingInfo.getMappedDeviceName()) == false)
			{
				throw new UnsupportedOperationException();
			}
			else // located at the same device
			{
				// TODO: this part should handle heterogeneous computing devices, so processor name check is also needed
				// currently only the first mapped processor is used for checking the both tasks are located at the same processor pool.
				if(srcTaskMappingInfo.getMappedProcessorList().get(0).getProcessorId() == 
						dstTaskMappingInfo.getMappedProcessorList().get(0).getProcessorId())
				{
					channel.setCommunicationType(CommunicationType.SHARED_MEMORY);
				}
				else
				{
					throw new UnsupportedOperationException();
				}
			}
		} catch (InvalidDataInMetadataFileException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
		
	private boolean isDataLoopTask(Task task) {
		boolean isDataLoop = false;
		
		while(task.getParentTaskGraphName().equals(Constants.TOP_TASKGRAPH_NAME) == false)
		{
			if(task.getLoopStruct().getLoopType() == TaskLoopType.DATA)
			{
				isDataLoop = true;
				break;
			}
			
			task = this.taskMap.get(task.getParentTaskGraphName());
		}
		
		return isDataLoop;
	}

	// ports and tasks are the most lower-level 
	private void setChannelType(Channel channel, Port srcPort, Port dstPort, Task srcTask, Task dstTask) {
		boolean isDstDataLoop = false;
		boolean isSrcDataLoop = false;
		boolean isDstDistributing = false;
		boolean isSrcDistributing = false;
		
		isDstDataLoop = isDataLoopTask(dstTask);
		isSrcDataLoop = isDataLoopTask(srcTask);
		isSrcDistributing = srcPort.isDistributingPort();
		isDstDistributing = dstPort.isDistributingPort();
		
		if(isSrcDataLoop == true  && isDstDataLoop == true && 
			isSrcDistributing == true && isDstDistributing == true)
		{
			channel.setChannelType(ChannelArrayType.FULL_ARRAY);
		}
		else if(isDstDataLoop == true && isDstDistributing == true)
		{
			channel.setChannelType(ChannelArrayType.INPUT_ARRAY);
		}
		else if(isSrcDataLoop == true && isSrcDistributing == true)
		{
			channel.setChannelType(ChannelArrayType.OUTPUT_ARRAY);
		}
		else
		{
			channel.setChannelType(ChannelArrayType.GENERAL);
		}
	}
	
	public void makeChannelInformation(CICAlgorithmType algorithm_metadata)
	{
		algorithm_metadata.getChannels().getChannel();
		int index = 0;
		
		for(ChannelType channelMetadata: algorithm_metadata.getChannels().getChannel())
		{
			Channel channel = new Channel(index, channelMetadata.getSize().intValue());
			
			// index 0 is only used
			// TODO: src element in XML schema file must be single occurrence.
			ChannelPortType channelSrcPort = channelMetadata.getSrc().get(0);
			ChannelPortType channelDstPort = channelMetadata.getDst().get(0);
			
			Port srcPort = this.portInfo.get(channelSrcPort.getTask() + Constants.NAME_SPLITER + channelSrcPort.getPort() + Constants.NAME_SPLITER + Constants.PortDirection.OUTPUT);
			Port dstPort = this.portInfo.get(channelDstPort.getTask() + Constants.NAME_SPLITER + channelDstPort.getPort() + Constants.NAME_SPLITER + Constants.PortDirection.INPUT);
			
			Task srcTask = this.taskMap.get(srcPort.getTaskName());
			Task dstTask = this.taskMap.get(dstPort.getTaskName());
			
			// channel type
			setChannelType(channel, srcPort, dstPort, srcTask, dstTask);			
			
			// communication type (device information)
			setChannelCommunicationType(channel, channelSrcPort, channelDstPort);
			
			// input/output port (port information)
			channel.setOutputPort(srcPort.getMostUpperPortInfo());
			channel.setInputPort(dstPort.getMostUpperPortInfo());
			
			// maximum chunk number
			channel.setMaximumChunkNum(this.taskMap);
			
			this.channelList.add(channel);
			index++;
		}
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
		
		for(Device device: this.deviceInfo.values())
		{
			for(Processor processor: device.getProcessorList())
			{
				if(processorName.equals(processor.getName()))
				{
					processorId = processor.getId();
					break;
				}
			}
			
			if(processorId != Constants.INVALID_ID_VALUE)
				break;
		}
		
		if(processorId == Constants.INVALID_ID_VALUE)
			throw new InvalidDataInMetadataFileException("There is no processor name called " + processorName);
		
		return processorId; 
	}
	
	private int getModeIdByName(String taskName, String modeName) throws InvalidDataInMetadataFileException
	{
		int modeId;
		Task task;
		
		task = this.taskMap.get(taskName);
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
	
	private String findDeviceNameFromProcessorName(String processorName) {
		String deviceName = "";
		
		for(Device device: this.deviceInfo.values())
		{
			for(Processor proc: device.getProcessorList())
			{
				if(processorName.equals(proc.getName()))
				{
					deviceName = device.getName();
					break;
				}
			}
			
			if(deviceName.length() > 0)
				break;
		}
		
		return deviceName;
	}
	
	private void makeMultipleCompositeTaskMapping(String[] splitedFileName, File scheduleFile) throws CICXMLException, InvalidDataInMetadataFileException 
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
		if(taskName.equals("SDF_" + this.taskMap.size())) // it means top-level graph
		{
			taskName = Constants.TOP_TASKGRAPH_NAME;
		}
		modeName = splitedFileName[ScheduleFileNameOffset.MODE_NAME.getValue()];
		modeId = getModeIdByName(taskName, modeName);
		throughputConstraint = getThroughputConstraintFromScheduleFileName(splitedFileName);
		task = this.taskMap.get(taskName);
		
		if(taskName.equals(Constants.TOP_TASKGRAPH_NAME))
			compositeMappingInfo = getCompositeMappingInfo(taskName, Constants.INVALID_ID_VALUE);
		else
			compositeMappingInfo = getCompositeMappingInfo(taskName, task.getId());	
		
		for(TaskGroupForScheduleType taskGroup: scheduleDOM.getTaskGroups().getTaskGroup())
		{
			for(ScheduleGroupType scheduleGroup : taskGroup.getScheduleGroup())
			{
				procId = getProcessorIdByName(scheduleGroup.getPoolName());
				// TODO: currently the last-picked processor name is used for searching the device name
				// get processor name from schedule information
				processorName = scheduleGroup.getPoolName();
				
				CompositeTaskMappedProcessor mappedProcessor = new CompositeTaskMappedProcessor(procId, 
																scheduleGroup.getLocalId().intValue(), modeId, sequenceId);
				CompositeTaskSchedule taskSchedule = new CompositeTaskSchedule(scheduleId, throughputConstraint);
				
				fillCompositeTaskSchedule(taskSchedule, scheduleGroup) ;				
				mappedProcessor.putCompositeTaskSchedule(taskSchedule);
				compositeMappingInfo.putProcessor(mappedProcessor);
				sequenceId++;
			}
		}
		
		compositeMappingInfo.setMappedDeviceName(findDeviceNameFromProcessorName(processorName));
	}
	

	
	private void makeCompositeTaskMappingInfo(String scheduleFolderPath) throws FileNotFoundException, InvalidScheduleFileNameException, InvalidDataInMetadataFileException {
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
				makeMultipleCompositeTaskMapping(splitedFileName, file);
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
	
	private void recursivePutTask(ArrayList<ScheduleItem> scheduleItemList, TaskModeTransition targetTaskModeTransition, 
									CompositeTaskMappedProcessor compositeMappedProc) {		
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
				targetTaskModeTransition.putRelatedChildTask(compositeMappedProc.getProcessorId(), compositeMappedProc.getProcessorLocalId(), 
															compositeMappedProc.getModeId(), task.getTaskName());
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
	
	// set isStaticScheduled and mode's related task list
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
	
	// scheduleFolderPath : output + /convertedSDF3xml/
	public void makeMappingInformation(CICMappingType mapping_metadata, CICProfileType profile_metadata, CICConfigurationType config_metadata, String scheduleFolderPath)
	{
		//config_metadata.getCodeGeneration().getRuntimeExecutionPolicy().equals(anObject)
		ExecutionPolicy executionPolicy = ExecutionPolicy.fromValue(config_metadata.getCodeGeneration().getRuntimeExecutionPolicy());

		try {
			switch(executionPolicy)
			{
			// TODO: fully static is not supported now
			case FULLY_STATIC: // Need schedule with time information (needed file: mapping, profile, schedule)
			case SELF_TIMED: // Need schedule (needed file: mapping, schedule)
				makeCompositeTaskMappingInfo(scheduleFolderPath);
				makeGeneralTaskMappingInfo(mapping_metadata);
				setTaskExtraInformationFromMappingInfo();
				setRelatedChildTasksOfMTMTask();
				setNumOfProcsOfTasks();
				setScheduleListIndexAndTaskFuncId();
				break;
			case STATIC_ASSIGNMENT: // Need mapping only (needed file: mapping)
				makeGeneralTaskMappingInfo(mapping_metadata);
				setNumOfProcsOfTasks();
				break;
			// TODO: fully dynamic is not supported now
			case FULLY_DYNAMIC: // Need mapped device information (needed file: mapping)
				makeGeneralTaskMappingInfo(mapping_metadata);
				break;
			}
		}
		catch(FileNotFoundException e) {
			e.printStackTrace();
		} 
		catch (InvalidScheduleFileNameException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
		catch (InvalidDataInMetadataFileException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public TaskGraphType getApplicationGraphProperty() {
		return applicationGraphProperty;
	}

	public void setApplicationGraphProperty(TaskGraphType applicationGraphProperty) {
		this.applicationGraphProperty = applicationGraphProperty;
	}

	public ArrayList<Channel> getChannelList() {
		return channelList;
	}

	public HashMap<String, Task> getTaskMap() {
		return taskMap;
	}

	public HashMap<String, TaskGraph> getTaskGraphMap() {
		return taskGraphMap;
	}

	public HashMap<String, GeneralTaskMappingInfo> getGeneralMappingInfo() {
		return generalMappingInfo;
	}

	public HashMap<String, Device> getDeviceInfo() {
		return deviceInfo;
	}

	public HashMap<String, HWElementType> getElementTypeHash() {
		return elementTypeHash;
	}

	public HashMap<String, Port> getPortInfo() {
		return portInfo;
	}

	public HashMap<String, CompositeTaskMappingInfo> getStaticScheduleMappingInfo() {
		return staticScheduleMappingInfo;
	}
}
