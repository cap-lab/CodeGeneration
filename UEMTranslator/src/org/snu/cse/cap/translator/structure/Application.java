package org.snu.cse.cap.translator.structure;


import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;

import org.snu.cse.cap.translator.Constants;
import org.snu.cse.cap.translator.ExecutionTime;
import org.snu.cse.cap.translator.structure.communication.InMemoryAccessType;
import org.snu.cse.cap.translator.structure.communication.PortDirection;
import org.snu.cse.cap.translator.structure.communication.channel.Channel;
import org.snu.cse.cap.translator.structure.communication.channel.ChannelArrayType;
import org.snu.cse.cap.translator.structure.communication.channel.ChannelCommunicationType;
import org.snu.cse.cap.translator.structure.communication.channel.ChannelPort;
import org.snu.cse.cap.translator.structure.communication.channel.ConnectionRoleType;
import org.snu.cse.cap.translator.structure.communication.channel.LoopPortType;
import org.snu.cse.cap.translator.structure.communication.channel.PortSampleRate;
import org.snu.cse.cap.translator.structure.communication.channel.RemoteCommunicationMethodType;
import org.snu.cse.cap.translator.structure.communication.multicast.MulticastCommunicationType;
import org.snu.cse.cap.translator.structure.communication.multicast.MulticastGroup;
import org.snu.cse.cap.translator.structure.communication.multicast.MulticastPort;
import org.snu.cse.cap.translator.structure.device.Device;
import org.snu.cse.cap.translator.structure.device.DeviceCommunicationType;
import org.snu.cse.cap.translator.structure.device.EnvironmentVariable;
import org.snu.cse.cap.translator.structure.device.HWCategory;
import org.snu.cse.cap.translator.structure.device.HWElementType;
import org.snu.cse.cap.translator.structure.device.NoProcessorFoundException;
import org.snu.cse.cap.translator.structure.device.Processor;
import org.snu.cse.cap.translator.structure.device.ProcessorElementType;
import org.snu.cse.cap.translator.structure.device.connection.Connection;
import org.snu.cse.cap.translator.structure.device.connection.ConnectionPair;
import org.snu.cse.cap.translator.structure.device.connection.ConstrainedSerialConnection;
import org.snu.cse.cap.translator.structure.device.connection.DeviceConnection;
import org.snu.cse.cap.translator.structure.device.connection.IPConnection;
import org.snu.cse.cap.translator.structure.device.connection.InvalidDeviceConnectionException;
import org.snu.cse.cap.translator.structure.device.connection.ProtocolType;
import org.snu.cse.cap.translator.structure.device.connection.SerialConnection;
import org.snu.cse.cap.translator.structure.device.connection.TCPConnection;
import org.snu.cse.cap.translator.structure.device.connection.UDPConnection;
import org.snu.cse.cap.translator.structure.device.connection.UnconstrainedSerialConnection;
import org.snu.cse.cap.translator.structure.library.Argument;
import org.snu.cse.cap.translator.structure.library.Function;
import org.snu.cse.cap.translator.structure.library.Library;
import org.snu.cse.cap.translator.structure.library.LibraryConnection;
import org.snu.cse.cap.translator.structure.mapping.InvalidScheduleFileNameException;
import org.snu.cse.cap.translator.structure.mapping.MappingInfo;
import org.snu.cse.cap.translator.structure.module.Module;
import org.snu.cse.cap.translator.structure.task.Task;
import org.snu.cse.cap.translator.structure.task.TaskLoopType;
import org.snu.cse.cap.translator.structure.task.TaskMode;

import hopes.cic.xml.ArchitectureConnectType;
import hopes.cic.xml.ArchitectureConnectionSlaveType;
import hopes.cic.xml.ArchitectureDeviceType;
import hopes.cic.xml.ArchitectureElementType;
import hopes.cic.xml.ArchitectureElementTypeType;
import hopes.cic.xml.CICAlgorithmType;
import hopes.cic.xml.CICArchitectureType;
import hopes.cic.xml.CICConfigurationType;
import hopes.cic.xml.CICGPUSetupType;
import hopes.cic.xml.CICMappingType;
import hopes.cic.xml.CICProfileType;
import hopes.cic.xml.ChannelPortType;
import hopes.cic.xml.ChannelType;
import hopes.cic.xml.DeviceConnectionListType;
import hopes.cic.xml.EnvironmentVariableType;
import hopes.cic.xml.IPConnectionType;
import hopes.cic.xml.LibraryFunctionArgumentType;
import hopes.cic.xml.LibraryFunctionType;
import hopes.cic.xml.LibraryLibraryConnectionType;
import hopes.cic.xml.LibraryType;
import hopes.cic.xml.ModeTaskType;
import hopes.cic.xml.ModeType;
import hopes.cic.xml.ModuleType;
import hopes.cic.xml.MulticastGroupType;
import hopes.cic.xml.MulticastPortType;
import hopes.cic.xml.PortMapType;
import hopes.cic.xml.SerialConnectionType;
import hopes.cic.xml.TaskLibraryConnectionType;
import hopes.cic.xml.TaskPortType;
import hopes.cic.xml.TaskRateType;
import hopes.cic.xml.TaskType;
import mapss.dif.csdf.sdf.SDFEdgeWeight;
import mapss.dif.csdf.sdf.SDFGraph;
import mapss.dif.csdf.sdf.SDFNodeWeight;
import mapss.dif.csdf.sdf.sched.MinBufferStrategy;
import mapss.dif.csdf.sdf.sched.TwoNodeStrategy;
import mocgraph.Edge;
import mocgraph.Node;
import mocgraph.sched.Firing;
import mocgraph.sched.ScheduleElement;

public class Application {
	// Overall metadata information
	private ArrayList<Channel> channelList;
	private HashMap<String, Task> taskMap; // Task name : Task class
	private HashMap<String, TaskGraph> fullTaskGraphMap; // Task name : Task class
	private HashMap<String, ChannelPort> portInfo; // Key: taskName/portName/direction, ex) MB_Y/inMB_Y/input
	private HashMap<String, MulticastPort> multicastPortInfo; // Key: taskName/portName/direction, ex) MB_Y/inMB_Y/input
	private HashMap<String, Device> deviceInfo; // device name: Device class
	private HashMap<String, DeviceConnection> deviceConnectionMap;
	private HashMap<String, HWElementType> elementTypeHash; // element type name : HWElementType class
	private TaskGraphType applicationGraphProperty;
	private HashMap<String, Library> libraryMap; // library name : Library class
	private ExecutionTime executionTime;

	public Application()
	{
		this.channelList = new ArrayList<Channel>();
		this.taskMap = new HashMap<String, Task>();
		this.portInfo = new HashMap<String, ChannelPort>();
		this.multicastPortInfo = new HashMap<String, MulticastPort>();
		this.deviceInfo = new HashMap<String, Device>();
		this.elementTypeHash = new HashMap<String, HWElementType>();
		this.applicationGraphProperty = null;
		this.deviceConnectionMap = new HashMap<String, DeviceConnection>();
		this.libraryMap = new HashMap<String, Library>();
		this.fullTaskGraphMap = new HashMap<String, TaskGraph>();
		this.executionTime = null;
	}

	private void putPortInfoFromTask(TaskType task_metadata, int taskId, String taskName) {
		for(TaskPortType portType: task_metadata.getPort())
		{
			PortDirection direction = PortDirection.fromValue(portType.getDirection().value());
			ChannelPort channelPort = new ChannelPort(taskId, taskName, portType.getName(), portType.getSampleSize().intValue(), portType.getType().value(), direction);

			if(portType.getDescription() != null && portType.getDescription().trim().length() > 0) {
				channelPort.setDescription(portType.getDescription());
			}

			this.portInfo.put(channelPort.getPortKey(), channelPort);

			if(portType.getRate() != null) {
				for(TaskRateType taskRate: portType.getRate()) {
					PortSampleRate sampleRate = new PortSampleRate(taskRate.getMode(), taskRate.getRate().intValue());
					channelPort.putSampleRate(sampleRate);
				}
			}
			else {
				// variable sample rate, do nothing
			}
		}
	}

	private void putMulticastPortInfoFromTask(TaskType task_metadata, int taskId, String taskName)
	{
		for(MulticastPortType multicastPortType: task_metadata.getMulticastPort())
		{
			PortDirection direction = PortDirection.fromValue(multicastPortType.getDirection().value());

			MulticastPort multicastPort = new MulticastPort(taskId, taskName, multicastPortType.getName(), multicastPortType.getGroup(), direction);
			this.multicastPortInfo.put(multicastPort.getPortKey(), multicastPort);
		}

	}

	private void setLoopDesignatedTaskIdFromTaskName()
	{
		for(Task task : this.taskMap.values())
		{
			Task designatedTask;
			if(task.getLoopStruct()!= null && task.getLoopStruct().getLoopType() == TaskLoopType.CONVERGENT)
			{
				designatedTask = this.taskMap.get(task.getLoopStruct().getDesignatedTaskName());
				task.getLoopStruct().setDesignatedTaskId(designatedTask.getId());
			}

		}
	}

	private void fillBasicTaskMapAndGraphInfo(CICAlgorithmType algorithm_metadata)
	{
		int taskId = 0;
		Task task;

		for(TaskType task_metadata: algorithm_metadata.getTasks().getTask())
		{
			task = new Task(taskId, task_metadata);

			this.taskMap.put(task.getName(), task);

			putPortInfoFromTask(task_metadata, taskId, task.getName());

			putMulticastPortInfoFromTask(task_metadata, taskId, task.getName());

			taskId++;
		}

		// this function must be called after setting all the task ID information
		setLoopDesignatedTaskIdFromTaskName();

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
			PortDirection direction = PortDirection.fromValue(portMapType.getDirection().value());
			Task task = this.taskMap.get(portMapType.getTask());
			ChannelPort port = this.portInfo.get(portMapType.getTask() + Constants.NAME_SPLITER + portMapType.getPort() +
										Constants.NAME_SPLITER + direction);

			if(portMapType.getChildTask() != null && portMapType.getChildTaskPort() != null)
			{
				ChannelPort childPort = this.portInfo.get(portMapType.getChildTask() + Constants.NAME_SPLITER + portMapType.getChildTaskPort() +
						Constants.NAME_SPLITER + direction);

				port.setSubgraphPort(childPort);
				childPort.setUpperGraphPort(port);
			}

			port.setLoopPortType(LoopPortType.fromValue(portMapType.getType().value()));

			if(task.getLoopStruct() != null && direction == PortDirection.INPUT)
			{
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
	}

	// taskMap, taskGraphList
	public void makeTaskInformation(CICAlgorithmType algorithm_metadata)
	{
		Task task;

		this.applicationGraphProperty = TaskGraphType.fromValue(algorithm_metadata.getProperty());

		fillBasicTaskMapAndGraphInfo(algorithm_metadata);

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
		if(connectionList.getSerialConnection() != null)
		{
			for(SerialConnectionType connectionType: connectionList.getSerialConnection())
			{
				SerialConnection connection;
				switch(device.getPlatform())
				{
				case LINUX:
				case WINDOWS:
					connection = new UnconstrainedSerialConnection(connectionType.getName(), connectionType.getRole().toString(),
																connectionType.getNetwork(), connectionType.getPortAddress());
					break;
				case ARDUINO:
					connection = new ConstrainedSerialConnection(connectionType.getName(), connectionType.getRole().toString(),
																connectionType.getNetwork(),
																connectionType.getBoardTXPinNumber().intValue(),
																connectionType.getBoardRXPinNumber().intValue());
					break;
				default:
					throw new IllegalArgumentException();
				}

				device.putConnection(connection);
			}
		}

		if(connectionList.getIPConnection() != null)
		{
			for(IPConnectionType connectionType: connectionList.getIPConnection())
			{
				IPConnection connection = null;
				switch(connectionType.getProtocol())
				{
				case TCP:
					connection = new TCPConnection(connectionType.getName(), connectionType.getRole().toString(),
							connectionType.getIp(), connectionType.getPort().intValue());
					break;
				case UDP:
					connection = new UDPConnection(connectionType.getName(), connectionType.getRole().toString(),
							connectionType.getIp(), connectionType.getPort().intValue());
					break;
				default:
					throw new IllegalArgumentException();
				}

				if(connection != null)
				{
					device.putConnection(connection);
				}
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
				if(this.deviceConnectionMap.containsKey(connectType.getMaster()))
				{
					deviceConnection = this.deviceConnectionMap.get(connectType.getMaster());
				}
				else
				{
					deviceConnection = new DeviceConnection(connectType.getMaster());
					this.deviceConnectionMap.put(connectType.getMaster(), deviceConnection);
				}

				try {
					master = findConnection(connectType.getMaster(), connectType.getConnection());

					for(ArchitectureConnectionSlaveType slaveType: connectType.getSlave())
					{
						Connection slave;

						slave = findConnection(slaveType.getDevice(), slaveType.getConnection());
						deviceConnection.putMasterToSlaveConnection(master, slaveType.getDevice(), slave);
						deviceConnection.putSlaveToMasterConnection(slaveType.getDevice(), slave, master);
					}
				} catch (InvalidDeviceConnectionException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
	}

	private void insertDeviceModules(Device device, List<ModuleType> moduleList, HashMap<String, Module> moduleMap)
	{
		for(ModuleType moduleType: moduleList)
		{
			Module module = moduleMap.get(moduleType.getName());

			if(module != null)
			{
				device.getModuleList().add(module);
			}
		}
	}

	private void insertEnvironmentVariables(Device device, List<EnvironmentVariableType> envVarList)
	{
		for(EnvironmentVariableType envVar: envVarList)
		{
			EnvironmentVariable evnVar = new EnvironmentVariable(envVar.getName(), envVar.getValue());

			device.getEnvironmentVariableList().add(evnVar);
		}
	}

	public void makeDeviceInformation(CICArchitectureType architecture_metadata, HashMap<String, Module> moduleMap)
	{
		makeHardwareElementInformation(architecture_metadata);
		int processId = 0;
		int deviceId = 0;

		if(architecture_metadata.getDevices() != null)
		{
			for(ArchitectureDeviceType device_metadata: architecture_metadata.getDevices().getDevice())
			{

				Device device = new Device(device_metadata.getName(), deviceId, device_metadata.getArchitecture(),
											device_metadata.getPlatform(), device_metadata.getRuntime());

				deviceId++;

				for(ArchitectureElementType elementType: device_metadata.getElements().getElement())
				{
					// only handles the elements which use defined types
					if(this.elementTypeHash.containsKey(elementType.getType()) == true)
					{
						ProcessorElementType elementInfo = (ProcessorElementType) this.elementTypeHash.get(elementType.getType());
						device.putProcessingElement(processId, elementType.getName(), elementInfo.getSubcategory(), elementType.getPoolSize().intValue());
						processId++;
					}
				}

				if(device_metadata.getConnections() != null)
				{
					putConnectionsOnDevice(device, device_metadata.getConnections());
				}

				if(device_metadata.getModules() != null)
				{
					insertDeviceModules(device, device_metadata.getModules().getModule(), moduleMap);
				}

				if(device_metadata.getEnvironmentVariables() != null)
				{
					insertEnvironmentVariables(device, device_metadata.getEnvironmentVariables().getVariable());
				}

				this.deviceInfo.put(device_metadata.getName(), device);
			}

			makeDeviceConnectionInformation(architecture_metadata);
		}
	}

	private MappingInfo findMappingInfoByTaskName(String taskName) throws InvalidDataInMetadataFileException
	{
		Task task;
		MappingInfo mappingInfo = null;

		for(Device device: this.deviceInfo.values())
		{
			// task is mapped in this device
			if(device.getTaskMap().containsKey(taskName))
			{
				mappingInfo = device.getGeneralMappingInfo().get(taskName);
				if(mappingInfo == null)
				{
					task = device.getTaskMap().get(taskName);
					mappingInfo = device.getStaticScheduleMappingInfo().get(task.getParentTaskGraphName());

					while(mappingInfo == null && task != null)
					{
						task = device.getTaskMap().get(task.getParentTaskGraphName());
						if(task != null)
						{
							mappingInfo = device.getStaticScheduleMappingInfo().get(task.getParentTaskGraphName());
						}
					}
				}
				break;
			}
		}

		if(mappingInfo == null)
		{
			throw new InvalidDataInMetadataFileException();
		}

		return mappingInfo;
	}


	private void setRemoteCommunicationMethodType(Channel channel, ConnectionPair connectionPair)
	{
		switch(connectionPair.getMasterConnection().getProtocol())
		{
		case SERIAL:
			switch(connectionPair.getMasterConnection().getNetwork())
			{
			case BLUETOOTH:
				channel.setRemoteMethodType(RemoteCommunicationMethodType.BLUETOOTH);
				break;
			case USB:
			case WIRE:
				channel.setRemoteMethodType(RemoteCommunicationMethodType.SERIAL);
				break;
			case ETHERNET_WI_FI:
			default:
				throw new UnsupportedOperationException();
			}
			break;
		case TCP:
			channel.setRemoteMethodType(RemoteCommunicationMethodType.TCP);

			break;
		default:
			throw new UnsupportedOperationException();
		}
	}

	private void setChannelConnectionRoleType(Channel channel, ConnectionPair connectionPair, String taskName)
	{
		if(connectionPair.getMasterConnection().getProtocol() == ProtocolType.TCP) {
			if(connectionPair.getMasterDeviceName().equals(taskName) == true) {
				channel.setConnectionRoleType(ConnectionRoleType.SERVER);
			}
			else {
				channel.setConnectionRoleType(ConnectionRoleType.CLIENT);
			}
		}
		else {
			if(connectionPair.getMasterDeviceName().equals(taskName) == true) {
				channel.setConnectionRoleType(ConnectionRoleType.MASTER);
			}
			else {
				channel.setConnectionRoleType(ConnectionRoleType.SLAVE);
			}
		}
	}

	private void setSourceRemoteCommunicationType(Channel channel, String srcTaskDevice, String dstTaskDevice) throws InvalidDeviceConnectionException
	{
		DeviceConnection srcTaskConnection = this.deviceConnectionMap.get(srcTaskDevice);
		DeviceConnection dstTaskConnection = this.deviceConnectionMap.get(dstTaskDevice);
		ConnectionPair connectionPair = null;

		if(srcTaskConnection != null) {// source is master
			connectionPair = srcTaskConnection.findOneConnectionToAnotherDevice(dstTaskDevice);
		}

		if(connectionPair == null && dstTaskConnection != null) {// destination is master
			connectionPair = dstTaskConnection.findOneConnectionToAnotherDevice(srcTaskDevice);
		}

		if(connectionPair == null) {
			throw new InvalidDeviceConnectionException();
		}

		channel.setCommunicationType(ChannelCommunicationType.REMOTE_WRITER);

		setRemoteCommunicationMethodType(channel, connectionPair);
		setChannelConnectionRoleType(channel, connectionPair, srcTaskDevice);
	}

	private void setInMemoryAccessTypeOfRemoteChannel(Channel channel, MappingInfo taskMappingInfo, boolean isSrcTask)
	{
		int procId;
		boolean isCPU = false;

		Device device = this.deviceInfo.get(taskMappingInfo.getMappedDeviceName());
		procId = taskMappingInfo.getMappedProcessorList().get(0).getProcessorId();

		for(Processor processor : device.getProcessorList()) {
			if(procId == processor.getId()) {
				isCPU = processor.getIsCPU();
				break;
			}
		}

		if(isCPU == false) {
			if(isSrcTask == true)
			{
				channel.setAccessType(InMemoryAccessType.GPU_CPU);
			}
			else // isSrcTask == false
			{
				channel.setAccessType(InMemoryAccessType.CPU_GPU);
			}
		}
		else
		{
			channel.setAccessType(InMemoryAccessType.CPU_ONLY);
		}
	}

	// TODO: Only support CPU and GPU cases
	private void setInDeviceCommunicationType(Channel channel, MappingInfo srcTaskMappingInfo, MappingInfo dstTaskMappingInfo)
	{
		boolean srcCPU = false;
		boolean dstCPU = false;
		int srcProcId, dstProcId;
		int srcProcLocalId, dstProcLocalId;

		srcProcId = srcTaskMappingInfo.getMappedProcessorList().get(0).getProcessorId();
		dstProcId = dstTaskMappingInfo.getMappedProcessorList().get(0).getProcessorId();
		srcProcLocalId = srcTaskMappingInfo.getMappedProcessorList().get(0).getProcessorLocalId();
		dstProcLocalId = dstTaskMappingInfo.getMappedProcessorList().get(0).getProcessorLocalId();

		Device device = this.deviceInfo.get(srcTaskMappingInfo.getMappedDeviceName());
		for(Processor processor : device.getProcessorList()) {
			if(srcProcId == processor.getId()) {
				srcCPU = processor.getIsCPU();
			}

			if(dstProcId == processor.getId()) {
				dstCPU = processor.getIsCPU();
			}
		}

		channel.setCommunicationType(ChannelCommunicationType.SHARED_MEMORY);
		channel.setProcesserId(srcProcId);

		if(srcCPU == false && dstCPU == true) {
			channel.setAccessType(InMemoryAccessType.GPU_CPU);
		}
		else if(srcCPU == false && dstCPU == false) {
			if(srcProcId == dstProcId && srcProcLocalId == dstProcLocalId) {
				channel.setAccessType(InMemoryAccessType.GPU_GPU);
			}
			else {
				channel.setAccessType(InMemoryAccessType.GPU_GPU_DIFFERENT);
			}
		}
		else if(dstCPU == true) { // && srcCPU == true
			if(srcProcId == dstProcId) {
				channel.setAccessType(InMemoryAccessType.CPU_ONLY);
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               			}
			else {
				throw new UnsupportedOperationException();
			}
		}
		else { // dstCPU == true && srcCPU == true
			channel.setAccessType(InMemoryAccessType.CPU_GPU);
		}
	}

	private void setChannelCommunicationType(Channel channel, MappingInfo srcTaskMappingInfo, MappingInfo dstTaskMappingInfo) throws InvalidDeviceConnectionException
	{
		// Two tasks are connected on different devices
		if(srcTaskMappingInfo.getMappedDeviceName().equals(dstTaskMappingInfo.getMappedDeviceName()) == false)
		{
			// it only set channel communication type of source task
			setSourceRemoteCommunicationType(channel, srcTaskMappingInfo.getMappedDeviceName(), dstTaskMappingInfo.getMappedDeviceName());
			setInMemoryAccessTypeOfRemoteChannel(channel, srcTaskMappingInfo, true);
		}
		else // located at the same device
		{
			setInDeviceCommunicationType(channel, srcTaskMappingInfo, dstTaskMappingInfo);
		}
	}

	private boolean isDataLoopTask(Task task) {
		boolean isDataLoop = false;

		while(task != null)
		{
			if(task.getLoopStruct() != null && task.getLoopStruct().getLoopType() == TaskLoopType.DATA)
			{
				isDataLoop = true;
				break;
			}

			task = this.taskMap.get(task.getParentTaskGraphName());
		}

		return isDataLoop;
	}

	// ports and tasks are the most lower-level
	private void setChannelType(Channel channel, ChannelPort srcPort, ChannelPort dstPort) {
		boolean isDstDataLoop = false;
		boolean isSrcDataLoop = false;
		boolean isDstDistributing = false;
		boolean isDstBroadcasting = false;
		Task srcTask;
		Task dstTask;

		srcTask = this.taskMap.get(srcPort.getTaskName());
		dstTask = this.taskMap.get(dstPort.getTaskName());
		isDstDataLoop = isDataLoopTask(dstTask);
		isSrcDataLoop = isDataLoopTask(srcTask);
		isDstDistributing = dstPort.isDistributingPort();
		isDstBroadcasting = dstPort.isBroadcastingPort();

		//2019.07.31 make channel connecting two tasks inside D-typeloop tasks as CHANNEL_TYPE_FULL_ARRAY.
		if(isSrcDataLoop == true && isDstDataLoop == true)
		{
			channel.setChannelType(ChannelArrayType.FULL_ARRAY);
		} // even a task uses broadcasting it needs to be input_array type to manage available number
		else if((isSrcDataLoop == false && isDstDistributing == true) || isDstBroadcasting == true)
		{
			channel.setChannelType(ChannelArrayType.INPUT_ARRAY);
		}
		else if(isSrcDataLoop == true && isDstDataLoop == false && isDstDistributing == false && isDstBroadcasting == false)
		{
			channel.setChannelType(ChannelArrayType.OUTPUT_ARRAY);
		}
		else
		{
			channel.setChannelType(ChannelArrayType.GENERAL);
		}
	}

	private void putPortIntoDeviceHierarchically(Device device, ChannelPort port, PortDirection direction)
	{
		// key: taskName/portName/direction
		ChannelPort currentPort;
		String key;

		currentPort = port;
		while(currentPort != null)
		{
			key = currentPort.getPortKey();
			if(device.getPortKeyToIndex().containsKey(key) == false)
			{
				device.getPortKeyToIndex().put(key, new Integer(device.getPortList().size()));
				device.getPortList().add(currentPort);

			}
			currentPort = currentPort.getSubgraphPort();
		}
	}

	private ConnectionRoleType getDstTaskConnectionRoleType(ConnectionRoleType srcTaskConnectionRoleType)
	{
		ConnectionRoleType dstTaskConnectionRoleType;

		switch(srcTaskConnectionRoleType)
		{
		case CLIENT:
			dstTaskConnectionRoleType = ConnectionRoleType.SERVER;
			break;
		case SERVER:
			dstTaskConnectionRoleType = ConnectionRoleType.CLIENT;
			break;
		case MASTER:
			dstTaskConnectionRoleType = ConnectionRoleType.SLAVE;
			break;
		case SLAVE:
			dstTaskConnectionRoleType = ConnectionRoleType.MASTER;
			break;
		default:
			throw new UnsupportedOperationException();
		}

		return dstTaskConnectionRoleType;
	}

	private void setSocketIndexFromTCPConnection(Channel channel, Device targetDevice, ConnectionPair connectionPair) throws InvalidDeviceConnectionException
	{
		int index = 0;
		TCPConnection connection = null;
		ArrayList<TCPConnection> connectionList = null;

		switch(channel.getRemoteMethodType())
		{
		case TCP:
			switch(channel.getConnectionRoleType())
			{
			case SERVER:
				connectionList = targetDevice.getTcpServerList();
				connection = (TCPConnection) connectionPair.getMasterConnection();
				break;
			case CLIENT:
				connectionList = targetDevice.getTcpClientList();
				connection = (TCPConnection) connectionPair.getSlaveConnection();
				break;
			default:
				throw new InvalidDeviceConnectionException();
			}
			break;
		default:
			throw new InvalidDeviceConnectionException();
		}

		connection.incrementChannelAccessNum();

		for(index = 0 ; index < connectionList.size(); index++)
		{
			TCPConnection tcpConnection = connectionList.get(index);

			if(tcpConnection.getName().equals(connection.getName()) == true)
			{
				// same connection name
				channel.setSocketInfoIndex(index);
				break;
			}
		}
	}

	private void setSocketIndexFromConstrainedSerialConnection(Channel channel, Device targetDevice, ConnectionPair connectionPair) throws InvalidDeviceConnectionException
	{
		int index = 0;
		ConstrainedSerialConnection connection = null;
		ArrayList<ConstrainedSerialConnection> connectionList = null;

		switch(channel.getRemoteMethodType())
		{
		case BLUETOOTH:
		case SERIAL:
			switch(channel.getConnectionRoleType())
			{
			case SLAVE:
				connectionList = targetDevice.getSerialConstrainedSlaveList();
				connection = (ConstrainedSerialConnection) connectionPair.getSlaveConnection();
				break;
			default:
				throw new InvalidDeviceConnectionException();
			}
			break;
		default:
			throw new InvalidDeviceConnectionException();
		}

		connection.incrementChannelAccessNum();

		for(index = 0 ; index < connectionList.size(); index++)
		{
			ConstrainedSerialConnection serialConnection = connectionList.get(index);

			if(serialConnection.getName().equals(connection.getName()) == true)
			{
				// same connection name
				channel.setSocketInfoIndex(index);
				break;
			}
		}
	}

	private void setSocketIndexFromUnconstrainedSerialConnection(Channel channel, Device targetDevice, ConnectionPair connectionPair) throws InvalidDeviceConnectionException
	{
		int index = 0;
		UnconstrainedSerialConnection connection = null;
		ArrayList<UnconstrainedSerialConnection> connectionList = null;

		switch(channel.getRemoteMethodType())
		{
		case BLUETOOTH:
			switch(channel.getConnectionRoleType())
			{
			case MASTER:
				connectionList = targetDevice.getBluetoothMasterList();
				connection = (UnconstrainedSerialConnection) connectionPair.getMasterConnection();
				break;
			case SLAVE:
				connectionList = targetDevice.getBluetoothUnconstrainedSlaveList();
				connection = (UnconstrainedSerialConnection) connectionPair.getSlaveConnection();
				break;
			default:
				throw new InvalidDeviceConnectionException();
			}
			break;
		case SERIAL:
			switch(channel.getConnectionRoleType())
			{
			case MASTER:
				connectionList = targetDevice.getSerialMasterList();
				connection = (UnconstrainedSerialConnection) connectionPair.getMasterConnection();
				break;
			case SLAVE:
				connectionList = targetDevice.getSerialUnconstrainedSlaveList();
				connection = (UnconstrainedSerialConnection) connectionPair.getSlaveConnection();
				break;
			default:
				throw new InvalidDeviceConnectionException();
			}
			break;
		default:
			throw new InvalidDeviceConnectionException();
		}

		connection.incrementChannelAccessNum();

		for(index = 0 ; index < connectionList.size(); index++)
		{
			UnconstrainedSerialConnection serialConnection = connectionList.get(index);

			if(serialConnection.getName().equals(connection.getName()) == true)
			{
				// same connection name
				channel.setSocketInfoIndex(index);
				break;
			}
		}
	}

	private void findAndSetSocketInfoIndex(Channel channel, Device srcDevice, Device dstDevice) throws InvalidDeviceConnectionException
	{
		DeviceConnection srcTaskConnection = this.deviceConnectionMap.get(srcDevice.getName());
		DeviceConnection dstTaskConnection = this.deviceConnectionMap.get(dstDevice.getName());
		ConnectionPair connectionPair = null;
		Device targetDevice;

		switch(channel.getConnectionRoleType())
		{
		case MASTER:
		case SLAVE:
		case CLIENT:
		case SERVER:
			switch(channel.getCommunicationType())
			{
			case REMOTE_READER:
				targetDevice = dstDevice;
				break;
			case REMOTE_WRITER:
				targetDevice = srcDevice;
				break;
			case SHARED_MEMORY:
				return; // do nothing with shared memory
			default:
				throw new InvalidDeviceConnectionException();
			}
			break;
		case NONE:
			return;
		default:
			throw new InvalidDeviceConnectionException();
		}

		if(srcTaskConnection != null) {// source is master
			connectionPair = srcTaskConnection.findOneConnectionToAnotherDevice(dstDevice.getName());
		}

		if(connectionPair == null && dstTaskConnection != null) {// destination is master
			connectionPair = dstTaskConnection.findOneConnectionToAnotherDevice(srcDevice.getName());
		}

		if(connectionPair == null) {
			throw new InvalidDeviceConnectionException();
		}

		switch(targetDevice.getPlatform())
		{
		case ARDUINO:
			setSocketIndexFromConstrainedSerialConnection(channel, targetDevice, connectionPair);
			break;
		case LINUX:
			if(connectionPair.getMasterConnection().getProtocol() == ProtocolType.TCP) {
				setSocketIndexFromTCPConnection(channel, targetDevice, connectionPair);
			}
			else {
				setSocketIndexFromUnconstrainedSerialConnection(channel, targetDevice, connectionPair);
			}
			break;
		case UCOS3:
		case WINDOWS:
		default:
			break;

		}
	}

	private void addChannelAndPortInfoToDevice(Channel channel, MappingInfo srcTaskMappingInfo, MappingInfo dstTaskMappingInfo) throws CloneNotSupportedException, InvalidDeviceConnectionException
	{
		Device srcDevice = this.deviceInfo.get(srcTaskMappingInfo.getMappedDeviceName());
		Device dstDevice = this.deviceInfo.get(dstTaskMappingInfo.getMappedDeviceName());

		// hierarchical put port information

		// src and dst is same device
		putPortIntoDeviceHierarchically(srcDevice, channel.getInputPort(), PortDirection.INPUT);
		putPortIntoDeviceHierarchically(srcDevice, channel.getOutputPort(), PortDirection.OUTPUT);

		findAndSetSocketInfoIndex(channel, srcDevice, dstDevice);

		srcDevice.getChannelList().add(channel);

		// if src and dst are different put same information to dst device
		if(srcTaskMappingInfo.getMappedDeviceName().equals(dstTaskMappingInfo.getMappedDeviceName()) == false)
		{
			Channel channelInDevice = channel.clone();
			putPortIntoDeviceHierarchically(dstDevice, channel.getInputPort(), PortDirection.INPUT);
			putPortIntoDeviceHierarchically(dstDevice, channel.getOutputPort(), PortDirection.OUTPUT);

			channelInDevice.setCommunicationType(ChannelCommunicationType.REMOTE_READER);
			channelInDevice.setConnectionRoleType(getDstTaskConnectionRoleType(channel.getConnectionRoleType()));
			setInMemoryAccessTypeOfRemoteChannel(channelInDevice, dstTaskMappingInfo, false);

			findAndSetSocketInfoIndex(channelInDevice, srcDevice, dstDevice);

			dstDevice.getChannelList().add(channelInDevice);
		}
	}

	private void setNextChannelId(HashMap<String, ArrayList<Channel>> portToChannelConnection, ChannelPort srcPort, Channel curChannel)
	{
		String portStartKey;
		ArrayList<Channel> sameSourceChannelList;
		Channel prevChannel;

		portStartKey = srcPort.getTaskId() + Constants.NAME_SPLITER + srcPort.getPortName();
		if(portToChannelConnection.containsKey(portStartKey) == false)
		{
			portToChannelConnection.put(portStartKey, new ArrayList<Channel>());
		}

		sameSourceChannelList = portToChannelConnection.get(portStartKey);

		if(sameSourceChannelList.size() > 0)
		{
			prevChannel = sameSourceChannelList.get(sameSourceChannelList.size() - 1);
			prevChannel.setNextChannelIndex(curChannel.getIndex());
		}

		sameSourceChannelList.add(curChannel);
	}

	private void addNode(SDFGraph graph, ArrayList<Task> taskList, HashMap<String, Node> unconnectedSDFTaskMap)
	{
		int instanceid = 0;
		for (int i = 0; i < taskList.size(); i++) {
			String nodeName = taskList.get(i).getName();
			SDFNodeWeight weight = new SDFNodeWeight(nodeName, instanceid++);
			Node node = new Node(weight);
			graph.addNode(node);
			unconnectedSDFTaskMap.put(nodeName, node);
			graph.setName(node, nodeName);
		}
	}

	private boolean addEdge(SDFGraph graph, TaskMode mode, Channel channel, HashMap<String, Node> unconnectedSDFTaskMap)
	{
		boolean isSDF = true;
		Task srcTask;
		Task dstTask;
		int srcRate = 0;
		int dstRate = 0;
		srcTask = this.taskMap.get(channel.getOutputPort().getTaskName());
		dstTask = this.taskMap.get(channel.getInputPort().getTaskName());

		if(channel.getOutputPort().getPortSampleRateList().size() == 1 || mode == null)
		{
			srcRate = channel.getOutputPort().getPortSampleRateList().get(0).getSampleRate();
		}
		else
		{
			srcRate = Constants.INVALID_VALUE;
			for(PortSampleRate rate: channel.getOutputPort().getPortSampleRateList())
			{
				if(mode.getName().equals(rate.getModeName()) == true)
				{
					srcRate = rate.getSampleRate();
					break;
				}
			}
		}


		if(channel.getInputPort().getPortSampleRateList().size() == 1 || mode == null)
		{
			dstRate = channel.getInputPort().getPortSampleRateList().get(0).getSampleRate();
		}
		else
		{
			dstRate = Constants.INVALID_VALUE;
			for(PortSampleRate rate: channel.getInputPort().getPortSampleRateList())
			{
				if(mode.getName().equals(rate.getModeName()) == true)
				{
					dstRate = rate.getSampleRate();
					break;
				}
			}
		}

		if(srcRate > 0 && dstRate > 0)
		{
			Node srcNode = null;
			Node dstNode = null;
			int initialData = channel.getInitialDataLen()/channel.getChannelSampleSize();

			if(srcTask.getLoopStruct() != null && srcTask.getChildTaskGraphName() == null)
			{
				srcRate = srcRate / srcTask.getLoopStruct().getLoopCount();
			}

			if(dstTask.getLoopStruct() != null && dstTask.getLoopStruct().getLoopType() == TaskLoopType.DATA &&
				dstTask.getChildTaskGraphName() == null && channel.getInputPort().getLoopPortType() == LoopPortType.DISTRIBUTING)
			{
				dstRate = dstRate / dstTask.getLoopStruct().getLoopCount();
			}
			else if(dstTask.getLoopStruct() != null &&
				dstTask.getChildTaskGraphName() == null)
			{
				if(dstRate / dstTask.getLoopStruct().getLoopCount() == 0)
				{
					srcRate = srcRate * dstTask.getLoopStruct().getLoopCount();
					initialData = initialData * dstTask.getLoopStruct().getLoopCount();
				}
				else
				{
					dstRate = dstRate / dstTask.getLoopStruct().getLoopCount();
				}
			}

			if(initialData == 0)
			{
				for(Object nodeObj : graph.nodes())
				{
					Node node = (Node) nodeObj;
					if(graph.getName(node).equals(srcTask.getName()) == true)
					{
						srcNode = node;
						unconnectedSDFTaskMap.remove(srcTask.getName());
					}

					if(graph.getName(node).equals(dstTask.getName()) == true)
					{
						dstNode = node;
						unconnectedSDFTaskMap.remove(dstTask.getName());
					}
				}

				SDFEdgeWeight weight = new SDFEdgeWeight(channel.getOutputPort().getPortName(), channel.getInputPort().getPortName(), srcRate,
					dstRate, initialData);

				Edge edge = new Edge(srcNode, dstNode);
				edge.setWeight(weight);
				graph.addEdge(edge);
				graph.setName(edge, channel.getOutputPort().getPortName() + "to" + channel.getInputPort().getPortName());
			}


		}
		else if(srcRate == Constants.INVALID_VALUE || dstRate == Constants.INVALID_VALUE)
		{
			isSDF = false;
		}
		else
		{
			// do nothing
		}

		return isSDF;
	}

	private SDFGraph makeSDFGraph(ArrayList<Task> taskList, ArrayList<Channel> channelList, TaskMode mode)
	{
		SDFGraph graph = new SDFGraph(taskList.size(), channelList.size());
		HashMap<String, Node> unconnectedSDFTaskMap = new HashMap<String, Node>();
		boolean isSDF = false;

		addNode(graph, taskList, unconnectedSDFTaskMap);

		for (Channel channel : channelList)
		{
			isSDF = addEdge(graph, mode, channel, unconnectedSDFTaskMap);
			if(isSDF == false)
			{
				break;
			}
		}

		if(isSDF == false)
		{
			graph = null;
		}
		else
		{
			for(Node node : unconnectedSDFTaskMap.values())
			{
				graph.removeNode(node);
			}
		}

		return graph;
	}

	private void handleScheduleElement(SDFGraph graph, mocgraph.sched.Schedule schedule, int modeId)
	{
		Task task;
		int taskRep;
		Iterator iterator = schedule.iterator();
		while (iterator.hasNext()) {
			ScheduleElement scheduleElement = (ScheduleElement) iterator.next();
			if (scheduleElement instanceof mocgraph.sched.Schedule)
			{
				mocgraph.sched.Schedule innerSchedule = (mocgraph.sched.Schedule) scheduleElement;
				handleScheduleElement(graph, innerSchedule, modeId);
			} else if (scheduleElement instanceof Firing) {
				Firing firing = (Firing) scheduleElement;
				String taskName = graph.getName((Node) firing.getFiringElement());
				task = this.taskMap.get(taskName);
				if(task.getIterationCountList().containsKey(modeId+"") == true)
				{
					taskRep = task.getIterationCountList().get(modeId+"").intValue() + firing.getIterationCount();
				}
				else
				{
					taskRep = firing.getIterationCount();
				}

				task.getIterationCountList().put(modeId+"", taskRep);
			}
		}
	}

	private void setIndividualIterationCount(ArrayList<Task> taskList, SDFGraph graph, int modeId)
	{
		mocgraph.sched.Schedule schedule;

		if(taskList.size() <= 2)
		{
			TwoNodeStrategy st = new TwoNodeStrategy(graph);
			schedule = st.schedule();
		}
		else
		{
			MinBufferStrategy st = new MinBufferStrategy(graph);
			schedule = st.schedule();
		}

		handleScheduleElement(graph, schedule, modeId);
	}

	private TaskGraph getTaskGraphCanBeMerged(TaskGraph taskGraph, HashMap<String, TaskGraph> taskGraphMap, HashMap<String, TaskGraph> mergedTaskGraphMap)
	{
		TaskGraph mergedParentTaskGraph = null;
		TaskGraph currentTaskGraph;
		TaskGraph parentTaskGraph;

		currentTaskGraph = taskGraph;

		while(currentTaskGraph.getParentTask() != null)
		{
			parentTaskGraph = taskGraphMap.get(currentTaskGraph.getParentTask().getParentTaskGraphName());

			if(parentTaskGraph != null)
			{
				if(currentTaskGraph.getTaskGraphType() == TaskGraphType.DATAFLOW &&
						parentTaskGraph.getTaskGraphType() == TaskGraphType.PROCESS_NETWORK)
				{
					mergedParentTaskGraph = currentTaskGraph.clone();
					mergedTaskGraphMap.put(mergedParentTaskGraph.getName(), mergedParentTaskGraph);
					break;
				}
				else
				{
					mergedParentTaskGraph = mergedTaskGraphMap.get(parentTaskGraph.getName());
					if(mergedParentTaskGraph != null)
					{
						break;
					}
				}
			}
			currentTaskGraph = parentTaskGraph;
		}

		return mergedParentTaskGraph;
	}

	private HashMap<String, TaskGraph> mergeTaskGraph(HashMap<String, TaskGraph> taskGraphMap)
	{
		HashMap<String, TaskGraph> mergedTaskGraphMap = new HashMap<String, TaskGraph>();
		Task parentTask = null;
		TaskGraph mergedTaskGraph = null;
		TaskGraph mergedParentTaskGraph;

		for(TaskGraph taskGraph: taskGraphMap.values())
		{
			parentTask = taskGraph.getParentTask();
			if(parentTask == null)
			{
				mergedTaskGraph = taskGraph.clone();
				mergedTaskGraphMap.put(taskGraph.getName(), mergedTaskGraph);
			}
			else if(parentTask.getModeTransition() != null || parentTask.getLoopStruct() != null)
			{
				mergedTaskGraph = taskGraph.clone();
				mergedTaskGraphMap.put(taskGraph.getName(), mergedTaskGraph);
			}
		}

		for(TaskGraph taskGraph: taskGraphMap.values())
		{
			parentTask = taskGraph.getParentTask();
			if(parentTask != null && parentTask.getModeTransition() == null && parentTask.getLoopStruct() == null) {
				mergedParentTaskGraph = getTaskGraphCanBeMerged(taskGraph, taskGraphMap, mergedTaskGraphMap);
				if(mergedParentTaskGraph.getName().equals(taskGraph.getName()) == false) {
					mergedParentTaskGraph.mergeChildTaskGraph(taskGraph);
				}
			}
		}

		return mergedTaskGraphMap;
	}

	private void setIterationCount(HashMap<String, TaskGraph> taskGraphMap)
	{
		HashMap<String, TaskGraph> mergedTaskGraphMap;

		mergedTaskGraphMap = mergeTaskGraph(taskGraphMap);

		for(TaskGraph taskGraph: mergedTaskGraphMap.values())
		{
			if(taskGraph.getTaskGraphType() == TaskGraphType.DATAFLOW)
			{
				SDFGraph graph = null;
				ArrayList<Channel> channelList = new ArrayList<Channel>();

				for(Channel channel : this.channelList)
				{
					boolean findSrcTask = false;
					boolean findDstTask = false;

					for (Task task : taskGraph.getTaskList())
					{
						Task srcTask = null;
						Task dstTask = null;
						ChannelPort inputPort = channel.getInputPort();
						ChannelPort outputPort = channel.getOutputPort();

						while(outputPort != null && findSrcTask == false)
						{
							srcTask = this.taskMap.get(outputPort.getTaskName());
							if(srcTask.getName().equals(task.getName()))
							{
								findSrcTask = true;
								break;
							}
							outputPort = outputPort.getSubgraphPort();
						}

						while(inputPort != null && findDstTask == false)
						{
							dstTask = this.taskMap.get(inputPort.getTaskName());
							if(dstTask.getName().equals(task.getName()))
							{
								findDstTask = true;
								break;
							}
							inputPort = inputPort.getSubgraphPort();
						}

						if(findSrcTask == true && findDstTask == true)
						{
							channelList.add(channel);
							break;
						}
					}
				}

				if(taskGraph.getParentTask() != null && taskGraph.getParentTask().getModeTransition() != null)
				{
					for(TaskMode mode : taskGraph.getParentTask().getModeTransition().getModeMap().values())
					{
						graph = makeSDFGraph(taskGraph.getTaskList(), channelList, mode);

						if(graph != null)
						{
							setIndividualIterationCount(taskGraph.getTaskList(), graph, mode.getId());
						}
					}
				}
				else
				{
					graph = makeSDFGraph(taskGraph.getTaskList(), channelList, null);

					if(graph != null)
					{
						setIndividualIterationCount(taskGraph.getTaskList(), graph, 0);
					}
				}
			}
		}
	}

	private void makeFullTaskGraph()
	{
		Task parentTask;

		// make global task graph list
		for(Task task : this.taskMap.values())
		{
			TaskGraph taskGraph;
			if(this.fullTaskGraphMap.containsKey(task.getParentTaskGraphName()) == false)
			{
				parentTask = this.taskMap.get(task.getParentTaskGraphName());

				if(parentTask != null)
				{
					taskGraph = new TaskGraph(task.getParentTaskGraphName(), parentTask.getTaskGraphProperty());
					taskGraph.setParentTask(parentTask);
				}
				else
				{
					taskGraph = new TaskGraph(task.getParentTaskGraphName(), this.applicationGraphProperty.getString());
				}

				this.fullTaskGraphMap.put(task.getParentTaskGraphName(), taskGraph);
			}
			else
			{
				taskGraph = this.fullTaskGraphMap.get(task.getParentTaskGraphName());
			}

			taskGraph.putTask(task);
		}
	}

	private void makeSDFTaskIterationCount()
	{
		setIterationCount(this.fullTaskGraphMap);
	}

	private void setMulticastPortMemoryAccessType(ArrayList<MulticastPort> portList, String deviceName, PortDirection direction) throws InvalidDataInMetadataFileException
	{
		Device device = this.deviceInfo.get(deviceName);
		int procId;
		int index = 0;

		for(index = 0; index<portList.size() ; index++)
		{
			MappingInfo taskMappingInfo = findMappingInfoByTaskName(portList.get(index).getTaskName());
			procId = taskMappingInfo.getMappedProcessorList().get(0).getProcessorId();
			for(Processor processor : device.getProcessorList())
			{
				if(procId == processor.getId())
				{
					if(processor.getIsCPU() == true)
					{
						portList.get(index).setMemoryAccessType(InMemoryAccessType.CPU_ONLY);
					}
					else
					{
						if(direction == PortDirection.INPUT)
						{
							portList.get(index).setMemoryAccessType(InMemoryAccessType.CPU_GPU);
						}
						else
						{
							portList.get(index).setMemoryAccessType(InMemoryAccessType.GPU_CPU);
						}
					}
					break;
				}
			}
		}
	}

	private void setMulticastCommunicationType(MulticastGroup multicastGroup, String deviceName, HashSet<String> deviceListMappedWithCounterPorts, PortDirection portDirection) throws InvalidDeviceConnectionException
	{
		Device device = this.deviceInfo.get(deviceName);

		if(deviceListMappedWithCounterPorts.contains(deviceName) == true)
		{
			if(portDirection == PortDirection.INPUT)
			{
				multicastGroup.putInputCommunicationType(MulticastCommunicationType.MULTICAST_COMMUNICATION_TYPE_SHARED_MEMORY);
			}
			else
			{
				multicastGroup.putOutputCommunicationType(MulticastCommunicationType.MULTICAST_COMMUNICATION_TYPE_SHARED_MEMORY);
			}
		}
		for(String counterDeviceName : deviceListMappedWithCounterPorts)
		{
			Device counterDevice = this.deviceInfo.get(counterDeviceName);
			boolean connectionSearched = false;
			if(deviceName.equals(counterDeviceName))
			{
				continue;
			}
			for (DeviceCommunicationType communicationType : device.getRequiredCommunicationSet())
			{
				if (MulticastCommunicationType.getMulticastAvailableCommunicationTypeList().contains(communicationType))
				{
					if (counterDevice.getRequiredCommunicationSet().contains(communicationType))
					{
						connectionSearched = true;
						if(portDirection == PortDirection.INPUT)
						{
							multicastGroup.putInputCommunicationType(MulticastCommunicationType.getMulticastCommunicationTypeByDeviceCommunicationType(communicationType));
						}
						else
						{
							multicastGroup.putOutputCommunicationType(MulticastCommunicationType.getMulticastCommunicationTypeByDeviceCommunicationType(communicationType));
						}
					}
				}
			}
			if (connectionSearched == false)
			{
				throw new InvalidDeviceConnectionException();
			}
		}
	}

	private void addMulticastGroupToDevice(MulticastGroup multicastGroup, String deviceName) throws InvalidDataInMetadataFileException, CloneNotSupportedException
	{
		Device device = this.deviceInfo.get(deviceName);
		device.putMulticastGroup(multicastGroup);
	}

	public void makeMulticastGroupInformation(CICAlgorithmType algorithm_metadata) throws InvalidDataInMetadataFileException, InvalidDeviceConnectionException, CloneNotSupportedException
	{
		int groupId = 0;

		if(algorithm_metadata.getMulticastGroups() == null)
		{
			return;
		}

		for(MulticastGroupType multicastMetadata: algorithm_metadata.getMulticastGroups().getMulticastGroup())
		{
			HashSet<String> deviceListMappedWithInputPort = new HashSet<String>();
			HashSet<String> deviceListMappedWithOutputPort = new HashSet<String>();
			int portId = 0;

			HashMap<String, MulticastGroup> multicastGroups = new HashMap<String, MulticastGroup>();

			for(MulticastPort multicastPort : this.multicastPortInfo.values())
			{
				if(multicastMetadata.getGroupName().equals(multicastPort.getGroupName()) == true)
				{
					String deviceName = findMappingInfoByTaskName(multicastPort.getTaskName()).getMappedDeviceName();
					if(!multicastGroups.containsKey(deviceName))
					{
						multicastGroups.put(deviceName, new MulticastGroup(groupId, multicastMetadata.getGroupName(), multicastMetadata.getSize().intValue()));
					}
					multicastPort.setPortId(portId);

					switch(multicastPort.getDirection())
					{
					case INPUT:
						multicastGroups.get(deviceName).putInputPort(multicastPort);
						deviceListMappedWithInputPort.add(deviceName);
						break;
					case OUTPUT:
						multicastGroups.get(deviceName).putOutputPort(multicastPort);
						deviceListMappedWithOutputPort.add(deviceName);
						break;
					default:
						throw new InvalidDeviceConnectionException();
					}
					portId++;
				} // end of if(multicastMetadata.getGroupName().equals(multicastPort.getGroupName()) == true)
			} // end of for(MulticastPort multicastPort : this.multicastPortInfo.values())
			for(String deviceName : multicastGroups.keySet())
			{
				MulticastGroup multicastGroup = multicastGroups.get(deviceName);

				ArrayList<MulticastPort> inputPortList = multicastGroup.getInputPortList();
				setMulticastCommunicationType(multicastGroup, deviceName, deviceListMappedWithOutputPort, PortDirection.INPUT);
				setMulticastPortMemoryAccessType(inputPortList, deviceName, PortDirection.INPUT);

				ArrayList<MulticastPort> outputPortList = multicastGroup.getOutputPortList();
				setMulticastCommunicationType(multicastGroup, deviceName, deviceListMappedWithInputPort, PortDirection.OUTPUT);
				setMulticastPortMemoryAccessType(outputPortList, deviceName, PortDirection.OUTPUT);

				addMulticastGroupToDevice(multicastGroup, deviceName);
			}

			groupId++;
		} // end of for(MulticastGroupType multicastMetadata: algorithm_metadata.getMulticastGroups().getMulticastGroup())
	}

	public void makeChannelInformation(CICAlgorithmType algorithm_metadata) throws InvalidDataInMetadataFileException, InvalidDeviceConnectionException, CloneNotSupportedException
	{
		int index = 0;
		HashMap<String, ArrayList<Channel>> portToChannelConnection = new HashMap<String, ArrayList<Channel>>();

		makeFullTaskGraph();

		if(algorithm_metadata.getChannels() == null)
		{
			return;
		}

		for(ChannelType channelMetadata: algorithm_metadata.getChannels().getChannel())
		{
			Channel channel = new Channel(index, channelMetadata.getSize().intValue() * channelMetadata.getSampleSize().intValue(),
					channelMetadata.getInitialDataSize().intValue() * channelMetadata.getSampleSize().intValue(), channelMetadata.getSampleSize().intValue());

			// index 0 is only used
			// TODO: src element in XML schema file must be single occurrence.
			ChannelPortType channelSrcPort = channelMetadata.getSrc().get(0);
			ChannelPortType channelDstPort = channelMetadata.getDst().get(0);
			MappingInfo srcTaskMappingInfo;
			MappingInfo dstTaskMappingInfo;

			ChannelPort srcPort = this.portInfo.get(channelSrcPort.getTask() + Constants.NAME_SPLITER + channelSrcPort.getPort() + Constants.NAME_SPLITER + PortDirection.OUTPUT);
			ChannelPort dstPort = this.portInfo.get(channelDstPort.getTask() + Constants.NAME_SPLITER + channelDstPort.getPort() + Constants.NAME_SPLITER + PortDirection.INPUT);

			// This information is used for single port multiple channel connection cases
			setNextChannelId(portToChannelConnection, srcPort, channel);

			// channel type
			setChannelType(channel, srcPort, dstPort);

			// input/output port (port information)
			channel.setOutputPort(srcPort.getMostUpperPort());
			channel.setInputPort(dstPort.getMostUpperPort());

			srcTaskMappingInfo = findMappingInfoByTaskName(channelSrcPort.getTask());
			dstTaskMappingInfo = findMappingInfoByTaskName(channelDstPort.getTask());

			// maximum chunk number
			channel.setMaximumChunkNum(this.taskMap, channelSrcPort.getTask(), channelDstPort.getTask(), srcTaskMappingInfo, dstTaskMappingInfo);

			// communication type (device information)
			setChannelCommunicationType(channel, srcTaskMappingInfo, dstTaskMappingInfo);
			addChannelAndPortInfoToDevice(channel, srcTaskMappingInfo, dstTaskMappingInfo);

			this.channelList.add(channel);
			index++;
		}


		for(Device device: this.deviceInfo.values())
		{
			// set source task of composite task which can be checked after setting channel information
			device.setSrcTaskOfMTM();
			// set channel input/output port index after setting channel information
			device.setChannelPortIndex();
		}

		makeSDFTaskIterationCount();
	}

	// scheduleFolderPath : output + /convertedSDF3xml/
	public void makeMappingAndTaskInformationPerDevices(CICMappingType mapping_metadata, CICProfileType profile_metadata, CICConfigurationType config_metadata, String scheduleFolderPath, CICGPUSetupType gpusetup_metadata)
	{
		//config_metadata.getCodeGeneration().getRuntimeExecutionPolicy().equals(anObject)
		ExecutionPolicy executionPolicy = ExecutionPolicy.fromValue(config_metadata.getCodeGeneration().getRuntimeExecutionPolicy());

		for(Device device: this.deviceInfo.values())
		{
			try
			{
				device.putInDeviceTaskInformation(this.taskMap, scheduleFolderPath, mapping_metadata, executionPolicy, this.applicationGraphProperty, gpusetup_metadata);
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (InvalidScheduleFileNameException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (InvalidDataInMetadataFileException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (NoProcessorFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	private void checkLibraryMasterIsC(Library library)
	{
		for(LibraryConnection connection : library.getLibraryConnectionList())
		{
			if(connection.isMasterLibrary() == true)
			{
				Library masterLibrary = this.libraryMap.get(connection.getMasterName());
				if(masterLibrary.getLanguage() == ProgrammingLanguage.C)
				{
					library.setMasterLanguageC(true);
					break;
				}
			}
			else // task
			{
				Task masterTask = this.taskMap.get(connection.getMasterName());
				if(masterTask.getLanguage() == ProgrammingLanguage.C)
				{
					library.setMasterLanguageC(true);
					break;
				}
			}
		}
	}

	private void setLibraryConnectionInformation(CICAlgorithmType algorithm_metadata)
	{
		if(algorithm_metadata.getLibraryConnections() != null)
		{
			if(algorithm_metadata.getLibraryConnections().getTaskLibraryConnection() != null)
			{
				for(TaskLibraryConnectionType connectionType : algorithm_metadata.getLibraryConnections().getTaskLibraryConnection())
				{
					Library library = this.libraryMap.get(connectionType.getSlaveLibrary());
					LibraryConnection libraryConnection = new LibraryConnection(connectionType.getMasterTask(),
														connectionType.getMasterPort(), false);
					library.getLibraryConnectionList().add(libraryConnection);
				}
			}

			if(algorithm_metadata.getLibraryConnections().getLibraryLibraryConnection() != null)
			{
				for(LibraryLibraryConnectionType connectionType : algorithm_metadata.getLibraryConnections().getLibraryLibraryConnection())
				{
					Library library = this.libraryMap.get(connectionType.getSlaveLibrary());
					LibraryConnection libraryConnection = new LibraryConnection(connectionType.getMasterLibrary(),
														connectionType.getMasterPort(), true);
					library.getLibraryConnectionList().add(libraryConnection);
				}
			}

			//this.taskMap;
			for(Library library : this.libraryMap.values())
			{
				checkLibraryMasterIsC(library);
			}
		}
	}

	private void setLibraryFunction(Library library, LibraryType libraryType)
	{
		for(LibraryFunctionType functionType : libraryType.getFunction())
		{
			Function function = new Function(functionType.getName(), functionType.getReturnType());

			if(functionType.getDescription() != null && functionType.getDescription().trim().length() > 0) {
				function.setDescription(functionType.getDescription());
			}

			for(LibraryFunctionArgumentType argType: functionType.getArgument())
			{
				Argument argument = new Argument(argType.getName(), argType.getType());
				function.getArgumentList().add(argument);

				if(argType.getDescription() != null && argType.getDescription().trim().length() > 0) {
					argument.setDescription(argType.getDescription());
				}
			}

			library.getFunctionList().add(function);
		}
	}

	public void makeLibraryInformation(CICAlgorithmType algorithm_metadata)
	{
		if(algorithm_metadata.getLibraries() != null && algorithm_metadata.getLibraries().getLibrary() != null)
		{
			for(LibraryType libraryType: algorithm_metadata.getLibraries().getLibrary())
			{
				Library library = new Library(libraryType.getName(), libraryType.getType(), libraryType.getFile(), libraryType.getHeader());

				if(libraryType.getDescription() != null && libraryType.getDescription().trim().length() > 0) {
					library.setDescription(libraryType.getDescription());
				}

				setLibraryFunction(library, libraryType);
				library.setExtraHeaderSet(libraryType.getExtraHeader());
				library.setExtraSourceSet(libraryType.getExtraSource());
				library.setLanguageAndFileExtension(libraryType.getLanguage());

				if(libraryType.getCflags() != null) {
					library.setcFlags(libraryType.getCflags());
				}

				if(libraryType.getLdflags() != null) {
					library.setLdFlags(libraryType.getLdflags());
				}

				this.libraryMap.put(libraryType.getName(), library);
			}

			setLibraryConnectionInformation(algorithm_metadata);
			setLibraryInfoPerDevices();
		}
	}

	public void makeConfigurationInformation(CICConfigurationType configuration_metadata)
	{
		this.executionTime = new ExecutionTime(configuration_metadata.getSimulation().getExecutionTime().getValue().intValue(),
												configuration_metadata.getSimulation().getExecutionTime().getMetric().value());
	}

	private void setLibraryInfoPerDevices() {
		for(Device device : this.deviceInfo.values())
		{
			device.putInDeviceLibraryInformation(this.libraryMap);
		}

		//this.libraryMap;
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

	public HashMap<String, Device> getDeviceInfo() {
		return deviceInfo;
	}

	public HashMap<String, HWElementType> getElementTypeHash() {
		return elementTypeHash;
	}

	public HashMap<String, ChannelPort> getPortInfo() {
		return portInfo;
	}

	public ExecutionTime getExecutionTime() {
		return executionTime;
	}

	public HashMap<String, TaskGraph> getFullTaskGraphMap() {
		return fullTaskGraphMap;
	}

	public HashMap<String, Library> getLibraryMap() {
		return libraryMap;
	}

	public HashMap<String, DeviceConnection> getDeviceConnectionMap() {
		return deviceConnectionMap;
	}
}
