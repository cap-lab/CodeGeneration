package org.snu.cse.cap.translator.structure;


import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.snu.cse.cap.translator.Constants;
import org.snu.cse.cap.translator.ExecutionTime;
import org.snu.cse.cap.translator.structure.communication.*;
import org.snu.cse.cap.translator.structure.communication.channel.*;
import org.snu.cse.cap.translator.structure.communication.multicast.MulticastCommunicationType;
import org.snu.cse.cap.translator.structure.communication.multicast.MulticastGroup;
import org.snu.cse.cap.translator.structure.communication.multicast.MulticastPort;
import org.snu.cse.cap.translator.structure.device.Device;
import org.snu.cse.cap.translator.structure.device.DeviceCommunicationType;
import org.snu.cse.cap.translator.structure.device.DeviceEncryptionType;
import org.snu.cse.cap.translator.structure.device.EnvironmentVariable;
import org.snu.cse.cap.translator.structure.device.HWCategory;
import org.snu.cse.cap.translator.structure.device.HWElementType;
import org.snu.cse.cap.translator.structure.device.NoProcessorFoundException;
import org.snu.cse.cap.translator.structure.device.Processor;
import org.snu.cse.cap.translator.structure.device.ProcessorElementType;
import org.snu.cse.cap.translator.structure.device.SchedulingMethod;
import org.snu.cse.cap.translator.structure.device.connection.Connection;
import org.snu.cse.cap.translator.structure.device.connection.ConnectionPair;
import org.snu.cse.cap.translator.structure.device.connection.ConstrainedSerialConnection;
import org.snu.cse.cap.translator.structure.device.connection.DeviceConnection;
import org.snu.cse.cap.translator.structure.device.connection.EncryptionInfo;
import org.snu.cse.cap.translator.structure.device.connection.InvalidDeviceConnectionException;
import org.snu.cse.cap.translator.structure.device.connection.ProtocolType;
import org.snu.cse.cap.translator.structure.device.connection.SSLTCPConnection;
import org.snu.cse.cap.translator.structure.device.connection.SerialConnection;
import org.snu.cse.cap.translator.structure.device.connection.TCPConnection;
import org.snu.cse.cap.translator.structure.device.connection.UDPConnection;
import org.snu.cse.cap.translator.structure.device.connection.UnconstrainedSerialConnection;
import org.snu.cse.cap.translator.structure.library.Argument;
import org.snu.cse.cap.translator.structure.library.Function;
import org.snu.cse.cap.translator.structure.library.Library;
import org.snu.cse.cap.translator.structure.library.LibraryConnection;
import org.snu.cse.cap.translator.structure.mapping.InvalidScheduleFileNameException;
import org.snu.cse.cap.translator.structure.mapping.MappedProcessor;
import org.snu.cse.cap.translator.structure.mapping.MappingInfo;
import org.snu.cse.cap.translator.structure.module.Module;
import org.snu.cse.cap.translator.structure.task.Task;
import org.snu.cse.cap.translator.structure.task.TaskLoopType;
import org.snu.cse.cap.translator.structure.task.TaskMode;

import hopes.cic.xml.*;
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
	private List<Channel> channelList;
	private Map<String, Task> taskMap; // Task name : Task class
	private Map<String, TaskGraph> fullTaskGraphMap; // Task name : Task class
	private Map<String, ChannelPort> portInfo; // Key: taskName/portName/direction, ex) MB_Y/inMB_Y/input
	private Map<String, MulticastPort> multicastPortInfo; // Key: taskName/portName/direction, ex) MB_Y/inMB_Y/input
	private Map<String, Device> deviceInfo; // device name: Device class
	private Map<String, DeviceConnection> deviceConnectionMap;
	private Map<String, HWElementType> elementTypeHash; // element type name : HWElementType class
	private TaskGraphType applicationGraphProperty;
	private Map<String, Library> libraryMap; // library name : Library class
	private ExecutionTime executionTime;

	public Application() {
		this.channelList = new ArrayList<Channel>();
		this.taskMap = new HashMap<String, Task>();
		this.portInfo = new HashMap<String, ChannelPort>();
		this.multicastPortInfo = new HashMap<String, MulticastPort>();
		this.deviceInfo = new HashMap<String, Device>();
		this.elementTypeHash = new HashMap<String, HWElementType>();
		this.deviceConnectionMap = new HashMap<String, DeviceConnection>();
		this.libraryMap = new HashMap<String, Library>();
		this.fullTaskGraphMap = new HashMap<String, TaskGraph>();
	}

	private void gatherPortInfo(TaskType taskMetadata, int taskId) {
		for (TaskPortType portType : taskMetadata.getPort()) {
			ChannelPort channelPort = new ChannelPort(taskId, taskMetadata.getName(), portType);
			portInfo.put(channelPort.getPortKey(), channelPort);
		}
	}

	private void gatherMulticastPortInfo(TaskType taskMetadata, int taskId) {
		for (MulticastPortType multicastPortType : taskMetadata.getMulticastPort()) {
			MulticastPort multicastPort = new MulticastPort(taskId, taskMetadata.getName(), multicastPortType);
			this.multicastPortInfo.put(multicastPort.getPortKey(), multicastPort);
		}
	}

	private void gatherTaskGraphInfo(CICAlgorithmType algorithmMetadata) {
		int taskId = 0;
		for (TaskType taskMetadata : algorithmMetadata.getTasks().getTask()) {
			taskMap.put(taskMetadata.getName(), new Task(taskId, taskMetadata));
			gatherPortInfo(taskMetadata, taskId);
			gatherMulticastPortInfo(taskMetadata, taskId);
			taskId++;
		}
		setLoopDesignatedTaskIdFromTaskName();
		if (algorithmMetadata.getPortMaps() != null) {
			setPortMapInformation(algorithmMetadata);
		}
	}

	// subgraphPort, upperGraphPort, maxAvailableNum
	private void setPortMapInformation(CICAlgorithmType algorithmMetadata)
	{
		for (PortMapType portMapType : algorithmMetadata.getPortMaps().getPortMap())
		{
			PortDirection direction = PortDirection.fromValue(portMapType.getDirection().value());

			ChannelPort port = portInfo.get(getPortKey(portMapType));

			if(portMapType.getChildTask() != null && portMapType.getChildTaskPort() != null)
			{
				ChannelPort childPort = portInfo.get(getChildPortKey(portMapType));

				port.setSubgraphPort(childPort);
				childPort.setUpperGraphPort(port);
			}

			port.setLoopPortType(LoopPortType.fromValue(portMapType.getType().value()));

			Task task = taskMap.get(portMapType.getTask());
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

	private String getPortKey(PortMapType portMapType) {
		return getPortKey(portMapType, portMapType.getTask(), portMapType.getPort());
	}

	private String getChildPortKey(PortMapType portMapType) {
		return getPortKey(portMapType, portMapType.getChildTask(), portMapType.getChildTaskPort());
	}

	private String getPortKey(PortMapType portMapType, String taskName, String portName) {
		PortDirection direction = PortDirection.fromValue(portMapType.getDirection().value());
		return taskName + Constants.NAME_SPLITER + portName + Constants.NAME_SPLITER + direction;
	}

	// taskMap, taskGraphList
	public void makeTaskInformation(CICAlgorithmType algorithmMetadata)
	{
		Task task;

		this.applicationGraphProperty = TaskGraphType.fromValue(algorithmMetadata.getProperty());

		gatherTaskGraphInfo(algorithmMetadata);

		// It only uses single modes - mode information in XML
		ModeType mode = algorithmMetadata.getModes().getMode().get(0);

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
	
	private void putSerialConnectionsOnDevice(Device device, DeviceConnectionListType connectionList)
	{
		if(connectionList.getSerialConnection() == null)
		{
			return;
		}
		for(SerialConnectionType connectionType : connectionList.getSerialConnection())
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
	
	private void putTCPConnectionsOnDevice(Device device, DeviceConnectionListType connectionList)
	{
		if(connectionList.getTCPConnection() == null)
		{
			return;
		}		
		for(TCPConnectionType connectionType : connectionList.getTCPConnection())
		{
			if (!connectionType.isSecure()) {
				TCPConnection connection = null;
				connection = new TCPConnection(connectionType.getName(), connectionType.getRole()
						.toString(),
					connectionType.getIp(), connectionType.getPort().intValue());
				device.putConnection(connection);
			}
		}
	}

	private void putSSLTCPConnectionsOnDevice(Device device, DeviceConnectionListType connectionList) {
		if (connectionList.getTCPConnection() == null) {
			return;
		}
		for (TCPConnectionType connectionType : connectionList.getTCPConnection()) {
			if (connectionType.isSecure()) {
				SSLTCPConnection connection = null;
				connection = new SSLTCPConnection(connectionType.getName(), connectionType.getRole()
						.toString(),
					connectionType.getIp(), connectionType.getPort().intValue(), connectionType.getCaPublicKey(),
					connectionType.getPublicKey(), connectionType.getPrivateKey());
				device.putConnection(connection);
			}
		}
	}
	

	private void putSupportedConnectionTypeListOnDevice(Device device, DeviceConnectionListType connectionList) 
	{
		if(connectionList.getSerialConnection() != null)
		{
			if(connectionList.getSerialConnection().stream().filter(x -> x.getNetwork().equals(NetworkType.BLUETOOTH)).count() > 0) 
			{
				device.putSupportedConnectionType(DeviceCommunicationType.BLUETOOTH);
			}
			if(connectionList.getSerialConnection().stream().filter(x -> x.getNetwork().equals(NetworkType.WIRE)).count() > 0) 
			{
				device.putSupportedConnectionType(DeviceCommunicationType.SERIAL);
			}
			if(connectionList.getSerialConnection().stream().filter(x -> x.getNetwork().equals(NetworkType.USB)).count() > 0) 
			{
				device.putSupportedConnectionType(DeviceCommunicationType.SERIAL);
			}
		}
		if(connectionList.getTCPConnection() != null && connectionList.getTCPConnection().size() > 0)
		{
			device.putSupportedConnectionType(DeviceCommunicationType.TCP);
			for (TCPConnectionType connectionType : connectionList.getTCPConnection()) {
				if (connectionType.isSecure()) {
					device.putSupportedConnectionType(DeviceCommunicationType.SECURE_TCP);
					break;
				}
			}
		}
		if(connectionList.getUDPConnection() != null && connectionList.getUDPConnection().size() > 0)
		{
			device.putSupportedConnectionType(DeviceCommunicationType.UDP);
		}
	}




	private void putConnectionsOnDevice(Device device, DeviceConnectionListType connectionList)
	{
		putSerialConnectionsOnDevice(device, connectionList);
		putTCPConnectionsOnDevice(device, connectionList);
		putSupportedConnectionTypeListOnDevice(device, connectionList);
		putSSLTCPConnectionsOnDevice(device, connectionList);
	}

	private Connection findConnection(String deviceName, String connectionName)
			throws InvalidDeviceConnectionException
	{
		Device device;
		Connection connection;

		device = this.deviceInfo.get(deviceName);
		connection = device.getConnection(connectionName);
	
		return connection;
	}
	
	private void putSupportedEncryptionTypeListOnDevice(Device device, EncryptionInfo encryption) {
		if (encryption.getEncryptionType().contentEquals(EncryptionType.LEA.toString())) {
			device.putSupportedEncryptionType(DeviceEncryptionType.LEA);
		}
		else if (encryption.getEncryptionType().contentEquals(EncryptionType.HIGHT.toString())) {
			device.putSupportedEncryptionType(DeviceEncryptionType.HIGHT);
		}
		else if (encryption.getEncryptionType().contentEquals(EncryptionType.SEED.toString())) {
			device.putSupportedEncryptionType(DeviceEncryptionType.SEED);
		}
	}

	private String setInitializationVectorLen(String initializationVector, EncryptionInfo encryption) {
		String iv;
		int index = 0;
		
		if (encryption.getEncryptionType().contentEquals(EncryptionType.LEA.toString())) {
			index = DeviceEncryptionType.LEA.getBlockSize();
		} else if (encryption.getEncryptionType().contentEquals(EncryptionType.HIGHT.toString())) {
			index = DeviceEncryptionType.HIGHT.getBlockSize();
		} else if (encryption.getEncryptionType().contentEquals(EncryptionType.SEED.toString())) {
			index = DeviceEncryptionType.SEED.getBlockSize();
		}
		
		iv = initializationVector.substring(0, index);
		
		return iv;
	}

	private boolean checkEncryptionKeySize(String encryptionType, int keyLen) {
		boolean result = true;

		if (encryptionType.contentEquals(DeviceEncryptionType.LEA.toString())) {
			if (keyLen != DeviceEncryptionType.KEY128_BYTE_SIZE && keyLen != DeviceEncryptionType.KEY192_BYTE_SIZE
					&& keyLen != DeviceEncryptionType.KEY256_BYTE_SIZE) {
				result = false;
			}
		} else if (encryptionType.contentEquals(DeviceEncryptionType.HIGHT.toString())) {
			if (keyLen != DeviceEncryptionType.KEY64_BYTE_SIZE) {
				result = false;
			}
		} else if (encryptionType.contentEquals(DeviceEncryptionType.SEED.toString())) {
			if (keyLen != DeviceEncryptionType.KEY128_BYTE_SIZE) {
				result = false;
			}
		}

		return result;
	}

	private void setEncryption(String deviceName, String connectionName, String encryptionType,
			String userKey,
			String initializationVector)
			throws InvalidDeviceConnectionException {
		Device device;
		Connection connection;
		String iv;
		int index = -1;
		
		try {
			if (!checkEncryptionKeySize(encryptionType, userKey.length())) {
				throw new InvalidDeviceConnectionException();
			}
		} catch (InvalidDeviceConnectionException e) {
			System.out.println("ERROR key length is wrong, key length: " + userKey.length());
			return;
		}

		device = this.deviceInfo.get(deviceName);

		connection = device.getConnection(connectionName);

		index = device.getEncryptionIndex(encryptionType, userKey);

		if (index == -1) {
			EncryptionInfo encryption = new EncryptionInfo(encryptionType, userKey);

			iv = setInitializationVectorLen(initializationVector, encryption);

			encryption.setEncryptionType(encryptionType);
			encryption.setUserKey(userKey);
			encryption.setUserKeyLen(userKey.length());
			encryption.setInitializationVector(iv);

			putSupportedEncryptionTypeListOnDevice(device, encryption);

			device.putEncryptionList(encryption);
			index = device.getEncryptionList().size() - 1;
		}

		connection.setEncryptionListIndex(index);
	}


	private String generateInitializationVector(int n) {
		String str = "";
		for(int i = 0; i < n; i++) {
			str += (int) Math.floor(Math.random() * 10);
		}
		return str;
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

					String initializationVector = generateInitializationVector(DeviceEncryptionType.MAX_BLOCK_SIZE);

					for(ArchitectureConnectionSlaveType slaveType: connectType.getSlave())
					{
						Connection slave;

						slave = findConnection(slaveType.getDevice(), slaveType.getConnection());

						if (connectType.getEncryption() != null && !connectType.getEncryption().toString()
								.contentEquals(EncryptionType.NO.toString()))
						{
							setEncryption(slaveType.getDevice(), slaveType.getConnection(),
									connectType.getEncryption().toString(), connectType.getUserkey(),
									initializationVector);
							setEncryption(connectType.getMaster(), master.getName(),
									connectType.getEncryption().toString(), connectType.getUserkey(),
									initializationVector);
						}

						deviceConnection.putMasterToSlaveConnection(master, slaveType.getDevice(), slave);
						deviceConnection.putSlaveToMasterConnection(slaveType.getDevice(), slave, master,
								((connectType.getEncryption() == null) ? EncryptionType.NO.toString()
										: connectType.getEncryption().toString()),
								connectType.getUserkey());
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
						device_metadata.getPlatform(), device_metadata.getRuntime(),
						((device_metadata.getScheduler() == null) ? SchedulingMethod.OTHER.toString()
								: device_metadata.getScheduler().toString()));

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
				channel.setRemoteMethodType(RemoteCommunicationType.BLUETOOTH);
				break;
			case USB:
			case WIRE:
				channel.setRemoteMethodType(RemoteCommunicationType.SERIAL);
				break;
			case ETHERNET_WI_FI:
			default:
				throw new UnsupportedOperationException();
			}
			break;
		case TCP:
			channel.setRemoteMethodType(RemoteCommunicationType.TCP);
			break;
		case SECURE_TCP:
			channel.setRemoteMethodType(RemoteCommunicationType.SECURE_TCP);
			break;
		default:
			throw new UnsupportedOperationException();
		}
	}

	private void setChannelConnectionRoleType(Channel channel, ConnectionPair connectionPair, String taskName)
	{
		if (connectionPair.getMasterConnection().getProtocol() == ProtocolType.TCP
				|| connectionPair.getMasterConnection().getProtocol() == ProtocolType.SECURE_TCP) {
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

		channel.setCommunicationType(CommunicationType.REMOTE_WRITER);

		setRemoteCommunicationMethodType(channel, connectionPair);
		setChannelConnectionRoleType(channel, connectionPair, srcTaskDevice);

		setEncryptionIndexInfo(channel, connectionPair);
	}

	private void setEncryptionIndexInfo(Channel channel, ConnectionPair connectionPair) {

		int encryptionListIndex = connectionPair.getSlaveConnection().getEncryptionListIndex();

		channel.setEncryptionListIndex(encryptionListIndex);
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

	// TODO: Only support CPU, GPU, and VIRTUAL cases
	private void setInDeviceCommunicationType(Channel channel, MappingInfo srcTaskMappingInfo, MappingInfo dstTaskMappingInfo)
	{
		boolean srcCPU = false, dstCPU = false;
		boolean srcVirtual = false, dstVirtual = false;
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
				srcVirtual = processor.getIsVirtual();
			}

			if(dstProcId == processor.getId()) {
				dstCPU = processor.getIsCPU();
				dstVirtual = processor.getIsVirtual();
			}
		}

		channel.setCommunicationType(CommunicationType.SHARED_MEMORY);
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
			else if (srcVirtual || dstVirtual) {
				channel.setAccessType(InMemoryAccessType.CPU_ONLY);
			}
			else {
				throw new UnsupportedOperationException();
			}
		}
		else { // dstCPU == false && srcCPU == true
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
				device.getPortKeyToIndex().put(key, Integer.valueOf(device.getPortList().size()));
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
		TCPConnection tcpConnection = null;
		ArrayList<TCPConnection> tcpConnectionList = null;
		SSLTCPConnection sslTcpconnection = null;
		ArrayList<SSLTCPConnection> sslTcpConnectionList = null;

		switch (channel.getRemoteMethodType()) {
		case TCP:
			switch (channel.getConnectionRoleType()) {
			case SERVER:
				tcpConnectionList = targetDevice.getTcpServerList();
				tcpConnection = (TCPConnection) connectionPair.getMasterConnection();
				break;
			case CLIENT:
				tcpConnectionList = targetDevice.getTcpClientList();
				tcpConnection = (TCPConnection) connectionPair.getSlaveConnection();
				break;
			default:
				throw new InvalidDeviceConnectionException();
			}

			tcpConnection.incrementChannelAccessNum();

			for (index = 0; index < tcpConnectionList.size(); index++) {
				TCPConnection curConnection = tcpConnectionList.get(index);

				if (curConnection.getName().equals(tcpConnection.getName()) == true) {
					// same connection name
					channel.setSocketInfoIndex(index);
					break;
				}
			}
			break;
		case SECURE_TCP:
			switch (channel.getConnectionRoleType()) {
			case SERVER:
				sslTcpConnectionList = targetDevice.getSecureTcpServerList();
				sslTcpconnection = (SSLTCPConnection) connectionPair.getMasterConnection();
				break;
			case CLIENT:
				sslTcpConnectionList = targetDevice.getSecureTcpClientList();
				sslTcpconnection = (SSLTCPConnection) connectionPair.getSlaveConnection();
				break;
			default:
				throw new InvalidDeviceConnectionException();
			}

			sslTcpconnection.incrementChannelAccessNum();

			for (index = 0; index < sslTcpConnectionList.size(); index++) {
				SSLTCPConnection curConnection = sslTcpConnectionList.get(index);

				if (curConnection.getName().equals(sslTcpconnection.getName()) == true) {
					// same connection name
					channel.setSocketInfoIndex(index);
					break;
				}
			}
			break;
		default:
			throw new InvalidDeviceConnectionException();
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
			case MASTER:
				connectionList = targetDevice.getSerialConstrainedMasterList();
				connection = (ConstrainedSerialConnection) connectionPair.getMasterConnection();
				break;
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
				connectionList = targetDevice.getSerialUnconstrainedMasterList();
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
			if (connectionPair.getMasterConnection().getProtocol() == ProtocolType.TCP
					|| connectionPair.getMasterConnection().getProtocol() == ProtocolType.SECURE_TCP) {
				setSocketIndexFromTCPConnection(channel, targetDevice, connectionPair);
			}
			else {
				setSocketIndexFromUnconstrainedSerialConnection(channel, targetDevice, connectionPair);
			}
			break;
		case WINDOWS:
			if (connectionPair.getMasterConnection().getProtocol() == ProtocolType.TCP
					|| connectionPair.getMasterConnection().getProtocol() == ProtocolType.SECURE_TCP) {
				setSocketIndexFromTCPConnection(channel, targetDevice, connectionPair);
			} else {
				setSocketIndexFromUnconstrainedSerialConnection(channel, targetDevice, connectionPair);
			}
			break;
		case UCOS3:
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

			channelInDevice.setCommunicationType(CommunicationType.REMOTE_READER);
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
	private Task getTaskOfMergedGraph(ChannelPort port)
	{
		Task task = null;
		ChannelPort currentPort = null;
		
		currentPort = port;
		task = this.taskMap.get(currentPort.getTaskName());
		
		while(currentPort != null) {
			
			if(task.getLoopStruct() != null || task.getModeTransition() != null) {
				break;
			}
			task = this.taskMap.get(currentPort.getTaskName());
			currentPort = currentPort.getSubgraphPort();
		}
		
		return task;
	}

	private boolean addEdge(SDFGraph graph, TaskMode mode, Channel channel, HashMap<String, Node> unconnectedSDFTaskMap)
	{
		boolean isSDF = true;
		Task srcTask;
		Task dstTask;
		int srcRate = 0;
		int dstRate = 0;		
		
		srcTask = getTaskOfMergedGraph(channel.getOutputPort());
		dstTask = getTaskOfMergedGraph(channel.getInputPort());

		if(channel.getOutputPort().getPortSampleRateList().size() == 1 || mode == null) {
			srcRate = channel.getOutputPort().getPortSampleRateList().get(0).getSampleRate();
		}
		else {
			srcRate = Constants.INVALID_VALUE;
			for(PortSampleRate rate: channel.getOutputPort().getPortSampleRateList()) {
				if(mode.getName().equals(rate.getModeName()) == true) {
					srcRate = rate.getSampleRate();
					break;
				}
			}
		}

		if(channel.getInputPort().getPortSampleRateList().size() == 1 || mode == null) {
			dstRate = channel.getInputPort().getPortSampleRateList().get(0).getSampleRate();
		}
		else {
			dstRate = Constants.INVALID_VALUE;
			for(PortSampleRate rate: channel.getInputPort().getPortSampleRateList()) {
				if(mode.getName().equals(rate.getModeName()) == true) {
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

			if(srcTask.getLoopStruct() != null && srcTask.getChildTaskGraphName() == null) {
				srcRate = srcRate / srcTask.getLoopStruct().getLoopCount();
			}

			if(dstTask.getLoopStruct() != null && dstTask.getLoopStruct().getLoopType() == TaskLoopType.DATA &&
				dstTask.getChildTaskGraphName() == null && 
				channel.getInputPort().getLoopPortType() == LoopPortType.DISTRIBUTING) {
				dstRate = dstRate / dstTask.getLoopStruct().getLoopCount();
			}
			else if(dstTask.getLoopStruct() != null &&
				dstTask.getChildTaskGraphName() == null) {
				if(dstRate / dstTask.getLoopStruct().getLoopCount() == 0) {
					srcRate = srcRate * dstTask.getLoopStruct().getLoopCount();
					initialData = initialData * dstTask.getLoopStruct().getLoopCount();
				}
				else {
					dstRate = dstRate / dstTask.getLoopStruct().getLoopCount();
				}
			}

			if(initialData == 0) {
				for(Object nodeObj : graph.nodes()) {
					Node node = (Node) nodeObj;
					if(graph.getName(node).equals(srcTask.getName()) == true) {
						srcNode = node;
						unconnectedSDFTaskMap.remove(srcTask.getName());
					}

					if(graph.getName(node).equals(dstTask.getName()) == true) {
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

	private void setLoopDesignatedTaskIdFromTaskName() {
		for (Task task : this.taskMap.values()) {
			Task designatedTask;
			if (task.getLoopStruct() != null && task.getLoopStruct().getLoopType() == TaskLoopType.CONVERGENT) {
				designatedTask = this.taskMap.get(task.getLoopStruct().getDesignatedTaskName());
				task.getLoopStruct().setDesignatedTaskId(designatedTask.getId());
			}

		}
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

	private TaskGraph getTaskGraphCanBeMerged(TaskGraph taskGraph, Map<String, TaskGraph> taskGraphMap,
			Map<String, TaskGraph> mergedTaskGraphMap)
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

	private HashMap<String, TaskGraph> mergeTaskGraph(Map<String, TaskGraph> taskGraphMap)
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

	private void setIterationCount(Map<String, TaskGraph> taskGraphMap)
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

	private Boolean checkMulticastPortMappedToCPU(MulticastPort multicastPort, Device device)  throws InvalidDataInMetadataFileException
	{
		for(MappedProcessor mappedProcessor : findMappingInfoByTaskName(multicastPort.getTaskName()).getMappedProcessorList())
		{
			for(Processor processor : device.getProcessorList())
			{
				if(mappedProcessor.getProcessorId() == processor.getId())
				{
					return processor.getIsCPU();
				}
			}
		}
		throw new InvalidDataInMetadataFileException();
	}
	
	private void setMulticastPortMemoryAccessType(HashMap<String, MulticastGroup> multicastGroupListPerDevice) throws InvalidDataInMetadataFileException
	{
		for(String deviceName : multicastGroupListPerDevice.keySet())
		{
			MulticastGroup multicastGroup = multicastGroupListPerDevice.get(deviceName);
			
			for(MulticastPort multicastPort : multicastGroup.getPortList())
			{
				if (checkMulticastPortMappedToCPU(multicastPort, this.getDeviceInfo().get(deviceName)) == true)
				{
					multicastPort.setMemoryAccessType(InMemoryAccessType.CPU_ONLY);
				} 
				else
				{
					if(multicastPort.getDirection() == PortDirection.INPUT)
					{
						multicastPort.setMemoryAccessType(InMemoryAccessType.CPU_GPU);
					}
					else
					{
						multicastPort.setMemoryAccessType(InMemoryAccessType.GPU_CPU);
					}
				}
			}
		}
	}

	private void addMulticastGroupsToDevice(HashMap<String, MulticastGroup> multicastGroupListPerDevice) throws InvalidDataInMetadataFileException, CloneNotSupportedException
	{
		for(String deviceName : multicastGroupListPerDevice.keySet())
		{
			Device device = this.deviceInfo.get(deviceName);
			device.putMulticastGroup(multicastGroupListPerDevice.get(deviceName));
		}
	}
	
	private void setMulticastPortId(HashMap<String, MulticastGroup> multicastGroupListPerDevice)
	{
		int portId = 0;
		for(MulticastGroup multicastGroup : multicastGroupListPerDevice.values())
		{
			for(MulticastPort multicastPort : multicastGroup.getPortList())
			{
				multicastPort.setPortId(portId);
				portId++;
			}
		}
	}
	
	private HashMap<String, MulticastGroup> makeMulticastGroupsPerDevice(MulticastGroup groupInfo) throws InvalidDataInMetadataFileException, CloneNotSupportedException {
		HashMap<String, MulticastGroup> multicastGroupListPerDevice = new HashMap<String, MulticastGroup>();
		
		for(MulticastPort multicastPort : this.multicastPortInfo.values())
		{
			if(groupInfo.getGroupName().equals(multicastPort.getGroupName()) == true)
			{
				String deviceName = findMappingInfoByTaskName(multicastPort.getTaskName()).getMappedDeviceName();
				if(!multicastGroupListPerDevice.containsKey(deviceName))
				{
					multicastGroupListPerDevice.put(deviceName, new MulticastGroup(groupInfo.getMulticastGroupId(), groupInfo.getGroupName(), groupInfo.getBufferSize()));
				}

				multicastGroupListPerDevice.get(deviceName).putPort(multicastPort.getDirection(), multicastPort);
			}
		}

		return multicastGroupListPerDevice;
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
			HashMap<String, MulticastGroup> multicastGroupPerDevice = makeMulticastGroupsPerDevice(
					new MulticastGroup(groupId, multicastMetadata.getGroupName(), multicastMetadata.getSize().intValue()));
			
			setMulticastPortId(multicastGroupPerDevice);
			
			setMulticastPortMemoryAccessType(multicastGroupPerDevice);
			
			addMulticastGroupsToDevice(multicastGroupPerDevice);
			
			groupId++;
		} 
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

			// System.out.println(channelSrcPort.getTask() + Constants.NAME_SPLITER +
			// channelSrcPort.getPort()
			// + Constants.NAME_SPLITER + PortDirection.OUTPUT);
			portInfo.keySet().forEach(System.out::println);
			ChannelPort srcPort = this.portInfo.get(channelSrcPort.getTask() + Constants.NAME_SPLITER + channelSrcPort.getPort() + Constants.NAME_SPLITER + PortDirection.OUTPUT);
			ChannelPort dstPort = this.portInfo.get(channelDstPort.getTask() + Constants.NAME_SPLITER + channelDstPort.getPort() + Constants.NAME_SPLITER + PortDirection.INPUT);

			// This information is used for single port multiple channel connection cases
			setNextChannelId(portToChannelConnection, srcPort, channel);

			// channel type
			setChannelType(channel, srcPort, dstPort);

			// input/output port (port information)
			channel.setOutputPort(srcPort.getMostUpperPort());
			channel.setInputPort(dstPort.getMostUpperPort());

			// System.out.println("ChannelSrcPort: " + channelSrcPort.getTask() + "/" +
			// channelSrcPort.getPort());
			// System.out.println("ChannelDstPort: " + channelDstPort.getTask() + "/" +
			// channelDstPort.getPort());
			srcTaskMappingInfo = findMappingInfoByTaskName(channelSrcPort.getTask());
			dstTaskMappingInfo = findMappingInfoByTaskName(channelDstPort.getTask());

			// maximum chunk number
			channel.setMaximumChunkNum(taskMap, channelSrcPort.getTask(), channelDstPort.getTask(), srcTaskMappingInfo,
					dstTaskMappingInfo);

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
	
	private void makeChannelConnectionMappingInfo(CICMappingType mapping_metadata)
	{
		//TODO : implement it after UI
	}
	
	private void checkSharedMemoryIsUsedAndSet(MulticastGroup multicastGroup)
	{
		if(multicastGroup.getInputPortNum() > 0)
		{
			multicastGroup.putInputCommunicationType(MulticastCommunicationType.SHARED_MEMORY);
			if(multicastGroup.getOutputPortNum() > 0)
			{
				multicastGroup.putOutputCommunicationType(MulticastCommunicationType.SHARED_MEMORY);
			}
		}
	}
	
	private void putCommunicationTypeOfMulticast(MulticastGroup multicastGroup, Connection connection, MulticastCommunicationType communicationType) 
	{
		if(multicastGroup.getOutputPortNum() > 0)
		{
			multicastGroup.putOutputCommunicationType(communicationType);
			connection.putMulticastSender(multicastGroup.getMulticastGroupId());
		}
		if(multicastGroup.getInputPortNum() > 0)
		{
			multicastGroup.putInputCommunicationType(communicationType);
			connection.putMulticastReceiver(multicastGroup.getMulticastGroupId());
		}
	}
	
	private void checkRemoteCommunicationIsUsedAndSet(MulticastGroup multicastGroup, Device device, MappingMulticastConnectionType connectionTypeList) 
	{
		HashSet<DeviceCommunicationType> supportedConnectionType = device.getRequiredCommunicationSet();
		
		if(connectionTypeList.getUDP() != null && supportedConnectionType.contains(DeviceCommunicationType.UDP))
		{
			if(device.checkUDPConnectionIsCreated(connectionTypeList.getUDP().getIp()) == false) 
			{
				device.putUDPConnection(connectionTypeList.getUDP().getIp(), new UDPConnection(null, null, connectionTypeList.getUDP().getIp(), connectionTypeList.getUDP().getPort().intValue()));
			}
			putCommunicationTypeOfMulticast(multicastGroup, device.getUDPConnection(connectionTypeList.getUDP().getIp()), MulticastCommunicationType.UDP);
		}
	}
	
	private void setMulticastConnectionType(MappingMulticastConnectionType connectionTypeList, MulticastGroup multicastGroup, Device device)
	{
		checkSharedMemoryIsUsedAndSet(multicastGroup);
		
		checkRemoteCommunicationIsUsedAndSet(multicastGroup, device, connectionTypeList);
	}
	
	private void makeMulticastConnectionMappingInfo(CICMappingType mapping_metadata)
	{
		for(Device device : this.deviceInfo.values())
		{
			for(MappingMulticastType multicastConnectionInfo : mapping_metadata.getMulticast())	
			{
				if(device.getMulticastGroupMap().containsKey(multicastConnectionInfo.getGroupName()))
				{
					setMulticastConnectionType(multicastConnectionInfo.getConnectionType(), device.getMulticastGroupMap().get(multicastConnectionInfo.getGroupName()), device);
				}
			}
			
		}
	}
	
	public void makeConnectionMappingInfo(CICMappingType mapping_metadata)
	{
		makeChannelConnectionMappingInfo(mapping_metadata);
		
		makeMulticastConnectionMappingInfo(mapping_metadata);
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
				device.putInDeviceTaskInformation(taskMap, scheduleFolderPath, mapping_metadata, executionPolicy,
						applicationGraphProperty, gpusetup_metadata);
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
		// System.out.println("Libraries " + algorithm_metadata.getLibraries());
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
		for (Device device : deviceInfo.values())
		{
			device.putInDeviceLibraryInformation(libraryMap);
		}

		//this.libraryMap;
	}

	public TaskGraphType getApplicationGraphProperty() {
		return applicationGraphProperty;
	}

	public void setApplicationGraphProperty(TaskGraphType applicationGraphProperty) {
		this.applicationGraphProperty = applicationGraphProperty;
	}

	public List<Channel> getChannelList() {
		return channelList;
	}

	public Map<String, Task> getTaskMap() {
		return taskMap;
	}

	public Map<String, Device> getDeviceInfo() {
		return deviceInfo;
	}

	public ExecutionTime getExecutionTime() {
		return executionTime;
	}

	public Map<String, TaskGraph> getFullTaskGraphMap() {
		return fullTaskGraphMap;
	}

	public Map<String, Library> getLibraryMap() {
		return libraryMap;
	}

	public Map<String, DeviceConnection> getDeviceConnectionMap() {
		return deviceConnectionMap;
	}
}
