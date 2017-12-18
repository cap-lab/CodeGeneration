package org.snu.cse.cap.translator.structure;


import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;

import org.snu.cse.cap.translator.Constants;
import org.snu.cse.cap.translator.Constants.PortDirection;
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
import org.snu.cse.cap.translator.structure.device.NoProcessorFoundException;
import org.snu.cse.cap.translator.structure.device.ProcessorElementType;
import org.snu.cse.cap.translator.structure.device.TCPConnection;
import org.snu.cse.cap.translator.structure.device.connection.DeviceConnection;
import org.snu.cse.cap.translator.structure.device.connection.InvalidDeviceConnectionException;
import org.snu.cse.cap.translator.structure.library.Argument;
import org.snu.cse.cap.translator.structure.library.Function;
import org.snu.cse.cap.translator.structure.library.Library;
import org.snu.cse.cap.translator.structure.mapping.CompositeTaskMappingInfo;
import org.snu.cse.cap.translator.structure.mapping.GeneralTaskMappingInfo;
import org.snu.cse.cap.translator.structure.mapping.InvalidScheduleFileNameException;
import org.snu.cse.cap.translator.structure.mapping.MappingInfo;
import org.snu.cse.cap.translator.structure.task.Task;
import org.snu.cse.cap.translator.structure.task.TaskLoopType;

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
import hopes.cic.xml.ChannelPortType;
import hopes.cic.xml.ChannelType;
import hopes.cic.xml.DeviceConnectionListType;
import hopes.cic.xml.LibraryFunctionArgumentType;
import hopes.cic.xml.LibraryFunctionType;
import hopes.cic.xml.LibraryType;
import hopes.cic.xml.ModeTaskType;
import hopes.cic.xml.ModeType;
import hopes.cic.xml.PortMapType;
import hopes.cic.xml.TCPConnectionType;
import hopes.cic.xml.TaskPortType;
import hopes.cic.xml.TaskRateType;
import hopes.cic.xml.TaskType;

public class Application {
	// Overall metadata information
	private ArrayList<Channel> channelList;
	private HashMap<String, Task> taskMap; // Task name : Task class
	private HashMap<String, Port> portInfo; // Key: taskName/portName/direction, ex) MB_Y/inMB_Y/input
	private HashMap<String, Device> deviceInfo; // device name: Device class
	private HashMap<String, DeviceConnection> deviceConnectionList;
	private HashMap<String, HWElementType> elementTypeHash; // element type name : HWElementType class
	private TaskGraphType applicationGraphProperty;	
	private HashMap<String, Library> libraryMap;
	
	public Application()
	{
		this.channelList = new ArrayList<Channel>();
		this.taskMap = new HashMap<String, Task>();
		this.portInfo = new HashMap<String, Port>();
		this.deviceInfo = new HashMap<String, Device>();
		this.elementTypeHash = new HashMap<String, HWElementType>();
		this.applicationGraphProperty = null;
		this.deviceConnectionList = new HashMap<String, DeviceConnection>();
		this.libraryMap = new HashMap<String, Library>();
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
		
		for(TaskType task_metadata: algorithm_metadata.getTasks().getTask())
		{
			task = new Task(taskId, task_metadata);
						
			this.taskMap.put(task.getName(), task);
			
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
	
	private void setChannelCommunicationType(Channel channel, MappingInfo srcTaskMappingInfo, MappingInfo dstTaskMappingInfo) 
	{
		// Two tasks are connected on different devices
		// TODO: multi device connection must be supported
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
	private void setChannelType(Channel channel, Port srcPort, Port dstPort) {
		boolean isDstDataLoop = false;
		boolean isSrcDataLoop = false;
		boolean isDstDistributing = false;
		boolean isSrcDistributing = false;
		Task srcTask; 
		Task dstTask;
		
		srcTask = this.taskMap.get(srcPort.getTaskName());
		dstTask = this.taskMap.get(dstPort.getTaskName());
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
	
	private void putPortIntoDeviceHierarchically(Device device, Port port, PortDirection direction)
	{
		// key: taskName/portName/direction
		Port currentPort;
		String key;
		
		currentPort = port;
		while(currentPort != null)
		{
			key = currentPort.getTaskName() + Constants.NAME_SPLITER + currentPort.getPortName() + Constants.NAME_SPLITER + direction;
			device.getPortInfo().put(key, currentPort);
			currentPort = currentPort.getSubgraphPort();
		}
	}
	
	private void addChannelAndPortInfoToDevice(Channel channel, MappingInfo srcTaskMappingInfo, MappingInfo dstTaskMappingInfo)
	{
		Device srcDevice = this.deviceInfo.get(srcTaskMappingInfo.getMappedDeviceName());
		Device dstDevice = this.deviceInfo.get(dstTaskMappingInfo.getMappedDeviceName());
		
		// hierarchical put port information
		// key: taskName/portName/direction
		putPortIntoDeviceHierarchically(srcDevice, channel.getInputPort(), PortDirection.INPUT);
		putPortIntoDeviceHierarchically(srcDevice, channel.getOutputPort(), PortDirection.OUTPUT);
		
		putPortIntoDeviceHierarchically(dstDevice, channel.getInputPort(), PortDirection.INPUT);
		putPortIntoDeviceHierarchically(dstDevice, channel.getOutputPort(), PortDirection.OUTPUT);
		
		srcDevice.getChannelList().add(channel);
		dstDevice.getChannelList().add(channel);
	}
	
	public void makeChannelInformation(CICAlgorithmType algorithm_metadata) throws InvalidDataInMetadataFileException
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
			MappingInfo srcTaskMappingInfo;
			MappingInfo dstTaskMappingInfo;	
			
			Port srcPort = this.portInfo.get(channelSrcPort.getTask() + Constants.NAME_SPLITER + channelSrcPort.getPort() + Constants.NAME_SPLITER + Constants.PortDirection.OUTPUT);
			Port dstPort = this.portInfo.get(channelDstPort.getTask() + Constants.NAME_SPLITER + channelDstPort.getPort() + Constants.NAME_SPLITER + Constants.PortDirection.INPUT);
			
			// channel type
			setChannelType(channel, srcPort, dstPort);
			
			// input/output port (port information)
			channel.setOutputPort(srcPort.getMostUpperPortInfo());
			channel.setInputPort(dstPort.getMostUpperPortInfo());
			
			// maximum chunk number
			channel.setMaximumChunkNum(this.taskMap);
			
			srcTaskMappingInfo = findMappingInfoByTaskName(channelSrcPort.getTask());
			dstTaskMappingInfo = findMappingInfoByTaskName(channelDstPort.getTask());
			
			// communication type (device information)
			setChannelCommunicationType(channel, srcTaskMappingInfo, dstTaskMappingInfo);
			addChannelAndPortInfoToDevice(channel, srcTaskMappingInfo, dstTaskMappingInfo);
			
			this.channelList.add(channel);
			index++;
		}
	}

	// scheduleFolderPath : output + /convertedSDF3xml/
	public void makeMappingAndTaskInformationPerDevices(CICMappingType mapping_metadata, CICProfileType profile_metadata, CICConfigurationType config_metadata, String scheduleFolderPath)
	{
		//config_metadata.getCodeGeneration().getRuntimeExecutionPolicy().equals(anObject)
		ExecutionPolicy executionPolicy = ExecutionPolicy.fromValue(config_metadata.getCodeGeneration().getRuntimeExecutionPolicy());
		
		for(Device device: this.deviceInfo.values())
		{
			try 
			{
				device.putInDeviceTaskInformation(this.taskMap, scheduleFolderPath, mapping_metadata, executionPolicy);
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
	
	public void makeLibraryInformation(CICAlgorithmType algorithm_metadata) 
	{
		if(algorithm_metadata.getLibraries() != null && algorithm_metadata.getLibraries().getLibrary() != null)
		{
			for(LibraryType libraryType: algorithm_metadata.getLibraries().getLibrary())
			{
				Library library = new Library(libraryType.getName(), libraryType.getType(), libraryType.getFile(), libraryType.getHeader());
				
				for(LibraryFunctionType functionType : libraryType.getFunction())
				{
					Function function = new Function(functionType.getName(), functionType.getReturnType());

					for(LibraryFunctionArgumentType argType: functionType.getArgument())
					{
						Argument argument = new Argument(argType.getName(), argType.getType());
						function.getArgumentList().add(argument);
					}
					
					library.getFunctionList().add(function);
				}
				
				this.libraryMap.put(libraryType.getName(), library);
			}
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

	public HashMap<String, Device> getDeviceInfo() {
		return deviceInfo;
	}

	public HashMap<String, HWElementType> getElementTypeHash() {
		return elementTypeHash;
	}

	public HashMap<String, Port> getPortInfo() {
		return portInfo;
	}
}
