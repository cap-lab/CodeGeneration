package org.snu.cse.cap.translator.structure;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Locale.Category;

import org.snu.cse.cap.translator.structure.channel.Channel;
import org.snu.cse.cap.translator.structure.device.BluetoothConnection;
import org.snu.cse.cap.translator.structure.device.Device;
import org.snu.cse.cap.translator.structure.device.HWCategory;
import org.snu.cse.cap.translator.structure.device.HWElementType;
import org.snu.cse.cap.translator.structure.device.ProcessorElementType;
import org.snu.cse.cap.translator.structure.device.TCPConnection;
import org.snu.cse.cap.translator.structure.mapping.MappingInfo;
import org.snu.cse.cap.translator.structure.task.Task;

import Translators.Constants;
import hopes.cic.xml.ArchitectureDeviceType;
import hopes.cic.xml.ArchitectureElementType;
import hopes.cic.xml.ArchitectureElementTypeType;
import hopes.cic.xml.BluetoothConnectionType;
import hopes.cic.xml.CICAlgorithmType;
import hopes.cic.xml.CICArchitectureType;
import hopes.cic.xml.CICMappingType;
import hopes.cic.xml.CICProfileType;
import hopes.cic.xml.CICScheduleType;
import hopes.cic.xml.ModeTaskType;
import hopes.cic.xml.ModeType;
import hopes.cic.xml.TCPConnectionType;
import hopes.cic.xml.TaskType;

public class Application {
	private ArrayList<Channel> channel;
	private HashMap<String, Task> taskMap;
	private HashMap<String, TaskGraph> taskGraphList;
	private ArrayList<MappingInfo> mappingInfo;
	private ArrayList<Device> deviceInfo;
	private HashMap<String, HWElementType> elementTypeList;
	
	public Application()
	{
		this.channel = new ArrayList<Channel>();	
		this.taskMap = new HashMap<String, Task>();
		this.taskGraphList = new HashMap<String, TaskGraph>();
		this.mappingInfo = new ArrayList<MappingInfo>();
		this.deviceInfo = new ArrayList<Device>();
		this.elementTypeList = new HashMap<String, HWElementType>();
	}
	
	// taskMap, taskGraphList
	public void makeTaskInformation(CICAlgorithmType algorithm_metadata)
	{
		int loop = 0;
		Task task;
		int inGraphIndex = 0;
		
		for(TaskType task_metadata: algorithm_metadata.getTasks().getTask())
		{
			TaskGraph taskGraph;
			task = new Task(loop, task_metadata);
						
			this.taskMap.put(task.getName(), task);
			
			if(this.taskGraphList.containsKey(task.getParentTaskGraphName()) == false)
			{
				taskGraph = new TaskGraph();				
				this.taskGraphList.put(task.getParentTaskGraphName(), taskGraph);
			}
			else // == true
			{
				taskGraph = this.taskGraphList.get(task.getParentTaskGraphName());
			}
			
			inGraphIndex = taskGraph.getNumOfTasks();
			task.setInGraphIndex(inGraphIndex);
			
			taskGraph.putTask(task);

			loop++;
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
			if(HWCategory.PROCESSOR.equals(elementType.getCategory().toString()) == true)
			{
				ProcessorElementType elementInfo = new ProcessorElementType(elementType.getName(), elementType.getModel(), 
																			elementType.getSubcategory());
				this.elementTypeList.put(elementType.getName(), elementInfo);
			}
			else
			{
				// do nothing, ignore information
			}
		}
	}

	public void makeDeviceInformation(CICArchitectureType architecture_metadata)
	{
		int loop = 0;
		
		makeHardwareElementInformation(architecture_metadata);
		
		for(ArchitectureDeviceType device_metadata: architecture_metadata.getDevices().getDevice())
		{
			Device device = new Device(device_metadata.getName(), device_metadata.getPlatform(), 
									device_metadata.getArchitecture(),device_metadata.getRuntime());
			
			for(ArchitectureElementType element: device_metadata.getElements().getElement())
			{
				// only handles the elements which use defined types
				if(elementTypeList.containsKey(element.getType()) == true)
				{
					ProcessorElementType elementInfo = (ProcessorElementType) elementTypeList.get(element.getType());
					device.putProcessingElement(element.getName(), elementInfo.getSubcategory(), element.getPoolSize().intValue());
				}
			}
			
			for(BluetoothConnectionType connectionType: device_metadata.getConnections().getBluetoothConnection())
			{
				BluetoothConnection connection = new BluetoothConnection(connectionType.getName(), connectionType.getRole().toString(), 
												connectionType.getFriendlyName(), connectionType.getMAC());
				device.putConnection(connection);
			}
			
			for(TCPConnectionType connectionType: device_metadata.getConnections().getTCPConnection())
			{
				TCPConnection connection = new TCPConnection(connectionType.getName(), connectionType.getRole().toString(), 
													connectionType.getIp(), connectionType.getPort().intValue());
				device.putConnection(connection);
			}
			
			this.deviceInfo.add(device);
		}
	}
	
	public void makeChannelInformation(CICAlgorithmType algorithm_metadata)
	{
		
	}
	
	public void makeMappingInformation(CICMappingType mapping_metadata, CICScheduleType schedule_metadata, CICProfileType profile_metadata)
	{
		
	}
}
