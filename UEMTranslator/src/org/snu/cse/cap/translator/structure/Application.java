package org.snu.cse.cap.translator.structure;

import java.io.File;
import java.io.FileFilter;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.zip.DataFormatException;

import org.snu.cse.cap.translator.structure.channel.Channel;
import org.snu.cse.cap.translator.structure.device.BluetoothConnection;
import org.snu.cse.cap.translator.structure.device.Device;
import org.snu.cse.cap.translator.structure.device.HWCategory;
import org.snu.cse.cap.translator.structure.device.HWElementType;
import org.snu.cse.cap.translator.structure.device.ProcessorElementType;
import org.snu.cse.cap.translator.structure.device.TCPConnection;
import org.snu.cse.cap.translator.structure.mapping.CompositeTaskMappingInfo;
import org.snu.cse.cap.translator.structure.mapping.CompositeTaskSchedule;
import org.snu.cse.cap.translator.structure.mapping.InvalidScheduleFileNameException;
import org.snu.cse.cap.translator.structure.mapping.MappingInfo;
import org.snu.cse.cap.translator.structure.mapping.ScheduleFileFilter;
import org.snu.cse.cap.translator.structure.task.Task;

import Translators.Constants;
import hopes.cic.exception.CICXMLException;
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
import hopes.cic.xml.MappingDeviceType;
import hopes.cic.xml.MappingProcessorIdType;
import hopes.cic.xml.MappingTaskType;
import hopes.cic.xml.ModeTaskType;
import hopes.cic.xml.ModeType;
import hopes.cic.xml.TCPConnectionType;
import hopes.cic.xml.TaskType;

enum ScheduleFileNameOffset {
	TASK_NAME(0),
	MODE_NAME(1),
	SCHEDULE_ID(2),
	THROUGHPUT_CONSTRAINT(3),
	SCHEUDLE_XML(4),
	;
	
	private final int value;
	
    private ScheduleFileNameOffset(int value) {
        this.value = value;
    }
    
    public int getValue() {
    	return this.value;
    }
}


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
			 if (value.equals(value)) {
				 return c;
			 }
		 }
		 throw new IllegalArgumentException(value.toString());
	}
}

public class Application {
	private ArrayList<Channel> channel;
	private HashMap<String, Task> taskMap; // Task name : Task class
	private HashMap<String, TaskGraph> taskGraphList; // Task graph name : TaskGraph class
	private HashMap<String, MappingInfo> mappingInfo; // Task name : MappingInfo class
	private HashMap<String, Device> deviceInfo; // device name: Device class
	private HashMap<String, HWElementType> elementTypeList; // element type name : HWElementType class
	private TaskGraphType applicationGraphProperty;
	
	public Application()
	{
		this.channel = new ArrayList<Channel>();	
		this.taskMap = new HashMap<String, Task>();
		this.taskGraphList = new HashMap<String, TaskGraph>();
		this.mappingInfo = new HashMap<String, MappingInfo>();
		this.deviceInfo = new HashMap<String, Device>();
		this.elementTypeList = new HashMap<String, HWElementType>();
		this.applicationGraphProperty = null;
	}
	
	// taskMap, taskGraphList
	public void makeTaskInformation(CICAlgorithmType algorithm_metadata)
	{
		int loop = 0;
		Task task;
		int inGraphIndex = 0;
		
		this.applicationGraphProperty = TaskGraphType.fromValue(algorithm_metadata.getProperty());

		
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
		makeHardwareElementInformation(architecture_metadata);
		
		if(architecture_metadata.getDevices() != null)
		{
			for(ArchitectureDeviceType device_metadata: architecture_metadata.getDevices().getDevice())
			{
				Device device = new Device(device_metadata.getName(), device_metadata.getPlatform(), 
						device_metadata.getArchitecture(),device_metadata.getRuntime());

				for(ArchitectureElementType elementType: device_metadata.getElements().getElement())
				{
					// only handles the elements which use defined types
					if(this.elementTypeList.containsKey(elementType.getType()) == true)
					{
						ProcessorElementType elementInfo = (ProcessorElementType) this.elementTypeList.get(elementType.getType());
						device.putProcessingElement(elementType.getName(), elementInfo.getSubcategory(), elementType.getPoolSize().intValue());
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

				this.deviceInfo.put(device_metadata.getName(), device);
			}
		}
	}
	
	public void makeChannelInformation(CICAlgorithmType algorithm_metadata)
	{
		
	}
	
	private CompositeTaskSchedule makeCompositeTaskSchedule(String[] splitedFileName, File scheduleFile) 
	{ 
		CompositeTaskSchedule taskSchedule;
		int scheduleId;
		CICScheduleTypeLoader scheduleLoader = new CICScheduleTypeLoader();
		CICScheduleType scheduleDOM;
		
		scheduleId = Integer.parseInt(splitedFileName[ScheduleFileNameOffset.SCHEDULE_ID.getValue()]);
		
		if(splitedFileName.length == ScheduleFileNameOffset.values().length)
		{
			int throughputConstraint = Integer.parseInt(splitedFileName[ScheduleFileNameOffset.THROUGHPUT_CONSTRAINT.getValue()]);
			taskSchedule = new CompositeTaskSchedule(scheduleId, throughputConstraint);
		}
		else // throughput constraint is missing (splitedFileName.length == ScheduleFileNameOffset.values().length - 1)
		{
			taskSchedule = new CompositeTaskSchedule(scheduleId);
		}
		
		try {
			scheduleDOM = scheduleLoader.loadResource(scheduleFile.getAbsolutePath());
		} catch (CICXMLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		//scheduleDOM.getTaskGroups().getTaskGroup().get
		
		return taskSchedule;
	}
	
	private void makeCompositeTaskMappingInfo(String scheduleFolderPath) throws FileNotFoundException, InvalidScheduleFileNameException {
		ScheduleFileFilter scheduleXMLFilefilter = new ScheduleFileFilter(); 
		String[] splitedFileName = null;
		File scheduleFolder = new File(scheduleFolderPath);
		String taskName;
		String modeName;
		int modeId;
		CompositeTaskMappingInfo compositeMappingInfo;		
		
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
			
			CompositeTaskSchedule taskSchedule = makeCompositeTaskSchedule(splitedFileName, file);
			
			taskName = splitedFileName[ScheduleFileNameOffset.TASK_NAME.getValue()];
			modeName = splitedFileName[ScheduleFileNameOffset.MODE_NAME.getValue()];
			Task task = this.taskMap.get(taskName);
			if(task.getModeTransition() == null)
			{
				modeId = 0;
			}
			else
			{
				modeId = task.getModeTransition().getModeIdFromName(modeName);
			}
			
			if(this.mappingInfo.containsKey(taskName) == false)
			{
				compositeMappingInfo = new CompositeTaskMappingInfo(taskName, modeId);
				this.mappingInfo.put(taskName, compositeMappingInfo);							
			}
			else
			{
				compositeMappingInfo = (CompositeTaskMappingInfo) this.mappingInfo.get(taskName);
			}
			
		}
		
	}
	
	// scheduleFolderPath : output + /convertedSDF3xml/
	public void makeMappingInformation(CICMappingType mapping_metadata, CICProfileType profile_metadata, CICConfigurationType config_metadata, String scheduleFolderPath)
	{
		//config_metadata.getCodeGeneration().getRuntimeExecutionPolicy().equals(anObject)
		ExecutionPolicy executionPolicy = ExecutionPolicy.fromValue(config_metadata.getCodeGeneration().getRuntimeExecutionPolicy());
		ArrayList<File> fileArrayList = new ArrayList<File>();
		
		ScheduleFileFilter scheduleXMLFilefilter = new ScheduleFileFilter(); 
		
		try {
			switch(executionPolicy)
			{
			case FULLY_STATIC: // Need schedule with time information (needed file: mapping, profile, schedule)
				File scheduleFolder = new File(scheduleFolderPath);
				if(scheduleFolder.exists() == false || scheduleFolder.isDirectory() == false)
				{
					throw new FileNotFoundException();
				}
				File[] fileList = scheduleFolder.listFiles(scheduleXMLFilefilter);
				for(File file : fileList) 
				{
					fileArrayList.add(file);
					String[] fileNameSplit = file.getName().split(Constants.SCHEDULE_FILE_SPLITER);
					
					if(fileNameSplit.length == ScheduleFileNameOffset.values().length || fileNameSplit.length == ScheduleFileNameOffset.values().length - 1)
					{
						int scheduleId = Integer.parseInt(fileNameSplit[ScheduleFileNameOffset.SCHEDULE_ID.getValue()]);
						String taskName = fileNameSplit[ScheduleFileNameOffset.TASK_NAME.getValue()];
						String modeName = fileNameSplit[ScheduleFileNameOffset.MODE_NAME.getValue()];
						int modeId;
						
						Task task = this.taskMap.get(taskName);
						if(task.getModeTransition() == null)
						{
							modeId = 0;
						}
						else
						{
							modeId = task.getModeTransition().getModeIdFromName(modeName);
						}
						
						if(this.mappingInfo.containsKey(taskName) == false)
						{
							CompositeTaskMappingInfo compositeMappingInfo = new CompositeTaskMappingInfo(taskName, modeId);
							
							this.mappingInfo.put(taskName, compositeMappingInfo);							
						}
						

						
						// all offset values are contained in the file name 
						if(fileNameSplit.length == ScheduleFileNameOffset.values().length)
						{
							int throughputConstraint = Integer.parseInt(fileNameSplit[ScheduleFileNameOffset.THROUGHPUT_CONSTRAINT.getValue()]);
							CompositeTaskSchedule taskSchedule = new CompositeTaskSchedule(scheduleId, throughputConstraint);
						}
						else if(fileNameSplit.length == ScheduleFileNameOffset.values().length - 1) // throughput constraint is missing
						{
							CompositeTaskSchedule taskSchedule = new CompositeTaskSchedule(scheduleId);
						}					
					}
					else
					{
						// error
					}
					
				}
				
				
				break;
			case SELF_TIMED: // Need schedule (needed file: mapping, schedule)
				break;
			case STATIC_ASSIGNMENT: // Need mapping only (needed file: mapping)
				
				break;
			case FULLY_DYNAMIC: // Need mapped device information (needed file: mapping)
				
				break;
			}
		}
		catch(FileNotFoundException e) {
			e.printStackTrace();
		}
		
		
		for(MappingTaskType taskType: mapping_metadata.getTask())
		{
			taskType.getName();
			for(MappingDeviceType deviceType: taskType.getDevice())
			{
				deviceType.getName();
				for(MappingProcessorIdType mappedProcessor: deviceType.getProcessor())
				{
					mappedProcessor.getPool();
					mappedProcessor.getLocalId();
				}
			}
		}
		

		
		//config_metadata.getCodeGeneration().getThreadOrFunctioncall();
		//config_metadata.getCodeGeneration().getRuntimeExecutionPolicy();
	}
}
