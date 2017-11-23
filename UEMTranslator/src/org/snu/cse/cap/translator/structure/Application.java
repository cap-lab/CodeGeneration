package org.snu.cse.cap.translator.structure;

import java.util.ArrayList;
import java.util.HashMap;

import org.snu.cse.cap.translator.structure.channel.Channel;
import org.snu.cse.cap.translator.structure.device.Device;
import org.snu.cse.cap.translator.structure.mapping.MappingInfo;
import org.snu.cse.cap.translator.structure.task.Task;

import Translators.Constants;
import hopes.cic.xml.CICAlgorithmType;
import hopes.cic.xml.CICArchitectureType;
import hopes.cic.xml.CICMappingType;
import hopes.cic.xml.CICProfileType;
import hopes.cic.xml.CICScheduleType;
import hopes.cic.xml.ModeTaskType;
import hopes.cic.xml.ModeType;
import hopes.cic.xml.TaskType;

public class Application {
	private ArrayList<Channel> channel;
	private HashMap<String, Task> taskMap;
	private HashMap<String, TaskGraph> taskGraphList;
	private ArrayList<MappingInfo> mappingInfo;
	private ArrayList<Device> deviceInfo;
	
	public Application()
	{
		channel = new ArrayList<Channel>();	
		taskMap = new HashMap<String, Task>();
		taskGraphList = new HashMap<String, TaskGraph>();
		mappingInfo = new ArrayList<MappingInfo>();
		deviceInfo = new ArrayList<Device>();
	}
	
	// taskMap, taskGraphList
	public void makeTaskInformation(CICAlgorithmType algorithm_metadata)
	{
		int loop = 0;
		Task task;
		int inGraphIndex = 0;
		
		for(TaskType task_metadata: algorithm_metadata.getTasks().getTask())
		{
			task = new Task(loop, task_metadata);
			TaskGraph taskGraph;
						
			taskMap.put(task.getName(), task);
			
			if(taskGraphList.containsKey(task.getParentTaskGraphName()) == false)
			{
				taskGraph = new TaskGraph();				
				taskGraphList.put(task.getParentTaskGraphName(), taskGraph);
			}
			else // == true
			{
				taskGraph = taskGraphList.get(task.getParentTaskGraphName());
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
			task = taskMap.get(modeTask.getName());
			
			task.setExtraInformationFromModeInfo(modeTask);
		}
	}

	public void makeDeviceInformation(CICArchitectureType architecture_metadata)
	{
		
	}
	
	public void makeChannelInformation(CICAlgorithmType algorithm_metadata)
	{
		
	}
	
	public void makeMappingInformation(CICMappingType mapping_metadata, CICScheduleType schedule_metadata, CICProfileType profile_metadata)
	{
		
	}
}
