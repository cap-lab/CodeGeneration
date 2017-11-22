package org.snu.cse.cap.translator.structure;

import java.util.ArrayList;
import java.util.HashMap;

import org.snu.cse.cap.translator.structure.channel.Channel;
import org.snu.cse.cap.translator.structure.device.Device;
import org.snu.cse.cap.translator.structure.mapping.MappingInfo;
import org.snu.cse.cap.translator.structure.task.Task;

import hopes.cic.xml.CICAlgorithmType;
import hopes.cic.xml.CICArchitectureType;
import hopes.cic.xml.CICMappingType;
import hopes.cic.xml.CICProfileType;
import hopes.cic.xml.CICScheduleType;
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
		
		for(hopes.cic.xml.TaskType task_metadata: algorithm_metadata.getTasks().getTask())
		{
			Task task = new Task(loop, task_metadata); 
			

						
			loop++;
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
