package org.snu.cse.cap.translator.structure.communication.channel;

import java.util.ArrayList;
import java.util.HashMap;

import org.snu.cse.cap.translator.structure.communication.Port;
import org.snu.cse.cap.translator.structure.communication.PortDirection;
import org.snu.cse.cap.translator.structure.mapping.MappingInfo;
import org.snu.cse.cap.translator.structure.task.Task;
import org.snu.cse.cap.translator.structure.task.TaskLoopType;


public class ChannelPort extends Port {
	private PortSampleRateType portSampleRateType;
	private ArrayList<PortSampleRate> portSampleRateList;
	private int sampleSize;
	private PortType portType;
	private ChannelPort subgraphPort;
	private ChannelPort upperGraphPort;
	private LoopPortType loopPortType;
	private int maximumChunkNum;
	private String description;
	
	public enum PortSampleRateType {
		FIXED,
		VARIABLE,
		MULTIPLE,
	}
	
	public enum PortType {
		QUEUE("fifo"),
		BUFFER("overwritable"),
		;
		
		private final String value;
		
		private PortType(final String value) {
			this.value = value;
		}
		
		public static PortType fromValue(String value) {
			 for (PortType c : PortType.values()) {
				 if (c.value.equals(value)) {
					 return c;
				 }
			 }
			 throw new IllegalArgumentException(value.toString());
		}	
	}
	
	public ChannelPort(int taskId, String taskName, String portName, int sampleSize, String portType, PortDirection direction) {
		super(taskId, taskName, portName, direction);
		this.sampleSize = sampleSize;
		this.portType = PortType.fromValue(portType);
		this.portSampleRateType = PortSampleRateType.VARIABLE;
		this.portSampleRateList = new ArrayList<PortSampleRate>();
		this.subgraphPort = null;
		this.upperGraphPort = null;
		this.loopPortType = null;
		this.maximumChunkNum = 1;
		this.description = "";
	}
	
	public ChannelPort getMostUpperPort()
	{
		ChannelPort upperPort = this;
		
		while(upperPort.getUpperGraphPort() != null)
		{
			upperPort = upperPort.getUpperGraphPort();
		}
		
		return upperPort;
	}
	
	public ChannelPort getMostInnerPort()
	{
		ChannelPort innerPort = this;
		
		while(innerPort.getSubgraphPort() != null)
		{
			innerPort = innerPort.getSubgraphPort();
		}
		
		return innerPort;
	}
	
	public void setMaximumParallelNumberInDTypeLoopTask(HashMap<String, Task> taskMap, String taskName, MappingInfo taskMappingInfo) {
		//maxParallel : tasks' coreNum.
		int maxParallel = 1;						
		Task task = null;
		Task parentTask = null;
		boolean isSubTaskofDTypeLoopTask = false;
		
		task = taskMap.get(taskName);		
		parentTask = taskMap.get(task.getParentTaskGraphName());		
		
		while(parentTask != null)
		{
			//if parentTask is DType Loop task
			if(parentTask.getLoopStruct() != null && parentTask.getLoopStruct().getLoopType() == TaskLoopType.DATA)
			{
				isSubTaskofDTypeLoopTask = true;
				break;
			}			
			parentTask = taskMap.get(parentTask.getParentTaskGraphName());
		}
			
		//if task of this port is SubTaskofDTypeLoopTask.
		if(isSubTaskofDTypeLoopTask) 
		{			
			maxParallel = taskMappingInfo.getMappedProcessorList().size();									
		}				
		this.maximumChunkNum = maxParallel;		
	}
	
	
	public void setMaximumParallelNumberInBorderLine(HashMap<String, Task> taskMap, String taskName) {		
		//maxParallel : multiple of all parent DTypeLoopTasks' nLoopCount.
		int maxParallel = 1;						
		Task task;
		Task parentTask = null;
		
		task = taskMap.get(taskName);	
		parentTask = taskMap.get(task.getParentTaskGraphName());
		
		while(parentTask != null)
		{
			if(parentTask.getLoopStruct() != null && parentTask.getLoopStruct().getLoopType() == TaskLoopType.DATA)
			{
				maxParallel *= parentTask.getLoopStruct().getLoopCount(); //multiple of parent LoopCount.
			}			
			parentTask = taskMap.get(parentTask.getParentTaskGraphName());
		}		
		this.maximumChunkNum = maxParallel;		
	}
	
	private boolean checkPort(LoopPortType checkPortType)
	{
		ChannelPort port;
		boolean isPortTypeMatch = false;
		
		port = this;
		while(port != null)
		{
			if(port.getPortType() == PortType.QUEUE && port.getLoopPortType() == checkPortType)
			{
				isPortTypeMatch = true;
				break;
			}
			port = port.getUpperGraphPort();
		}
		
		if(isPortTypeMatch == false)
		{
			port = this.getSubgraphPort();
			while(port != null)
			{
				if(port.getPortType() == PortType.QUEUE && port.getLoopPortType() == checkPortType)
				{
					isPortTypeMatch = true;
					break;
				}
				port = port.getSubgraphPort();
			}
		}
		
		return isPortTypeMatch;		
	}
	
	public boolean isDistributingPort() {
		boolean isDistributingPort = false;
		
		isDistributingPort = checkPort(LoopPortType.DISTRIBUTING);
		
		return isDistributingPort;
	}
	
	public boolean isBroadcastingPort() {
		boolean isBroadcastingPort = false;
		
		isBroadcastingPort = checkPort(LoopPortType.BROADCASTING);
		
		return isBroadcastingPort;
	}
	
	public void putSampleRate(PortSampleRate portSampleRate) {
		this.portSampleRateList.add(portSampleRate);
		if( this.portSampleRateList.size() > 1)
		{
			this.portSampleRateType = PortSampleRateType.MULTIPLE;
		}
		else // this.portSampleRateList.size() == 1
		{
			this.portSampleRateType = PortSampleRateType.FIXED;
		}
	}
	
	public PortSampleRateType getPortSampleRateType() {
		return portSampleRateType;
	}
	
	public void setPortSampleRateType(PortSampleRateType portSampleRateType) {
		this.portSampleRateType = portSampleRateType;
	}
	
	public int getSampleSize() {
		return sampleSize;
	}
	
	public void setSampleSize(int sampleSize) {
		this.sampleSize = sampleSize;
	}
	
	public PortType getPortType() {
		return portType;
	}
	
	public void setPortType(PortType portType) {
		this.portType = portType;
	}
	
	public ChannelPort getSubgraphPort() {
		return (ChannelPort) subgraphPort;
	}

	public void setSubgraphPort(ChannelPort subgraphPort) {
		this.subgraphPort = subgraphPort;
	}

	public ArrayList<PortSampleRate> getPortSampleRateList() {
		return portSampleRateList;
	}

	public ChannelPort getUpperGraphPort() {
		return (ChannelPort) upperGraphPort;
	}

	public void setUpperGraphPort(ChannelPort uppergraphPort) {
		this.upperGraphPort = uppergraphPort;
	}

	public LoopPortType getLoopPortType() {
		return loopPortType;
	}

	public void setLoopPortType(LoopPortType loopPortType) {
		this.loopPortType = loopPortType;
	}

	public int getMaximumChunkNum() {
		return maximumChunkNum;
	}
	
	public void setMaximumChunkNum(int maximumChunkNum) {
		this.maximumChunkNum = maximumChunkNum;
	}

	public String getDescription() {
		return description;
	}

	public void setDescription(String description) {
		this.description = description;
	}
}
