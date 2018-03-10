package org.snu.cse.cap.translator.structure.channel;

import java.util.ArrayList;
import java.util.HashMap;

import org.snu.cse.cap.translator.Constants;
import org.snu.cse.cap.translator.structure.task.Task;
import org.snu.cse.cap.translator.structure.task.TaskLoopType;

enum PortSampleRateType {
	FIXED,
	VARIABLE,
	MULTIPLE,
}

enum PortType {
	QUEUE("fifo"),
	BUFFER("buffer"),
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

public class Port {
	private int taskId;
	private String taskName;
	private String portName;
	private PortSampleRateType portSampleRateType;
	private ArrayList<PortSampleRate> portSampleRateList;
	private int sampleSize;
	private PortType portType;
	private Port subgraphPort;
	private Port upperGraphPort;
	private LoopPortType loopPortType;
	private PortDirection direction;
	private String portKey;
	private int maximumChunkNum;
	
	
	public Port(int taskId, String taskName, String portName, int sampleSize, String portType, PortDirection direction) {
		this.taskId = taskId;
		this.taskName = taskName;
		this.portName = portName;
		this.sampleSize = sampleSize;
		this.portType = PortType.fromValue(portType);
		this.portSampleRateType = PortSampleRateType.VARIABLE;
		this.portSampleRateList = new ArrayList<PortSampleRate>();
		this.subgraphPort = null;
		this.upperGraphPort = null;
		this.loopPortType = null;
		this.direction = direction;
		this.portKey = taskName + Constants.NAME_SPLITER + portName + Constants.NAME_SPLITER + direction;
		this.maximumChunkNum = 1;
	}
	
	public Port getMostUpperPortInfo()
	{
		Port upperPort = this;
		
		while(upperPort.getUpperGraphPort() != null)
		{
			upperPort = upperPort.getUpperGraphPort();
		}
		
		return upperPort;
	}
	
	private int setOutputMaximumParallelNumber(HashMap<String, Task> taskMap) {
		int maxParallel = 1;
		Port port;
		Task task;
		
		port = this;
		while(port != null)
		{
			task = taskMap.get(port.getTaskName());
			if(task.getLoopStruct() != null && task.getLoopStruct().getLoopType() == TaskLoopType.DATA)
			{
				maxParallel *= task.getLoopStruct().getLoopCount();
			}
			port = port.getUpperGraphPort();
		}
		
		port = this.getSubgraphPort();
		while(port != null)
		{
			task = taskMap.get(port.getTaskName());
			if(task.getLoopStruct() != null && task.getLoopStruct().getLoopType() == TaskLoopType.DATA)
			{
				maxParallel *= task.getLoopStruct().getLoopCount();
			}
			port = port.getSubgraphPort();
		}
		
		return maxParallel;
	}
	
	private int setInputMaximumParallelNumber(HashMap<String, Task> taskMap) {
		int maxParallel = 1;
		Port port;
		Task task;
		
		port = this;
		while(port != null)
		{
			if(port.getPortType() == PortType.QUEUE && port.getLoopPortType() == LoopPortType.DISTRIBUTING)
			{
				task = taskMap.get(port.getTaskName());
				maxParallel *= task.getLoopStruct().getLoopCount();
			}
			port = port.getUpperGraphPort();
		}
		
		port = this.getSubgraphPort();
		while(port != null)
		{
			if(port.getPortType() == PortType.QUEUE && port.getLoopPortType() == LoopPortType.DISTRIBUTING)
			{
				task = taskMap.get(port.getTaskName());
				maxParallel *= task.getLoopStruct().getLoopCount();
			}
			port = port.getSubgraphPort();
		}
		
		return maxParallel;
	}
	
	public void setMaximumParallelNumber(HashMap<String, Task> taskMap) {
		int maxParallel = 1;
		
		if(this.direction == PortDirection.INPUT)
		{
			maxParallel = setInputMaximumParallelNumber(taskMap);
		}
		else
		{
			maxParallel = setOutputMaximumParallelNumber(taskMap);
		}

		
		this.maximumChunkNum = maxParallel;
	}
	
	public boolean isDistributingPort() {
		Port port;
		boolean isDistributingPort = false;
		
		port = this;
		while(port != null)
		{
			if(port.getPortType() == PortType.QUEUE && port.getLoopPortType() == LoopPortType.DISTRIBUTING)
			{
				isDistributingPort = true;
				break;
			}
			port = port.getUpperGraphPort();
		}
		
		if(isDistributingPort == false)
		{
			port = this.getSubgraphPort();
			while(port != null)
			{
				if(port.getPortType() == PortType.QUEUE && port.getLoopPortType() == LoopPortType.DISTRIBUTING)
				{
					isDistributingPort = true;
					break;
				}
				port = port.getSubgraphPort();
			}
		}
		
		return isDistributingPort;
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
	
	public int getTaskId() {
		return taskId;
	}
	
	public void setTaskId(int taskId) {
		this.taskId = taskId;
	}
	
	public String getPortName() {
		return portName;
	}
	
	public void setPortName(String portName) {
		this.portName = portName;
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
	
	public Port getSubgraphPort() {
		return subgraphPort;
	}

	public void setSubgraphPort(Port subgraphPort) {
		this.subgraphPort = subgraphPort;
	}

	public ArrayList<PortSampleRate> getPortSampleRateList() {
		return portSampleRateList;
	}

	public Port getUpperGraphPort() {
		return upperGraphPort;
	}

	public void setUpperGraphPort(Port uppergraphPort) {
		this.upperGraphPort = uppergraphPort;
	}

	public String getTaskName() {
		return taskName;
	}

	public void setTaskName(String taskName) {
		this.taskName = taskName;
	}

	public LoopPortType getLoopPortType() {
		return loopPortType;
	}

	public void setLoopPortType(LoopPortType loopPortType) {
		this.loopPortType = loopPortType;
	}

	public PortDirection getDirection() {
		return direction;
	}

	public String getPortKey() {
		return portKey;
	}

	public int getMaximumChunkNum() {
		return maximumChunkNum;
	}
}
