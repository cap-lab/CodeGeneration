package org.snu.cse.cap.translator.structure.channel;

import java.util.ArrayList;

enum PortSampleRateType {
	FIXED,
	VARIABLE,
	MULITPLE,
}

enum PortType {
	QUEUE("fifo"),
	BUFFER("buffer"),
	;
	
	private final String value;
	
	private PortType(final String value) {
		this.value = value;
	}
	
	@Override
	public String toString() {
		return value;
	}
	
	public static PortType fromValue(String value) {
		 for (PortType c : PortType.values()) {
			 if (value.equals(value)) {
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
	
	public Port(int taskId, String taskName, String portName, int sampleSize, String portType) {
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
	
	public boolean isDistributingPort() {
		Port port;
		boolean isDistributingPort = false;
		
		port = this;
		while(port.getUpperGraphPort() != null)
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
			port = this;
			while(port.getSubgraphPort() != null)
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
			this.portSampleRateType = PortSampleRateType.MULITPLE;
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
}
