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
	private String portName;
	private PortSampleRateType portSampleRateType;
	private ArrayList<PortSampleRate> portSampleRateList;
	private int sampleSize;
	private PortType portType;
	private int subgraphPortIndex;
	
	public Port(int taskId, String portName, int sampleSize, String portType) {
		this.taskId = taskId;
		this.portName = portName;
		this.sampleSize = sampleSize;
		this.portType = PortType.fromValue(portType);
		this.portSampleRateType = PortSampleRateType.VARIABLE;
		this.portSampleRateList = new ArrayList<PortSampleRate>();
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
	
	public int getSubgraphPortIndex() {
		return subgraphPortIndex;
	}
	
	public void setSubgraphPortIndex(int subgraphPortIndex) {
		this.subgraphPortIndex = subgraphPortIndex;
	}
}
