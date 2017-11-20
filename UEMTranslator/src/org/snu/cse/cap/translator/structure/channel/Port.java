package org.snu.cse.cap.translator.structure.channel;

import java.util.ArrayList;

enum PortSampleRateType {
	FIXED,
	VARIABLE,
	MULITPLE,
}

enum PortType {
	QUEUE,
	BUFFER,
}

public class Port {
	private int taskId;
	private String portName;
	private PortSampleRateType portSampleRateType;
	private ArrayList<PortSampleRate> portSampleRateList;
	private int nSampleSize;
	private PortType portType;
	private int subgraphPortIndex;
	
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
	
	public int getnSampleSize() {
		return nSampleSize;
	}
	
	public void setnSampleSize(int nSampleSize) {
		this.nSampleSize = nSampleSize;
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
