package org.snu.cse.cap.translator.structure.communication;

import org.snu.cse.cap.translator.Constants;

public class Port {
	protected int taskId;
	protected String taskName;
	protected String portName;
	protected String portKey;
	protected PortDirection direction;
	
	public Port(int taskId, String taskName, String portName,PortDirection direction) {
		this.taskId = taskId;
		this.taskName = taskName;
		this.portName = portName;
		this.portKey = taskName + Constants.NAME_SPLITER + portName + Constants.NAME_SPLITER + direction;
		this.direction = direction;
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

	public String getPortKey() {
		return portKey;
	}
	
	public String getTaskName() {
		return taskName;
	}

	public void setTaskName(String taskName) {
		this.taskName = taskName;
	}

	public PortDirection getDirection() {
		return direction;
	}
}
