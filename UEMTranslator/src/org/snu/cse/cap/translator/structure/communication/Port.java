package org.snu.cse.cap.translator.structure.communication;

import org.snu.cse.cap.translator.Constants;

import hopes.cic.xml.TaskPortType;

public class Port {
	protected int taskId;
	protected String taskName;
	protected String portName;
	protected PortDirection direction;
	
	public Port(int taskId, String taskName, TaskPortType portType) {
		this(taskId, taskName, portType.getName());
		this.direction = PortDirection.fromValue(portType.getDirection().value());
	}

	protected Port(int taskId, String taskName, String portName) {
		this.taskId = taskId;
		this.taskName = taskName;
		this.portName = portName;
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
		return taskName + Constants.NAME_SPLITER + portName + Constants.NAME_SPLITER + direction;
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
