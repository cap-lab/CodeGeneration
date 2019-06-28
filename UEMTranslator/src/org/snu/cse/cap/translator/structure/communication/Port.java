package org.snu.cse.cap.translator.structure.communication;

public class Port {
	protected int taskId;
	protected String taskName;
	protected String portName;
	protected Port subgraphPort;
	protected Port upperGraphPort;
	protected PortDirection direction;
	
	public Port(int taskId, String taskName, String portName,PortDirection direction) {
		this.taskId = taskId;
		this.taskName = taskName;
		this.portName = portName;
		this.subgraphPort = null;
		this.upperGraphPort = null;
		this.direction = direction;
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
	
	public Port getSubgraphPort() {
		return subgraphPort;
	}

	public void setSubgraphPort(Port subgraphPort) {
		this.subgraphPort = subgraphPort;
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

	public PortDirection getDirection() {
		return direction;
	}
}
