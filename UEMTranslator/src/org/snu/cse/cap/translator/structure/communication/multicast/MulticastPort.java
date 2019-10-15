package org.snu.cse.cap.translator.structure.communication.multicast;

import org.snu.cse.cap.translator.structure.communication.InMemoryAccessType;
import org.snu.cse.cap.translator.structure.communication.Port;
import org.snu.cse.cap.translator.structure.communication.PortDirection;

public class MulticastPort extends Port {
	private String groupName;
	private int portId;
	private InMemoryAccessType inMemoryAccessType;
		
	public MulticastPort(int taskId, String taskName, String portName, String groupName, PortDirection direction) {
		super(taskId, taskName, portName, direction);
		this.groupName = groupName;
	}
	
	public void setPortId(int id) {
		this.portId = id;
	}
	
	public int getPortId() {
		return this.portId;
	}
	
	public String getGroupName() {
		return groupName;
	}
	
	public void setMemoryAccessType(InMemoryAccessType accessType) {
		this.inMemoryAccessType = accessType;
	}
	
	public InMemoryAccessType getInMemoryAccessType() {
		return this.inMemoryAccessType;
	}
}
