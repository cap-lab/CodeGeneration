package org.snu.cse.cap.translator.structure.communication.multicast;

import org.snu.cse.cap.translator.structure.communication.InMemoryAccessType;
import org.snu.cse.cap.translator.structure.communication.Port;
import org.snu.cse.cap.translator.structure.communication.PortDirection;

import hopes.cic.xml.MulticastPortType;

public class MulticastPort extends Port {
	private String groupName;
	private int portId;
	private InMemoryAccessType inMemoryAccessType;
		
	public MulticastPort(int taskId, String taskName, MulticastPortType multicastPortType) {
		super(taskId, taskName, multicastPortType.getName());
		this.groupName = multicastPortType.getGroup();
		this.direction = PortDirection.fromValue(multicastPortType.getDirection().value());
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
