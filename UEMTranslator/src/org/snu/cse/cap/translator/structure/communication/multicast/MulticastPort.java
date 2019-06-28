package org.snu.cse.cap.translator.structure.communication.multicast;

import org.snu.cse.cap.translator.Constants;
import org.snu.cse.cap.translator.structure.communication.Port;
import org.snu.cse.cap.translator.structure.communication.PortDirection;
import org.snu.cse.cap.translator.structure.communication.InMemoryAccessType;
import org.snu.cse.cap.translator.structure.communication.multicast.MulticastCommunicationType;

public class MulticastPort extends Port {
	private String groupName;
	private String portKey;
	private MulticastCommunicationType communicationType;
	private InMemoryAccessType accessType;
		
	public MulticastPort(int taskId, String taskName, String portName, String groupName, PortDirection direction) {
		super(taskId, taskName, portName, direction);
		this.portKey = taskName + Constants.NAME_SPLITER + portName + Constants.NAME_SPLITER + direction;
		this.groupName = groupName;
	}
	
	public String getGroupName() {
		return groupName;
	}
	
	public void setCommunicationType(MulticastCommunicationType communicationType) {
		this.communicationType = communicationType;
	}
	
	public MulticastCommunicationType getCommunicationType() {
		return this.communicationType;
	}
	public String getPortKey() {
		return portKey;
	}
}
