package org.snu.cse.cap.translator.structure.communication.multicast;

import java.util.ArrayList;
import java.util.HashSet;

import org.snu.cse.cap.translator.structure.communication.PortDirection;
import org.snu.cse.cap.translator.structure.device.DeviceCommunicationType;

public class MulticastGroup{
	private int multicastGroupId;
	private String groupName;
	private int bufferSize;
	private ArrayList<MulticastPort> inputPortList; 
	private ArrayList<MulticastPort> outputPortList;
	private HashSet<MulticastCommunicationType> inputCommunicationTypeList;
	private HashSet<MulticastCommunicationType> outputCommunicationTypeList;

	public MulticastGroup(int index, String groupName, int bufferSize) {
		this.bufferSize = bufferSize;
		this.multicastGroupId = index;
		this.groupName = groupName;
		this.inputPortList = new ArrayList<MulticastPort>();
		this.outputPortList = new ArrayList<MulticastPort>();
		this.inputCommunicationTypeList = new HashSet<MulticastCommunicationType>();
		this.outputCommunicationTypeList = new HashSet<MulticastCommunicationType>();
	}
	
	public int getMulticastGroupId() {
		return this.multicastGroupId;
	}
	
	public int getBufferSize() {
		return this.bufferSize;
	}
	
	public String getGroupName() {
		return this.groupName;
	}
	
	public ArrayList<MulticastPort> getPortList(PortDirection direction) {
		ArrayList<MulticastPort> portList = null;
		switch(direction) {
		case INPUT:
			portList = this.inputPortList;
			break;
		case OUTPUT:
			portList = this.outputPortList;
			break;
		}
		return portList;
	}
	
	public ArrayList<MulticastPort> getPortList() {
		ArrayList<MulticastPort> portList = new ArrayList<MulticastPort>();
		portList.addAll(inputPortList);
		portList.addAll(outputPortList);
		return portList;
	}

	public ArrayList<MulticastPort> getInputPortList() {
		return this.inputPortList;
	}

	public ArrayList<MulticastPort> getOutputPortList() {
		return this.outputPortList;
	}
	
	public int getInputPortNum() {
		return inputPortList.size();
	}
	
	public int getOutputPortNum() {
		return outputPortList.size();
	}
	
	public void putPort(PortDirection direction, MulticastPort port) {
		getPortList(direction).add(port);
	}

	public void putInputCommunicationType(MulticastCommunicationType communicationType) {
		inputCommunicationTypeList.add(communicationType);
	}
	
	public void putOutputCommunicationType(MulticastCommunicationType communicationType) {
		outputCommunicationTypeList.add(communicationType);
	}
	
	public HashSet<MulticastCommunicationType> getInputCommunicationType() {
		return inputCommunicationTypeList;
	}
	
	public HashSet<MulticastCommunicationType> getOutputCommunicationType() {
		return outputCommunicationTypeList;
	}
	
	public HashSet<MulticastCommunicationType> getCommunicationTypeList() {
		HashSet<MulticastCommunicationType> communicationTypeList = new HashSet<MulticastCommunicationType>();
		communicationTypeList.addAll(inputCommunicationTypeList);
		communicationTypeList.addAll(outputCommunicationTypeList);
		return communicationTypeList;
	}
}
