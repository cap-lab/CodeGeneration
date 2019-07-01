package org.snu.cse.cap.translator.structure.communication.multicast;

import java.util.ArrayList;

public class MulticastGroup implements Cloneable {
	private int muticastGroupId;
	private String groupName;
	private int size;
	private ArrayList<MulticastPort> inputPortList; 
	private int inputPortNum;
	private ArrayList<MulticastCommunicationType> inputCommunicationTypeList;
	private ArrayList<MulticastPort> outputPortList; 
	private int outputPortNum;
	private ArrayList<MulticastCommunicationType> outputCommunicationTypeList;

	public MulticastGroup(int index, String groupName, int size) {
		this.size = size;
		this.muticastGroupId = index;
		this.groupName = groupName;
		this.inputPortNum = 0;
		this.outputPortNum = 0;
	}
	
	// Does not need to clone inputPort and outputPort  
	public MulticastGroup clone() throws CloneNotSupportedException {
		MulticastGroup multicastGroup;
		
		multicastGroup = (MulticastGroup) super.clone();
		multicastGroup.muticastGroupId = this.muticastGroupId;
		multicastGroup.size = this.size;
		
		// Shallow copy for these two objects
		multicastGroup.inputPortList = this.inputPortList;
		multicastGroup.outputPortList = this.outputPortList;
		
		return multicastGroup;
	}
	
	public int getMulticastGroupId() {
		return muticastGroupId;
	}
	
	public int getSize() {
		return size;
	}
	
	public String getGroupName() {
		return groupName;
	}
	
	public void setSize(int size) {
		this.size = size;
	}

	public ArrayList<MulticastPort> getInputPortList() {
		return inputPortList;
	}

	public ArrayList<MulticastPort> getOutputPortList() {
		return outputPortList;
	}

	public void setInputPortList(ArrayList<MulticastPort> inputPortList) {
		this.inputPortList = inputPortList;
	}
	
	public void putInputPort(MulticastPort inputPort) {
		this.inputPortList.add(inputPort);
	}
	
	public void putOutputPort(MulticastPort outputPort) {
		this.outputPortList.add(outputPort);
	}

	public void setOutputPortList(ArrayList<MulticastPort> outputPortList) {
		this.outputPortList = outputPortList;
	}

	public void clearInputPort() {
		this.inputPortList.clear();
	}
	
	public void clearOutputPort() {
		this.outputPortList.clear();
	}
	
	public void setInputCommunicationTypeList(ArrayList<MulticastCommunicationType> inputCommunicationTypeList) {
		this.inputCommunicationTypeList = inputCommunicationTypeList;
		this.inputPortNum = this.inputCommunicationTypeList.size();
	}
	
	public void putInputCommunicationType(MulticastCommunicationType inputCommunicationType) {
		this.inputCommunicationTypeList.add(inputCommunicationType);
		this.inputPortNum += 1;
	}
	
	public ArrayList<MulticastCommunicationType> getInputCommunicationTypeList() {
		return this.inputCommunicationTypeList;
	}
	
	public void putOutputCommunicationType(MulticastCommunicationType outputCommunicationType) {
		this.outputCommunicationTypeList.add(outputCommunicationType);
		this.outputPortNum += 1;
	}

	public void setOutputCommunicationTypeList(ArrayList<MulticastCommunicationType> outputCommunicationTypeList) {
		this.outputCommunicationTypeList = outputCommunicationTypeList;
		this.outputPortNum = this.outputCommunicationTypeList.size();
	}
	
	public ArrayList<MulticastCommunicationType> getOutputCommunicationTypeList() {
		return this.outputCommunicationTypeList;
	}
}
