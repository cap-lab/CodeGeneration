package org.snu.cse.cap.translator.structure.communication.multicast;

import java.util.ArrayList;

public class MulticastGroup implements Cloneable {
	private int multicastGroupId;
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
		this.multicastGroupId = index;
		this.groupName = groupName;
		this.inputPortList = new ArrayList<MulticastPort>();
		this.outputPortList = new ArrayList<MulticastPort>();
		this.inputCommunicationTypeList = new ArrayList<MulticastCommunicationType> ();
		this.outputCommunicationTypeList = new ArrayList<MulticastCommunicationType> ();
		this.inputPortNum = 0;
		this.outputPortNum = 0;
	}
	
	// Does not need to clone inputPort and outputPort  
	public MulticastGroup clone() throws CloneNotSupportedException {
		MulticastGroup multicastGroup;
		
		multicastGroup = (MulticastGroup) super.clone();
		multicastGroup.multicastGroupId = this.multicastGroupId;
		multicastGroup.size = this.size;
		
		// Shallow copy for these two objects
		multicastGroup.inputPortList = this.inputPortList;
		multicastGroup.outputPortList = this.outputPortList;
		
		return multicastGroup;
	}
	
	public int getMulticastGroupId() {
		return this.multicastGroupId;
	}
	
	public int getSize() {
		return this.size;
	}
	
	public String getGroupName() {
		return this.groupName;
	}
	
	public void setSize(int size) {
		this.size = size;
	}

	public ArrayList<MulticastPort> getInputPortList() {
		return this.inputPortList;
	}

	public ArrayList<MulticastPort> getOutputPortList() {
		return this.outputPortList;
	}
	
	public int getInputPortNum() {
		return this.inputPortNum;
	}
	
	public int getOutputPortNum() {
		return this.outputPortNum;
	}

	public void setInputPortList(ArrayList<MulticastPort> inputPortList) {
		this.inputPortList = inputPortList;
		this.inputPortNum = this.inputPortList.size();
	}
	
	public void putInputPort(MulticastPort inputPort) {
		this.inputPortList.add(inputPort);
		this.inputPortNum++;
	}
	
	public void putOutputPort(MulticastPort outputPort) {
		this.outputPortList.add(outputPort);
		this.outputPortNum++;
	}

	public void setOutputPortList(ArrayList<MulticastPort> outputPortList) {
		this.outputPortList = outputPortList;
		this.outputPortNum = this.outputPortList.size();
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
	}
	
	public ArrayList<MulticastCommunicationType> getInputCommunicationTypeList() {
		return this.inputCommunicationTypeList;
	}
	
	public void putOutputCommunicationType(MulticastCommunicationType outputCommunicationType) {
		this.outputCommunicationTypeList.add(outputCommunicationType);
	}

	public void setOutputCommunicationTypeList(ArrayList<MulticastCommunicationType> outputCommunicationTypeList) {
		this.outputCommunicationTypeList = outputCommunicationTypeList;
		this.outputPortNum = this.outputCommunicationTypeList.size();
	}
	
	public ArrayList<MulticastCommunicationType> getOutputCommunicationTypeList() {
		return this.outputCommunicationTypeList;
	}
}
