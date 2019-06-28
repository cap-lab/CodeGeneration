package org.snu.cse.cap.translator.structure.communication.multicast;

import java.util.ArrayList;
import org.snu.cse.cap.translator.Constants;
import org.snu.cse.cap.translator.structure.communication.InMemoryAccessType;

public class MulticastGroup implements Cloneable {
	private int index;
	private String groupName;
	private int size;
	private ArrayList<MulticastPort> inputPortList; // the most input port is set here
	private ArrayList<MulticastPort> outputPortList; // the most outer port is set here
	private int nextMulticastGroupIndex;

	public MulticastGroup(int index, String groupName, int size) {
		this.size = size;
		this.index = index;
		this.groupName = groupName;
		this.nextMulticastGroupIndex = Constants.INVALID_ID_VALUE;
	}
	
	// Does not need to clone inputPort and outputPort  
	public MulticastGroup clone() throws CloneNotSupportedException {
		MulticastGroup multicastGroup;
		
		multicastGroup = (MulticastGroup) super.clone();
		multicastGroup.index = this.index;
		multicastGroup.size = this.size;
		
		multicastGroup.nextMulticastGroupIndex = this.nextMulticastGroupIndex;
		
		// Shallow copy for these two objects
		multicastGroup.inputPortList = this.inputPortList;
		multicastGroup.outputPortList = this.outputPortList;
		
		return multicastGroup;
	}
	
	public int getIndex() {
		return index;
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

	public int getNextMulticastGroupIndex() {
		return nextMulticastGroupIndex;
	}

	public void setNextMulticastGroupIndex(int nextMulticastGroupIndex) {
		this.nextMulticastGroupIndex = nextMulticastGroupIndex;
	}

	public void clearInputPort() {
		this.inputPortList.clear();
	}
	
	public void clearOutputPort() {
		this.outputPortList.clear();
	}
}
