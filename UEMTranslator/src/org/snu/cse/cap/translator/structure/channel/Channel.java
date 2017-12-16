package org.snu.cse.cap.translator.structure.channel;

import java.util.HashMap;

import org.snu.cse.cap.translator.structure.task.Task;

public class Channel {
	private int index;
	private CommunicationType communicationType;
	private ChannelArrayType channelType;
	private int size;
	private Port inputPort; // the most outer port is set here
	private Port outputPort; // the most outer port is set here
	private int maximumChunkNum;
	
	public Channel(int index, int size) {
		this.size = size;
		this.index = index;
		this.channelType = ChannelArrayType.GENERAL;
		this.maximumChunkNum = 1;
	}
	
	public int getIndex() {
		return index;
	}
	
	public CommunicationType getCommunicationType() {
		return communicationType;
	}
	
	public ChannelArrayType getChannelType() {
		return channelType;
	}
	
	public int getSize() {
		return size;
	}
	
	public int getMaximumChunkNum() {
		return maximumChunkNum;
	}
	
	public void setIndex(int channelIndex) {
		this.index = channelIndex;
	}
	
	public void setCommunicationType(CommunicationType communicationType) {
		this.communicationType = communicationType;
	}
	
	public void setChannelType(ChannelArrayType channelType) {
		this.channelType = channelType;
	}
	
	public void setSize(int channelSize) {
		this.size = channelSize;
	}
	
	public void setMaximumChunkNum(HashMap<String, Task> taskMap) {		
		this.maximumChunkNum = this.inputPort.getMaximumParallelNumber(taskMap);
	}

	public Port getInputPort() {
		return inputPort;
	}

	public Port getOutputPort() {
		return outputPort;
	} 

	public void setInputPort(Port inputPort) {
		this.inputPort = inputPort;
		
		// also set maximum chunk number here
	}

	public void setOutputPort(Port outputPort) {
		this.outputPort = outputPort;
	}
}
