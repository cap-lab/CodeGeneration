package org.snu.cse.cap.translator.structure.channel;

import java.util.HashMap;

import org.snu.cse.cap.translator.Constants;
import org.snu.cse.cap.translator.structure.task.Task;

public class Channel {
	private int index;
	private CommunicationType communicationType;
	private ChannelArrayType channelType;
	private int size;
	private Port inputPort; // the most outer port is set here
	private Port outputPort; // the most outer port is set here
	private int initialDataLen;
	private int nextChannelIndex;
	
	public Channel(int index, int size, int initialDataLen) {
		this.size = size;
		this.index = index;
		this.channelType = ChannelArrayType.GENERAL;
		this.initialDataLen = initialDataLen;
		this.nextChannelIndex = Constants.INVALID_ID_VALUE;
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
	
	public void setMaximumChunkNum(HashMap<String, Task> taskMap)
	{
		this.inputPort.setMaximumParallelNumber(taskMap);
		this.outputPort.setMaximumParallelNumber(taskMap);
	}

	public void setOutputPort(Port outputPort) {
		this.outputPort = outputPort;
	}

	public int getInitialDataLen() {
		return initialDataLen;
	}

	public void setInitialDataLen(int initialDataLen) {
		this.initialDataLen = initialDataLen;
	}

	public int getNextChannelIndex() {
		return nextChannelIndex;
	}

	public void setNextChannelIndex(int nextChannelIndex) {
		this.nextChannelIndex = nextChannelIndex;
	}
}
