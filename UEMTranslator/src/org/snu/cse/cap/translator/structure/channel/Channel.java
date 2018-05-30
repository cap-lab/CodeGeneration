package org.snu.cse.cap.translator.structure.channel;

import java.util.HashMap;

import org.snu.cse.cap.translator.Constants;
import org.snu.cse.cap.translator.structure.task.Task;

public class Channel implements Cloneable {
	private int index;
	private CommunicationType communicationType;
	private ChannelArrayType channelType;
	private int size;
	private Port inputPort; // the most outer port is set here
	private Port outputPort; // the most outer port is set here
	private int initialDataLen;
	private int nextChannelIndex;
	private int channelSampleSize;
	
	public Channel(int index, int size, int initialDataLen, int sampleSize) {
		this.size = size;
		this.index = index;
		this.channelType = ChannelArrayType.GENERAL;
		this.initialDataLen = initialDataLen;
		this.nextChannelIndex = Constants.INVALID_ID_VALUE;
		this.channelSampleSize = sampleSize;
	}
	
	// Does not need to clone inputPort and outputPort  
	public Channel clone() throws CloneNotSupportedException {
		Channel channel;
		
		channel = (Channel) super.clone();
		channel.index = this.index;
		channel.communicationType = this.communicationType;
		channel.channelType = this.channelType;
		channel.size = this.size;
		// Shallow copy for these two objects
		channel.inputPort = this.inputPort;
		channel.outputPort = this.outputPort;
		channel.initialDataLen = this.initialDataLen;
		channel.nextChannelIndex = this.nextChannelIndex;
		channel.channelSampleSize = this.channelSampleSize;
		
		return channel;
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
		
		// update initial data length depending on port sample rate
		/*
		if(inputPort.getPortSampleRateType() == PortSampleRateType.FIXED)
		{
			this.initialDataLen = this.initialDataLen * inputPort.getPortSampleRateList().get(0).getSampleRate();	
		}
		else if(inputPort.getPortSampleRateType() == PortSampleRateType.MULTIPLE)
		{
			// TODO: how can I get port sample rate from MTM task?
			System.out.println("Initial data cannot be unknown at this time. Please be careful to use on MTM task");
			this.initialDataLen = this.initialDataLen * inputPort.getPortSampleRateList().get(0).getSampleRate();
		}
		else // inputPort.getPortSampleRateType() == PortSampleRateType.VARIABLE
		{
			// do nothing
		}
		*/
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

	public int getChannelSampleSize() {
		return channelSampleSize;
	}
}
