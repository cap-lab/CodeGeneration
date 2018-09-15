package org.snu.cse.cap.translator.structure.channel;

import java.util.ArrayList;
import java.util.HashMap;

import org.snu.cse.cap.translator.Constants;
import org.snu.cse.cap.translator.structure.task.Task;

public class Channel implements Cloneable {
	private int index;
	private CommunicationType communicationType;
	private InMemoryAccessType accessType;
	private ChannelArrayType channelType;
	private int size;
	private Port inputPort; // the most outer port is set here
	private int  inputPortIndex; // port index used in channel data generation
	private Port outputPort; // the most outer port is set here
	private int  outputPortIndex; // port index used in channel data generation
	private int initialDataLen;
	private int nextChannelIndex;
	private int channelSampleSize;
	private int tcpClientIndex;
	
	public Channel(int index, int size, int initialDataLen, int sampleSize) {
		this.size = size;
		this.index = index;
		this.channelType = ChannelArrayType.GENERAL;
		this.initialDataLen = initialDataLen;
		this.nextChannelIndex = Constants.INVALID_ID_VALUE;
		this.channelSampleSize = sampleSize;
		this.tcpClientIndex = Constants.INVALID_ID_VALUE;
		this.inputPortIndex = Constants.INVALID_VALUE;
		this.outputPortIndex = Constants.INVALID_VALUE;
	}
	
	// Does not need to clone inputPort and outputPort  
	public Channel clone() throws CloneNotSupportedException {
		Channel channel;
		
		channel = (Channel) super.clone();
		channel.index = this.index;
		channel.communicationType = this.communicationType;
		channel.accessType = this.accessType;
		channel.channelType = this.channelType;
		channel.size = this.size;
		
		channel.initialDataLen = this.initialDataLen;
		channel.nextChannelIndex = this.nextChannelIndex;
		channel.channelSampleSize = this.channelSampleSize;
		channel.tcpClientIndex = this.tcpClientIndex;
		
		// Shallow copy for these two objects
		channel.inputPort = this.inputPort;
		channel.outputPort = this.outputPort;
		
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
	
	public void setPortIndexByPortList(ArrayList<Port> portList)
	{
		int index = 0;
		int listSize = portList.size();
		Port port;
		for(index = 0; index < listSize ; index++)
		{
			if(this.inputPortIndex != Constants.INVALID_VALUE && this.outputPortIndex != Constants.INVALID_VALUE)
			{
				break;
			}
			
			port = portList.get(index);
			if(port.getTaskId() == this.inputPort.getTaskId() && 
					port.getPortName().equals(this.inputPort.getPortName()) == true)
			{
				this.inputPortIndex = index;
			}				
			else if(port.getTaskId() == this.outputPort.getTaskId() && 
					port.getPortName().equals(this.outputPort.getPortName()) == true)
			{
				this.outputPortIndex = index;
			}
		}
	}
	
	public int getInputPortIndex() {
		return inputPortIndex;
	}

	public int getOutputPortIndex() {
		return outputPortIndex;
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

	public int getTcpClientIndex() {
		return tcpClientIndex;
	}

	public void setTcpClientIndex(int tcpClientIndex) {
		this.tcpClientIndex = tcpClientIndex;
	}

	public InMemoryAccessType getAccessType() {
		return accessType;
	}

	public void setAccessType(InMemoryAccessType accessType) {
		this.accessType = accessType;
	}
}
