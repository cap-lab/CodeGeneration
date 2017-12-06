package org.snu.cse.cap.translator.structure.channel;

public class Channel {
	private int index;
	private CommunicationType communicationType;
	private ChannelArrayType channelType;
	private int size;
	private Port inputPort;
	private Port outputPort;
	private int maximumChunkNum;
	
	public Channel(int index, int size) {
		this.size = size;
		this.index = index;
		this.channelType = ChannelArrayType.GENERAL;
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
	
	public void setMaximumChunkNum(int maximumChunkNum) {
		this.maximumChunkNum = maximumChunkNum;
	}

	public Port getInputPort() {
		return inputPort;
	}

	public Port getOutputPort() {
		return outputPort;
	}

	public void setInputPort(Port inputPort) {
		this.inputPort = inputPort;
	}

	public void setOutputPort(Port outputPort) {
		this.outputPort = outputPort;
	}
}
