package org.snu.cse.cap.translator.structure.channel;

enum CommunicationType {
	SHARED_MEMORY,
	TCP_CLIENT,
	TCP_SERVER,
}

enum ChannelType {
	GENERAL,
	INPUT_ARRAY,
	OUTPUT_ARRAY,
	FULL_ARRAY,
}

public class Channel {
	private int index;
	private CommunicationType communicationType;
	private ChannelType channelType;
	private int size;
	private Port inputPort;
	private Port outputPort;
	private int maximumChunkNum;
	
	public Channel(int index, int size) {
		this.size = size;
		this.index = index;
	}
	
	public int getIndex() {
		return index;
	}
	
	public CommunicationType getCommunicationType() {
		return communicationType;
	}
	
	public ChannelType getChannelType() {
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
	
	public void setChannelType(ChannelType channelType) {
		this.channelType = channelType;
	}
	
	public void setSize(int channelSize) {
		this.size = channelSize;
	}
	
	public void setMaximumChunkNum(int maximumChunkNum) {
		this.maximumChunkNum = maximumChunkNum;
	}
}
