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
	private int channelIndex;
	private CommunicationType communicationType;
	private ChannelType channelType;
	private int channelSize;
	private Port inputPort;
	private Port outputPort;
	private int maximumChunkNum;
}
