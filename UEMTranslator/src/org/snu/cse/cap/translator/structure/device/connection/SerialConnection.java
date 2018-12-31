package org.snu.cse.cap.translator.structure.device.connection;

import hopes.cic.xml.NetworkType;

public abstract class SerialConnection extends Connection {
	protected int channelAccessNum;

	public static final String ROLE_MASTER= "master";
	public static final String ROLE_SLAVE = "slave";

	public SerialConnection(String name, String role, NetworkType network) {
		super(name, role, network, ProtocolType.SERIAL);
		this.channelAccessNum = 0;
	}

	public int getChannelAccessNum() {
		return channelAccessNum;
	}
	
	public void incrementChannelAccessNum() {
		this.channelAccessNum = this.channelAccessNum + 1; 
	}
}
