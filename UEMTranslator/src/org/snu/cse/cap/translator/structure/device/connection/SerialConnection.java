package org.snu.cse.cap.translator.structure.device.connection;

import hopes.cic.xml.NetworkType;

public abstract class SerialConnection extends Connection {
	protected int channelAccessNum;
	protected SerialConnectionType connectionType;

	public static final String ROLE_MASTER= "master";
	public static final String ROLE_SLAVE = "slave";

	public SerialConnection(String name, String role, NetworkType network) {
		super(name, role, network, ProtocolType.SERIAL);
		this.channelAccessNum = 0;
		switch (network) {
		case BLUETOOTH:
			this.connectionType = SerialConnectionType.BLUETOOTH;
			break;
		case WIRE:
			this.connectionType = SerialConnectionType.WIRE;
			break;
		case USB:
			this.connectionType = SerialConnectionType.USB;
			break;
		default:
			throw new IllegalArgumentException(network.toString() + " is not supported for serial connection");
		}
	}

	public int getChannelAccessNum() {
		return channelAccessNum;
	}
	
	public void incrementChannelAccessNum() {
		this.channelAccessNum = this.channelAccessNum + 1; 
	}

	public SerialConnectionType getConnectionType() {
		return connectionType;
	}
}
