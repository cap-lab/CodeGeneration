package org.snu.cse.cap.translator.structure.device.connection;

public class TCPConnection extends IPConnection {
	private int channelAccessNum;
	public TCPConnection(String name, String role, String IP, int port) 
	{
		super(name, role, IP, port, ProtocolType.TCP);
		this.channelAccessNum = 0;
	}
	
	public int getChannelAccessNum() {
		return channelAccessNum;
	}
	
	public void incrementChannelAccessNum() {
		this.channelAccessNum = this.channelAccessNum + 1; 
	}
}
