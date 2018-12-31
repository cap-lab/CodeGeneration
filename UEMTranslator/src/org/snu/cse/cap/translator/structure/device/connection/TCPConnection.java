package org.snu.cse.cap.translator.structure.device.connection;

import hopes.cic.xml.NetworkType;

public class TCPConnection extends Connection {
	private String IP;
	private int port;
	
	public static final String ROLE_SERVER = "server";
	public static final String ROLE_CLIENT = "client";
	
	public TCPConnection(String name, String role, String IP, int port) 
	{
		super(name, role, NetworkType.ETHERNET_WI_FI, ProtocolType.TCP);
		this.IP = IP;
		this.port = port;
	}
	
	public String getIP() {
		return IP;
	}
	
	public void setIP(String iP) {
		IP = iP;
	}
	
	public int getPort() {
		return port;
	}
	
	public void setPort(int port) {
		this.port = port;
	}
	
}
