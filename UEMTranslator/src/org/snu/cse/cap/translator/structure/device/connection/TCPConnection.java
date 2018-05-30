package org.snu.cse.cap.translator.structure.device.connection;

public class TCPConnection extends Connection {
	private String IP;
	private int port;
	
	public TCPConnection(String name, String role, String IP, int port) 
	{
		super(name, role, ConnectionType.TCP);
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
