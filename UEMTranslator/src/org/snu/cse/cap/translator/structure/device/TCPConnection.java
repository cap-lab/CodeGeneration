package org.snu.cse.cap.translator.structure.device;

public class TCPConnection extends Connection {
	private String IP;
	private int port;
	
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
