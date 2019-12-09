package org.snu.cse.cap.translator.structure.device.connection;

public class UDPConnection extends IPConnection {
	public UDPConnection(String name, String role, String IP, int port) 
	{
		super(name, role, IP, port, ProtocolType.UDP);
	}
	public UDPConnection() 
	{
		super(null, null, null, -1, ProtocolType.UDP);
	}
	
	public String getUDPId() {
		return IP.replaceAll("[.]", "_") + "_" + port;
	}
}
