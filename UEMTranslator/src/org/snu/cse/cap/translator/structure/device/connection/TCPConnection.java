package org.snu.cse.cap.translator.structure.device.connection;

public class TCPConnection extends IPConnection {
	public TCPConnection(String name, String role, String IP, int port) 
	{
		super(name, role, IP, port, ProtocolType.TCP);
	}
}
