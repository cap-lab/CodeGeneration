package org.snu.cse.cap.translator.structure.device.connection;

import hopes.cic.xml.NetworkType;

public abstract class Connection {
	protected NetworkType network;
	protected ProtocolType protocol;
	protected String name;
	protected String role;
	
	public Connection(String name, String role, NetworkType network, ProtocolType protocol) 
	{
		this.name = name;
		this.role = role;
		this.network = network;
		this.protocol = protocol;
	}
	
	public String getName() {
		return name;
	}
	
	public String getRole() {
		return role;
	}
	
	
	public void setName(String name) {
		this.name = name;
	}
	
	public void setRole(String role) {
		this.role = role;
	}

	public NetworkType getNetwork() {
		return network;
	}

	public ProtocolType getProtocol() {
		return protocol;
	}
}
