package org.snu.cse.cap.translator.structure.device.connection;

public abstract class Connection {
	protected ConnectionType type;
	protected String name;
	protected String role;
	
	public Connection(String name, String role, ConnectionType type) 
	{
		this.name = name;
		this.role = role;
		this.type = type;
	}
	
	public ConnectionType getType() {
		return type;
	}
	
	public String getName() {
		return name;
	}
	
	public String getRole() {
		return role;
	}
	
	public void setType(ConnectionType connectionType) {
		this.type = connectionType;
	}
	
	public void setName(String name) {
		this.name = name;
	}
	
	public void setRole(String role) {
		this.role = role;
	}
}
