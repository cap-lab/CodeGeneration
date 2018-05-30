package org.snu.cse.cap.translator.structure.device.connection;

public class ConnectionPair {
	String masterDeviceName;
	Connection masterConnection;
	String slaveDeviceName;
	Connection slaveConnection;
	
	public ConnectionPair(String masterDeviceName, Connection masterConnection, String slaveDeviceName,
			Connection slaveConnection) {
		this.masterDeviceName = masterDeviceName;
		this.masterConnection = masterConnection;
		this.slaveDeviceName = slaveDeviceName;
		this.slaveConnection = slaveConnection;
	}

	public String getMasterDeviceName() {
		return masterDeviceName;
	}
	public Connection getMasterConnection() {
		return masterConnection;
	}
	public String getSlaveDeviceName() {
		return slaveDeviceName;
	}
	public Connection getSlaveConnection() {
		return slaveConnection;
	}
}
