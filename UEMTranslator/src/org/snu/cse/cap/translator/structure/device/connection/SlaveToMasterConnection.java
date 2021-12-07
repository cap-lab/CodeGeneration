package org.snu.cse.cap.translator.structure.device.connection;

public class SlaveToMasterConnection {
	private Connection slave;
	private Connection master;
	private String slaveDeviceName;
	private String encryptionType;
	private String userKey;
	
	public SlaveToMasterConnection(String slaveDeviceName, Connection slave, Connection master, String encryptionType, String userKey)
	{
		this.slave = slave;
		this.master = master;
		this.slaveDeviceName = slaveDeviceName;
		this.encryptionType = encryptionType;
		this.userKey = userKey;
	}

	public Connection getSlave() {
		return slave;
	}

	public Connection getMaster() {
		return master;
	}

	public String getSlaveDeviceName() {
		return slaveDeviceName;
	}

	public String getEncryptionType() {
		return encryptionType;
	}

	public String getUserKey() {
		return userKey;
	}

}
