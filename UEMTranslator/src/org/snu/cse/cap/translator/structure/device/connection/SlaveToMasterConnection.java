package org.snu.cse.cap.translator.structure.device.connection;

public class SlaveToMasterConnection {
	private Connection slave;
	private Connection master;
	private String masterDeviceName;
	
	public SlaveToMasterConnection(Connection slave, String masterDeviceName, Connection master)
	{
		this.slave = slave;
		this.master = master;
		this.masterDeviceName = masterDeviceName;
	}

	public Connection getSlave() {
		return slave;
	}

	public Connection getMaster() {
		return master;
	}

	public String getMasterDeviceName() {
		return masterDeviceName;
	}
}
