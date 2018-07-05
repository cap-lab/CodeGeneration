package org.snu.cse.cap.translator.structure.device.connection;

public class SlaveToMasterConnection {
	private Connection slave;
	private Connection master;
	private String slaveDeviceName;
	
	public SlaveToMasterConnection(String slaveDeviceName, Connection slave, Connection master)
	{
		this.slave = slave;
		this.master = master;
		this.slaveDeviceName = slaveDeviceName;
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

}
