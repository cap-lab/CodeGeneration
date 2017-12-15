package org.snu.cse.cap.translator.structure.device.connection;

import org.snu.cse.cap.translator.structure.device.Connection;

public class SlaveToMasterConnection {
	private Connection slave;
	private Connection master;
	
	public SlaveToMasterConnection(Connection slave, Connection master)
	{
		this.slave = slave;
		this.master = master;
	}

	public Connection getSlave() {
		return slave;
	}

	public Connection getMaster() {
		return master;
	}
}
