package org.snu.cse.cap.translator.structure.device.connection;

import java.util.ArrayList;

import org.snu.cse.cap.translator.structure.device.Connection;

public class MasterToSlaveConnection {
	private Connection master;
	private ArrayList<Connection> slaveList;
	
	public MasterToSlaveConnection(Connection master) {
		this.master = master;
		this.slaveList = new ArrayList<Connection>(); 
	}
	
	public void putSlave(Connection slave)
	{
		this.slaveList.add(slave);
	}

}
