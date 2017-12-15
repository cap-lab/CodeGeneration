package org.snu.cse.cap.translator.structure.device.connection;

import java.util.HashMap;

import org.snu.cse.cap.translator.structure.device.Connection;

public class DeviceConnection {
	private String deviceName;
	private HashMap<String, MasterToSlaveConnection> connectionToSlaveList;
	private HashMap<String, SlaveToMasterConnection> connectionToMasterList;
	
	public DeviceConnection(String deviceName) {
		this.deviceName = deviceName;
		this.connectionToSlaveList = new HashMap<String, MasterToSlaveConnection>();
		this.connectionToMasterList = new HashMap<String, SlaveToMasterConnection>();
	}
	
	public void putMasterToSlaveConnection(Connection master, Connection slave)
	{
		MasterToSlaveConnection connection;
		if(this.connectionToSlaveList.containsKey(master.getName()))
		{
			connection = this.connectionToSlaveList.get(master.getName());
		}
		else
		{
			connection = new MasterToSlaveConnection(master);
			
			this.connectionToSlaveList.put(master.getName(), connection);
		}
		
		connection.putSlave(slave);
	}
	
	public void putSlaveToMasterConnection(Connection slave, Connection master) throws InvalidDeviceConnectionException
	{
		SlaveToMasterConnection connection;
		if(this.connectionToMasterList.containsKey(slave.getName()))
		{
			throw new InvalidDeviceConnectionException();
		}
		else
		{
			connection = new SlaveToMasterConnection(slave, master);
			
			this.connectionToMasterList.put(slave.getName(), connection);
		}
		
	}
}


