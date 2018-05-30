package org.snu.cse.cap.translator.structure.device.connection;

import java.util.ArrayList;
import java.util.HashMap;

public class DeviceConnection {
	private String deviceName;
	private HashMap<String, MasterToSlaveConnection> connectionToSlaveMap;
	private HashMap<String, SlaveToMasterConnection> connectionToMasterMap;
	
	public DeviceConnection(String deviceName) {
		this.deviceName = deviceName;
		this.connectionToSlaveMap = new HashMap<String, MasterToSlaveConnection>(); // key: master connection name
		this.connectionToMasterMap = new HashMap<String, SlaveToMasterConnection>(); // key slave connection name
	}
	
	public void putMasterToSlaveConnection(Connection master, String slaveDeviceName, Connection slave)
	{
		MasterToSlaveConnection connection;
		if(this.connectionToSlaveMap.containsKey(master.getName()))
		{
			connection = this.connectionToSlaveMap.get(master.getName());
		}
		else
		{
			connection = new MasterToSlaveConnection(master);
			
			this.connectionToSlaveMap.put(master.getName(), connection);
		}
		
		connection.putSlave(slave, slaveDeviceName);
	}
	
	public void putSlaveToMasterConnection(Connection slave, String masterDeviceName, Connection master) throws InvalidDeviceConnectionException
	{
		SlaveToMasterConnection connection;
		if(this.connectionToMasterMap.containsKey(slave.getName()))
		{
			throw new InvalidDeviceConnectionException();
		}
		else
		{
			connection = new SlaveToMasterConnection(slave, masterDeviceName, master);
			
			this.connectionToMasterMap.put(slave.getName(), connection);
		}
	}

	// TODO: connection - channel mapping is needed to support multiple types of connections between devices 
	// Only single connection is allowed between devices
	public ConnectionPair findOneConnectionToAnotherDevice(String deviceName)
	{
		ConnectionPair connectionPair = null;
		//Connection connection;
		for(MasterToSlaveConnection connection : this.connectionToSlaveMap.values())
		{
			for(String slaveDeviceName : connection.getSlaveDeviceToConnectionMap().keySet())
			{
				if(slaveDeviceName.equals(deviceName) == true)
				{
					ArrayList<Connection> connectionList;
					connectionList = connection.getSlaveDeviceToConnectionMap().get(slaveDeviceName);
					if(connectionList.size() > 0)
					{
						// TODO: Currently, only supports single connection between two devices
						connectionPair = new ConnectionPair(this.deviceName, connection.getMaster(), slaveDeviceName, connectionList.get(0));
						break;
					}
				}				
			}
		}
		
		if(connectionPair != null)
		{
			for(SlaveToMasterConnection connection: this.connectionToMasterMap.values())
			{
				if(connection.getMasterDeviceName().equals(deviceName) == true)
				{
					connectionPair = new ConnectionPair(deviceName, connection.getMaster(), this.deviceName, connection.getSlave());
					break;
				}
			}
		}
		
		return connectionPair;
	}

	public HashMap<String, MasterToSlaveConnection> getConnectionToSlaveMap() {
		return connectionToSlaveMap;
	}

	public HashMap<String, SlaveToMasterConnection> getConnectionToMasterMap() {
		return connectionToMasterMap;
	}

	public String getDeviceName() {
		return deviceName;
	}
}


