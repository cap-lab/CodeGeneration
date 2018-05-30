package org.snu.cse.cap.translator.structure.device.connection;

import java.util.ArrayList;
import java.util.HashMap;

public class MasterToSlaveConnection {
	private Connection master;
	HashMap<String, ArrayList<Connection>> slaveDeviceToConnectionMap; // key: slave device name
	
	public MasterToSlaveConnection(Connection master) {
		this.master = master;
		this.slaveDeviceToConnectionMap = new HashMap<String, ArrayList<Connection>>(); 
	}
	
	public void putSlave(Connection slave, String slaveDeviceName)
	{
		ArrayList<Connection> connectionList;
		if(this.slaveDeviceToConnectionMap.containsKey(slaveDeviceName) == false)
		{
			connectionList = new ArrayList<Connection>();
			this.slaveDeviceToConnectionMap.put(slaveDeviceName, connectionList);
		}

		connectionList = this.slaveDeviceToConnectionMap.get(slaveDeviceName);
		connectionList.add(slave);
	}

	public Connection getMaster() {
		return master;
	}

	public HashMap<String, ArrayList<Connection>> getSlaveDeviceToConnectionMap() {
		return slaveDeviceToConnectionMap;
	}

}
