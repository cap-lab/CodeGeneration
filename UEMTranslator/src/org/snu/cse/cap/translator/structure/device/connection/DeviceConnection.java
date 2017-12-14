package org.snu.cse.cap.translator.structure.device.connection;

import java.util.ArrayList;

public class DeviceConnection {
	private String deviceName;
	private ArrayList<MasterToSlaveConnection> connectionToSlaveList;
	private ArrayList<SlaveToMasterConnection> connectionToMasterList;
	
	public DeviceConnection(String deviceName) {
		this.deviceName = deviceName;
		this.connectionToSlaveList = new ArrayList<MasterToSlaveConnection>();
		this.connectionToMasterList = new ArrayList<SlaveToMasterConnection>();
	}
}


