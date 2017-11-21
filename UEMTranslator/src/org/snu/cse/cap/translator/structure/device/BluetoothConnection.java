package org.snu.cse.cap.translator.structure.device;

public class BluetoothConnection extends Connection {
	private String friendlyName;
	private String macAddress;
	
	public String getFriendlyName() {
		return friendlyName;
	}
	
	public String getMacAddress() {
		return macAddress;
	}
	
	public void setFriendlyName(String friendlyName) {
		this.friendlyName = friendlyName;
	}
	
	public void setMacAddress(String macAddress) {
		this.macAddress = macAddress;
	}
}
