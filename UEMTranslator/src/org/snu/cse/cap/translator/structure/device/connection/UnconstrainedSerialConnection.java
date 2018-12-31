package org.snu.cse.cap.translator.structure.device.connection;

import hopes.cic.xml.NetworkType;

public class UnconstrainedSerialConnection extends SerialConnection {
	private String portAddress;

	public UnconstrainedSerialConnection(String name, String role, NetworkType network, String portAddress) {
		super(name, role, network);
		this.portAddress = portAddress;
	}

	public String getPortAddress() {
		return portAddress;
	}

}
