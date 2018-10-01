package org.snu.cse.cap.translator.structure.device.connection;

import hopes.cic.xml.NetworkType;

public abstract class SerialConnection extends Connection {

	public SerialConnection(String name, String role, NetworkType network) {
		super(name, role, network, ProtocolType.SERIAL);
	}

}
