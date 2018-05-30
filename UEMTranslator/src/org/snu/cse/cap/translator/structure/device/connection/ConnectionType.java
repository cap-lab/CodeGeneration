package org.snu.cse.cap.translator.structure.device.connection;

public enum ConnectionType {
	TCP("tcp"),
	BLUETOOTH("bluetooth"),
	;

	private final String value;
	
	private ConnectionType(final String value) {
		this.value = value;
	}
	
	@Override
	public String toString() {
		return value;
	}
}