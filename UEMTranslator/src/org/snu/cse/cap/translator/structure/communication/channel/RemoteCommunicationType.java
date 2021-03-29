package org.snu.cse.cap.translator.structure.communication.channel;

public enum RemoteCommunicationType {
	NONE("NONE"),
	BLUETOOTH("BLUETOOTH"),
	TCP("TCP"),
	SERIAL("SERIAL"),
	SECURE_TCP("SecureTCP"),
	;

	private final String value;

	private RemoteCommunicationType(final String value) {
		this.value = value;
	}

	@Override
	public String toString() {
		return value;
	}

	public static RemoteCommunicationType fromValue(String value) {
		for (RemoteCommunicationType c : RemoteCommunicationType.values()) {
			if (c.value.equals(value)) {
				return c;
			}
		}
		throw new IllegalArgumentException(value.toString());
	}
}
