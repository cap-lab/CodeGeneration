package org.snu.cse.cap.translator.structure.device;

public enum DeviceCommunicationType {
	NONE("none"),
	SHARED_MEMORY("shared_memory"),
	BLUETOOTH("bluetooth"),
	TCP("tcp"),
	UDP("udp"),
	SERIAL("serial"),
	;

	private final String value;
	
	private DeviceCommunicationType(final String value) {
		this.value = value;
	}
	
	@Override
	public String toString() {
		return value;
	}
	
	public static DeviceCommunicationType fromValue(String value) {
		 for (DeviceCommunicationType c : DeviceCommunicationType.values()) {
			 if (c.value.equalsIgnoreCase(value)) {
				 return c;
			 }
		 }
		 throw new IllegalArgumentException(value.toString());
	}
}
