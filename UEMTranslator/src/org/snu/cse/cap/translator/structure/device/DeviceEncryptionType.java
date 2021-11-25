package org.snu.cse.cap.translator.structure.device;

public enum DeviceEncryptionType {
	NO("NO"),
	LEA("LEA"),
	HIGHT("HIGHT"),
	SEED("SEED"),
	;

	public static final int MAX_BLOCK_SIZE = 16;

	private final String value;
	private int blockSize;
	
	private DeviceEncryptionType(final String value) {
		this.value = value;
		setBlockSize(0);

		if (value.contentEquals("LEA") || value.contentEquals("SEED")) {
			setBlockSize(16);
		} else if (value.equals("HIGHT")) {
			setBlockSize(8);
		}
	}
	
	@Override
	public String toString() {
		return value;
	}
	
	public static DeviceEncryptionType fromValue(String value) {
		 for (DeviceEncryptionType c : DeviceEncryptionType.values()) {
			 if (c.value.equalsIgnoreCase(value)) {
				 return c;
			 }
		 }
		 throw new IllegalArgumentException(value.toString());
	}

	public int getBlockSize() {
		return blockSize;
	}

	public void setBlockSize(int blockSize) {
		this.blockSize = blockSize;
	}
}
