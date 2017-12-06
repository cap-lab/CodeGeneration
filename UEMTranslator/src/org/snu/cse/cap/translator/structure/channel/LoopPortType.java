package org.snu.cse.cap.translator.structure.channel;

public enum LoopPortType {
	DISTRIBUTING("distributing"),
	BROADCASTING("broadcasting"),
	;
	
	private final String value;
	
	private LoopPortType(String value) {
		this.value = value;
	}
	
	@Override
	public String toString() {
		return value;
	}
	
	public static LoopPortType fromValue(String value) {
		 for (LoopPortType c : LoopPortType.values()) {
			 if (c.value.equals(value)) {
				 return c;
			 }
		 }
		 throw new IllegalArgumentException(value.toString());
	}
}