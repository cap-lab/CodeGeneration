package org.snu.cse.cap.translator.structure.channel;

public enum PortDirection {
	INPUT("input"),
	OUTPUT("output"),
	;
	
	private final String value;
	
	private PortDirection(String value) {
		this.value = value;
	}
	
	@Override
	public String toString() {
		return value;
	}
	
	public static PortDirection fromValue(String value) {
		 for (PortDirection c : PortDirection.values()) {
			 if (c.value.equals(value)) {
				 return c;
			 }
		 }
		 throw new IllegalArgumentException(value.toString());
	}
}