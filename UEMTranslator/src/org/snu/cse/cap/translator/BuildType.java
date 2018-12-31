package org.snu.cse.cap.translator;

public enum BuildType {
	MAKEFILE("makefile"),
	AUTOMAKE("automake"),
	;
	
	private final String value;
	
	private BuildType(final String value) {
		this.value = value;
	}
	
	@Override
	public String toString() {
		return value;
	}
	
	public static BuildType fromValue(String value) {
		 for (BuildType c : BuildType.values()) {
			 if (c.value.equals(value)) {
				 return c;
			 }
		 }
		 throw new IllegalArgumentException(value.toString());
	}
}
