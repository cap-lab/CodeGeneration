package org.snu.cse.cap.translator.structure.task;

public enum TaskShapeType {
	COMPUTATIONAL("computational"),
	CONTROL("control"),
	LOOP("loop"),
	EXTERNAL("external"),
	COMPOSITE("composite"), // Composite is not used by task type which is used for mapping and scheduling information
	;

	private final String value;
	
	private TaskShapeType(final String value) {
		this.value = value;
	}
	
	public static TaskShapeType fromValue(String value) {
		 for (TaskShapeType c : TaskShapeType.values()) {
			 if (c.value.equalsIgnoreCase(value)) {
				 return c;
			 }
		 }
		 throw new IllegalArgumentException(value.toString());
	}
}

