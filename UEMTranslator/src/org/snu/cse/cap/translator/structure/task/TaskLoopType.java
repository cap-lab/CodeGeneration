package org.snu.cse.cap.translator.structure.task;

public enum TaskLoopType {
	CONVERGENT("convergent"),
	DATA("data"),
	;
	
	private final String value;
	
	private TaskLoopType(final String value) {
		this.value = value;
	}
	
	public static TaskLoopType fromValue(String value) {
		 for (TaskLoopType c : TaskLoopType.values()) {
			 if (c.value.equals(value)) {
				 return c;
			 }
		 }
		 throw new IllegalArgumentException(value.toString());
	}	
}