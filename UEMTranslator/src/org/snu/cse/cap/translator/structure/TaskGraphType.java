package org.snu.cse.cap.translator.structure;

public enum TaskGraphType {
	PROCESS_NETWORK("ProcessNetwork"),
	DATAFLOW("DataFlow"),
	HYBRID("Hybrid"),
	;

	private final String value;
	
	private TaskGraphType(final String value) {
		this.value = value;
	}
	
	public static TaskGraphType fromValue(String value) {
		 for (TaskGraphType c : TaskGraphType.values()) {
			 if (c.value.equalsIgnoreCase(value)) {
				 return c;
			 }
		 }
		 throw new IllegalArgumentException(value.toString());
	}
}

