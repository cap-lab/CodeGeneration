package org.snu.cse.cap.translator.structure.task;

public enum TaskLoopType {
	CONVERGENT("convergent"),
	DATA("data"),
	;
	
	private final String value;
	
	private TaskLoopType(final String value) {
		this.value = value;
	}
	
	@Override
	public String toString() {
		return value;
	}
}