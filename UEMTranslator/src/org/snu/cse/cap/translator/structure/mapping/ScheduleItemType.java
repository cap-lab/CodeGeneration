package org.snu.cse.cap.translator.structure.mapping;

import org.snu.cse.cap.translator.structure.TaskGraphType;

public enum ScheduleItemType {
	LOOP("LOOP"),
	TASK("TASK"),
	
	;

	private final String value;
	
	private ScheduleItemType(final String value) {
		this.value = value;
	}
	
	public static ScheduleItemType fromValue(String value) {
		 for (ScheduleItemType c : ScheduleItemType.values()) {
			 if (c.value.equals(value)) {
				 return c;
			 }
		 }
		 throw new IllegalArgumentException(value.toString());
	}

	
	@Override
	public String toString() {
		return value;
	}
}