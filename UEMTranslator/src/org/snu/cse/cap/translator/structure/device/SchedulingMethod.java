package org.snu.cse.cap.translator.structure.device;

public enum SchedulingMethod {
	OTHER("other"), FIFO("fifo"), RR("rr"), HIGH("high"), REALTIME("realtime");

	private final String value;
	
	private SchedulingMethod(final String value) {
		this.value = value;
	}
	
	@Override
	public String toString() {
		return value;
	}
	
	public static SchedulingMethod fromValue(String value) {
		 for (SchedulingMethod c : SchedulingMethod.values()) {
			 if (c.value.equalsIgnoreCase(value)) {
				 return c;
			 }
		 }
		 throw new IllegalArgumentException(value.toString());
	}
}