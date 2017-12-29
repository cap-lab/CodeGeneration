package org.snu.cse.cap.translator.structure.task;

public enum TimeMetric {
	CYCLE("cycle"),
	COUNT("count"), 
	MICROSEC("us"),
	MILLISEC("ms"),
	SEC("s"),
	MINUTE("m"),
	HOUR("h"),
	;
	
	private final String value;
	
	private TimeMetric(String value) {
		this.value = value;
	}
	
	public static TimeMetric fromValue(String value) {
		 for (TimeMetric c : TimeMetric.values()) {
			 if (c.value.equals(value)) {
				 return c;
			 }
		 }
		 throw new IllegalArgumentException(value.toString());
	}
}