package org.snu.cse.cap.translator.structure.mapping;

public enum ScheduleFileNameOffset {
	TASK_NAME(0),
	MODE_NAME(1),
	NUM_OF_USABLE_CPU(2),
	THROUGHPUT_CONSTRAINT(3),
	SCHEUDLE_XML(4),
	;
	
	private final int value;
	
    private ScheduleFileNameOffset(int value) {
        this.value = value;
    }
    
    public int getValue() {
    	return this.value;
    }
}