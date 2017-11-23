package org.snu.cse.cap.translator.structure.task;

public class TaskIntegerParameter extends TaskParameter {
	private int value;
	
	public TaskIntegerParameter(String name, int value) {
		super(name, ParameterType.INT);
		this.value = value;
	}

	public int getValue() {
		return value;
	}

	public void setValue(int value) {
		this.value = value;
	}
}
