package org.snu.cse.cap.translator.structure.task;

import org.snu.cse.cap.translator.structure.task.TaskParameter;

public class TaskDoubleParameter extends TaskParameter {
	private double value;
	
	public TaskDoubleParameter(String name, double value) {
		super(name, ParameterType.DOUBLE);
		this.value = value;
	}

	public double getValue() {
		return value;
	}

	public void setValue(double value) {
		this.value = value;
	}
}
