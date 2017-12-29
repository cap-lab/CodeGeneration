package org.snu.cse.cap.translator;

import org.snu.cse.cap.translator.structure.task.TimeMetric;

public class ExecutionTime {
	private int value;
	private TimeMetric metric;
	
	private static final int DEFAULT_EXECUTION_TIME = 500;
	private static final TimeMetric DEFAULT_EXECUTION_TIME_METRIC = TimeMetric.MILLISEC;
	
	public ExecutionTime()
	{
		this.value = DEFAULT_EXECUTION_TIME;
		this.metric = DEFAULT_EXECUTION_TIME_METRIC;
	}
	
	public ExecutionTime(int executionTime, String timeMetric)
	{
		this.value = executionTime;
		this.metric = TimeMetric.fromValue(timeMetric);
	}
	
	public int getValue() {
		return value;
	}
	
	public TimeMetric getMetric() {
		return metric;
	}
}
