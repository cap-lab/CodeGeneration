package org.snu.cse.cap.translator.structure.mapping;

import java.util.ArrayList;

public class CompositeTaskSchedule {
	private int scheduleId;
	private ArrayList<ScheduleItem> scheduleList;
	private int throughputConstraint;
	private int maxLoopVariableNum;
	
	public CompositeTaskSchedule(int scheduleId) {
		this.scheduleId = scheduleId;
		this.scheduleList = new ArrayList<ScheduleItem>();
		this.throughputConstraint = 0; // throughput is not defined
	}
	
	public CompositeTaskSchedule(int scheduleId, int throughputConstraint) {
		this.scheduleId = scheduleId;
		this.scheduleList = new ArrayList<ScheduleItem>();
		this.throughputConstraint = throughputConstraint;
	}
	
	public ArrayList<ScheduleItem> getScheduleList() {
		return scheduleList;
	}
	
	public void putScheduleItem(ScheduleItem item) {
		this.scheduleList.add(item);
	}
	
	public int getScheduleId() {
		return scheduleId;
	}
	
	public void setScheduleId(int scheduleId) {
		this.scheduleId = scheduleId;
	}
	
	public int getThroughputConstraint() {
		return throughputConstraint;
	}
	
	public void setThroughputConstraint(int throughputConstraint) {
		this.throughputConstraint = throughputConstraint;
	}
}
