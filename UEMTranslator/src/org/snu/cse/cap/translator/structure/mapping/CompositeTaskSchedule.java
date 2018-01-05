package org.snu.cse.cap.translator.structure.mapping;

import java.util.ArrayList;

public class CompositeTaskSchedule {
	private ArrayList<ScheduleItem> scheduleList;
	private int numOfUsableCPU;
	private int throughputConstraint;
	private int maxLoopVariableNum;
	
	public int getMaxLoopVariableNum() {
		return maxLoopVariableNum;
	}

	public void setMaxLoopVariableNum(int maxLoopVariableNum) {
		this.maxLoopVariableNum = maxLoopVariableNum;
	}
	
	public CompositeTaskSchedule(int numOfUsableCPU, int throughputConstraint) {
		this.scheduleList = new ArrayList<ScheduleItem>();
		this.throughputConstraint = throughputConstraint;
	}
	
	public ArrayList<ScheduleItem> getScheduleList() {
		return scheduleList;
	}
	
	public void putScheduleItem(ScheduleItem item) {
		this.scheduleList.add(item);
	}
	
	public int getThroughputConstraint() {
		return throughputConstraint;
	}
	
	public void setThroughputConstraint(int throughputConstraint) {
		this.throughputConstraint = throughputConstraint;
	}

	public int getNumOfUsableCPU() {
		return numOfUsableCPU;
	}
}
