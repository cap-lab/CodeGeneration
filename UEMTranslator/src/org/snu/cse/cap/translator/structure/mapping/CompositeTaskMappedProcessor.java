package org.snu.cse.cap.translator.structure.mapping;

import java.util.ArrayList;

public class CompositeTaskMappedProcessor extends MappedProcessor {
	private ArrayList<CompositeTaskSchedule> scheduleLists;
	private int modeId;
	
	public CompositeTaskMappedProcessor(int processorId, int processorLocalId, int modeId) {
		super(processorId, processorLocalId);
		this.modeId = modeId;
		this.scheduleLists = new ArrayList<CompositeTaskSchedule>();
	}
	
	public ArrayList<CompositeTaskSchedule> getCompositeTaskScheduleList() {
		return scheduleLists;
	}
	
	public void putCompositeTaskSchedule(CompositeTaskSchedule schedule) 
	{
		this.scheduleLists.add(schedule);
	}
	
	public int getModeId() {
		return modeId;
	}
	
	public void setModeId(int modeId) {
		this.modeId = modeId;
	}
}
