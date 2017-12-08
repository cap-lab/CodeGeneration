package org.snu.cse.cap.translator.structure.mapping;

import java.util.ArrayList;

public class CompositeTaskMappedProcessor extends MappedProcessor {
	private ArrayList<CompositeTaskSchedule> compositeTaskScheduleList;
	private int modeId;
	private int sequenceIdInMode;
	
	public CompositeTaskMappedProcessor(int processorId, int processorLocalId, int modeId, int sequenceId) {
		super(processorId, processorLocalId);
		this.modeId = modeId;
		this.compositeTaskScheduleList = new ArrayList<CompositeTaskSchedule>();
		this.sequenceIdInMode = sequenceId;
	}
	
	public ArrayList<CompositeTaskSchedule> getCompositeTaskScheduleList() {
		return compositeTaskScheduleList;
	}
	
	public void putCompositeTaskSchedule(CompositeTaskSchedule schedule) 
	{
		this.compositeTaskScheduleList.add(schedule);
	}
	
	public int getModeId() {
		return modeId;
	}
	
	public void setModeId(int modeId) {
		this.modeId = modeId;
	}

	public int getSequenceIdInMode() {
		return sequenceIdInMode;
	}
}
