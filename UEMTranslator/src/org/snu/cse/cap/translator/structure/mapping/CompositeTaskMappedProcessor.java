package org.snu.cse.cap.translator.structure.mapping;

import java.util.ArrayList;
import java.util.HashMap;

import org.snu.cse.cap.translator.structure.task.Task;

public class CompositeTaskMappedProcessor extends MappedProcessor {
	private ArrayList<CompositeTaskSchedule> compositeTaskScheduleList;
	private int modeId;
	private int sequenceIdInMode;
	private int inArrayIndex = 0;
	private HashMap<String, Task> srcTaskMap;
	
	public CompositeTaskMappedProcessor(int processorId, int processorLocalId, int modeId, int sequenceId) {
		super(processorId, processorLocalId);
		this.modeId = modeId;
		this.compositeTaskScheduleList = new ArrayList<CompositeTaskSchedule>();
		this.sequenceIdInMode = sequenceId;
		this.srcTaskMap = new HashMap<String, Task>();
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

	public int getInArrayIndex() {
		return inArrayIndex;
	}

	public void setInArrayIndex(int inArrayIndex) {
		this.inArrayIndex = inArrayIndex;
	}

	public HashMap<String, Task> getSrcTaskMap() {
		return srcTaskMap;
	}
}
