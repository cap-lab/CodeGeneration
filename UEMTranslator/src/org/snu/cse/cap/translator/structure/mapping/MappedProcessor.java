package org.snu.cse.cap.translator.structure.mapping;

import java.util.ArrayList;

public class MappedProcessor {
	protected int processorId;
	protected int processorLocalId;
	
	public MappedProcessor(int processorId, int processorLocalId)
	{
		this.processorId = processorId;
		this.processorLocalId = processorLocalId;
	}
	
	public int getProcessorId() {
		return processorId;
	}
	
	public int getProcessorLocalId() {
		return processorLocalId;
	}
	
	public void setProcessorId(int processorId) {
		this.processorId = processorId;
	}
	
	public void setProcessorLocalId(int processorLocalId) {
		this.processorLocalId = processorLocalId;
	}
}
