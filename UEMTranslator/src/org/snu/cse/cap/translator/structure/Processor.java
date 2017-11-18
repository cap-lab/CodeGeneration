package org.snu.cse.cap.translator.structure;

public class Processor {
	private int processorId;
	private boolean isCPU;
	private int processorPoolSize;
	
	public int getProcessorId() {
		return processorId;
	}
	
	public void setProcessorId(int processorId) {
		this.processorId = processorId;
	}
	
	public boolean isCPU() {
		return isCPU;
	}
	
	public void setCPU(boolean isCPU) {
		this.isCPU = isCPU;
	}
	
	public int getProcessorPoolSize() {
		return processorPoolSize;
	}
	
	public void setProcessorPoolSize(int processorPoolSize) {
		this.processorPoolSize = processorPoolSize;
	}
}
