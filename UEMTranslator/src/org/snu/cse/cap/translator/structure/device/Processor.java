package org.snu.cse.cap.translator.structure.device;

public class Processor {
	private int id;
	private boolean isCPU;
	private String name;
	private int poolSize;
	
	public Processor(String name, ProcessorCategory type, int poolSize) 
	{
		this.name = name;
		if(type == ProcessorCategory.CPU)
		{
			isCPU = true;
		}
		else
		{
			isCPU = false;
		}
		this.poolSize = poolSize;
	}
	
	public int getId() {
		return id;
	}
	
	public boolean isCPU() {
		return isCPU;
	}
	
	public String getName() {
		return name;
	}
	
	public int getPoolSize() {
		return poolSize;
	}
	
	public void setId(int id) {
		this.id = id;
	}
	
	public void setCPU(boolean isCPU) {
		this.isCPU = isCPU;
	}
	
	public void setName(String name) {
		this.name = name;
	}
	
	public void setPoolSize(int poolSize) {
		this.poolSize = poolSize;
	}
}
