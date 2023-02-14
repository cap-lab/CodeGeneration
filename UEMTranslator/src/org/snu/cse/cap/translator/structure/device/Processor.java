package org.snu.cse.cap.translator.structure.device;

public class Processor {
	private int id;
	private boolean isCPU;
	private boolean isVirtual;
	private String name;
	private int poolSize;
	
	public Processor(int id, String name, ProcessorCategory type, int poolSize) 
	{
		this.id = id;
		this.name = name;
		this.isVirtual = false;
		if (type == ProcessorCategory.CPU)
		{
			isCPU = true;
		}
		else if (type == ProcessorCategory.GPU)
		{
			isCPU = false;
		}
		else {
			isCPU = true;
			isVirtual = true;
		}
		this.poolSize = poolSize;
	}
	
	public int getId() {
		return id;
	}
	
	public boolean getIsCPU() {
		return isCPU;
	}

	public boolean getIsVirtual() {
		return isVirtual;
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
