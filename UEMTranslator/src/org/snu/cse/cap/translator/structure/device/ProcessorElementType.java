package org.snu.cse.cap.translator.structure.device;

enum ProcessorCategory {
	CPU("CPU"),
	GPU("GPU"),
	VIRTUAL("VIRTUAL"),
	;

	private final String value;
	
	private ProcessorCategory(final String value) {
		this.value = value;
	}
	
	@Override
	public String toString() {
		return value;
	}
}

public class ProcessorElementType extends HWElementType {
	private ProcessorCategory subcategory;
	private String model;
	
	// default category is CPU
	public ProcessorElementType(String name, String model) {
		super(name, HWCategory.PROCESSOR);
		this.model = model;
		this.subcategory = ProcessorCategory.CPU;
	}
	
	public ProcessorElementType(String name, String model, String subcategory) {
		super(name, HWCategory.PROCESSOR);
		this.model = model;
		if(subcategory != null)
		{
			this.subcategory = ProcessorCategory.valueOf(subcategory);
		}
		else
		{
			this.subcategory = ProcessorCategory.CPU;
		}
	}

	public ProcessorCategory getSubcategory() {
		return subcategory;
	}

	public void setSubcategory(ProcessorCategory subcategory) {
		this.subcategory = subcategory;
	}

	public String getModel() {
		return model;
	}

	public void setModel(String model) {
		this.model = model;
	}
}
