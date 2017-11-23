package org.snu.cse.cap.translator.structure.task;

enum ParameterType {
	DOUBLE("float"),
	INT("int"),
	;

	private final String value;
	
	private ParameterType(final String value) {
		this.value = value;
	}
	
	@Override
	public String toString() {
		return value;
	}
}

public abstract class TaskParameter {
	protected int id;
	protected ParameterType type;
	protected String name;
	
	public TaskParameter(String name, ParameterType type) {
		this.name = name;
		this.type = type;
	}
	
	public int getId() {
		return id;
	}
	
	public void setId(int paramId) {
		this.id = paramId;
	}
	
	public ParameterType getType() {
		return type;
	}
	
	public void setType(ParameterType type) {
		this.type = type;
	}
	
	public String getName() {
		return name;
	}
	
	public void setName(String paramName) {
		this.name = paramName;
	}
}
