package org.snu.cse.cap.translator.structure.task;

enum ParameterType {
	DOUBLE("float"),
	INT("int"),
	;

	private final String value;
	
	private ParameterType(final String value) {
		this.value = value;
	}
	
	public static ParameterType fromValue(String value) {
		 for (ParameterType c : ParameterType.values()) {
			 if (c.value.equals(value)) {
				 return c;
			 }
		 }
		 throw new IllegalArgumentException(value.toString());
	}	
}

public abstract class TaskParameter {
	protected int id;
	protected ParameterType type;
	protected String name;
	protected String description;
	
	public TaskParameter(String name, ParameterType type) {
		this.name = name;
		this.type = type;
		this.description = "";
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

	public String getDescription() {
		return description;
	}

	public void setDescription(String description) {
		this.description = description;
	}
}
