package org.snu.cse.cap.translator.structure;

enum ParameterType {
	DOUBLE,
	INT,
}

public abstract class TaskParameter {
	protected int paramId;
	protected ParameterType type;
	protected String paramName;
	
	public int getParamId() {
		return paramId;
	}
	
	public void setParamId(int paramId) {
		this.paramId = paramId;
	}
	
	public ParameterType getType() {
		return type;
	}
	
	public void setType(ParameterType type) {
		this.type = type;
	}
	
	public String getParamName() {
		return paramName;
	}
	
	public void setParamName(String paramName) {
		this.paramName = paramName;
	}
}
