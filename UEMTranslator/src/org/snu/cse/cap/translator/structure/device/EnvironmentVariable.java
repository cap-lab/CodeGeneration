package org.snu.cse.cap.translator.structure.device;

public class EnvironmentVariable {
	private String name;
	private String value;
	
	public EnvironmentVariable(String name, String value)
	{
		this.name = name.trim();
		this.value = value.trim();
	}
	
	public String getName() {
		return name;
	}
	public String getValue() {
		return value;
	}
}
