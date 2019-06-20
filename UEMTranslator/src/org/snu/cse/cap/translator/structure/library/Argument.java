package org.snu.cse.cap.translator.structure.library;

public class Argument {
	private String name;
	private String type;
	private String description;
	
	public Argument(String name, String type)
	{
		this.name = name;
		this.type = type;
		this.description = "";
		
	}
	
	public String getName() {
		return name;
	}

	public String getType() {
		return type;
	}

	public String getDescription() {
		return description;
	}

	public void setDescription(String description) {
		this.description = description;
	}

}
