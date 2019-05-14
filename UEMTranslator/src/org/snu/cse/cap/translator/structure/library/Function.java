package org.snu.cse.cap.translator.structure.library;

import java.util.ArrayList;

public class Function {
	private String name;
	private String returnType;
	private ArrayList<Argument> argumentList;
	private String description;
	
	public Function(String name, String returnType) {
		this.name = name;
		this.returnType = returnType;
		this.argumentList = new ArrayList<Argument>();
		this.description = "";
	}
	
	public String getName() {
		return name;
	}

	public String getReturnType() {
		return returnType;
	}

	public ArrayList<Argument> getArgumentList() {
		return argumentList;
	}

	public String getDescription() {
		return description;
	}

	public void setDescription(String description) {
		this.description = description;
	}
}
