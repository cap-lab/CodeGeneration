package org.snu.cse.cap.translator.structure.library;

import java.util.ArrayList;

public class Function {
	private String name;
	private String returnType;
	private ArrayList<Argument> argumentList;
	
	public Function(String name, String returnType) {
		this.name = name;
		this.returnType = returnType;
		this.argumentList = new ArrayList<Argument>();
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
}
