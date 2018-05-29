package org.snu.cse.cap.translator.structure;

public enum ProgrammingLanguage {
	C("C"),
	CPP("C++"),
	;
	
	private final String value;
	
	private ProgrammingLanguage(String value) {
		this.value = value;
	}
	
	@Override
	public String toString() {
		return value;
	}
}