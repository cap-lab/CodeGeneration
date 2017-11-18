package org.snu.cse.cap.translator.structure;

import java.util.ArrayList;

public class Transition {
	private String srcMode;
	private String dstMode;
	private ArrayList<Condition> conditionList;
	
	public String getSrcMode() {
		return srcMode;
	}
	
	public void setSrcMode(String srcMode) {
		this.srcMode = srcMode;
	}
	
	public String getDstMode() {
		return dstMode;
	}
	
	public void setDstMode(String dstMode) {
		this.dstMode = dstMode;
	}
}
