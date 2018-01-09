package org.snu.cse.cap.translator.structure.task;

import java.util.ArrayList;
import java.util.List;

public class Transition {
	private String srcMode;
	private String dstMode;
	private ArrayList<Condition> conditionList;
	
	public Transition(String srcMode, String dstMode, ArrayList<Condition> conditionList) 
	{
		this.srcMode = srcMode;
		this.dstMode = dstMode;
		this.conditionList = conditionList;
	}
	
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

	public ArrayList<Condition> getConditionList() {
		return conditionList;
	}
}
