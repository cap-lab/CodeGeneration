package org.snu.cse.cap.translator.structure;

import java.util.ArrayList;

public class TaskMode {
	private int modeId;
	private String modeName;
	private ArrayList<String> relatedChildTaskList;
	
	public int getModeId() {
		return modeId;
	}
	
	public void setModeId(int modeId) {
		this.modeId = modeId;
	}
	
	public String getModeName() {
		return modeName;
	}
	
	public void setModeName(String modeName) {
		this.modeName = modeName;
	}
}
