package org.snu.cse.cap.translator.structure.task;

import java.util.ArrayList;

public class TaskMode {
	private int id;
	private String name;
	private ArrayList<String> relatedChildTaskList;
	
	public TaskMode(int modeId, String modeName) {
		this.id = modeId;
		this.name = modeName;
		this.relatedChildTaskList = new ArrayList<String>();
	}
	
	public int getId() {
		return id;
	}
	
	public void setId(int modeId) {
		this.id = modeId;
	}
	
	public String getName() {
		return name;
	}
	
	public void setName(String modeName) {
		this.name = modeName;
	}
}
