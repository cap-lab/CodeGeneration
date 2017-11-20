package org.snu.cse.cap.translator.structure.task;

import java.util.ArrayList;
import java.util.HashMap;

public class TaskModeTransition {
	private int taskId;
	private HashMap<String, TaskMode> modeMap;
	private HashMap<String, Integer> variableMap;
	private ArrayList<Transition> transitionList;
	
	public int getTaskId() {
		return taskId;
	}
	
	public void setTaskId(int taskId) {
		this.taskId = taskId;
	}
}
