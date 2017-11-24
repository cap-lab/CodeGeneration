package org.snu.cse.cap.translator.structure.task;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

public class TaskModeTransition {
	private int taskId;
	private HashMap<String, TaskMode> modeMap;
	private HashMap<String, String> variableMap;
	private ArrayList<Transition> transitionList;
	
	public TaskModeTransition(int taskId) {
		this.taskId = taskId;
		modeMap = new HashMap<String, TaskMode>();
		variableMap = new HashMap<String, String>();
		transitionList = new ArrayList<Transition>();
	}
	
	public void putMode(int modeId, String modeName) 
	{
		TaskMode mode = new TaskMode(modeId, modeName);
		this.modeMap.put(modeName, mode);
	}
	
	public void putTransition(String srcMode, String dstMode, ArrayList<Condition> conditionList) 
	{
		Transition transition = new Transition(srcMode, dstMode, conditionList);
		
		this.transitionList.add(transition);
	}

	// Variable type is not used in translated code
	public void putVariable(String variableName, String variableType) 
	{		
		this.variableMap.put(variableName, variableType);
	}
	
	public int getTaskId() {
		return taskId;
	}
	
	public void setTaskId(int taskId) {
		this.taskId = taskId;
	}
}
