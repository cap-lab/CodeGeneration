package org.snu.cse.cap.translator.structure.task;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

public class TaskModeTransition {
	private int taskId;
	private HashMap<String, TaskMode> modeMap;
	private HashMap<String, TaskMode> modeIdMap;
	private HashMap<String, String> variableMap;
	private ArrayList<Transition> transitionList;
	
	public TaskModeTransition(int taskId) {
		this.taskId = taskId;
		this.modeMap = new HashMap<String, TaskMode>();
		this.modeIdMap = new HashMap<String, TaskMode>();
		this.variableMap = new HashMap<String, String>();
		this.transitionList = new ArrayList<Transition>();
	}
	
	public void putMode(int modeId, String modeName) 
	{
		TaskMode mode = new TaskMode(modeId, modeName);
		this.modeMap.put(modeName, mode);
		this.modeIdMap.put("" + modeId, mode);
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
	
	public int getModeIdFromName(String modeName)
	{
		TaskMode mode = this.modeMap.get(modeName);
		
		return mode.getId();
	}

	public HashMap<String, TaskMode> getModeMap() {
		return modeMap;
	}
	
	public void putRelatedChildTask(int procId, int procLocalId, int modeId, Task task)
	{
		TaskMode mode;
		mode = this.modeIdMap.get("" + modeId);
		mode.putChildTask(procId, procLocalId, task);
	}
	
	public HashMap<String, TaskMode> getModeIdMap() {
		return modeIdMap;
	}

	public void setModeMap(HashMap<String, TaskMode> modeMap) {
		this.modeMap = modeMap;
	}

	public HashMap<String, String> getVariableMap() {
		return variableMap;
	}
	
}
