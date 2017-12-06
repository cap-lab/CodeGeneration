package org.snu.cse.cap.translator.structure.task;

import java.util.HashMap;
import java.util.HashSet;

public class TaskMode {
	private int id;
	private String name;
	private HashSet<String> relatedChildTaskSet;
	private HashMap<String, String> relatedChildTaskProcMap;
	private static final String TASK_PROC_MAP_SEPARATOR = "/";
	private static final int TASK_PROC_MAP_PROC_ID_INDEX = 0;
	private static final int TASK_PROC_MAP_LOCAL_ID_INDEX = 1;
	private static final int TASK_PROC_MAP_TASK_NAME_INDEX = 2;
	
	public TaskMode(int modeId, String modeName) {
		this.id = modeId;
		this.name = modeName;
		this.relatedChildTaskSet = new HashSet<String>();
		this.relatedChildTaskProcMap = new HashMap<String, String>();
	}
	
	public interface ChildTaskTraverseCallback<T> {
		 public void traverseCallback(String taskName, int procId, int procLocalId, T userData);
	}

	
	public void putChildTask(int procId, int procLocalId, String taskName)
	{
		this.relatedChildTaskSet.add(taskName);
		this.relatedChildTaskProcMap.put(procId + TASK_PROC_MAP_SEPARATOR  + procLocalId + TASK_PROC_MAP_SEPARATOR + taskName, taskName);
	}
	
	public void traverseRelatedChildTask(ChildTaskTraverseCallback childTaskCallback, Object userData)
	{
		for(String combinedKey : this.relatedChildTaskProcMap.keySet())
		{
			String[] splitedKey;
			splitedKey = combinedKey.split(TASK_PROC_MAP_SEPARATOR);
			childTaskCallback.traverseCallback(splitedKey[TASK_PROC_MAP_TASK_NAME_INDEX], Integer.parseInt(splitedKey[TASK_PROC_MAP_PROC_ID_INDEX]), 
					Integer.parseInt(splitedKey[TASK_PROC_MAP_LOCAL_ID_INDEX]), userData);	
		}
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
