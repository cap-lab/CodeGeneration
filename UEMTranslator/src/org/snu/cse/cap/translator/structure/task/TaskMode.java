package org.snu.cse.cap.translator.structure.task;

import java.util.HashMap;
import java.util.HashSet;

public class TaskMode {
	private int id;
	private String name;
	private HashSet<Task> relatedChildTaskSet;
	private HashMap<String, String> relatedChildTaskProcMap;
	private static final String TASK_PROC_MAP_SEPARATOR = "/";
	private static final int TASK_PROC_MAP_PROC_ID_INDEX = 0;
	private static final int TASK_PROC_MAP_LOCAL_ID_INDEX = 1;
	private static final int TASK_PROC_MAP_TASK_NAME_INDEX = 2;
	
	private HashMap<Integer, HashMap<String, String>> taskProcMapWithThroughput; // key: throughput constraint, value: (procKey : Task name) 
	
	public TaskMode(int modeId, String modeName) {
		this.id = modeId;
		this.name = modeName;
		this.relatedChildTaskSet = new HashSet<Task>();
		this.relatedChildTaskProcMap = new HashMap<String, String>();
		this.taskProcMapWithThroughput = new HashMap<Integer, HashMap<String, String>>();
	}
	
	public interface ChildTaskTraverseCallback<T> {
		 public void traverseCallback(String taskName, int procId, int procLocalId, T userData);
	}

	
	public void putChildTask(int procId, int procLocalId, int throughputConstraint, Task task)
	{
		HashMap<String, String> taskProcMap = null;
		String procMapKey = null;
		
		procMapKey = procId + TASK_PROC_MAP_SEPARATOR  + procLocalId + TASK_PROC_MAP_SEPARATOR + task.getName();
		
		this.relatedChildTaskSet.add(task);
		this.relatedChildTaskProcMap.put(procMapKey, task.getName());
		
		taskProcMap = this.taskProcMapWithThroughput.get(new Integer(throughputConstraint));
		if(taskProcMap == null)
		{
			taskProcMap = new HashMap<String, String>();
			this.taskProcMapWithThroughput.put(new Integer(throughputConstraint), taskProcMap);
		}
		taskProcMap.put(procMapKey, task.getName());
	}
	
	public <T> void traverseRelatedChildTask(Integer throughputConstraint, ChildTaskTraverseCallback<T> childTaskCallback, T userData)
	{
		HashMap<String, String> taskProcMap = this.taskProcMapWithThroughput.get(throughputConstraint);
		for(String combinedKey : taskProcMap.keySet())
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

	public HashSet<Task> getRelatedChildTaskSet() {
		return relatedChildTaskSet;
	}

	public HashMap<Integer, HashMap<String, String>> getTaskProcMapWithThroughput() {
		return taskProcMapWithThroughput;
	}
}

