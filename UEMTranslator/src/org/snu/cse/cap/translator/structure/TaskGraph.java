package org.snu.cse.cap.translator.structure;

import java.util.ArrayList;

import org.snu.cse.cap.translator.structure.task.Task;

public class TaskGraph {
	private String taskGraphName;
	private ArrayList<Task> taskList;
	private TaskGraphType taskGraphType;
	private Task parentTask;
	
	public TaskGraph(String graphName) {
		this.taskList = new ArrayList<Task>();
		this.taskGraphName = graphName;
		this.parentTask = null;
	}
	
	public void putTask(Task task) {
		this.taskList.add(task);
	}
	
	public int getNumOfTasks() {
		return this.taskList.size();
	}
	
	public String getTaskGraphName() {
		return taskGraphName;
	}
	
	public void setTaskGraphName(String taskGraphName) {
		this.taskGraphName = taskGraphName;
	}
	
	public ArrayList<Task> getTaskList() {
		return taskList;
	}
	
	public void setTaskList(ArrayList<Task> taskList) {
		this.taskList = taskList;
	}
	
	public TaskGraphType getTaskGraphType() {
		return taskGraphType;
	}
	
	public void setTaskGraphType(TaskGraphType taskGraphType) {
		this.taskGraphType = taskGraphType;
	}
	
	public Task getParentTask() {
		return parentTask;
	}
	
	public void setParentTask(Task parentTask) {
		this.parentTask = parentTask;
	}
	
	
}
