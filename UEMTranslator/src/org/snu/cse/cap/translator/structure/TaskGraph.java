package org.snu.cse.cap.translator.structure;

import java.util.ArrayList;

import org.snu.cse.cap.translator.structure.task.Task;

public class TaskGraph {
	private String name;
	private ArrayList<Task> taskList;
	private TaskGraphType taskGraphType;
	private Task parentTask;
	
	public TaskGraph(String graphName) {
		this.taskList = new ArrayList<Task>();
		this.name = graphName;
		this.parentTask = null;
		this.taskGraphType = TaskGraphType.PROCESS_NETWORK;
	}
	
	public TaskGraph(String graphName, String taskGraphType) {
		this.taskList = new ArrayList<Task>();
		this.name = graphName;
		this.parentTask = null;
		if(taskGraphType != null)
		{
			this.taskGraphType = TaskGraphType.fromValue(taskGraphType);	
		}
		else
		{
			this.taskGraphType = TaskGraphType.PROCESS_NETWORK;
		}
	}
	
	public void putTask(Task task) {
		this.taskList.add(task);
	}
	
	public int getNumOfTasks() {
		return this.taskList.size();
	}
	
	public String getName() {
		return name;
	}
	
	public void setName(String taskGraphName) {
		this.name = taskGraphName;
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
	
	public TaskGraph clone()
	{
		TaskGraph clonedGraph;
		
		clonedGraph = new TaskGraph(this.name);
		clonedGraph.setTaskGraphType(this.taskGraphType);
		clonedGraph.setParentTask(this.parentTask);
		
		for(Task task : this.taskList)
		{
			clonedGraph.putTask(task);
		}
		
		return clonedGraph;
	}
	
	public void mergeChildTaskGraph(TaskGraph subgraph)
	{
		// remove task with "subgraph name"
		for(Task subGraphParentTask : this.taskList)
		{
			if(subGraphParentTask.getName().equals(subgraph.getName()) == true)
			{
				this.taskList.remove(subGraphParentTask);
				break;
			}
		}
		
		// put tasks in subgraph
		for(Task subgraphTask : subgraph.getTaskList())
		{
			this.taskList.add(subgraphTask);
		}
	}
}
