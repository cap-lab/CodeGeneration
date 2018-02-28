package org.snu.cse.cap.translator.structure.task;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import javax.management.modelmbean.InvalidTargetObjectTypeException;

import org.snu.cse.cap.translator.Constants;
import org.snu.cse.cap.translator.structure.library.Library;

import hopes.cic.xml.LibraryMasterPortType;
import hopes.cic.xml.LoopStructureType;
import hopes.cic.xml.MTMConditionType;
import hopes.cic.xml.MTMModeType;
import hopes.cic.xml.MTMTransitionType;
import hopes.cic.xml.MTMType;
import hopes.cic.xml.MTMVariableType;
import hopes.cic.xml.ModeTaskType;
import hopes.cic.xml.TaskParameterType;
import hopes.cic.xml.TaskType;

enum TaskRunCondition {
	DATA_DRIVEN("data-driven"),
	TIME_DRIVEN("time-driven"),
	CONTROL_DRIVEN("control-driven"),
	;
	
	private final String value;
	
	private TaskRunCondition(String value) {
		this.value = value;
	}
	
	@Override
	public String toString() {
		return value;
	}
	
	public static TaskRunCondition fromValue(String value) {
		 for (TaskRunCondition c : TaskRunCondition.values()) {
			 if (c.value.equals(value)) {
				 return c;
			 }
		 }
		 throw new IllegalArgumentException(value.toString());
	}
}

public class Task {
	private int id;
	private String name;
	private TaskShapeType type;
	private int taskFuncNum;
	private int runRate;
	private int period;
	private TimeMetric periodMetric;
	private String parentTaskGraphName;
	private int inGraphIndex;
	private String childTaskGraphName;
	private TaskModeTransition modeTransition;
	private TaskLoop loopStruct;
	private ArrayList<TaskParameter> taskParamList;
	private boolean staticScheduled;
	private TaskRunCondition runCondition;
	private String taskCodeFile;
	private HashMap<String, Library> masterPortToLibraryMap;
	private HashSet<String> extraHeaderSet;
	private HashSet<String> extraSourceSet;
	private String ldFlags;
	
	public Task(int id, TaskType xmlTaskData)
	{
		this.taskParamList = new ArrayList<TaskParameter>();
		this.loopStruct = null;
		this.modeTransition = null;
		this.staticScheduled = false; // default is false
		this.masterPortToLibraryMap = new HashMap<String, Library>();
		this.ldFlags = null;
		this.extraHeaderSet = new HashSet<String>();
		this.extraSourceSet = new HashSet<String>();
	
		setId(id);
		setName(xmlTaskData.getName());
		setType(xmlTaskData.getTaskType(), xmlTaskData.getLoopStructure());
		setParentTaskGraphName(xmlTaskData.getParentTask());
		setRunCondition(xmlTaskData.getRunCondition().value());
		setTaskCodeFile(xmlTaskData.getFile(), xmlTaskData.getHasSubGraph());
		setParameters(xmlTaskData.getParameter());
		setLoop(xmlTaskData.getLoopStructure());
		setModeTransition(xmlTaskData.getMtm(), xmlTaskData.getHasMTM());
		setExtraHeaderSet(xmlTaskData.getExtraHeader());
		setExtraSourceSet(xmlTaskData.getExtraSource());
		
		if(xmlTaskData.getLdflags() != null)
		{
			this.ldFlags = xmlTaskData.getLdflags();
		}
	}
	
	private void setExtraHeaderSet(List<String> extraHeaderList)
	{
		for(String extraHeaderFile: extraHeaderList)
		{
			this.extraHeaderSet.add(extraHeaderFile);
		}
	}
	
	
	private void setExtraSourceSet(List<String> extraSourceList)
	{
		for(String extraSourceFile: extraSourceList)
		{
			this.extraSourceSet.add(extraSourceFile);
		}
	}
		
	private void setModeTransition(MTMType mtm, String hasMTM) 
	{
		int loop = 0;
		if(mtm != null && hasMTM.equals(Constants.XML_YES))
		{
			this.modeTransition = new TaskModeTransition(this.id);
			
			for(MTMModeType mtmMode: mtm.getModeList().getMode())
			{
				this.modeTransition.putMode(loop, mtmMode.getName());
				loop++;
			}
			
			for(MTMTransitionType mtmTransition: mtm.getTransitionList().getTransition()) 
			{
				ArrayList<Condition> conditionList = new ArrayList<Condition>();
				for(MTMConditionType mtmCondition : mtmTransition.getConditionList().getCondition())
				{
					Condition cond = new Condition(mtmCondition.getVariable(), mtmCondition.getValue(), 
													mtmCondition.getComparator());
					conditionList.add(cond);
				}
				
				this.modeTransition.putTransition(mtmTransition.getSrcMode(), mtmTransition.getDstMode(), conditionList);
			}
			
			for(MTMVariableType mtmVariable : mtm.getVariableList().getVariable())
			{
				this.modeTransition.putVariable(mtmVariable.getName(), mtmVariable.getType());
			}
		}
	}
	
	private void setLoop(LoopStructureType loopStruct) 
	{
		if(loopStruct != null) 
		{
			this.loopStruct = new TaskLoop(loopStruct.getType().value(), loopStruct.getLoopCount().intValue(), 
								loopStruct.getDesignatedTask());
		}
	}
	
	private void setParameters(List<TaskParameterType> paramList) 
	{
		for(TaskParameterType param : paramList)
		{
			TaskParameter taskParam;
			
			try {
				if(ParameterType.fromValue(param.getType()) == ParameterType.DOUBLE)
				{
					taskParam = new TaskDoubleParameter(param.getName(), Double.parseDouble(param.getValue()));
				}
				else if(ParameterType.fromValue(param.getType()) == ParameterType.INT)
				{
					taskParam = new TaskIntegerParameter(param.getName(), Integer.parseInt(param.getValue()));
				}
				else{
					throw new InvalidTargetObjectTypeException();
				}
				
				this.taskParamList.add(taskParam);
			}
			catch(InvalidTargetObjectTypeException e) {
				e.printStackTrace();
			}
			
		}
	}
	
	public void setExtraInformationFromModeInfo(ModeTaskType modeTaskInfo) 
	{
		if(modeTaskInfo.getRunRate() != null)
		{
			this.runRate = modeTaskInfo.getRunRate().intValue();
		}
		else
		{
			this.runRate = 1;
		}
		
		if(modeTaskInfo.getPeriod() != null)
		{
			this.period = modeTaskInfo.getPeriod().getValue().intValue();
			this.periodMetric = TimeMetric.fromValue(modeTaskInfo.getPeriod().getMetric().value());
		}
		else
		{
			this.period = 1;
			this.periodMetric = TimeMetric.MILLISEC;
		}
			
	}
	
	public int getId() {
		return id;
	}
	
	public void setId(int taskId) {
		this.id = taskId;
	}
	
	public String getName() {
		return name;
	}
	
	public void setName(String taskName) {
		this.name = taskName;
	}
	
	public TaskShapeType getType() {
		return type;
	}
	
	public void setType(String taskType, LoopStructureType loopStructure) {
		
		if(TaskShapeType.fromValue(taskType) == TaskShapeType.LOOP)
		{
			this.type = TaskShapeType.LOOP;		
		}
		else if(TaskShapeType.fromValue(taskType) == TaskShapeType.CONTROL) 
		{
			this.type = TaskShapeType.CONTROL;
		}
		else if(TaskShapeType.fromValue(taskType) == TaskShapeType.COMPUTATIONAL)
		{
			this.type = TaskShapeType.COMPUTATIONAL;
		}
	}
	
	public int getTaskFuncNum() {
		return taskFuncNum;
	}
	
	public void setTaskFuncNum(int taskFuncNum) {
		this.taskFuncNum = taskFuncNum;
	}
	
	public int getRunRate() {
		return runRate;
	}
	
	public void setRunRate(int runRate) {
		this.runRate = runRate;
	}
	
	public int getPeriod() {
		return period;
	}
	
	public void setPeriod(int period) {
		this.period = period;
	}
	
	public TimeMetric getPeriodMetric() {
		return periodMetric;
	}
	
	public void setPeriodMetric(TimeMetric periodMetric) {
		this.periodMetric = periodMetric;
	}
	
	public String getParentTaskGraphName() {
		return parentTaskGraphName;
	}
	
	public void setParentTaskGraphName(String parentTaskGraphName) {
		if(parentTaskGraphName.equals(this.name))
		{
			this.parentTaskGraphName = Constants.TOP_TASKGRAPH_NAME;
		}
		else
		{
			this.parentTaskGraphName = parentTaskGraphName;	
		}
	}
	
	public int getInGraphIndex() {
		return inGraphIndex;
	}
	
	public void setInGraphIndex(int inGraphIndex) {
		this.inGraphIndex = inGraphIndex;
	}
	
	public String getChildTaskGraphName() {
		return childTaskGraphName;
	}
	
	public TaskModeTransition getModeTransition() {
		return modeTransition;
	}
	
	public TaskLoop getLoopStruct() {
		return loopStruct;
	}
	
	public void setLoopStruct(TaskLoop loop) {
		this.loopStruct = loop;
	}
	
	public boolean isStaticScheduled() {
		return staticScheduled;
	}
	
	public void setStaticScheduled(boolean staticScheduled) {
		this.staticScheduled = staticScheduled;
	}

	public String getRunCondition() {
		return runCondition.name();
	}

	public void setRunCondition(String runCondition) {
		this.runCondition = TaskRunCondition.fromValue(runCondition);
	}

	public String getTaskCodeFile() {
		return taskCodeFile;
	}

	public void setTaskCodeFile(String cicFile, String hasSubgraph) {
		this.taskCodeFile = cicFile;
		
		if(this.taskCodeFile.endsWith(Constants.XML_PREFIX) == true && hasSubgraph.equals(Constants.XML_YES))
		{
			// Task has subgraph
			// Because the parent task is "this" task, childTaskGraphName is same to "this" task's name
			this.childTaskGraphName = this.name;
			this.taskFuncNum = 0;
		}
		else // No subgraph
		{
			this.childTaskGraphName = null;
			this.taskFuncNum = 1;
		}
	}
	
	public ArrayList<TaskParameter> getTaskParamList() {
		return taskParamList;
	}

	public HashMap<String, Library> getMasterPortToLibraryMap() {
		return masterPortToLibraryMap;
	}

	public HashSet<String> getExtraHeaderSet() {
		return extraHeaderSet;
	}

	public String getLdFlags() {
		return ldFlags;
	}

	public void setLdFlags(String ldFlags) {
		this.ldFlags = ldFlags;
	}

	public HashSet<String> getExtraSourceSet() {
		return extraSourceSet;
	}
}
