// hs: need to delete before release
package hopes.cic.xml.handler;

import java.io.ByteArrayInputStream;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import hopes.cic.exception.CICXMLException;
import hopes.cic.xml.CICAlgorithmType;
import hopes.cic.xml.CICAlgorithmTypeLoader;
import hopes.cic.xml.ChannelListType;
import hopes.cic.xml.DataParallelType;
import hopes.cic.xml.ExternalTaskType;
import hopes.cic.xml.LibraryType;
import hopes.cic.xml.LoopStructureTypeType;
import hopes.cic.xml.ModeTaskType;
import hopes.cic.xml.ModeType;
import hopes.cic.xml.PortMapListType;
import hopes.cic.xml.TaskType;

public class CICAlgorithmXMLHandler extends CICXMLHandler {
	private CICAlgorithmTypeLoader loader;
	private CICAlgorithmType algorithm;
	private List<TaskType> taskList = new ArrayList<TaskType>();
	private List<ExternalTaskType> externalTaskList = new ArrayList<ExternalTaskType>();
	private List<LibraryType> libraryList = new ArrayList<LibraryType>();
	private List<TaskType> mtmtaskList = new ArrayList<TaskType>();
	
	public CICAlgorithmXMLHandler() {
		loader = new CICAlgorithmTypeLoader();
		algorithm = new CICAlgorithmType();
	}

	protected void storeResource(StringWriter writer) throws CICXMLException {
		loader.storeResource(algorithm, writer);
	}

	protected void loadResource(ByteArrayInputStream is) throws CICXMLException {
		algorithm = loader.loadResource(is);
	}

	public void init() {
		taskList.clear();
		externalTaskList.clear();
		libraryList.clear();
		mtmtaskList.clear();
		for (TaskType task : algorithm.getTasks().getTask()) {
			taskList.add(task);
			if (task.getHasSubGraph().equalsIgnoreCase("Yes")) {
				mtmtaskList.add(task);
			}
		}
		for (ExternalTaskType externalTask : algorithm.getTasks().getExternalTask()) {
			externalTaskList.add(externalTask);
		}
		if (algorithm.getLibraries() != null) {
			for (LibraryType library : algorithm.getLibraries().getLibrary()) {
				libraryList.add(library);
			}
		}
	}

	public List<TaskType> getTaskList() {
		return taskList;
	}

	public List<ExternalTaskType> getExternalTaskList() {
		return externalTaskList;
	}

	public List<LibraryType> getLibraryList() {
		return libraryList;
	}

	public List<TaskType> getMTMTaskList() {
		return mtmtaskList;
	}

	public PortMapListType getPortMaps() {
		return algorithm.getPortMaps();
	}

	public ChannelListType getChannels() {
		return algorithm.getChannels();
	}

	public String getProperty() {
		return algorithm.getProperty();
	}

	public CICAlgorithmType getAlgorithm()
	{
		return algorithm;
	}

	private boolean isInDataTypeLoop(TaskType task) {
		while (true) {
			if(task.getLoopStructure() != null && task.getLoopStructure().getType().equals(LoopStructureTypeType.DATA)){
				return true;
			} else if (task.getName().equals(task.getParentTask())) {
				return false;
			} else {
				task = findTaskByName(task.getParentTask());
			}
		}
	}
	
	public Map<String, DataParallelType> getMapParallelType() {
		Map<String, DataParallelType> mapParallelType = new HashMap<String, DataParallelType>();
		List<TaskType> parallelTaskList = taskList.stream().filter(t -> t.getDataParallel() != null)
				.collect(Collectors.toList());
		parallelTaskList.forEach(t -> mapParallelType.put(t.getName(), t.getDataParallel().getType()));
		return mapParallelType;
	}

	public Map<String, LoopStructureTypeType> getMapLoopType() {
		Map<String, LoopStructureTypeType> mapLoopType = new HashMap<String, LoopStructureTypeType>();
		for (TaskType taskType : taskList) {
			if (isInDataTypeLoop(taskType)) {
				mapLoopType.put(taskType.getName(), LoopStructureTypeType.DATA);
			}
		}
		return mapLoopType;
	}
	
	private Map<String, ArrayList<String>> makeHierarchicalTaskMap()
	{
		Map<String, ArrayList<String>> hierarchicalTaskMap = new HashMap<String, ArrayList<String>>();
		//make hierarchicalTaskMap.
		for (TaskType task : taskList)
		{
			String subTaskName = task.getName();
			TaskType currentTaskType = task;
			while(!currentTaskType.getParentTask().equals(currentTaskType.getName())) //while task has parent task.
			{
				TaskType parentTaskTaskType = findTaskByName(currentTaskType.getParentTask());				

				if(hierarchicalTaskMap.containsKey(parentTaskTaskType.getName()))
				{
					hierarchicalTaskMap.get(parentTaskTaskType.getName()).add(subTaskName);
				}
				else
				{
					ArrayList<String> newSubTasksList = new ArrayList<String>();
					newSubTasksList.add(subTaskName);
					hierarchicalTaskMap.put(parentTaskTaskType.getName(), newSubTasksList);
				}
				currentTaskType = parentTaskTaskType;
			}
		}
		
		return hierarchicalTaskMap;
	}		
	
	public Map<String, ArrayList<String>> getHierarchicalDATALoopAndSubTasksMap() 
	{		
		Map<String, ArrayList<String>> hierarchicalTaskMap = makeHierarchicalTaskMap();
		Map<String, ArrayList<String>> hierarchicalDATALoopAndSubTasksMap = new HashMap<String, ArrayList<String>>();
			
		for(String hierarchicalTaskName : hierarchicalTaskMap.keySet())
		{
			TaskType hierarchicalTaskType = findTaskByName(hierarchicalTaskName);
			if(hierarchicalTaskType.getLoopStructure() != null && hierarchicalTaskType.getLoopStructure().getType().equals(LoopStructureTypeType.DATA))
			{
				hierarchicalDATALoopAndSubTasksMap.put(hierarchicalTaskName, hierarchicalTaskMap.get(hierarchicalTaskName));
			}
		}	
		return hierarchicalDATALoopAndSubTasksMap;
	}

	public ModeTaskType findModeTaskTypeByTaskName(String taskName) {
		for(ModeType mode : algorithm.getModes().getMode()) {
			for (ModeTaskType modeTask : mode.getTask()) {
				if (modeTask.getName().equals(taskName)) {
					return modeTask;
				}
			}
		}
		throw new RuntimeException("Error : modeTask not found. " + taskName);
	}

	public ModeTaskType findModeTaskTypeByTaskNameInFirstIndex(String taskName) {
		for (ModeTaskType mt : algorithm.getModes().getMode().get(0).getTask()) {
			if (mt.getName().equals(taskName)) {
				return mt;
			}
		}
		return null;
	}

	public List<ModeTaskType> findModeTaskTypeListByTaskNameInFirstIndex(String taskName) {
		return algorithm.getModes().getMode().get(0).getTask().stream()
				.filter(mt -> mt.getName().equals(taskName))
				.collect(Collectors.toList());
	}

	public TaskType findTaskByName(String taskName) {
		return taskList.stream().filter(t -> t.getName().equals(taskName)).findFirst().orElse(null);
	}
}
	
