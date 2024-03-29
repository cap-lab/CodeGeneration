// hs: need to delete before release
package hopes.cic.xml.handler;

import java.io.ByteArrayInputStream;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import hopes.cic.exception.CICXMLException;
import hopes.cic.xml.CICAlgorithmType;
import hopes.cic.xml.CICAlgorithmTypeLoader;
import hopes.cic.xml.ChannelListType;
import hopes.cic.xml.DataParallelType;
import hopes.cic.xml.LibraryType;
import hopes.cic.xml.LoopStructureTypeType;
import hopes.cic.xml.ModeTaskType;
import hopes.cic.xml.ModeType;
import hopes.cic.xml.PortMapListType;
import hopes.cic.xml.TaskPortType;
import hopes.cic.xml.TaskType;

public class CICAlgorithmXMLHandler extends CICXMLHandler {
	private CICAlgorithmTypeLoader loader;
	private CICAlgorithmType algorithm;
	private List<TaskType> taskList = new ArrayList<TaskType>();
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
		libraryList.clear();
		mtmtaskList.clear();
		for (TaskType task : algorithm.getTasks().getTask()) {
			taskList.add(task);
			if (task.getHasSubGraph().equalsIgnoreCase("Yes")) {
				mtmtaskList.add(task);
			}
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

	public List<TaskType> getNonHierarchicalTaskList() {
		return taskList.stream().filter(t -> !t.getHasSubGraph().equalsIgnoreCase("Yes")).collect(Collectors.toList());
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

	public boolean isInDataTypeLoop(TaskType task) {
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
	
	public int getTotalDataTypeLoopCountNum(TaskType task) {
		int loopCountMul = 1;
		while (true) {
			if (task.getName().equals(task.getParentTask())) {
				return loopCountMul;
			}
			if(task.getLoopStructure() != null && task.getLoopStructure().getType().equals(LoopStructureTypeType.DATA)){
				loopCountMul *= task.getLoopStructure().getLoopCount().intValueExact();
			}
			task = findTaskByName(task.getParentTask());
		}
	}
	
	public Map<String, DataParallelType> getMapParallelType() {
		return taskList.stream().filter(t -> t.getDataParallel() != null)
				.collect(Collectors.toMap(TaskType::getName, task -> task.getDataParallel().getType()));
	}

	public Map<String, Boolean> getLoopTaskInfo() {
		Map<String, Boolean> loopTaskLevelInfo = new HashMap<String, Boolean>();
		for (TaskType taskType : taskList) {
			if (taskType.getLoopStructure() != null
					&& taskType.getLoopStructure().getType().equals(LoopStructureTypeType.DATA)) {
				loopTaskLevelInfo.put(taskType.getName(), true);
			} else if (isInDataTypeLoop(taskType)) {
				loopTaskLevelInfo.put(taskType.getName(), false);
			}
		}
		return loopTaskLevelInfo;
	}

	public Map<TaskType, List<TaskType>> getHierarchicalDATALoopTaskMap() {
		Map<TaskType, List<TaskType>> hierarchicalTaskMap = makeHierarchicalTaskMap();
		Map<TaskType, List<TaskType>> hierarchicalDATALoopTaskMap = new HashMap<TaskType, List<TaskType>>();
		for (TaskType task : hierarchicalTaskMap.keySet()) {
			if (task.getLoopStructure() != null
					&& task.getLoopStructure().getType().equals(LoopStructureTypeType.DATA)) {
				hierarchicalDATALoopTaskMap.put(task, hierarchicalTaskMap.get(task));
			}
		}
		return hierarchicalDATALoopTaskMap;
	}

	private Map<TaskType, List<TaskType>> makeHierarchicalTaskMap() {
		Map<TaskType, List<TaskType>> hierarchicalTaskMap = new HashMap<TaskType, List<TaskType>>();
		for (TaskType task : taskList) {
			// while task has parent task
			while (!task.getParentTask().equals(task.getName())) {
				TaskType parentTask = findTaskByName(task.getParentTask());
				hierarchicalTaskMap.getOrDefault(parentTask, new ArrayList<TaskType>()).add(task);
				task = parentTask;
			}
		}
		return hierarchicalTaskMap;
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

	public ModeTaskType findModeTaskTypeByTaskNameFirst(String taskName) {
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

	public TaskPortType findTaskPortByName(TaskType task, String portName) {
		return task.getPort().stream().filter(port -> port.getName().equals(portName)).findAny()
				.orElseThrow(IllegalArgumentException::new);
	}

	public Map<String, TaskType> getFullTaskMap() {
		return taskList.stream().collect(Collectors.toMap(TaskType::getName, Function.identity()));
	}
}
	
