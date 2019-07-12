// hs: need to delete before release
package hae.peace.container.cic.mapping.xml;

import java.io.ByteArrayInputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import hae.kernel.util.ObjectList;
import hae.peace.container.cic.mapping.MappingTask;
import hopes.cic.exception.CICXMLException;
import hopes.cic.xml.CICAlgorithmType;
import hopes.cic.xml.CICAlgorithmTypeLoader;
import hopes.cic.xml.DataParallelType;
import hopes.cic.xml.LoopStructureTypeType;
import hopes.cic.xml.TaskType;

public class CICAlgorithmXMLHandler extends CICXMLHandler {
	private CICAlgorithmTypeLoader loader;
	private CICAlgorithmType algorithm;
	
	public CICAlgorithmXMLHandler() {
		loader = new CICAlgorithmTypeLoader();
	}
	
	@Override
	public void setXMLString(String xmlString) throws CICXMLException {
		ByteArrayInputStream is = new ByteArrayInputStream(xmlString.getBytes());
		algorithm = loader.loadResource(is);
	}
	
	private HashMap<String, TaskType> getFullTaskMap(List<TaskType> taskList)
	{
		HashMap<String, TaskType> taskMap = new HashMap<String, TaskType>();
		for(TaskType task : taskList)
		{
			taskMap.put(task.getName(), task);
		}
		
		return taskMap;
	}
	
	private boolean isInDataTypeLoop(String taskName, HashMap<String, TaskType> taskMap)
	{
		boolean isDataTypeLoop = false;
		TaskType task;
		String currentTaskName;
		
		task = taskMap.get(taskName);
		
		do {
			currentTaskName = task.getName();
			
			if(task.getLoopStructure() != null && task.getLoopStructure().getType() == LoopStructureTypeType.DATA)
			{
				isDataTypeLoop = true;
				break;
			}
					
			task = taskMap.get(task.getParentTask());
			
		}while(task.getName().equals(currentTaskName) == false);
		
		return isDataTypeLoop;
	}
	
	
	public void updateTaskList(ObjectList taskList)
	{
		Map<String, DataParallelType> mapParallelType = new HashMap<String, DataParallelType>();
		Map<String, LoopStructureTypeType> mapLoopType = new HashMap<String, LoopStructureTypeType>();
		HashMap<String, TaskType> taskMap;
		
		taskMap = getFullTaskMap(algorithm.getTasks().getTask());
		
		for(TaskType taskType : algorithm.getTasks().getTask())
		{
			String taskName = taskType.getName();
			if(taskType.getDataParallel()!=null)
			{
				String key = taskName;
				mapParallelType.put(key, taskType.getDataParallel().getType());
			}
			
			if(isInDataTypeLoop(taskName, taskMap) == true)
			{
				String key = taskName;
				mapLoopType.put(key, LoopStructureTypeType.DATA);
			}
		}
		
		for(Object oSubtask : taskList)
		{
			if( oSubtask instanceof MappingTask ) {
				MappingTask task = (MappingTask)oSubtask;
				String key = task.getName();
				if(mapParallelType.containsKey(key))
					task.setParallelType(mapParallelType.get(key));
				if(mapLoopType.containsKey(key))
					task.setLoopType(mapLoopType.get(key));
			}
		}
	}
}
