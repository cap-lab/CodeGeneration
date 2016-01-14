// hs: need to delete before release
package hae.peace.container.cic.mapping.xml;

import java.io.ByteArrayInputStream;
import java.util.HashMap;
import java.util.Map;

import hae.kernel.util.ObjectList;
import hae.peace.container.cic.mapping.CICDSEPanel;
import hae.peace.container.cic.mapping.CICManualDSEPanel;
import hae.peace.container.cic.mapping.MappingTask;
import hopes.cic.exception.CICXMLException;
import hopes.cic.xml.CICAlgorithmType;
import hopes.cic.xml.CICAlgorithmTypeLoader;
import hopes.cic.xml.DataParallelType;
import hopes.cic.xml.TaskType;

public class CICAlgorithmXMLHandler {
	private CICAlgorithmTypeLoader loader;
	private CICAlgorithmType algorithm;
	
	public CICAlgorithmXMLHandler() {
		loader = new CICAlgorithmTypeLoader();
	}
	
	public void setXMLString(String xmlString, ObjectList taskList) throws CICXMLException {
		ByteArrayInputStream is = new ByteArrayInputStream(xmlString.getBytes());
		algorithm = loader.loadResource(is);
		updateTaskList(taskList);
	}
	
	public void updateTaskList(ObjectList taskList)
	{
		Map<String, DataParallelType> mapParallelType = new HashMap<String, DataParallelType>();
		
		for(TaskType taskType : algorithm.getTasks().getTask())
		{
			String taskName = taskType.getName();
			if(taskType.getDataParallel()!=null)
			{
				String key = taskName;
				mapParallelType.put(key, taskType.getDataParallel().getType());
			}
		}
		
		for(Object oSubtask : taskList)
		{
			if( oSubtask instanceof MappingTask ) {
				MappingTask task = (MappingTask)oSubtask;
				String key = task.getName();
				if(mapParallelType.containsKey(key))
					task.setParallelType(mapParallelType.get(key));
			}
		}
	}
}
