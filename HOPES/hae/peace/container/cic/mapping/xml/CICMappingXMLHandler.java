package hae.peace.container.cic.mapping.xml;

import hae.kernel.util.ObjectList;
import hae.peace.container.PeacePanel;
import hae.peace.container.PeaceTargetPanel;
import hae.peace.container.cic.mapping.CICManualDSEPanel;
import hae.peace.container.cic.mapping.MappingTask;
import hae.peace.container.cic.mapping.Processor;
import hopes.cic.exception.CICXMLException;
import hopes.cic.xml.CICMappingType;
import hopes.cic.xml.CICMappingTypeLoader;
import hopes.cic.xml.DataParallelType;
import hopes.cic.xml.MappingDeviceType;
import hopes.cic.xml.MappingLibraryType;
import hopes.cic.xml.MappingProcessorIdType;
import hopes.cic.xml.MappingTaskType;

import java.io.ByteArrayInputStream;
import java.io.StringWriter;
import java.math.BigInteger;
import java.util.Enumeration;
import java.util.List;

public class CICMappingXMLHandler extends CICXMLHandler {
	private CICMappingTypeLoader loader;
	private CICMappingType mapping;
	
	public CICMappingXMLHandler() {
		loader = new CICMappingTypeLoader();
	}

	public CICMappingType getMapping() {
		return mapping;
	}

	public void setMapping(CICMappingType mapping) {
		this.mapping = mapping;
	}

	public void setXMLString(String xmlString) throws CICXMLException {
		ByteArrayInputStream is = new ByteArrayInputStream(xmlString.getBytes());
		mapping = loader.loadResource(is);
		processed = false;		
	}

	public String getXMLString() throws CICXMLException {
		StringWriter writer = new StringWriter();
		loader.storeResource(mapping, writer);
		writer.flush();
		return writer.toString();
	}

	private ObjectList taskList = new ObjectList();
	private boolean processed = false;
	public void setTaskList(CICArchitectureXMLHandler architectureHandler) {
		if (!processed && mapping != null) {
			taskList.clear();
			for(MappingTaskType taskType: mapping.getTask()) {
				String taskName = taskType.getName();
				DataParallelType parallelType = taskType.getDataParallel() == null ? DataParallelType.NONE : 
				taskType.getDataParallel();
				MappingTask task = new MappingTask(taskName, parallelType);

				for (MappingProcessorIdType processorType : taskType.getDevice().get(0).getProcessor()) {
					String poolName = processorType.getPool();
					BigInteger localId = processorType.getLocalId();

					Processor processor = architectureHandler.getProcessor(poolName, localId);
					if (processor == null) {
						System.out.println("cannot find processor[" + poolName + ":" + localId + "]");
						continue;
					}
					task.addProcessorForce(processor);
				}
				taskList.add(task);
			}			
			//delete 2017.8.17
			//The taskList doesn't need to add LibararyTask. Because I don't want to display Librarytask on the mapping table.
			/*for( MappingLibraryType libraryType: mapping.getLibrary() ) {
				taskList.add(libraryType);
			}*/
			processed = true;
		}
	}
	
	public ObjectList getTaskList() {
		return taskList;
	}
	
	private void clearMap() {
		for (MappingTaskType task : mapping.getTask()) {
			task.getDevice().get(0).getProcessor().clear();
		}
	}

	private List<MappingProcessorIdType> getProcessIdList(String taskName) {
		for (MappingTaskType task : mapping.getTask()) {
			if (!taskName.equals(task.getName()))
				continue;
			else
				return task.getDevice().get(0).getProcessor();
		}
		return null;
	}
	
	private List<MappingDeviceType> getDeviceType(String taskName){
		for (MappingTaskType task : mapping.getTask()) {
			if (!taskName.equals(task.getName()))
				continue;
			else
				return task.getDevice();
		}
		return null;
	}
	
	public void update() {
		clearMap();
		for (Object obj : taskList) {
			if( obj instanceof MappingTask ) {
				MappingTask task = (MappingTask)obj;
				
				List<MappingDeviceType> device = getDeviceType(task.getName());
				
				List<MappingProcessorIdType> processors = getProcessIdList(task.getName());
				for (Object obj2 : task.getAssignedProcList()) {
					Processor processor = (Processor)obj2;
					MappingProcessorIdType processorType = new MappingProcessorIdType();
					processorType.setPool(processor.getName());
					processorType.setLocalId(BigInteger.valueOf(processor.getIndex()));
					processors.add(processorType);

					device.get(0).setName(processor.getParentDevice());
				}
			}
		}
	}
	
	public MappingTask findTaskByName(String taskName, CICArchitectureXMLHandler architectureHandler) {
		setTaskList(architectureHandler);
		Enumeration tasks = getTaskList().elements();
		while(tasks.hasMoreElements()){
			Object next = tasks.nextElement();
			if( next instanceof MappingTask ) {
				MappingTask task = (MappingTask)next;
				if(task.getName().equals(taskName)) {
					return task;
				}
			}
		}
		return null;
	}

	public MappingLibraryType findLibraryByName(String libraryName, CICArchitectureXMLHandler architectureHandler) {
		setTaskList(architectureHandler);
		Enumeration nodes = getTaskList().elements();
		while(nodes.hasMoreElements()){
			Object next = nodes.nextElement();
			if( next instanceof MappingLibraryType ) {
				MappingLibraryType library = (MappingLibraryType)next;
				if( library.getName().equals(libraryName) ) {
					return library;
				}
			}
		}
		return null;
	}

}
