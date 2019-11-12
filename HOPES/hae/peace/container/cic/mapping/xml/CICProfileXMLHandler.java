package hae.peace.container.cic.mapping.xml;

import java.io.ByteArrayInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.StringWriter;
import java.math.BigInteger;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import hae.kernel.util.ObjectList;
import hae.peace.container.PeaceTargetPanel;
import hae.peace.container.cic.mapping.CICDSEPanel;
import hae.peace.container.cic.mapping.CICManualDSEPanel;
import hae.peace.container.cic.mapping.MappingTask;
import hae.peace.container.cic.mapping.Processor;
import hopes.cic.exception.CICXMLException;
import hopes.cic.xml.ArchitectureElementCategoryType;
import hopes.cic.xml.ArchitectureElementType;
import hopes.cic.xml.ArchitectureElementTypeType;
import hopes.cic.xml.CICAlgorithmType;
import hopes.cic.xml.CICAlgorithmTypeLoader;
import hopes.cic.xml.CICProfileType;
import hopes.cic.xml.CICProfileTypeLoader;
import hopes.cic.xml.DataParallelType;
import hopes.cic.xml.MappingLibraryType;
import hopes.cic.xml.MappingProcessorIdType;
import hopes.cic.xml.MappingTaskType;
import hopes.cic.xml.ProfileTaskType;
import hopes.cic.xml.TaskType;

public class CICProfileXMLHandler extends CICXMLHandler{
	private CICProfileTypeLoader loader;
	private CICProfileType profile;
	
	public CICProfileXMLHandler() {
		loader = new CICProfileTypeLoader();
	}
	
	protected void storeResource(StringWriter writer) throws CICXMLException {
		loader.storeResource(profile, writer);
	}
	protected void loadResource(ByteArrayInputStream is) throws CICXMLException {
		profile = loader.loadResource(is);
	}
		
	public CICProfileType getProfile() 
	{
		return profile;		
	}
	

	private ObjectList taskList = new ObjectList();
	private boolean processed = false;
	public ObjectList getTaskList() {
		taskList.clear();
		for(ProfileTaskType taskType: profile.getTask()) {
			String taskName = taskType.getName();
			ProfileTaskType task = new ProfileTaskType();
			task.setName(taskName);
			taskList.add(task);
		}

		return taskList;
	}
	
	public ProfileTaskType findTaskByName(String taskName) {
		Enumeration tasks = getTaskList().elements();
		while(tasks.hasMoreElements()){
			Object next = tasks.nextElement();
			if( next instanceof ProfileTaskType ) {
				ProfileTaskType task = (ProfileTaskType)next;
				if(task.getName().equals(taskName)) {
					return task;
				}
			}
		}
		return null;
	}
	
}
