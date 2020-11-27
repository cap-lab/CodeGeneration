package hopes.cic.xml.handler;

import java.io.ByteArrayInputStream;
import java.io.StringWriter;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;

import hopes.cic.exception.CICXMLException;
import hopes.cic.xml.CICMappingType;
import hopes.cic.xml.CICMappingTypeLoader;
import hopes.cic.xml.MappingDeviceType;
import hopes.cic.xml.MappingExternalTaskType;
import hopes.cic.xml.MappingLibraryType;
import hopes.cic.xml.MappingMulticastType;
import hopes.cic.xml.MappingMulticastUDPType;
import hopes.cic.xml.MappingProcessorIdType;
import hopes.cic.xml.MappingTaskType;

public class CICMappingXMLHandler extends CICXMLHandler {
	private CICMappingTypeLoader loader;
	private CICMappingType mapping;
	private List<MappingTaskType> taskList = new ArrayList<MappingTaskType>();
	private List<MappingExternalTaskType> externalTaskList = new ArrayList<MappingExternalTaskType>();

	public CICMappingXMLHandler() {
		loader = new CICMappingTypeLoader();
		mapping = new CICMappingType();
	}

	protected void storeResource(StringWriter writer) throws CICXMLException {
		loader.storeResource(mapping, writer);
	}

	protected void loadResource(ByteArrayInputStream is) throws CICXMLException {
		mapping = loader.loadResource(is);
	}

	public void init() {
		taskList.clear();
		externalTaskList.clear();
		makeMappingTaskList();
		makeMappingExternalTaskList();
	}

	public List<MappingTaskType> getTaskList() {
		return taskList;
	}

	public void setTaskList(List<MappingTaskType> taskList) {
		this.taskList = taskList;
	}
	
	public void addTask(MappingTaskType task) {
		this.taskList.add(task);
		this.mapping.getTask().add(task);
	}

	public List<MappingExternalTaskType> getExternalTaskList() {
		return externalTaskList;
	}

	public void makeMappingTaskList() {
		for (MappingTaskType task : mapping.getTask()) {
			taskList.add(task);
		}
	}

	public void makeMappingExternalTaskList() {
		for (MappingExternalTaskType task : mapping.getExternalTask()) {
			externalTaskList.add(task);
		}
	}

	public CICMappingType getMapping() {
		return mapping;
	}

	public void setMapping(CICMappingType mapping) {
		this.mapping = mapping;
	}

	public void clearMap() {
		for (MappingTaskType task : taskList) {
			task.getDevice().get(0).getProcessor().clear();
		}
		for (MappingExternalTaskType externalTask : externalTaskList) {
			for (MappingTaskType task : externalTask.getChildTask()) {
				task.getDevice().get(0).getProcessor().clear();
			}
		}
	}

	public List<MappingProcessorIdType> getProcessIdList(String taskName) {
		for (MappingTaskType task : taskList) {
			if (taskName.equals(task.getName())) {
				return task.getDevice().get(0).getProcessor();
			}
		}
		for (MappingExternalTaskType externalTask : externalTaskList) {
			for (MappingTaskType task : externalTask.getChildTask()) {
				if (taskName.equals(task.getName())) {
					return task.getDevice().get(0).getProcessor();
				}
			}
		}
		return null;
	}
	
	public List<MappingDeviceType> getDeviceType(String taskName) {
		for (MappingTaskType task : taskList) {
			if (!taskName.equals(task.getName()))
				continue;
			else
				return task.getDevice();
		}
		for (MappingExternalTaskType externalTask : externalTaskList) {
			for (MappingTaskType task : externalTask.getChildTask()) {
				if (taskName.equals(task.getName())) {
					return task.getDevice();
				}
			}
		}
		return null;
	}
	
	public String getXMLString(CICMappingType xmlData) throws CICXMLException {
		StringWriter writer = new StringWriter();
		loader.storeResource(xmlData, writer);
		writer.flush();
		return writer.toString();
	}

	public boolean updateXMLFile(String fileName, String element) throws CICXMLException {
		CICMappingType originData = loader.loadResource(new ByteArrayInputStream(getLocalFile(fileName).getBytes()));
		if(element.equals("task")) {
			originData.getTask().clear();
			originData.getTask().addAll(mapping.getTask());
		} else if (element.equals("externalTask")) {
			originData.getExternalTask().clear();
			originData.getExternalTask().addAll(mapping.getExternalTask());
		}
		else if (element.equals("library")) {
			originData.getLibrary().clear();
			originData.getLibrary().addAll(mapping.getLibrary());
		} else if(element.equals("multicast")) {
			originData.getMulticast().clear();
			originData.getMulticast().addAll(mapping.getMulticast());
		} else {
			return false;
		}
		
		return putLocalFile(fileName, getXMLString(originData));
	}
	
	public ArrayList<String> getGroupList() {
		ArrayList<String> multicastGroups = new ArrayList<String>();
		for(MappingMulticastType group : mapping.getMulticast()) {
			multicastGroups.add(group.getGroupName());
		}
		return multicastGroups;
	}
	
	private MappingMulticastType findGroup(String groupName) {
		for(MappingMulticastType group : mapping.getMulticast()) {
			if(group.getGroupName().equals(groupName)) {
				return group;
			}
		}
		return null;
	}
	
	public ArrayList<String> getChoosedConnectionList(String groupName) {
		ArrayList<String> choosedConnections = new ArrayList<String>();
		MappingMulticastType selectedGroup = findGroup(groupName);
		
		if(selectedGroup == null) {
			return choosedConnections;
		}
		if(selectedGroup.getConnectionType().getUDP() != null) {
			choosedConnections.add("UDP");
		}
		return choosedConnections;
	}
	
	public void addMulticastConnection(String groupName, String connection) {
		MappingMulticastType selectedGroup = findGroup(groupName);
		if(selectedGroup == null) {
			return;
		}
		if(connection.equals("UDP")) {
			MappingMulticastUDPType udp = new MappingMulticastUDPType();
			udp.setIp("255.255.255.255");
			udp.setPort(new BigInteger("8080"));
			selectedGroup.getConnectionType().setUDP(udp);
		}
	}
	
	public void removeMulticastConnection(String groupName, String connection) {
		MappingMulticastType selectedGroup = findGroup(groupName);
		if(selectedGroup == null) {
			return;
		}
		if(connection.equals("UDP")) {
			selectedGroup.getConnectionType().setUDP(null);;
		}
	}
	
	public  MappingMulticastUDPType getUDPConnectionInfo(String groupName) {
		MappingMulticastType selectedGroup = findGroup(groupName);
		if(selectedGroup == null) {
			return null;
		}
		return selectedGroup.getConnectionType().getUDP();
	}
	
	public void xmlMulticastUpdated(String groupName, String connection, String infoType, String Content) {
		MappingMulticastType selectedGroup = findGroup(groupName);
		if(connection.equals("UDP")) {
			if(infoType.equals("ip")){
				selectedGroup.getConnectionType().getUDP().setIp(Content);
			} else {
				selectedGroup.getConnectionType().getUDP().setPort(new BigInteger(Content));
			}
		}
	}

	public MappingTaskType getTaskMappingByTaskName(String taskName) {
		MappingTaskType taskMapping = taskList.stream().filter(it -> it.getName().equals(taskName)).findFirst()
				.orElse(null);
		if (taskMapping == null) {
			taskMapping = externalTaskList.stream().map(MappingExternalTaskType::getChildTask).flatMap(List::stream)
					.filter(extTaskMapping -> extTaskMapping.getName().equals(taskName)).findFirst().orElse(null);
		}
		return taskMapping;
	}

	public List<MappingMulticastType> getMappingMulticastTypeList() {
		return mapping.getMulticast();
	}

	public void addExternalTask(MappingExternalTaskType mappingExternalTaskType) {
		this.externalTaskList.add(mappingExternalTaskType);
		this.mapping.getExternalTask().add(mappingExternalTaskType);
	}

	public void addLibrary(MappingLibraryType mappingLibrary) {
		mapping.getLibrary().add(mappingLibrary);
	}
}
