package hae.peace.container.cic.mapping.xml;

import java.io.ByteArrayInputStream;
import java.io.StringWriter;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;

import hae.peace.container.cic.mapping.MappingDLAppTask;
import hae.peace.container.cic.mapping.MappingTask;
import hae.peace.container.cic.mapping.Processor;
import hopes.cic.exception.CICXMLException;
import hopes.cic.xml.CICMappingType;
import hopes.cic.xml.CICMappingTypeLoader;
import hopes.cic.xml.DataParallelType;
import hopes.cic.xml.MappingDLAppTaskType;
import hopes.cic.xml.MappingDeviceType;
import hopes.cic.xml.MappingMulticastType;
import hopes.cic.xml.MappingMulticastUDPType;
import hopes.cic.xml.MappingProcessorIdType;
import hopes.cic.xml.MappingTaskType;

public class CICMappingXMLHandler extends CICXMLHandler {
	private CICMappingTypeLoader loader;
	private CICMappingType mapping;
	
	public CICMappingXMLHandler() {
		loader = new CICMappingTypeLoader();
	}
	
	protected void storeResource(StringWriter writer) throws CICXMLException {
		loader.storeResource(mapping, writer);
	}
	
	protected void loadResource(ByteArrayInputStream is) throws CICXMLException {
		mapping = loader.loadResource(is);
	}
	
	@Override
	public void setXMLString(String xmlString) throws CICXMLException {
		super.setXMLString(xmlString);
		processed = false;		
	}

	public CICMappingType getMapping() {
		return mapping;
	}

	public void setMapping(CICMappingType mapping) {
		this.mapping = mapping;
	}

	private List<MappingTask> taskList = new ArrayList<MappingTask>();
	private boolean processed = false;

	private void addTaskProcessorForce(MappingTask task, MappingTaskType taskType,
			CICArchitectureXMLHandler architectureHandler) {
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
	}

	public void makeTaskList(CICArchitectureXMLHandler architectureHandler) {
		if (processed || mapping == null) {
			return;
		}
		taskList.clear();
		for (MappingTaskType taskType : mapping.getTask()) {
			String taskName = taskType.getName();
			DataParallelType parallelType = taskType.getDataParallel() == null ? DataParallelType.NONE
					: taskType.getDataParallel();
			MappingTask task = new MappingTask(taskName, parallelType);
			addTaskProcessorForce(task, taskType, architectureHandler);
			taskList.add(task);
		}
		for (MappingDLAppTaskType taskType : mapping.getDlAppTask()) {
			String taskName = taskType.getName();
			DataParallelType parallelType = taskType.getDataParallel() == null ? DataParallelType.NONE
					: taskType.getDataParallel();
			MappingDLAppTask task = new MappingDLAppTask(taskName, parallelType);
			addTaskProcessorForce(task, taskType, architectureHandler);
			taskList.add(task);
		}
		processed = true;
	}
	
	public List<MappingTask> getTaskList() {
		return taskList;
	}
	
	private void clearMap() {
		for (MappingTaskType task : mapping.getTask()) {
			task.getDevice().get(0).getProcessor().clear();
		}
		for (MappingDLAppTaskType task : mapping.getDlAppTask()) {
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

		for (MappingDLAppTaskType task : mapping.getDlAppTask()) {
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

		for (MappingTaskType task : mapping.getDlAppTask()) {
			if (!taskName.equals(task.getName()))
				continue;
			else
				return task.getDevice();
		}
		return null;
	}
	
	public void update() {
		clearMap();
		for (MappingTask obj : taskList) {
				MappingTask task = (MappingTask)obj;
				
				List<MappingDeviceType> device = getDeviceType(task.getName());
				
				List<MappingProcessorIdType> processors = getProcessIdList(task.getName());
				for (Processor processor : task.getAssignedProcList()) {
					MappingProcessorIdType processorType = new MappingProcessorIdType();
					processorType.setPool(processor.getName());
					processorType.setLocalId(BigInteger.valueOf(processor.getIndex()));
					processors.add(processorType);
					device.get(0).setName(processor.getParentDevice());
				}
		}
	}
	
	public String getXMLString(CICMappingType xmlData) throws CICXMLException{
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
		} else if (element.equals("dlAppTask")) {
			originData.getDlAppTask().clear();
			originData.getDlAppTask().addAll(mapping.getDlAppTask());
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
	
	public MappingTask findTaskByName(String taskName, CICArchitectureXMLHandler architectureHandler) {
		makeTaskList(architectureHandler);
		for (MappingTask task : getTaskList()) {
			if (task.getName().equals(taskName)) {
				return task;
			}
		}
		return null;
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
}
