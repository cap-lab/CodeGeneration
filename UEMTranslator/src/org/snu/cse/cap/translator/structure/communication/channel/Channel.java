package org.snu.cse.cap.translator.structure.communication.channel;

import java.util.ArrayList;
import java.util.HashMap;

import org.snu.cse.cap.translator.Constants;
import org.snu.cse.cap.translator.structure.communication.CommunicationType;
import org.snu.cse.cap.translator.structure.communication.InMemoryAccessType;
import org.snu.cse.cap.translator.structure.device.DeviceCommunicationType;
import org.snu.cse.cap.translator.structure.mapping.MappingInfo;
import org.snu.cse.cap.translator.structure.task.Task;
import org.snu.cse.cap.translator.structure.task.TaskLoopType;

public class Channel implements Cloneable {
	private int index;
	private CommunicationType communicationType;
	private ConnectionRoleType connectionRoleType;
	private DeviceCommunicationType remoteMethodType;
	private InMemoryAccessType accessType;
	private ChannelArrayType channelType;
	private int size;
	private ChannelPort inputPort; // the most outer port is set here
	private int  inputPortIndex; // port index used in channel data generation
	private ChannelPort outputPort; // the most outer port is set here
	private int  outputPortIndex; // port index used in channel data generation
	private int initialDataLen;
	private int nextChannelIndex;
	private int channelSampleSize;
	private int socketInfoIndex;
	private int processerId;

	public int getProcesserId() {
		return processerId;
	}

	public void setProcesserId(int processerId) {
		this.processerId = processerId;
	}

	public Channel(int index, int size, int initialDataLen, int sampleSize) {
		this.size = size;
		this.index = index;
		this.channelType = ChannelArrayType.GENERAL;
		this.initialDataLen = initialDataLen;
		this.nextChannelIndex = Constants.INVALID_ID_VALUE;
		this.channelSampleSize = sampleSize;
		this.socketInfoIndex = Constants.INVALID_ID_VALUE;
		this.inputPortIndex = Constants.INVALID_VALUE;
		this.outputPortIndex = Constants.INVALID_VALUE;
		this.remoteMethodType = DeviceCommunicationType.NONE;
		this.connectionRoleType = ConnectionRoleType.NONE;

	}

	// Does not need to clone inputPort and outputPort
	@Override
	public Channel clone() throws CloneNotSupportedException {
		Channel channel;

		channel = (Channel) super.clone();
		channel.index = this.index;
		channel.communicationType = this.communicationType;
		channel.accessType = this.accessType;
		channel.channelType = this.channelType;
		channel.size = this.size;

		channel.initialDataLen = this.initialDataLen;
		channel.nextChannelIndex = this.nextChannelIndex;
		channel.channelSampleSize = this.channelSampleSize;
		channel.socketInfoIndex = this.socketInfoIndex;
		channel.remoteMethodType = this.remoteMethodType;
		channel.connectionRoleType = this.connectionRoleType;

		// Shallow copy for these two objects
		channel.inputPort = this.inputPort;
		channel.outputPort = this.outputPort;

		return channel;
	}

	public int getIndex() {
		return index;
	}

	public CommunicationType getCommunicationType() {
		return communicationType;
	}

	public ChannelArrayType getChannelType() {
		return channelType;
	}

	public int getSize() {
		return size;
	}

	public void setIndex(int channelIndex) {
		this.index = channelIndex;
	}

	public void setCommunicationType(CommunicationType communicationType) {
		this.communicationType = communicationType;
	}

	public void setChannelType(ChannelArrayType channelType) {
		this.channelType = channelType;
	}

	public void setSize(int channelSize) {
		this.size = channelSize;
	}

	public ChannelPort getInputPort() {
		return inputPort;
	}

	public ChannelPort getOutputPort() {
		return outputPort;
	}

	public void setInputPort(ChannelPort inputPort) {
		this.inputPort = inputPort;

		// update initial data length depending on port sample rate
		/*
		if(inputPort.getPortSampleRateType() == PortSampleRateType.FIXED)
		{
			this.initialDataLen = this.initialDataLen * inputPort.getPortSampleRateList().get(0).getSampleRate();
		}
		else if(inputPort.getPortSampleRateType() == PortSampleRateType.MULTIPLE)
		{
			// TODO: how can I get port sample rate from MTM task?
			System.out.println("Initial data cannot be unknown at this time. Please be careful to use on MTM task");
			this.initialDataLen = this.initialDataLen * inputPort.getPortSampleRateList().get(0).getSampleRate();
		}
		else // inputPort.getPortSampleRateType() == PortSampleRateType.VARIABLE
		{
			// do nothing
		}
		*/
	}

	public void setPortIndexByPortList(ArrayList<ChannelPort> portList)
	{
		int index = 0;
		int listSize = portList.size();
		ChannelPort port;
		for(index = 0; index < listSize ; index++)
		{
			if(this.inputPortIndex != Constants.INVALID_VALUE && this.outputPortIndex != Constants.INVALID_VALUE)
			{
				break;
			}

			port = portList.get(index);
			if(port.getTaskId() == this.inputPort.getTaskId() &&
					port.getPortName().equals(this.inputPort.getPortName()) == true)
			{
				this.inputPortIndex = index;
			}
			else if(port.getTaskId() == this.outputPort.getTaskId() &&
					port.getPortName().equals(this.outputPort.getPortName()) == true)
			{
				this.outputPortIndex = index;
			}
		}
	}

	public int getInputPortIndex() {
		return inputPortIndex;
	}

	public int getOutputPortIndex() {
		return outputPortIndex;
	}

	public boolean isDTypeLoopTaskorSubTaskofDTypeLoopTask(HashMap<String, Task> taskMap, String taskName)
	{
		Task task = taskMap.get(taskName);
		boolean isSubTaskofDTypeLoopTask = false;

		while(task != null)
		{
			if(task.getLoopStruct() != null && task.getLoopStruct().getLoopType() == TaskLoopType.DATA)
			{
				isSubTaskofDTypeLoopTask = true;
				break;
			}
			task = taskMap.get(task.getParentTaskGraphName());
		}
		return isSubTaskofDTypeLoopTask;
	}

	public boolean hasSameDTypeLoopParentTask(HashMap<String, Task> taskMap, String dstTaskName, String srcTaskName)
	{
		//Find the nearest ancestor DType Loop task for each of the two tasks.
		//if one of two tasks are not in DTypeLoopTask, return false.

		Task dstTask = taskMap.get(dstTaskName);
		Task dstParentTask = taskMap.get(dstTask.getParentTaskGraphName());
		Task srcTask = taskMap.get(srcTaskName);
		Task srcParentTask = taskMap.get(srcTask.getParentTaskGraphName());
		boolean isDstTaskSubTaskofDTypeLoopTask = false;
		boolean isSrcTaskSubTaskofDTypeLoopTask = false;

		while(dstParentTask != null)
		{
			if(dstParentTask.getLoopStruct() != null && dstParentTask.getLoopStruct().getLoopType() == TaskLoopType.DATA)
			{
				isDstTaskSubTaskofDTypeLoopTask = true;
				break;
			}
			dstParentTask = taskMap.get(dstParentTask.getParentTaskGraphName());
		}

		while(srcParentTask != null)
		{
			if(srcParentTask.getLoopStruct() != null && srcParentTask.getLoopStruct().getLoopType() == TaskLoopType.DATA)
			{
				isSrcTaskSubTaskofDTypeLoopTask = true;
				break;
			}
			srcParentTask = taskMap.get(srcParentTask.getParentTaskGraphName());
		}

		if(!isDstTaskSubTaskofDTypeLoopTask || !isSrcTaskSubTaskofDTypeLoopTask)
		{
			return false;
		}
		else if(!dstParentTask.getName().equals(srcParentTask.getName()))
		{
			return false;
		}
		else
		{
			return true;
		}
	}

	public void setMaximumChunkNum(HashMap<String, Task> taskMap, String srcTaskName, String dstTaskName, MappingInfo srcTaskMappingInfo, MappingInfo dstTaskMappingInfo)
	{
		//Two tasks on both sides of the channel are in common loop task.
		if(hasSameDTypeLoopParentTask(taskMap, dstTaskName, srcTaskName))
		{
			this.inputPort.setMaximumParallelNumberInDTypeLoopTask(taskMap, dstTaskName, dstTaskMappingInfo);
			this.outputPort.setMaximumParallelNumberInDTypeLoopTask(taskMap, srcTaskName, srcTaskMappingInfo);
		}
		//dstTask of the channel is in DTypeLoopTask and srcTask is in outside of DTypeLoopTask(the channel is crossing the boundary).
		else if(isDTypeLoopTaskorSubTaskofDTypeLoopTask(taskMap, dstTaskName) && !isDTypeLoopTaskorSubTaskofDTypeLoopTask(taskMap, srcTaskName))
		{
			this.inputPort.setMaximumParallelNumberInBorderLine(taskMap);
			this.outputPort.setMaximumParallelNumberInBorderLine(taskMap);
		}
		//srcTask of the channel is in DTypeLoopTask and dstTask is in outside of DTypeLoopTask(the channel is crossing the boundary).
		else if(isDTypeLoopTaskorSubTaskofDTypeLoopTask(taskMap, srcTaskName) && !isDTypeLoopTaskorSubTaskofDTypeLoopTask(taskMap, dstTaskName))
		{
			this.inputPort.setMaximumParallelNumberInBorderLine(taskMap);
			this.outputPort.setMaximumParallelNumberInBorderLine(taskMap);
		}
		else
		{
			this.inputPort.setMaximumChunkNum(1);
			this.outputPort.setMaximumChunkNum(1);
		}
	}

	public void setOutputPort(ChannelPort outputPort) {
		this.outputPort = outputPort;
	}

	public int getInitialDataLen() {
		return initialDataLen;
	}

	public void setInitialDataLen(int initialDataLen) {
		this.initialDataLen = initialDataLen;
	}

	public int getNextChannelIndex() {
		return nextChannelIndex;
	}

	public void setNextChannelIndex(int nextChannelIndex) {
		this.nextChannelIndex = nextChannelIndex;
	}

	public int getChannelSampleSize() {
		return channelSampleSize;
	}

	public int getSocketInfoIndex() {
		return socketInfoIndex;
	}

	public void setSocketInfoIndex(int socketInfoIndex) {
		this.socketInfoIndex = socketInfoIndex;
	}

	public InMemoryAccessType getAccessType() {
		return accessType;
	}

	public void setAccessType(InMemoryAccessType accessType) {
		this.accessType = accessType;
	}

	public ConnectionRoleType getConnectionRoleType() {
		return connectionRoleType;
	}

	public DeviceCommunicationType getRemoteMethodType() {
		return remoteMethodType;
	}

	public void setConnectionRoleType(ConnectionRoleType connectionRoleType) {
		this.connectionRoleType = connectionRoleType;
	}

	public void setRemoteMethodType(DeviceCommunicationType remoteMethodType) {
		this.remoteMethodType = remoteMethodType;
	}
}
