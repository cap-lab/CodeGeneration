package org.snu.cse.cap.translator.structure.mapping;

import java.math.BigInteger;

import org.snu.cse.cap.translator.structure.task.TaskShapeType;

import hopes.cic.xml.YesNoType;

public class TaskGPUMappingInfo extends MappingInfo {
	private String taskName;
	private boolean isClustering;
	private boolean isPipelining;
	private int inMaxStream;
	
	private int inBlockSizeWidth;
	private int inBlockSizeHeight;
	private int inBlockSizeDepth;
	
	private int inThreadSizeWidth;
	private int inThreadSizeHeight;
	private int inThreadSizeDepth;
	
	public TaskGPUMappingInfo(String taskName, TaskShapeType mappedTaskType, YesNoType isClustering, YesNoType isPipelining,int inMaxStrem) {
		super(mappedTaskType);
		this.taskName = taskName;
		if(isClustering.toString().equals("YES")) this.isClustering = true;
		else this.isClustering = false;
		if(isPipelining.toString().equals("YES")) this.isPipelining = true;
		else this.isPipelining = false;
		this.inMaxStream = inMaxStrem;
	}

	public String getTaskName() {
		return taskName;
	}

	public void setTaskName(String taskName) {
		this.taskName = taskName;
	}

	public boolean getIsClustering() {
		return isClustering;
	}

	public boolean getIsPipelining() {
		return isPipelining;
	}
	
	public int getMaxStream(){
		return inMaxStream;
	}
	
	public int getBlockSizeWidth(){
		return inBlockSizeWidth;
	}
	
	public int getBlockSizeHeight(){
		return inBlockSizeHeight;
	}
	
	public int getBlockSizeDepth(){
		return inBlockSizeDepth;
	}
	
	public int getThreadSizeWidth(){
		return inThreadSizeWidth;
	}
	
	public int getThreadSizeHeight(){
		return inThreadSizeHeight;
	}
	
	public int getThreadSizeDepth(){
		return inThreadSizeDepth;
	}

	public void setIsClustering(boolean isClustering) {
		this.isClustering = isClustering;
	}

	public void setIsPipelining(boolean isPipelining) {
		this.isPipelining = isPipelining;
	}
	
	public void setMaxStream(int inMaxStrem){
		this.inMaxStream = inMaxStrem;
	}
	
	public void setBlockSizeWidth(BigInteger inBlockSizeWidth){
		this.inBlockSizeWidth = inBlockSizeWidth.intValue();
	}
	
	public void setBlockSizeHeight(BigInteger inBlockSizeHeight){
		this.inBlockSizeHeight = inBlockSizeHeight.intValue();
	}
	
	public void setBlockSizeDepth(BigInteger inBlockSizeDepth){
		this.inBlockSizeDepth = inBlockSizeDepth.intValue();
	}
	
	public void setThreadSizeWidth(BigInteger inThreadSizeWidth){
		this.inThreadSizeWidth = inThreadSizeWidth.intValue();
	}
	
	public void setThreadSizeHeight(BigInteger inThreadSizeHeight){
		this.inThreadSizeHeight = inThreadSizeHeight.intValue();
	}
	
	public void setThreadSizeDepth(BigInteger inThreadSizeDepth){
		this.inThreadSizeDepth = inThreadSizeDepth.intValue();
	}

}
