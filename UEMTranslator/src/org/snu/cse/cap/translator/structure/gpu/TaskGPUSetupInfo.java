package org.snu.cse.cap.translator.structure.gpu;

import java.math.BigInteger;

import org.snu.cse.cap.translator.structure.task.TaskShapeType;

import hopes.cic.xml.YesNoType;

public class TaskGPUSetupInfo {
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
	
	public TaskGPUSetupInfo(String taskName, TaskShapeType mappedTaskType, YesNoType isClustering, YesNoType isPipelining,int inMaxStrem) {
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
	public int getInMaxStream() {
		return inMaxStream;
	}

	public int getInBlockSizeWidth() {
		return inBlockSizeWidth;
	}

	public int getInBlockSizeHeight() {
		return inBlockSizeHeight;
	}

	public int getInBlockSizeDepth() {
		return inBlockSizeDepth;
	}

	public int getInThreadSizeWidth() {
		return inThreadSizeWidth;
	}

	public int getInThreadSizeHeight() {
		return inThreadSizeHeight;
	}

	public int getInThreadSizeDepth() {
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
