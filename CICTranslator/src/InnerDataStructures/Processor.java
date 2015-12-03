package InnerDataStructures;

import java.util.*;
import hopes.cic.xml.*;

public class Processor {
	private int mIndex;
	private String mProcessorName;
	private String mPoolName;
	private int mLocalIndex;
	private int mControlQueueIndex;
	private String mSupportOS;
	private String mSched;
	private String mProcessorType;
	private String mArchType;
	//private List<Task> mTask;
	
	private int mTaskStartPrio;
	private int mCurTaskPrio;
	
	public Processor(int index, String processorName, String poolName, int localIndex, String supportOS, String sched, String processorType, String Arch){
		mIndex = index;
		mProcessorName = processorName;
		mPoolName = poolName;
		mLocalIndex = localIndex;
		mSupportOS = supportOS;
		mSched = sched;
		mProcessorType = processorType;
		mControlQueueIndex = -1;
		//mTask = new ArrayList<Task>();
		mTaskStartPrio = 4;
		mCurTaskPrio = 0;
		mArchType = Arch;
	}
	
	public void setControlQueueIndex(int controlQueueIndex)	{mControlQueueIndex = controlQueueIndex;}
	public void increaseTaskStartPrio(int prio)				{mTaskStartPrio += prio;}
	
	public int getIndex()				{return mIndex;}
	public String getProcName()			{return mProcessorName;}
	public String getArchType()			{return mArchType;}
	public String getPoolName()			{return mPoolName;}
	public int getLocalIndex()			{return mLocalIndex;}
	public String getSupportOS()		{return mSupportOS;}
	public String getSched()			{return mSched;}
	public String getProcType()			{return mProcessorType;}
	//public List<Task> getTask()			{return mTask;}
	public int getControlQueueIndex()	{return mControlQueueIndex;}
}
