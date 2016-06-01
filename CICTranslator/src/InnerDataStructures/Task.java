package InnerDataStructures;

import java.util.*;
import hopes.cic.xml.*;

public class Task {
	private int mIndex;
	private String mName;
	private String mCICFile;
	private Map<String, Map<String, List<Integer>>> mProc;
	private Map<String, Map<String, List<Integer>>> mCallCount;
	private String mCflag;
	private String mLDflag;
	private String mDataParallel;
	private int mWidth;
	private int mHeight;
	private List<VectorType> mDependencyList;
	private List<Integer> mFeedbackList;
	private List<Queue> mQueue;
	private Map<String, Map<String, Integer>> mInPortList;
	private Map<String, Map<String, Integer>> mOutPortList;
	private int mEventCompleteID;
	private int mEventStartID;
	private List<TaskParameterType> mParameters;
	private int mPeriod;
	private int mDeadline;
	private int mPriority;
	private String mPeriodMetric;
	private String mDeadlineMetric;
	private String mRunCondition;
	private Map<String, Integer>  mExecutionTimeValue;
	private Map<String, String> mExecutionTimeMetric;
	private boolean mIsSlaveTask;
	private List<String> mControllingTask;
	private int mPriority_TFS;
	private List<String> mExtraHeader;
	private List<String> mExtraSource;
	private List<LibraryMasterPortType> mLibraryPort;
	private List<TaskParameterType> mParameter;
	private List<String> mPrev;
	private List<String> mNext;
	private String mHasSubgraph;
	private boolean mHasMTM;
	private MTM mMTMInfo;
	private String mParentTask;
	private String mTaskType;
	private int mRunRate;
	private boolean mIsSrcTask;
	
	
	public Task(int index, String name, String cicfile, Map<String, Map<String, List<Integer>>> proc
			, String cflag, String ldflag, String dataParallel, int width, int height
			, List<VectorType> dependency, List<Integer> feedbackList, String runCondition
			, List<String> extraHeader, List<String> extraSource, List<LibraryMasterPortType> libraryPort
			, List<TaskParameterType> parameter, String hasSubgraph, boolean hasMTM, MTM mtmInfo
			, String parentTask, String taskType, Map<String, Map<String, Integer>> inPortList
			, Map<String, Map<String, Integer>> outPortList, boolean isSrcTask)
	{
		mIndex = index;
		mName = name;
		mProc = proc;
		mCflag = cflag;
		mLDflag = ldflag;
		mCICFile = cicfile;
		mDataParallel = dataParallel;
		mWidth = width;
		mHeight = height;
		mDependencyList = dependency;
		mFeedbackList = feedbackList;
		mEventCompleteID = 0;
		mEventStartID = 0;
		mRunCondition = runCondition;
		mExtraHeader = extraHeader;
		mExtraSource = extraSource;
		mLibraryPort = libraryPort;
		mParameter = parameter;
		mExecutionTimeValue = new HashMap<String, Integer>();
		mExecutionTimeMetric = new HashMap<String, String>();
		mIsSlaveTask = false;
		mPriority_TFS = 0;
		mPrev = new ArrayList<String>();
		mNext = new ArrayList<String>();
		mQueue = new ArrayList<Queue> ();
		mControllingTask = new ArrayList<String> ();
		mHasSubgraph = hasSubgraph;
		mHasMTM = hasMTM;
		mParentTask = parentTask;
		mTaskType = taskType;
		mInPortList = inPortList;
		mOutPortList = outPortList;
		mMTMInfo = mtmInfo;
		mRunRate = 1;
		mIsSrcTask = isSrcTask;
	}
	
	//constructor when generating virtual tasks (ex. task_proc_0, SDF_5)
	public Task(int index, String name, String parentTask, int runRate, String periodMetric, String runCondition, int period)
	{
		mIndex = index;
		mName = name;
		mProc = null;
		mCflag = "";
		mLDflag = "";
		mCICFile = "";
		mDataParallel = "";
		mWidth = 0;
		mHeight = 0;
		mDependencyList = null;
		mFeedbackList = null;
		mEventCompleteID = 0;
		mEventStartID = 0;
		mPeriod = period;
		mDeadline = 0;
		mPriority = 0;
		mPeriodMetric = periodMetric;
		mRunCondition = runCondition;
		mExtraHeader = new ArrayList<String>();
		mExtraSource = new ArrayList<String>();
		mLibraryPort = null;
		mParameter = null;
		mExecutionTimeValue = new HashMap<String, Integer>();
		mExecutionTimeMetric = new HashMap<String, String>();
		mIsSlaveTask = false;
		mPriority_TFS = 0;
		mPrev = new ArrayList<String>();
		mNext = new ArrayList<String>();
		mQueue = new ArrayList<Queue> ();
		mControllingTask = new ArrayList<String> ();
		mHasSubgraph = "Yes";
		mHasMTM = false;
		mParentTask = parentTask;
		mTaskType = "Computational";
		mInPortList = null;
		mOutPortList = null;
		mMTMInfo = null;
		mRunRate = runRate;
		mIsSrcTask = false;
	}
	
	public void setIndex(int index)					{mIndex = index;}
	public void setPeriod(int period)				{mPeriod = period;}
	public void setPeriodMetric(String metric)		{mPeriodMetric = metric;}
	public void setDeadline(int deadline)			{mDeadline = deadline;}
	public void setPriority(int priority)			{mPriority = priority;}
	public void setRunRate(int runRate)				{mRunRate = runRate;}

	public void setProc(Map<String, Map<String, List<Integer>>> proc)					{mProc = proc;}
	public void setExecutionTimeValue(Map<String, Integer> executionTime)	{mExecutionTimeValue = executionTime;}
	public void setExecutionTimeMetric(Map<String, String> metric)		{mExecutionTimeMetric = metric;}
	public void setIsSlaveTask(boolean isSlaveTask)		{mIsSlaveTask = isSlaveTask;}
	public void setParentTask(String parentTask)		{mParentTask = parentTask;}
	public void setIsSrcTask(boolean isSrc)				{mIsSrcTask = isSrc;}
	public void setCallCount(Map<String, Map<String, List<Integer>>> callCount)					{mCallCount = callCount;}
	
	public boolean getIsSrcTask()					{return mIsSrcTask;}
	public String getRunCondition()					{return mRunCondition;}
	public String getIndex()						{return Integer.toString(mIndex);}
	public String getName()							{return mName;}
	public String getCICFile()						{return mCICFile;}
	public String getPeriodMetric()					{return mPeriodMetric;}
	public String getPeriod()						{return Integer.toString(mPeriod);}
	public String getDeadlineMetric()				{return mDeadlineMetric;}
	public String getDeadline()						{return Integer.toString(mDeadline);}
	public String getRunRate()						{return Integer.toString(mRunRate);}
	public boolean getIsSlaveTask()					{return mIsSlaveTask;}
	public List<String> getPrev()					{return mPrev;}
	public List<String> getNext()					{return mNext;}
	public List<Queue> getQueue()					{return mQueue;}
	public Map<String, Map<String, List<Integer>>> getProc()		{return mProc;}
	public List<String> getControllingTask()		{return mControllingTask;}
	public List<TaskParameterType> getParameter()	{return mParameter;}
	public List<String> getExtraSource()			{return mExtraSource;}
	public List<String> getExtraHeader()			{return mExtraHeader;}
	public String getCflag()						{return mCflag;}
	public String getLDflag()						{return mLDflag;}
	public String getHasSubgraph()					{return mHasSubgraph;}
	public boolean getHasMTM()						{return mHasMTM;}
	public String getParentTask()					{return mParentTask;}
	public String getTaskType()						{return mTaskType;}
	public Map<String, Map<String, Integer>> getInPortList()	{return mInPortList;}
	public Map<String, Map<String, Integer>> getOutPortList()	{return mOutPortList;}
	public Map<String, Map<String, Integer>> getPortList(){
		Map<String, Map<String, Integer>> portList = new HashMap<String, Map<String, Integer>>();
		portList.putAll(mInPortList);
		portList.putAll(mOutPortList);
		return portList;
	}
	public List<LibraryMasterPortType> getLibraryPortList()	{return mLibraryPort;}
	public MTM getMTM()								{return mMTMInfo;}
	public Map<String, Integer> getExecutionTimeValue()	{return mExecutionTimeValue;}
	public Map<String, String > getExecutionTimeMetric()	{return mExecutionTimeMetric;}
	public Map<String, Map<String, List<Integer>>> getCallCount()		{return mCallCount;}
}
