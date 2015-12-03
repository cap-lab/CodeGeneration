package InnerDataStructures;

import java.util.List;

public class Transition {
	String mSrcMode;
	String mDstMode;
	String mName;
	List <Condition> mConditionList;
	
	public Transition(String name, String srcMode, String dstMode, List<Condition> conditionList){
		mName = name;
		mSrcMode = srcMode;
		mDstMode = dstMode;
		mConditionList = conditionList;
	}
	
	public String getName()						{return mName;}
	public String getSrcMode()					{return mSrcMode;}
	public String getDstMode()					{return mDstMode;}
	public List<Condition> getConditionList()	{return mConditionList;}
}
