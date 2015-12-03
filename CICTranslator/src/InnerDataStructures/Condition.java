package InnerDataStructures;

public class Condition {
	String mVariable;
	String mValue;
	String mComparator;
	
	public Condition(String var, String val, String comp) {
		mVariable = var;
		mValue = val;
		mComparator = comp;
	}
	
	public String getVariable()	{return mVariable;}
	public String getValue()		{return mValue;}
	public String getComparator()	{return mComparator;}
}

