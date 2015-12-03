package InnerDataStructures;

public class Variable {
	String mType;
	String mName;
	
	public Variable(String type, String name){
		mType = type;
		mName = name;
	}
	
	public String getName()	{return mName;}
	public String getType()	{return mType;}
}
