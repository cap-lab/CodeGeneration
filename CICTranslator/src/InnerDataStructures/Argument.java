package InnerDataStructures;

public class Argument {
	private int mIndex;
	private String mLibraryName;
	private String mFunctionName;
	private String mType;
	private String mVariableName;
	
	public Argument(int index, String libraryName, String functionName, String type, String variableName)
	{
		mIndex = index;
		mLibraryName = libraryName;
		mFunctionName = functionName;
		mType = type;
		mVariableName = variableName;
	}
	
	public int getIndex()					{return mIndex;}
	public String getLibraryName()			{return mLibraryName;}
	public String getFunctionName()			{return mFunctionName;}
	public String getType()					{return mType;}
	public String getVariableName()			{return mVariableName;}
}
