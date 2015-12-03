package InnerDataStructures;

import java.util.*;

public class Function {
	private int mIndex;
	private String mLibraryName;
	private String mFunctionName;
	private String mReturnType;
	private List<Argument> mArgumentList;

	public Function(int index, String libraryName, String functionName, String returnType, List<Argument> argumentList)
	{
		mIndex = index;
		mLibraryName = libraryName;
		mFunctionName = functionName;
		mReturnType = returnType;
		mArgumentList = argumentList;
	}
	
	public String getIndex()				{return Integer.toString(mIndex);}
	public String getLibraryName()			{return mLibraryName;}
	public String getFunctionName()			{return mFunctionName;}
	public String getReturnType()			{return mReturnType;}
	public List<Argument> getArgList()		{return mArgumentList;}
}
