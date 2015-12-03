package InnerDataStructures;

import hopes.cic.xml.*;

import java.util.*;

public class Library {
	private int mIndex;
	private String mType;
	private String mName;
	private String mHeader;
	private String mFile;
	private int mProc;
	private List<Function> mFunctionList;
	private List<Integer> mDiffMappedProcs;
	private List<LibraryStub> mStubList;
	private List<LibraryMasterPortType> mLibraryPort;
	private List<String> mExtraSource;
	private List<String> mExtraHeader;
	private String mCflag;
	private String mLDflag;
	
	public Library(int index, String name, String type, String header, String file, int proc, List<Function> functionList, List<Integer> diffMappedProcs, List<LibraryStub> stubList, List<LibraryMasterPortType>libraryPort, List<String> extraSource, List<String> extraHeader, String cflag, String ldflag){
		mIndex = index;
		mType = type;
		mName = name;
		mHeader = header;
		mFile = file;
		mProc = proc;
		mFunctionList = functionList;
		mDiffMappedProcs = diffMappedProcs;
		mStubList = stubList;
		mLibraryPort = libraryPort;
		mExtraSource = extraSource;
		mExtraHeader = extraHeader;
		mCflag = cflag;
		mLDflag = ldflag;
	}
	
	public String getIndex()						{return Integer.toString(mIndex);}
	public String getFile()							{return mFile;}
	public String getHeader()						{return mHeader;}
	public String getName()							{return mName;}
	public List<String> getExtraSource()			{return mExtraSource;}
	public List<String> getExtraHeader()			{return mExtraHeader;}
	public String getCflag()						{return mCflag;}
	public String getLDflag()						{return mLDflag;}
	public List<Function> getFuncList()				{return mFunctionList;}
	public int getProc()							{return mProc;}
}
