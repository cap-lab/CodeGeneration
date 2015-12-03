package InnerDataStructures;

import java.util.*;

public class LibraryStub {
	private String mName;
	private int mTaskIndex;
	private int mSendChannelId;
	private int mReceiveChannelId;
	private int mMyProc;
	private int mTargetProc;
	private List<Function> mFunc;
	
	public LibraryStub(String name, int taskIndex, int sendChannelId, int receiveChannelId, int myProc, int targetProc, List<Function> func){
		mName = name;
		mTaskIndex = taskIndex;
		mSendChannelId = sendChannelId;
		mReceiveChannelId = receiveChannelId;
		mMyProc = myProc;
		mTargetProc = targetProc;
		mFunc = func;
	}
}
