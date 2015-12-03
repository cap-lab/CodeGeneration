package InnerDataStructures;

import java.util.*;
import hopes.cic.xml.*;

public class Queue {
	private int mIndex;
	private String mSrc;
	private int mSrcPortId;
	private String mSrcPortName;
	private String mDst;
	private int mDstPortId;
	private String mDstPortName;
	private int mSize;
	private int mInitData;
	private int mSampleSize;
	private String mTypeName;
	private ChannelTypeType mOriginType;
	private String mSampleType;
	
	public Queue(int index, String src, int srcPortId, String srcPortName, String dst, int dstPortId, String dstPortName, int size, int initData, int sampleSize, String typeName, ChannelTypeType originType, String sampleType)
	{
		mIndex = index;
		mSrc = src;
		mSrcPortId = srcPortId;
		mSrcPortName = srcPortName;
		mDst = dst;
		mDstPortId = dstPortId;
		mDstPortName = dstPortName;
		mSize = size;
		mInitData = initData;
		mSampleSize = sampleSize;
		mTypeName = typeName;
		mOriginType = originType;
		mSampleType = sampleType;
	}
	
	public void setIndex(int index) {mIndex = index;}
	
	public String getIndex()		{return Integer.toString(mIndex);}
	public String getDst()			{return mDst;}
	public String getSrc()			{return mSrc;}
	public String getDstPortId()	{return Integer.toString(mDstPortId);}
	public String getSrcPortId()	{return Integer.toString(mSrcPortId);}
	public String getDstPortName()	{return mDstPortName;}
	public String getSrcPortName()	{return mSrcPortName;}
	public String getSampleSize()	{return Integer.toString(mSampleSize);}
	public String getSampleType()	{return mSampleType;}
	public String getTypeName()		{return mTypeName;}
	public String getSize()			{return Integer.toString(mSize);}
	public String getInitData()		{return Integer.toString(mInitData);}
}
