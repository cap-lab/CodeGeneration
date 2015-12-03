package InnerDataStructures;

import java.util.*;

import hopes.cic.xml.*;

public class Communication {
	
	public class BluetoothNode
	{
		public String mBluetoothName;
		private String mFriendlyName;
		private String mMac;
		
		public BluetoothNode(String name, String fname, String mac)
		{
			mBluetoothName = name; 
			mFriendlyName = fname;
			mMac = mac;
		}
		
		public String getMac()						{return mMac; }
		public String getFriendlyName()				{return mFriendlyName; }
	}
	public class BluetoothComm 
	{
		private BluetoothNode mMasterProcessor;
		private List<BluetoothNode> mSlaveProcessors;
		
		public BluetoothComm()
		{
			mSlaveProcessors = new ArrayList<BluetoothNode>();
		}
		
		public BluetoothNode getMasterProc()				{return mMasterProcessor;}
		public List<BluetoothNode> getSlaveProc()			{return mSlaveProcessors;}
		
		public void setMasterProc(String name, String fname, String mac)	
		{
			mMasterProcessor = new BluetoothNode(name, fname, mac);
		}
		public void addSlaveProc(String name, String fname, String mac)	
		{
			mSlaveProcessors.add(new BluetoothNode(name, fname, mac));
		}
	}
	
	public class I2CNode
	{
		public String mI2CName;
		private String mId;
		
		public I2CNode(String name, String id)
		{
			mI2CName = name;
			mId = id;
		}
		public String getId()						{return mId; }
	}
	public class I2CComm
	{
		private I2CNode mMasterProcessor;
		private I2CNode mSlaveProcessor;
		
		public I2CComm() {}
		
		public I2CNode getMasterProc()					{return mMasterProcessor;}
		public I2CNode getSlaveProc()					{return mSlaveProcessor;}
		
		public void setMasterProcName(String name, String id)	{mMasterProcessor = new I2CNode(name, id);}
		public void setSlaveProcName(String name, String id)	{mSlaveProcessor = new I2CNode(name, id);}
	}
	
	public class WIFINode
	{
		public String mWIFIName;
		private String mIp;
		
		public WIFINode(String name, String ip)
		{
			mWIFIName = name;
			mIp = ip;
		}
		public String getIp()						{return mIp; }
	}
	public class WIFIComm
	{
		private WIFINode mServerProcessor;
		private List<WIFINode> mClientProcessors;
		
		public WIFIComm()
		{
			mClientProcessors = new ArrayList<WIFINode>();
		}
		
		public WIFINode getServerProc()				{return mServerProcessor;}
		public List<WIFINode> getClientProc()		{return mClientProcessors;}
		
		public void setServerProc(String name, String ip)	
		{
			mServerProcessor = new WIFINode(name, ip);
		}
		public void addClientProc(String name)	
		{
			mClientProcessors.add(new WIFINode(name, null));
		}
	}
	
	public class VRepSharedNode
	{
		public String mVRepSharedNodeName;
		
		public VRepSharedNode(String name)
		{
			mVRepSharedNodeName = name;
		}
		
		public String getNodeName()				{return mVRepSharedNodeName; }
	}
	public class VRepSharedMemComm
	{
		private int key;
		private VRepSharedNode mMasterProcessor;
		private List<VRepSharedNode> mSlaveProcessors;
		
		public VRepSharedMemComm()
		{
			mSlaveProcessors = new ArrayList<VRepSharedNode>();
		}
		
		public int getKey()									{return key;}
		public VRepSharedNode getMasterProc()				{return mMasterProcessor;}
		public List<VRepSharedNode> getSlaveProc()			{return mSlaveProcessors;}
		
		public void setKey(int key)						{this.key = key;}
		public void setMasterProc(String name)	
		{
			mMasterProcessor = new VRepSharedNode(name);
		}
		public void addSlaveProc(String name)	
		{
			mSlaveProcessors.add(new VRepSharedNode(name));
		}
	}
	
	public static final int TYPE_BLUETOOTH = 0;
	public static final int TYPE_I2C = 1;
	public static final int TYPE_WIFI = 2;
	public static final int TYPE_VREPSHAREDBUS = 3;

	
	private I2CComm mI2C = null;
	private BluetoothComm mBluetooth = null;
	private WIFIComm mWifi = null;
	private VRepSharedMemComm mVrepShm = null;
	
	private int type; 
	
	
	public Communication(ArchitectureConnectionCategoryType type) 
	{
		if(type == ArchitectureConnectionCategoryType.BLUETOOTH)
		{
			this.type = TYPE_BLUETOOTH;
			mBluetooth = new BluetoothComm();
		}
		else if(type == ArchitectureConnectionCategoryType.I_2_C_BUS)
		{
			this.type = TYPE_I2C;
			mI2C = new I2CComm();
		}
		else if(type == ArchitectureConnectionCategoryType.WIFI)
		{	
			this.type = TYPE_WIFI;
			mWifi = new WIFIComm();			
		}
		else if(type == ArchitectureConnectionCategoryType.V_REP_SHARED_BUS)
		{
			this.type = TYPE_VREPSHAREDBUS;
			mVrepShm = new VRepSharedMemComm();
		}
		else
		{
			System.out.println("I don't know the communication type");
		}
	}
		
	public I2CComm getI2CComm()						{return mI2C;}
	public BluetoothComm getBluetoothComm()			{return mBluetooth;}
	public WIFIComm getWifiComm()					{return mWifi; }
	public VRepSharedMemComm getVRepSharedMemComm()	{return mVrepShm; }
	
	public int getType()						{return type; }
}
