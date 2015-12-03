package CommonLibraries;

import hopes.cic.exception.CICXMLException;
import hopes.cic.xml.*;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.swing.JOptionPane;

import InnerDataStructures.*;
import InnerDataStructures.Communication.BluetoothNode;
import InnerDataStructures.Communication.I2CNode;
import InnerDataStructures.Communication.WIFINode;
import InnerDataStructures.Library;
import InnerDataStructures.Communication.BluetoothComm;
import InnerDataStructures.Communication.I2CComm;
import InnerDataStructures.Communication.VRepSharedMemComm;
import InnerDataStructures.Communication.VRepSharedNode;
import InnerDataStructures.Communication.WIFIComm;

public class OutComm {
	
	public static final int TYPE_BLUETOOTH = 0;
	public static final int TYPE_I2C = 1;
	public static final int TYPE_WIFI = 2;
	public static final int TYPE_VREPSHAREDBUS = 3;
		
	public static void generateConmapHeader(String file, String mTemplateFile, Map<String, Task> mTask, Map<Integer, Queue> mQueue, List<Communication> mCommunication, Map<Integer, Processor> mProcessor, Processor mMyProcessor, CICMappingType mMapping)
	{
		File fileOut = new File(file);
		File templateFile = new File(mTemplateFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);
			
			outstream.write(translateConmapHeader(content, mTask, mQueue, mCommunication, mProcessor, mMyProcessor, mMapping).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	//private CICMappingType mMapping;
	public static String translateConmapHeader(String mContent, Map<String, Task> mTask, Map<Integer, Queue> mQueue, List<Communication> mCommunication, 
			Map<Integer, Processor> mProcessor, Processor mMyProcessor, CICMappingType mMapping /*, Map<String, Library> mLibrary, ArrayList<Library> mLibraryStubList, ArrayList<Library> mLibraryWrapperList*/)
	{
		String code = mContent;
		String connectionMapEntries="";		
		for(Processor proc: mProcessor.values())
		{
			System.out.println("proc>>> " + proc.getPoolName());
		}
		connectionMapEntries += "CIC_UT_CONNECTIONMAP conn_map[] = {";
		for(Queue queue: mQueue.values())
		{
			Processor srcProc = null;
			Task srcTask = null;
			Processor dstProc = null;
			Task dstTask = null;
			
			srcTask = mTask.get(queue.getSrc());
			srcProc = mProcessor.get(srcTask.getProc().get("Default").get("Default").get(0));	//Need to fix
			dstTask = mTask.get(queue.getDst());
			dstProc = mProcessor.get(dstTask.getProc().get("Default").get("Default").get(0));	//Need to fix
			
			/*
			for(Processor proc: mProcessor.values())
			{			
				List<Task> taskList = proc.getTask(); 
				for(Task t: taskList)
				{
					if(t.getName().equals(queue.getSrc()))
					{
						srcProc = proc;
						srcTask = t;
					}
					
					if(t.getName().equals(queue.getDst()))
					{
						dstProc = proc;
						dstTask = t;
					}
				}				
			}
			*/
						
			if((!srcProc.equals(dstProc)) && (srcProc.getPoolName().equals(mMyProcessor.getPoolName()) || dstProc.getPoolName().equals(mMyProcessor.getPoolName())))
			{
				connectionMapEntries +="\t{CON_NOR_CHANNEL," + queue.getIndex() +  ","; 
				int procId = -1; 				
				for(Processor proc: mProcessor.values())
				{
					procId++;
					//System.out.println("proc: " + proc.getPoolName() + "\t id::" + procId);
					if(mMyProcessor.getPoolName().equals(proc.getPoolName()))//to test equal? contains
						continue;
					else
					{
						for(MappingTaskType mtt: mMapping.getTask())
						{
							for(MappingProcessorIdType mpit: mtt.getProcessor())
							{
								if(mpit.getPool().equals(proc.getPoolName()) && (mtt.getName().equals(srcTask.getName()) || mtt.getName().equals(dstTask.getName())))
								{	
									//System.out.println("mpit: proc: " + mpit.getPool());
									connectionMapEntries += procId + ",";
								}
							}
						}
					}			
					
				}
				
				for(Communication comm : mCommunication)
				{
					int type = comm.getType();
					switch(type)
					{
					case TYPE_BLUETOOTH:
						BluetoothComm btcom = comm.getBluetoothComm();
						if(btcom.getMasterProc().mBluetoothName.equals(srcProc.getPoolName()))
						{
							for(BluetoothNode slave: btcom.getSlaveProc())
							{
								if(slave.mBluetoothName.equals(dstProc.getPoolName()))
								{
									connectionMapEntries += "BLUETOOTH_CONN),\n";
									break;
								}
							}
						}
						else if(btcom.getMasterProc().mBluetoothName.equals(dstProc.getPoolName()))
						{
							for(BluetoothNode slave: btcom.getSlaveProc())
							{
								if(slave.mBluetoothName.equals(srcProc.getPoolName()))
								{
									connectionMapEntries += "BLUETOOTH_CONN},\n";
									break;
								}
							}
						}
						break;
					case TYPE_I2C:
						I2CComm i2ccom = comm.getI2CComm();
						if(i2ccom.getMasterProc().mI2CName.equals(srcProc.getPoolName()))
						{
							if(i2ccom.getSlaveProc().mI2CName.equals(dstProc.getPoolName()))
							{
								connectionMapEntries += "I2C_CONN},\n";
								break;
							}							
						}
						else if(i2ccom.getMasterProc().mI2CName.equals(dstProc.getPoolName()))
						{
							if(i2ccom.getSlaveProc().mI2CName.equals(srcProc.getPoolName()))
							{
								connectionMapEntries += "I2C_CONN},\n";
								break;
							}
						}
						break;
					case TYPE_WIFI:
						WIFIComm wfcom = comm.getWifiComm();
						if(wfcom.getServerProc().mWIFIName.equals(srcProc.getPoolName()))
						{
							for(WIFINode client: wfcom.getClientProc())
							{
								if(client.mWIFIName.equals(dstProc.getPoolName()))
								{
									connectionMapEntries += "WIFI_CONN},\n";
									break;
								}
							}
						}
						else if(wfcom.getServerProc().mWIFIName.equals(dstProc.getPoolName()))
						{
							for(WIFINode client: wfcom.getClientProc())
							{
								if(client.mWIFIName.equals(srcProc.getPoolName()))
								{
									connectionMapEntries += "WIFI_CONN},\n";
									break;
								}
							}
						}
						break;
					case TYPE_VREPSHAREDBUS:
						VRepSharedMemComm shmcom = comm.getVRepSharedMemComm();
						if(shmcom.getMasterProc().mVRepSharedNodeName.equals(srcProc.getPoolName()))
						{
							for(VRepSharedNode slave: shmcom.getSlaveProc())
							{
								if(slave.mVRepSharedNodeName.equals(dstProc.getPoolName()))
								{
									connectionMapEntries += "SHARED_CONN},\n";
									break;
								}
							}
						}
						else if(shmcom.getMasterProc().mVRepSharedNodeName.equals(dstProc.getPoolName()))
						{
							for(VRepSharedNode slave: shmcom.getSlaveProc())
							{
								if(slave.mVRepSharedNodeName.equals(srcProc.getPoolName()))
								{
									connectionMapEntries += "SHARED_CONN},\n";
									break;
								}
							}
						}
						break;
					default:
						connectionMapEntries += "NON_CONN}, \n";
						break;							
					}
				}
			}
			
		}
		connectionMapEntries += "};";
		//Libraries are mapped on the same target proc
		/*for(Library library: mLibrary.values())
		{
			int procId = library.getProc();
			Processor proc = mProcessor.get(procId);
			
			if(target.equals(proc.getPoolName()))
			{
				
			}
		}
		
		//if(mLibraryStubList.size() > 0 || mLibraryWrapperList.size() > 0 )
		//{
			
		//}
		if(mLibraryStubList.size() > 0)
		{
			for(Library lib: mLibraryStubList)
			{
				lib.getProc()
				for(Processor proc: mProcessor.values())
				{			
					for(Task t: taskList)
					{
						if(t.equals(mTask.get(queue.getSrc())))
				}
		}
		
		for(InnerDataStructures.Library library: mWrapperList)
		{
			
			externFuncDeclCode += "extern void " + library.getName() + "_wrapper_init(void);\n";
			externFuncDeclCode += "extern void " + library.getName() + "_wrapper_go(void);\n";
			externFuncDeclCode += "extern void " + library.getName() + "_wrapper_wrapup(void);\n";
			
			wrapperEntriesCode += "\t{" + index++ + ", " + library.getProc() +  ", \"" + library.getName() + "_wrapper\", " + library.getName() + "_wrapper_init, "
					+ library.getName() + "_wrapper_go, " + library.getName() + "_wrapper_wrapup, 0},\n";
		}
		
		if(mStubList != null){
			for(InnerDataStructures.Library library: mStubList)
				wrapperEntriesCode += "\t{" + index++ + ", " + library.getProc() +  ", NULL, NULL, NULL, 0},\n";
		}
		
		for(InnerDataStructures.Library library: mWrapperList){
			externFuncDeclCode += "extern void " + library.getName() + "_wrapper_init(void);\n";
			externFuncDeclCode += "extern void " + library.getName() + "_wrapper_go(void);\n";
			externFuncDeclCode += "extern void " + library.getName() + "_wrapper_wrapup(void);\n";
			
			wrapperEntriesCode += "\t{" + index++ + ", " + library.getProc() +  ", \"" + library.getName() + "_wrapper\", " + library.getName() + "_wrapper_init, "
					+ library.getName() + "_wrapper_go, " + library.getName() + "_wrapper_wrapup, 0},\n";
		}
		
		if(mStubList != null){
			for(InnerDataStructures.Library library: mStubList)
				wrapperEntriesCode += "\t{" + index++ + ", " + library.getProc() +  ", NULL, NULL, NULL, 0},\n";
		}
		
				for(Library library: mLibrary.values())
				{
					int procId = library.getProc();
					Processor proc = mProcessor.get(procId);
					if(mMyProcessor.getPoolName().equals(proc.getPoolName()))
					{
						boolean hasRemoteConn = false;
						for(int i=0; i<mAlgorithm.getLibraryConnections().getTaskLibraryConnection().size(); i++){
							TaskLibraryConnectionType taskLibCon = mAlgorithm.getLibraryConnections().getTaskLibraryConnection().get(i);
							if(!mTask.containsKey(taskLibCon.getMasterTask())){
								hasRemoteConn = true;
								break;
							}
						}
						CommonLibraries.Library.generateLibraryCode(mOutputPath, library, mAlgorithm);
						if(hasRemoteConn)
						{	
							//바뀌어야 할 듯 
							CommonLibraries.Library.generateLibraryWrapperCode(mOutputPath, library, mAlgorithm);
							
							mLibraryWrapperList.add(library);
						}
					}
				}
				
				// Libraries are mapped on other target procs
				for(Task t: mTask.values())
				{
					String taskName = t.getName();
					String libPortName = "";
					String libName = "";
					Library library = null;
					if(t.getLibraryPortList().size() != 0)
					{
						List<LibraryMasterPortType> libportList = t.getLibraryPortList();
						for(int i=0; i<libportList.size(); i++)
						{
							LibraryMasterPortType libPort = libportList.get(i);
							libPortName = libPort.getName();
							break;
						}
						if(libPortName == "")
						{ 
							System.out.println("Library task does not exist!");
							System.exit(-1);
						}
						else
						{
							for(int i=0; i<mAlgorithm.getLibraryConnections().getTaskLibraryConnection().size(); i++)
							{
								TaskLibraryConnectionType taskLibCon = mAlgorithm.getLibraryConnections().getTaskLibraryConnection().get(i);
								if(taskLibCon.getMasterTask().equals(taskName) && taskLibCon.getMasterPort().equals(libPortName))
								{
									libName = taskLibCon.getSlaveLibrary();
									break;
								}
							}
							if(!mLibrary.containsKey(libName))
							{
								for(Library lib: mGlobalLibrary.values())
								{
									if(lib.getName().equals(libName))
									{
										library = lib;
										break;
									}
								}
								if(library != null)
								{
									Util.copyFile(mOutputPath + "/"+ library.getHeader(), mOutputPath + "/../" + library.getHeader());
									//바꿔야 할 듯
									CommonLibraries.Library.generateLibraryStubCode(mOutputPath, library, mAlgorithm, false);
									mLibraryStubList.add(library);
								}
							}
						}
					}
				}
		
		for(InnerDataStructures.Library library: mWrapperList){
			externFuncDeclCode += "extern void " + library.getName() + "_wrapper_init(void);\n";
			externFuncDeclCode += "extern void " + library.getName() + "_wrapper_go(void);\n";
			externFuncDeclCode += "extern void " + library.getName() + "_wrapper_wrapup(void);\n";
			
			wrapperEntriesCode += "\t{" + index++ + ", " + library.getProc() +  ", \"" + library.getName() + "_wrapper\", " + library.getName() + "_wrapper_init, "
					+ library.getName() + "_wrapper_go, " + library.getName() + "_wrapper_wrapup, 0},\n";
		}
		
		if(mStubList != null){
			for(InnerDataStructures.Library library: mStubList)
				wrapperEntriesCode += "\t{" + index++ + ", " + library.getProc() +  ", NULL, NULL, NULL, 0},\n";
		}
		*/
		code = code.replace("##CONNMAP_ENTRIES", connectionMapEntries);
		
		return code;
	}
	
	public static String generateConnIncludes(List<Communication> mCommunication, int sizeOfmLibraryStubList, int sizeOfmLibraryWrapperList)
	{		
		String code = "";
		if(mCommunication.size() > 0)
		{
			if(!(sizeOfmLibraryStubList > 0 || sizeOfmLibraryWrapperList > 0 ))
			{
				code += "#include \"LIB_port.h\"\n";
			}
			code += "#include \"cic_conmap.h\"\n#include \"conmap_def.h\"\n";
			
			int numBluetooth = 0, numI2C = 0, numWifi = 0, numSM = 0, numCommType = 0;			
			for(Communication com: mCommunication)
			{
				int type = com.getType();
							
				//should be change! 
				switch(type)
				{
				case TYPE_BLUETOOTH:
					numBluetooth++;
					break; 
				case TYPE_I2C:
					numI2C++;
					break;
				case TYPE_WIFI:
					numWifi++;
					break;
				case TYPE_VREPSHAREDBUS:
					numSM++;
					break;
				default: 
					break;
				}
			}
			if(numBluetooth > 0)
			{
				numCommType++;
				code += "#define NUM_BLUETOOTH_CONNECTION " + numBluetooth + "\n";
			}
			if(numI2C > 0)
			{
				numCommType++;
				code += "#define NUM_I2C_CONNECTION " + numI2C + "\n";
			}
			if(numWifi > 0)
			{
				numCommType++;
				code += "#define NUM_WIFI_CONNECTION " + numWifi + "\n";
			}
			if(numSM > 0)
			{
				numCommType++;
				code += "#define NUM_SHARED_CONNECTION " + numSM + "\n";
			}
			
			if(numCommType > 0)
			{
				//because of #receiver + 1Sender
				code += "#define NUM_OUT_CONNECTION_TYPE " + (numCommType+1) + "\n";	
				code += "THREAD_TYPE outCommThread[NUM_OUT_CONNECTION_TYPE];\n";	
				code += "#define num_connMap (int)(sizeof(connMap)/sizeof(connMap[0]))";
			}			
		}
		
		return code;
	}
	
	/*
	private int calculateRemoteConnection(List<Communication> mCommunication, int type, Map<Integer, Processor> mProcessor, String mProcessorName, CICMappingType mMapping){
		int ret = 0;
		
		
		ArrayList<Processor> connectList = new ArrayList<Processor>();
		Map<Processor, Processor> candidateList = new Map<Processor, Processor>();
		for(Queue queue: mQueue.values())
		{			
			Processor srcProc = null;
			Processor dstProc = null;
			for(Processor proc: mProcessor.values())
			{						
				List<Task> taskList = proc.getTask(); 
				for(Task t: taskList)
				{
					if(t.getName().equals(queue.getSrc()))
					{
						srcProc = proc;
					}					
					if(t.getName().equals(queue.getDst()))
					{
						dstProc = proc;
					}
				}				
			}
			if(srcProc == dstProc)
				continue;
			else 
			{
				
			}				
		}
		
		for(Communication comm: mCommunication)
		{
			if(type != comm.getType())
				continue;
			
			for(Processor proc: mProcessor.values())
			{
				if(proc.getPoolName().equals(mProcessorName))
					continue;
				else
				{
					if(srcProc.getPoolName().equals(proc.getPoolName()) && dstProc.getPoolName().equals(mProcessorName))
					{
						if(connectList.contains(proc))
						{
							
						}
						else
						{
							connectList.add(proc);
						}
					}
					else if(srcProc.getPoolName().equals(mProcessorName) && dstProc.getPoolName().equals(proc.getPoolName()))
					{
						
					}
					for(MappingTaskType mtt: mMapping.getTask())
					{
						for(MappingProcessorIdType mpit: mtt.getProcessor())
						{
							if(mpit.getPool().equals(proc.getPoolName()))
							{
								mtt.
							}
						}
					}
				}
			}
			
			for(Processor proc: mProcessor.values())
			{
				procId++;
				//System.out.println("proc: " + proc.getPoolName() + "\t id::" + procId);
				if(mMyProcessor.getPoolName().equals(proc.getPoolName()))//to test equal? contains
					continue;
				else
				{
					for(MappingTaskType mtt: mMapping.getTask())
					{
						for(MappingProcessorIdType mpit: mtt.getProcessor())
						{
							if(mpit.getPool().equals(proc.getPoolName()) && (mtt.getName().equals(srcTask.getName()) || mtt.getName().equals(dstTask.getName())))
							{	
								//System.out.println("mpit: proc: " + mpit.getPool());
								connectionMapEntries += procId + ",";
							}
						}
					}
					;
				}					
				//if(proc.getPoolName().contains(mMyProcessor.getPoolName()))
			}
			switch(type)
			{
			case TYPE_BLUETOOTH:
				//
				
				break; 
			case TYPE_I2C:
				break;
			case TYPE_WIFI:
				break;
			case TYPE_VREPSHAREDBUS:
				break;
			default: 
				break;
			}
		}
		
		// Need to add for normal channel
		
		ArrayList<TaskLibraryConnectionType> taskLibConn = new ArrayList<TaskLibraryConnectionType>();
		for(Library library: mLibrary.values()){
			for(int i=0; i<mAlgorithm.getLibraryConnections().getTaskLibraryConnection().size(); i++){
				TaskLibraryConnectionType tlc = mAlgorithm.getLibraryConnections().getTaskLibraryConnection().get(i);
				if(tlc.getSlaveLibrary().equals(library.getName()))	taskLibConn.add(tlc);
			}
		}
		
		for(Processor proc: mProcessor.values()){
			if(proc.getPoolName().contains("IRobotCreate"))	continue;
			else{
				for(TaskLibraryConnectionType tlc: taskLibConn){
					for(MappingTaskType mtt: mMapping.getTask()){
						for(MappingProcessorIdType mpit: mtt.getProcessor()){
							if(mpit.getPool().equals(proc.getPoolName()) && mtt.getName().equals(tlc.getMasterTask())){
								ret = ret + 1;
								break;
							}
						}
					}
				}
			}
		}
		
		for(Library library: mLibrary.values()){
			int procId = library.getProc();
			Processor proc = mProcessor.get(procId);
			if(proc.getPoolName().contains("IRobotCreate")){
				boolean hasRemoteConn = false;
				for(int i=0; i<mAlgorithm.getLibraryConnections().getTaskLibraryConnection().size(); i++){
					TaskLibraryConnectionType tlc = mAlgorithm.getLibraryConnections().getTaskLibraryConnection().get(i);
					if(tlc.getSlaveLibrary().equals(library.getName()))	taskLibConn.add(tlc);
				}
			}
		}
		
		
		return ret;
	}	
	*/
	
	public static String generateCommCode(String mTranslatorPath, String mProcessorName, List<Communication> mCommunication )
	{
		String code = "";
		String commCode = "";
		String templateFile;
		List<Integer> keys = new ArrayList<Integer>();
		boolean bluetooth = false, i2c = false, wifi = false, sm = false;
					
		if(mCommunication.size() > 0)
		{
			//현재 num~~ 값은 최대 1로 가정 
			int numBluetooth = 0, numI2C = 0, numWifi = 0, numSM = 0, numCommType = 0;	
			
			//각 통신당 연결된 로봇 수 //need to fix 
			int[] numConnect = new int[4];//0: TYPE_BLUETOOTH, 1: TYPE_I2C, 2: TYPE_VREPSHAREDBUS, 3: TYPE_WIFI
			
			for(Communication com: mCommunication)
			{
				int type = com.getType();
								
				switch(type)
				{
				case TYPE_BLUETOOTH:
					numBluetooth++;
					break; 
				case TYPE_I2C:
					numI2C++;
					break;
				case TYPE_WIFI:
					numWifi++;
					break;
				case TYPE_VREPSHAREDBUS:
					numSM++;
					keys.add(com.getVRepSharedMemComm().getKey());
					break;
				default: 
					break;
				}
			}
			
			for(Communication com : mCommunication)
			{
				int type = com.getType();
				
				if(sm == false && type == TYPE_VREPSHAREDBUS)
				{
					sm = true;
					System.out.println("SHAREDMEMORYT!!!");
					VRepSharedMemComm shmcom = com.getVRepSharedMemComm();
					System.out.println(">>> " + shmcom.getMasterProc().getNodeName() + " === " + mProcessorName);
					if(shmcom.getMasterProc().getNodeName().equals(mProcessorName))
					{
						templateFile = mTranslatorPath + "templates/common/communication_wrapper/shared_memory/general_linux_master.template";
						commCode = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##COMM_MASTER_CODE");
												
						String shmPortDef = ""; 
												
						shmPortDef += "#define SEND_PORT 0\n#define RECEIVE_PORT 1 //need to check when many shm connects b/w diff robots\n";
						System.out.println("!!!!!! " + shmPortDef);					
						commCode = commCode.replace("##SHM_PORT_DEFINE ", shmPortDef);
						
					}					
					else
					{
						for(VRepSharedNode slave: shmcom.getSlaveProc())
						{
							if(slave.getNodeName().equals(mProcessorName))
							{
								templateFile = mTranslatorPath + "templates/common/communication_wrapper/shared_memory/general_linux_slave.template";
								commCode = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##COMM_SLAVE_CODE");
								
								String shmPortDef = ""; 
								
								shmPortDef += "\t#define SEND_PORT 1\n\t#define RECEIVE_PORT 0 //need to check when many shm connects b/w diff robots\n";
														
								commCode = commCode.replace("##SHM_PORT_DEFINE", shmPortDef);
							}
						}
					}
					String shmChannelDef = "";
					String shmKey = "";
					String shmChannelAssign = "";
										
					shmChannelDef += "SHM_CHANNEL* shm_channels[" + (2*numSM) + "];// 2* NUM_SHARED_CONNECTION \n";
					
					shmKey += "static int KEY_NUM[NUM_SHARED_CONNECTION] = {";
					for(int key: keys)
					{
						shmKey += key + ","; 
					}
					shmKey += "};\n";
					
					for(int i = 0; i < (2*numSM); i++)
					{
						shmChannelAssign += "\t\tshm_channels[" + i + "] = (SHM_CHANNEL*)(shm_memory";
						if(i > 0)
						{
							shmChannelAssign += " + sizeof(SHM_CHANNEL) * " + i;
						}
						
						shmChannelAssign += ");\n";
						
					}
					
					commCode = commCode.replace("##SHM_CHANNEL_DEFINE", shmChannelDef);
					commCode = commCode.replace("##SHM_KEY", shmKey);
					commCode = commCode.replace("##SHM_CHANNEL_ASSIGN", shmChannelAssign);
					
					
					code += commCode;
				}
				else if(wifi == false && type == TYPE_WIFI)
				{
					wifi = true;
					System.out.println("TYPE_WIFI!!!");
					WIFIComm wificom = com.getWifiComm();					
					if(wificom.getServerProc().mWIFIName.equals(mProcessorName))
					{
						System.out.println(">>> " + wificom.getServerProc().mWIFIName + " === " + mProcessorName);
						templateFile = mTranslatorPath + "templates/common/communication_wrapper/wifi/general_linux_master.template";
						commCode = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##COMM_MASTER_CODE");						
					}					
					else
					{
						for(WIFINode client: wificom.getClientProc())
						{
							if(client.mWIFIName.equals(mProcessorName))
							{
								System.out.println(">>>>>>> " + client.mWIFIName + " === " + mProcessorName);
								templateFile = mTranslatorPath + "templates/common/communication_wrapper/wifi/general_linux_slave.template";
								commCode = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##COMM_SLAVE_CODE");
								
								String serverIpDef = ""; 
								
								serverIpDef += "char* ipaddr = \"" + wificom.getServerProc().getIp() + "\";\n";
														
								commCode = commCode.replace("##DEF_SERVER_IP", serverIpDef);
							}
						}
					}					
					
					code += commCode;
				}
				//else if() ;
			}
		}	
		
		return code;
	}	
	
	
	public static String generateSenderCode(String mTranslatorPath, List<Communication> mCommunication)
	{
		String code = "";
		String templateFile = mTranslatorPath + "templates/common/communication_wrapper/wrapper.template"; 
		String commSenderRoutineCode = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##COMM_WRAPPER_CODE");
		boolean bluetooth = false, i2c = false, sm = false, wifi = false;
		for(Communication comm : mCommunication)
		{
			int type = comm.getType();
			
			if(sm == false && type == TYPE_VREPSHAREDBUS)
			{
				sm = true;
				templateFile = mTranslatorPath + "templates/common/communication_wrapper/shared_memory/wrapper_receiver.template";
				
				String temp = "";
				String senderSpecific = "";
				String boolSpecific = "";
				String createReceiver = "";
								
				temp = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##SENDER_SPECIFIC");
				if(bluetooth || i2c || wifi)
				{
					senderSpecific += "else ";
				}
				senderSpecific += temp;
				boolSpecific = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##BOOL_SPECIFIC");
				temp = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##CREATE_SPECIFIC_RECEIVER");
				if(bluetooth || i2c || wifi)
				{
					createReceiver += "else ";
				}
				createReceiver += temp;

				commSenderRoutineCode = commSenderRoutineCode.replace("##SENDER_SPECIFIC", senderSpecific);
				commSenderRoutineCode = commSenderRoutineCode.replace("##BOOL_SPECIFIC", boolSpecific);
				commSenderRoutineCode = commSenderRoutineCode.replace("##CREATE_SPECIFIC_RECEIVER", createReceiver);
			}
			else if(wifi == false && type == TYPE_WIFI)
			{
				wifi = true;
				templateFile = mTranslatorPath + "templates/common/communication_wrapper/wifi/wrapper_receiver.template";
				
				String temp = "";
				String senderSpecific = "";
				String boolSpecific = "";
				String createReceiver = "";
								
				temp = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##SENDER_SPECIFIC");
				if(bluetooth || i2c || sm)
				{
					senderSpecific += "else ";
				}
				senderSpecific += temp;
				boolSpecific = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##BOOL_SPECIFIC");
				temp = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##CREATE_SPECIFIC_RECEIVER");
				if(bluetooth || i2c || sm)
				{
					createReceiver += "else ";
				}
				createReceiver += temp;

				commSenderRoutineCode = commSenderRoutineCode.replace("##SENDER_SPECIFIC", senderSpecific);
				commSenderRoutineCode = commSenderRoutineCode.replace("##BOOL_SPECIFIC", boolSpecific);
				commSenderRoutineCode = commSenderRoutineCode.replace("##CREATE_SPECIFIC_RECEIVER", createReceiver);
			}
			
			code += commSenderRoutineCode;
		}
		return code;
	}	
	
	public static String generateReceiverCode(String mTranslatorPath, List<Communication> mCommunication)
	{
		String code = "";
		String commReceiverRoutine = "";
		String templateFile;
		boolean bluetooth = false, i2c = false, sm = false, wifi = false;
		for(Communication comm: mCommunication)
		{
			int type = comm.getType();
			
			if(sm == false && type == TYPE_VREPSHAREDBUS)
			{
				sm = true;
				templateFile = mTranslatorPath + "templates/common/communication_wrapper/shared_memory/wrapper_receiver.template";
				commReceiverRoutine = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##RECEIVER_SPECIFIC");
				
			}
			else if(wifi == false && type == TYPE_WIFI)
			{
				wifi = true;
				templateFile = mTranslatorPath + "templates/common/communication_wrapper/wifi/wrapper_receiver.template";
				commReceiverRoutine = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##RECEIVER_SPECIFIC");				
			}
			code += commReceiverRoutine;
		}
		return code;
	}	
	
}
