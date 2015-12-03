package Translators;

import hopes.cic.xml.CICAlgorithmType;
import hopes.cic.xml.CICControlType;
import hopes.cic.xml.CICGPUSetupType;
import hopes.cic.xml.CICMappingType;
import hopes.cic.xml.CICScheduleType;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import InnerDataStructures.Library;
import InnerDataStructures.Processor;
import InnerDataStructures.Communication;
import InnerDataStructures.Queue;
import InnerDataStructures.Task;

public class CICDistributedRobotsTranslator implements CICTargetCodeTranslator {
	private String mTarget;
	private String mTranslatorPath;
	private String mOutputPath;
	private String mRootPath;
	private String mCICXMLFile;
	private int mGlobalPeriod;
	private String mGlobalPeriodMetric;
	private String mThreadVer;
	private String mCodeGenType;
	private String mLanguage;
	
	private Map<String, Task> mTask;
	private Map<Integer, Queue> mQueue;
	private Map<String, Library> mLibrary;
	private Map<Integer, Processor> mProcessor;
	private List<Communication> mCommunication;
	
	private Map<String, Task> mVTask;
	private Map<String, Task> mPVTask;
	
	private CICAlgorithmType mAlgorithm;
	private CICControlType mControl;
	private CICScheduleType mSchedule;
	private CICGPUSetupType mGpusetup;
	private CICMappingType mMapping;

	private String strategy;
	
	public int generateCodeWithComm(String target, String translatorPath, String outputPath, String rootPath, Map<Integer, Processor> processor, List<Communication> communication, Map<String, Task> task, Map<Integer, Queue> queue, Map<String, Library> library, Map<String, Library> globalLibrary, int globalPeriod, String globalPeriodMetric, String cicxmlfile, String language, String threadVer, CICAlgorithmType algorithm, CICControlType control, CICScheduleType schedule, CICGPUSetupType gpusetup, CICMappingType mapping, Map<Integer, List<Task>> connectedtaskgraph,  Map<Integer, List<List<Task>>> connectedsdftaskset, Map<String, Task> vtask, Map<String, Task> pvtask, String codegentype) throws FileNotFoundException
	{
		mTarget = target;
		mTranslatorPath = translatorPath;
		mOutputPath = outputPath;
		mRootPath = rootPath;
		mCICXMLFile = cicxmlfile;
		mGlobalPeriod = globalPeriod;
		mGlobalPeriodMetric = globalPeriodMetric;
		mThreadVer = threadVer;
		mCodeGenType = codegentype;
		mLanguage = language;
		
		mTask = task;
		mQueue = queue;
		mLibrary = library;
		mProcessor = processor;
		mCommunication = communication;
		
		mVTask = vtask;
		mPVTask = pvtask;
		
		mAlgorithm = algorithm;
		mControl = control;
		mSchedule = schedule;
		mGpusetup = gpusetup;
		mMapping = mapping;
		
		int ret = 0;
		
		
		for(Processor proc: mProcessor.values()){
			if(proc.getPoolName().contains("TIEvalbot")){
				Map<String, Task> mEvalbotTask = new HashMap<String, Task>();
				Map<Integer, Queue> mEvalbotQueue = new HashMap<Integer, Queue>();
				Map<String, Library> mEvalbotLibrary= new HashMap<String, Library>();
				
				seperateDataStructure("TIEvalbot", mEvalbotTask, mEvalbotQueue, mEvalbotLibrary);
				seperateDataStructure("Arduino", mEvalbotTask, mEvalbotQueue, mEvalbotLibrary);
				String mEvalbotOutputPath = mOutputPath + "TIEvalbot/";				
				File evalF = new File(mEvalbotOutputPath);			
				evalF.mkdir();

				CICTargetCodeTranslator translator = new CICTIEvalbotTranslator();   
				try {
					ret = translator.generateCodeWithComm(mTarget, mTranslatorPath, mEvalbotOutputPath, mRootPath, mProcessor, mCommunication , mEvalbotTask, mEvalbotQueue, mEvalbotLibrary, mLibrary, mGlobalPeriod, mGlobalPeriodMetric, mCICXMLFile, mLanguage, mThreadVer, mAlgorithm, mControl, mSchedule, mGpusetup, mMapping, null, null, mVTask, mPVTask, mCodeGenType);
				} catch (FileNotFoundException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
			else if(proc.getPoolName().contains("NXT")){
				Map<String, Task> mNXTTask = new HashMap<String, Task>();
				Map<Integer, Queue> mNXTQueue = new HashMap<Integer, Queue>();
				Map<String, Library> mNXTLibrary= new HashMap<String, Library>();
				
				seperateDataStructure("NXT", mNXTTask, mNXTQueue, mNXTLibrary);
				String mNXTOutputPath = mOutputPath + "NXT/";				
				File nxtF = new File(mNXTOutputPath);			
				nxtF.mkdir();

				CICTargetCodeTranslator translator = new CICNXTPCTranslator();
				try {
					ret = translator.generateCodeWithComm(mTarget, mTranslatorPath, mNXTOutputPath, mRootPath, mProcessor, mCommunication , mNXTTask, mNXTQueue, mNXTLibrary, mLibrary, mGlobalPeriod, mGlobalPeriodMetric, mCICXMLFile, mLanguage, mThreadVer, mAlgorithm, mControl, mSchedule, mGpusetup, mMapping, null, null, mVTask, mPVTask, mCodeGenType);
				} catch (FileNotFoundException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}

			}
			else if(proc.getPoolName().contains("IRobotCreate")){
				Map<String, Task> mIRobotCreateTask = new HashMap<String, Task>();
				Map<Integer, Queue> mIRobotCreateQueue = new HashMap<Integer, Queue>();
				Map<String, Library> mIRobotCreateLibrary= new HashMap<String, Library>();
				
				seperateDataStructure("IRobotCreate", mIRobotCreateTask, mIRobotCreateQueue, mIRobotCreateLibrary);
				String mIRobotCreateOutputPath = mOutputPath + "IRobotCreate/";				
				File nxtF = new File(mIRobotCreateOutputPath);			
				nxtF.mkdir();
				
				CICTargetCodeTranslator translator = new CICIRobotCreateTranslator();
				try {
					ret = translator.generateCodeWithComm(mTarget, mTranslatorPath, mIRobotCreateOutputPath, mRootPath, mProcessor, mCommunication, mIRobotCreateTask, mIRobotCreateQueue, mIRobotCreateLibrary, mLibrary, mGlobalPeriod, mGlobalPeriodMetric, mCICXMLFile, mLanguage, mThreadVer, mAlgorithm, mControl, mSchedule, mGpusetup, mMapping, null, null, mVTask, mPVTask, mCodeGenType);
				} catch (FileNotFoundException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
		 
	    return 0;
	}
		
	public void seperateDataStructure(String target, Map<String, Task> mTargetTask, Map<Integer, Queue> mTargetQueue, Map<String, Library> mTargetLibrary){
		for(Task t: mTargetTask.values()){
			Map<String, Map<String, List<Integer>>> plmapmap = t.getProc();
			for(Map<String, List<Integer>> plmap: plmapmap.values()){
				for(List<Integer> pl: plmap.values()){
					for(int p: pl){
						if(mProcessor.get(p).getPoolName().contains(target)){
							mTargetTask.put(t.getName(), t);
						}
					}
				}
			}
		}
		/*
		for(Processor proc: mProcessor.values()){
			List<Task> taskList = proc.getTask();
			if(proc.getPoolName().contains(target)){
				int index = 0;
				for(Task t: taskList){
					t.setIndex(index++);
					mTargetTask.put(t.getName(), t);
				}
			}
		}
		*/
		
		if(mLibrary != null){
			for(Library lib: mLibrary.values()){
				if(mProcessor.get(lib.getProc()).getPoolName().contains(target))			mTargetLibrary.put(lib.getName(), lib);
			}
		}
		
		int a_index = 0, t_index = 0;
		for(Queue q: mQueue.values()){
			if(mTargetTask.containsKey(q.getSrc()) || mTargetTask.containsKey(q.getDst()))	{mTargetQueue.put(Integer.parseInt(q.getIndex()), q);}
		}
	}

	@Override
	public int generateCode(String mTarget, String mTranslatorPath,
			String mOutputPath, String mRootPath,
			Map<Integer, Processor> mProcessor, Map<String, Task> mTask,
			Map<Integer, Queue> mQueue, Map<String, Library> mLibrary,
			Map<String, Library> mGlobalLibrary, int mGlobalPeriod,
			String mGlbalPeriodMetric, String mCICXMLFile, String language,
			String threadVer, CICAlgorithmType mAlgorithm,
			CICControlType mControl, CICScheduleType mSchedule,
			CICGPUSetupType mGpusetup, CICMappingType mMapping,
			Map<Integer, List<Task>> mConnectedTaskGraph,
			Map<Integer, List<List<Task>>> mConnectedSDFTaskSet,
			Map<String, Task> mVTask, Map<String, Task> mPTask, String mCodeGenType) throws FileNotFoundException {
		// TODO Auto-generated method stub
		return 0;
	}
	
}
