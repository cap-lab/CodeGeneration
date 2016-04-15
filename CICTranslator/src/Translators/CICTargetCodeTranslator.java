package Translators;

import java.io.FileNotFoundException;
import java.util.*;

import CommonLibraries.*;
import InnerDataStructures.*;
import InnerDataStructures.Communication;
import InnerDataStructures.Library;
import InnerDataStructures.Queue;
import hopes.cic.xml.*;

public interface CICTargetCodeTranslator {

	public int generateCode(String mTarget, String mTranslatorPath, String mOutputPath, String mRootPath,
			Map<Integer, Processor> mProcessor, Map<String, Task> mTask, Map<Integer, Queue> mQueue,
			Map<String, Library> mLibrary, Map<String, Library> mGlobalLibrary, int mGlobalPeriod,
			String mGlbalPeriodMetric, String mCICXMLFile, String language, CICAlgorithmType mAlgorithm,
			CICControlType mControl, CICScheduleType mSchedule, CICGPUSetupType mGpusetup, CICMappingType mMapping,
			Map<Integer, List<Task>> mConnectedTaskGraph, Map<Integer, List<List<Task>>> mConnectedSDFTaskSet,
			Map<String, Task> mVTask, Map<String, Task> mPVTask, String mRuntimeExecutionPolicy, String codeGenerationStyle)
					throws FileNotFoundException;

	public int generateCodeWithComm(String mTarget, String mTranslatorPath, String mOutputPath, String mRootPath,
			Map<Integer, Processor> mProcessor, List<Communication> mCommunication, Map<String, Task> mTask,
			Map<Integer, Queue> mQueue, Map<String, Library> mLibrary, Map<String, Library> mGlobalLibrary,
			int mGlobalPeriod, String mGlbalPeriodMetric, String mCICXMLFile, String language,
			CICAlgorithmType mAlgorithm, CICControlType mControl, CICScheduleType mSchedule, CICGPUSetupType mGpusetup,
			CICMappingType mMapping, Map<Integer, List<Task>> mConnectedTaskGraph,
			Map<Integer, List<List<Task>>> mConnectedSDFTaskSet, Map<String, Task> mVTask, Map<String, Task> mPVTask,
			String mRuntimeExecutionPolicy, String codeGenerationStyle) throws FileNotFoundException;

}
