package Translators;

import java.io.*;
import java.math.*;
import java.util.*;

import CommonLibraries.*;
import InnerDataStructures.*;
import InnerDataStructures.Library;
import InnerDataStructures.Queue;
import hopes.cic.xml.*;

public class CICCudaTranslator implements CICTargetCodeTranslator {
	private String mTarget;
	private String mCICXMLFile;
	private String mOutputPath;
	private String mTranslatorPath;
	private String mRootPath;
	private CICAlgorithmType mAlgorithm;
	private CICArchitectureType mArchitecture;
	private CICMappingType mMapping;
	private CICControlType mControl;
	private CICConfigurationType mConfiguration;
	private CICProfileType mProfile;
	private CICGPUSetupType mGpusetup;
	private CICScheduleType mSchedule;
	private CICDeviceIOType mDeviceIO;
	
	private Map<String, Task> mTask;
	private Map<String, Library> mLibrary;
	private Map<Integer, Queue> mQueue;
	private Map<Integer, Processor> mProcessor;
	private Map<String, LibraryStub> mLibraryStub;
	
	private Map<String, Task> mVTask;
	private Map<String, Task> mPVTask;
	
	private String mGlobalPeriodMetric;
	private int mGlobalPeriod;
	private int mTotalControlQueue;
	
	private String mThreadVer;
	private String mCodeGenType;
	private String mLanguage;
	
	
	@Override
	public int generateCode(String target, String translatorPath, String outputPath, String rootPath, Map<Integer, Processor> processor, Map<String, Task> task, Map<Integer, Queue> queue, Map<String, Library> library, Map<String, Library> globalLibrary, int globalPeriod, String globalPeriodMetric, String cicxmlfile, String language, String threadVer, CICAlgorithmType algorithm, CICControlType control, CICScheduleType schedule, CICGPUSetupType gpusetup, CICMappingType mapping, Map<Integer, List<Task>> connectedtaskgraph,  Map<Integer, List<List<Task>>> connectedsdftaskset, Map<String, Task> vtask, Map<String, Task> pvtask, String codegentype) throws FileNotFoundException
	{
		int ret = 0;
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
		
		mVTask = vtask;
		mPVTask = pvtask;
		
		mAlgorithm = algorithm;
		mControl = control;
		mSchedule = schedule;
		mGpusetup = gpusetup;
		mMapping = mapping;
		
		String fileOut = new String();
		String templateFile = new String();
				
		Util.copyFile(mOutputPath+"target_task_model.h", mTranslatorPath + "templates/common/task_model/pthread.template");
		Util.copyFile(mOutputPath+"target_system_model.h", mTranslatorPath + "templates/common/system_model/general_linux.template");
		
		// generate cic_tasks.h
		fileOut = mOutputPath + "task_def.h";
		templateFile = mTranslatorPath + "templates/common/common_template/task_def.h.template";
		CommonLibraries.CIC.generateTaskDataStructure(fileOut, templateFile, mTask, mGlobalPeriod, mGlobalPeriodMetric, mThreadVer, mCodeGenType, mVTask, mPVTask);
		
		
		// generate task_name.c (include task_name.cic)
		for(Task t: mTask.values()){
			int isGPU = 0;
			for(GPUTaskType gpuTask: mGpusetup.getTasks().getTask()){
				if(gpuTask.getName().equals(t.getName())){
					isGPU = 1;
					break;
				}
			}
			if(isGPU == 0){
				fileOut = mOutputPath + t.getName() + ".c";
				templateFile = mTranslatorPath + "templates/common/common_template/task_code_template.c";
				CommonLibraries.CIC.generateTaskCode(fileOut, templateFile, t, mAlgorithm, mControl);
			}
			else if(isGPU == 1)	{
				fileOut = mOutputPath + t.getName() + ".cu";
				templateFile = mTranslatorPath + "templates/target/cuda/task_code_template.c";
				generateTaskCode(fileOut, templateFile, t);
			}		
		}
		
		// Copy CIC_CUDA_port.h file
		Util.copyFile(mOutputPath+"CIC_CUDA_port.h", mTranslatorPath + "templates/target/cuda/CIC_CUDA_port.h");
		
		// Copy cic_gpuinfo.h file
		Util.copyFile(mOutputPath+"cic_gpuinfo.h", mTranslatorPath + "templates/target/cuda/cic_gpuinfo.h");		
		
		if(mGpusetup != null){
			for(GPUTaskType gpuTask: mGpusetup.getTasks().getTask()){
				if(gpuTask.getClustering().equals(gpuTask.getClustering().YES)){
					if(gpuTask.getPipelining().equals(gpuTask.getPipelining().YES)){
						if(gpuTask.getMaxStream().intValue() <= 2)
							Util.copyFile(mOutputPath+"CIC_port_cuda_cluster_async_lt2.c", mTranslatorPath + "templates/target/cuda/device_channel_manage/CIC_port_cuda_cluster_async_lt2.c");	
						else if(gpuTask.getMaxStream().intValue() == 3)
							Util.copyFile(mOutputPath+"CIC_port_cuda_cluster_async_eq3.c", mTranslatorPath + "templates/target/cuda/device_channel_manage/CIC_port_cuda_cluster_async_eq3.c");
						else if(gpuTask.getMaxStream().intValue() >= 4)
							Util.copyFile(mOutputPath+"CIC_port_cuda_cluster_async_mt4.c", mTranslatorPath + "templates/target/cuda/device_channel_manage/CIC_port_cuda_cluster_async_mt4.c");
					}
					else
						Util.copyFile(mOutputPath+"CIC_port_cuda_cluster_sync.c", mTranslatorPath + "templates/target/cuda/device_channel_manage/CIC_port_cuda_cluster_sync.c");
				}
				else{
					if(gpuTask.getPipelining().equals(gpuTask.getPipelining().YES)){
						if(gpuTask.getMaxStream().intValue() <= 2)
							Util.copyFile(mOutputPath+"CIC_port_cuda_bypass_async_lt2.c", mTranslatorPath + "templates/target/cuda/device_channel_manage/CIC_port_cuda_bypass_async_lt2.c");	
						else if(gpuTask.getMaxStream().intValue() == 3)
							Util.copyFile(mOutputPath+"CIC_port_cuda_bypass_async_eq3.c", mTranslatorPath + "templates/target/cuda/device_channel_manage/CIC_port_cuda_bypass_async_eq3.c");
						else if(gpuTask.getMaxStream().intValue() >= 4)
							Util.copyFile(mOutputPath+"CIC_port_cuda_bypass_async_mt4.c", mTranslatorPath + "templates/target/cuda/device_channel_manage/CIC_port_cuda_bypass_async_mt4.c");
					}
					else
						Util.copyFile(mOutputPath+"CIC_port_cuda_bypass_sync.c", mTranslatorPath + "templates/target/cuda/device_channel_manage/CIC_port_cuda_bypass_sync.c");
				}
			}
		}
		
		// generate proc.c
		fileOut = mOutputPath + "proc.cu";
		templateFile = mTranslatorPath + "templates/common/common_template/proc.c.template";
		generateProcCode(fileOut, templateFile);
	    
		// generate Makefile
	    fileOut = mOutputPath + "Makefile";
	    generateMakefile(fileOut);
		
		return 0;
	}
	
	public void generateTaskCode(String mDestFile, String mTemplateFile, Task task)
	{
		File fileOut = new File(mDestFile);
		File templateFile = new File(mTemplateFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);
			
			outstream.write(translateTaskCode(content, task).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public String translateTaskCode(String mContent, Task task){
		String code = mContent;
		String libraryDef = "";
		String cicInclude = "";
		String sysPortDef = "";
		
		sysPortDef = "#define SYS_REQ(x, ...) SYS_REQ_##x(__VA_ARGS__)\n ";

		if(mAlgorithm.getLibraries() != null){
			libraryDef += "#define LIBCALL(x, ...) LIBCALL_##x(__VA_ARGS__)\n";
			for(TaskLibraryConnectionType taskLibCon: mAlgorithm.getLibraryConnections().getTaskLibraryConnection()){
				if(taskLibCon.getMasterTask().equals(task.getName())){
					libraryDef += "\n#include \"" + taskLibCon.getSlaveLibrary()+".h\"\n";
					libraryDef += "#define LIBCALL_" + taskLibCon.getMasterPort() + "(f, ...) l_" + taskLibCon.getSlaveLibrary() + "_##f(__VA_ARGS__)\n";
				}
			}
		}
		
		int isGPU = 0;
		GPUTaskType gpuTask = null;
		for(GPUTaskType gTask: mGpusetup.getTasks().getTask()){
			if(gTask.getName().equals(task.getName())){
				isGPU = 1;
				gpuTask = gTask;
				break;
			}
		}
		
		if(isGPU == 1){
			String taskIndexDef = "";
			String syncDef = "";
			String devBufDef = "";
			String cicPortInclude = "";
			String dimmDef = "";
			String kernelDef = "";
			String prePostDef = "";
			
			prePostDef += "#define TASK_PREINIT void " + task.getName() + "_preinit()\n"
					   + "#define TASK_POSTWRAPUP void " + task.getName() + "_postwrapup()\n"
					   + "#define TASK_PREGO void " + task.getName() + "_prego()\n"
					   + "#define TASK_POSTGO void " + task.getName() + "_postgo()\n";
			
			int bufferCount = 0;
			int bufferFlag = 0;
			List<String> bufferPort = new ArrayList<String>();
			for(Queue queue: task.getQueue()){
				bufferFlag = 0;
				for(String tPort: bufferPort)
					if(queue.getSrc().equals(tPort))
						if(queue.getIndex() != mTask.get(tPort).getIndex())	bufferFlag = 1;
				if(bufferFlag == 0){
					bufferPort.add(queue.getSrc());
					bufferCount++;
				}
			}
	
			int globalPeriod = 0;
			if(mGlobalPeriodMetric.equalsIgnoreCase("h"))			globalPeriod = mGlobalPeriod * 3600 * 1000 * 1000;
			else if(mGlobalPeriodMetric.equalsIgnoreCase("m"))		globalPeriod = mGlobalPeriod * 60 * 1000 * 1000;
			else if(mGlobalPeriodMetric.equalsIgnoreCase("s"))		globalPeriod = mGlobalPeriod * 1000 * 1000;
			else if(mGlobalPeriodMetric.equalsIgnoreCase("ms"))		globalPeriod = mGlobalPeriod * 1000;
			else if(mGlobalPeriodMetric.equalsIgnoreCase("us"))		globalPeriod = mGlobalPeriod * 1;
			else{
				System.out.println("[makeTasks] Not supported metric of period");
				System.exit(-1);
			}
			int taskRunCount = globalPeriod / Integer.parseInt(task.getPeriod());
			
			taskIndexDef += "#define TASK_INDEX (" + task.getIndex() + ")\n";
			
			if(gpuTask.getClustering().equals(gpuTask.getClustering().YES)){
				if(gpuTask.getPipelining().equals(gpuTask.getPipelining().YES)){
					if(gpuTask.getMaxStream().intValue() == 2){
						syncDef += "static int __current_stream;\nstatic int __previous_1_stream;\n"
								+ "static int __run_count;\n";
					}
					else if(gpuTask.getMaxStream().intValue() == 3){
						syncDef += "static int __current_stream;\nstatic int __previous_1_stream;\nstatic int __oldest_stream;\n"
								+ "static int __run_count;\n";
					}
					else{
						syncDef += "static int __current_stream;\nstatic int __previous_1_stream;\n" 
								+ "static int __previous_2_stream;\nstatic int __oldest_stream;\n"
								+ "static int __run_count;\n";
					}
					
					syncDef += "static cudaStream_t __stream[" + gpuTask.getMaxStream() + "];\n";
					syncDef += "static int __num_of_streams = " + gpuTask.getMaxStream() + ";\n";
					syncDef += "static int __runs = " + taskRunCount + ";\n";
					syncDef += "static int __num_of_channels = " + bufferCount + ";\n\n";
					syncDef += "static unsigned char *__in_dev_buf[" + gpuTask.getMaxStream() + "];\n";
					syncDef += "static unsigned char *__out_dev_buf[" + gpuTask.getMaxStream() + "];\n";
				}
				else{
					syncDef += "static int __num_of_channels = " + bufferCount + ";\n\n";
					syncDef += "static unsigned char *__in_dev_buf;\n";
					syncDef += "static unsigned char *__out_dev_buf;\n";
				}
			}
			else{
				if(gpuTask.getPipelining().equals(gpuTask.getPipelining().YES)){
					if(gpuTask.getMaxStream().intValue() == 2){
						syncDef += "static int __current_stream;\nstatic int __previous_1_stream;\n"
								+ "static int __run_count;\n";
					}
					else if(gpuTask.getMaxStream().intValue() == 3){
						syncDef += "static int __current_stream;\nstatic int __previous_1_stream;\nstatic int __oldest_stream;\n"
								+ "static int __run_count;\n";
					}
					else{
						syncDef += "static int __current_stream;\nstatic int __previous_1_stream;\n" 
								+ "static int __previous_2_stream;\nstatic int __oldest_stream;\n"
								+ "static int __run_count;\n";
					}
					syncDef += "static cudaStream_t __stream[" + gpuTask.getMaxStream() + "];\n";
					syncDef += "static int __num_of_streams = " + gpuTask.getMaxStream() + ";\n";
					syncDef += "static int __runs = " + taskRunCount + ";\n";
					syncDef += "static int __num_of_channels = " + bufferCount + ";\n\n";
					syncDef += "static unsigned char *__dev_buf[" + bufferCount + "][" + gpuTask.getMaxStream() + "];\n";
				}
				else{
					syncDef += "static int __num_of_channels = " + bufferCount + ";\n\n";
					syncDef += "static unsigned char *__dev_buf[" + bufferCount + "];\n";
				}
			}
			
			int inBufIncr = 0;
			int outBufIncr = 0;
			int count = 0;
			List<String> srcPort = new ArrayList<String>();
			devBufDef += "#define DEVICE_BUFFER(name) DEVICE_BUFFER_##name\n\n";
			for(Queue queue: task.getQueue()){
				if(queue.getDst().equals(task.getName())){
					if(gpuTask.getClustering().equals(gpuTask.getClustering().YES)){
						if(gpuTask.getPipelining().equals(gpuTask.getPipelining().YES)){
							devBufDef += "#define DEVICE_BUFFER_" + queue.getDstPortName() + " (" + queue.getSampleType()
							          + "*)__in_dev_buf[__current_stream]+(" + outBufIncr + "/sizeof(" 
							          + queue.getSampleType() + "))\n";
						}
						else{
							devBufDef += "#define DEVICE_BUFFER_" + queue.getDstPortName() + " (" + queue.getSampleType()
					          + "*)__in_dev_buf+(" + outBufIncr + "/sizeof(" 
					          + queue.getSampleType() + "))\n";
						}
						outBufIncr += (Integer.parseInt(queue.getSize()) * Integer.parseInt(queue.getSampleSize())); 
					}
					else{
						if(gpuTask.getPipelining().equals(gpuTask.getPipelining().YES)){
							devBufDef += "#define DEVICE_BUFFER_" + queue.getDstPortName() + " (" + queue.getSampleType()
							          + "*)__dev_buf[" + count + "][__current_stream]\n";
						}
						else{
							devBufDef += "#define DEVICE_BUFFER_" + queue.getDstPortName() + " (" + queue.getSampleType()
					          + "*)__dev_buf[" + count + "]\n";
						}
						count++;
					}
				}
				else if(queue.getSrc().equals(task.getName())){
					int notAdd = 0;
					for(String srcPortId: srcPort){
						if(srcPortId.equals(queue.getSrcPortId())){
							notAdd = 1;
							break;
						}
					}
					if(notAdd == 0){
						srcPort.add(queue.getSrcPortId());
						
						if(gpuTask.getClustering().equals(gpuTask.getClustering().YES)){
							if(gpuTask.getPipelining().equals(gpuTask.getPipelining().YES)){
								if(gpuTask.getMaxStream().intValue() >= 4){
									devBufDef += "#define DEVICE_BUFFER_" + queue.getSrcPortName() + " (" + queue.getSampleType()
								          + "*)__out_dev_buf[__previous_2_stream]+(" + inBufIncr + "/sizeof(" 
								          + queue.getSampleType() + "))\n";
								}
								else{
									devBufDef += "#define DEVICE_BUFFER_" + queue.getSrcPortName() + " (" + queue.getSampleType()
							          + "*)__out_dev_buf[__previous_1_stream]+(" + inBufIncr + "/sizeof(" 
							          + queue.getSampleType() + "))\n";
								}
								inBufIncr += (Integer.parseInt(queue.getSize()) * Integer.parseInt(queue.getSampleSize())); 
							}
							else{
								devBufDef += "#define DEVICE_BUFFER_" + queue.getSrcPortName() + " (" + queue.getSampleType()
						          + "*)__out_dev_buf+(" + inBufIncr + "/sizeof(" 
						          + queue.getSampleType() + "))\n";
							}
						}
						else{
							if(gpuTask.getPipelining().equals(gpuTask.getPipelining().YES)){
								if(gpuTask.getMaxStream().intValue() >= 4){
									devBufDef += "#define DEVICE_BUFFER_" + queue.getSrcPortName() + " (" + queue.getSampleType()
								          + "*)__dev_buf[" + count + "][__previous_2_stream]\n";
								}
								else{
									devBufDef += "#define DEVICE_BUFFER_" + queue.getSrcPortName() + " (" + queue.getSampleType()
							          + "*)__dev_buf[" + count + "][__previous_1_stream]\n";
								}
							}
							else{
								devBufDef += "#define DEVICE_BUFFER_" + queue.getSrcPortName() + " (" + queue.getSampleType()
						          + "*)__dev_buf[" + count + "]\n";
							}
							count++;
						}
					}
				}
			}
			
			if(gpuTask.getClustering().equals(gpuTask.getClustering().YES)){
				if(gpuTask.getPipelining().equals(gpuTask.getPipelining().YES)){
					if(gpuTask.getMaxStream().intValue() <= 2)
						cicPortInclude = "#include \"CIC_port_cuda_cluster_async_lt2.c\"\n";
					else if(gpuTask.getMaxStream().intValue() == 3)
						cicPortInclude = "#include \"CIC_port_cuda_cluster_async_eq3.c\"\n";
					else if(gpuTask.getMaxStream().intValue() >= 4)
						cicPortInclude = "#include \"CIC_port_cuda_cluster_async_mt4.c\"\n";
				}
				else
					cicPortInclude += "#include \"CIC_port_cuda_cluster_sync.c\"\n";
			}
			else{
				if(gpuTask.getPipelining().equals(gpuTask.getPipelining().YES)){
					if(gpuTask.getMaxStream().intValue() <= 2)
						cicPortInclude += "#include \"CIC_port_cuda_bypass_async_lt2.c\"\n";
					else if(gpuTask.getMaxStream().intValue() == 3)
						cicPortInclude += "#include \"CIC_port_cuda_bypass_async_eq3.c\"\n";
					else if(gpuTask.getMaxStream().intValue() >= 4)
						cicPortInclude += "#include \"CIC_port_cuda_bypass_async_mt4.c\"\n";
				}
				else
					cicPortInclude += "#include \"CIC_port_cuda_bypass_sync.c\"\n";
			}
			
			int thread[] = {1, 1, 1};
			int grid[] = {1, 1, 1};
			
			count = 0;
			for(BigInteger value: gpuTask.getGlobalWorkSize().getValue()){
				grid[count] = value.intValue();
				count++;
			}
			count = 0;
			for(BigInteger value: gpuTask.getLocalWorkSize().getValue()){
				thread[count] = value.intValue();
				count++;
			}
			
			dimmDef += "#define THREADS __threads\n#define GRID __grid\n";
			dimmDef += "#define THREAD_X " + thread[0] + "\n";
			dimmDef += "#define THREAD_Y " + thread[1] + "\n";
			dimmDef += "#define THREAD_Z " + thread[2] + "\n";
			dimmDef += "#define GRID_X " + grid[0] + "\n";
			dimmDef += "#define GRID_Y " + grid[1] + "\n";
			dimmDef += "#define GRID_Z " + grid[2] + "\n";
			dimmDef += "static dim3 __threads(THREAD_X, THREAD_Y, THREAD_Z);\nstatic dim3 __grid(GRID_X, GRID_Y, GRID_Z);";
			
			if(gpuTask.getPipelining().equals(gpuTask.getPipelining().YES)){
				if(gpuTask.getMaxStream().intValue() <= 2)
					kernelDef += "#define KERNEL_CALL(x, ...) ({int __temp_stream = __previous_1_stream; __previous_1_stream = __current_stream; if(__run_count < __runs) x<<<GRID, THREADS, 0, __stream[__current_stream]>>>(__VA_ARGS__); __previous_1_stream = __temp_stream;})\n";
				else if(gpuTask.getMaxStream().intValue() == 3)
					kernelDef += "#define KERNEL_CALL(x, ...) ({int __temp_stream = __current_stream; __current_stream = __previous_1_stream; if(__run_count > 0 && __run_count <= __runs) x<<<GRID, THREADS, 0, __stream[__previous_1_stream]>>>(__VA_ARGS__); __current_stream = __temp_stream;})\n";
				else if(gpuTask.getMaxStream().intValue() >= 4)
					kernelDef += "#define KERNEL_CALL(x, ...) ({int __temp_1_stream = __current_stream; int __temp_2_stream = __previous_2_stream; __current_stream = __previous_1_stream; __previous_2_stream = __previous_1_stream; if(__run_count > 0 && __run_count <= __runs) x<<<GRID, THREADS, 0, __stream[__previous_1_stream]>>>(__VA_ARGS__); if(__run_count > 1 && __run_count <= __runs+1) CUDA_ERROR_CHECK(cudaStreamSynchronize(__stream[__previous_2_stream])); __current_stream = __temp_1_stream; __previous_2_stream = __temp_2_stream;})\n";
			}
			else
				kernelDef += "#define KERNEL_CALL(x, ...) x<<<GRID, THREADS>>>(__VA_ARGS__);\n";
			
			code = code.replace("##TASK_INDEX_DEFINITION", taskIndexDef);
			code = code.replace("##SYNC_DEFINITION", syncDef);
			code = code.replace("##DEVICE_BUFFER_DEFINITION", devBufDef);
			code = code.replace("##CIC_PORT_CUDA_INCLUDE", cicPortInclude);
			code = code.replace("##DIMM_DEFINITION", dimmDef);
			code = code.replace("##KERNEL_DEFINITION", kernelDef);
			code = code.replace("##PRE_POST_DEFINITION", prePostDef);
		}
		
		cicInclude += "\n#include \""+task.getCICFile()+"\"\n";
		
		code = code.replace("##LIBRARY_DEFINITION", libraryDef);
		code = code.replace("##TASK_NAME", task.getName());
		code = code.replace("##TASK_INSTANCE_NAME", task.getName());
		code = code.replace("##SYSPORT_DEFINITION", sysPortDef);
		code = code.replace("##CIC_INCLUDE", cicInclude);
		code = code.replace("##MTM_DEFINITION", "");
			
		return code;
	}

	public void generateProcCode(String mDestFile, String mTemplateFile){
		File fileOut = new File(mDestFile);
		File templateFile = new File(mTemplateFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);
			
			outstream.write(translateProcCode(content).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public String translateProcCode(String mContent)
	{
		String code = mContent;

		// TARGET_DEPENDENT_IMPLEMENTATION //
		String targetDependentImpl = "";
	
		// generate external pre post functions & task support entry code
		String externalPrePostFunctions = new String();
		String setPrePostFunction = new String();
		
		setPrePostFunction += "void setPrePostFunction(){\n";
		int index = 0;
		while(index < mTask.size()) {
			Task task = null;
			for(Task t: mTask.values()){
				if(Integer.parseInt(t.getIndex()) == index){
					task = t;
					break;
				}
			}
			int gpuFlag = 0;
			int asyncFlag = 0;
			for(GPUTaskType gpuTask: mGpusetup.getTasks().getTask()){
				if(gpuTask.getName().equals(task.getName())){
					gpuFlag = 1;
					if(gpuTask.getPipelining().equals(gpuTask.getPipelining().YES)){
						asyncFlag = 1;
						break;
					}
				}
			}
			if(gpuFlag == 1){
				externalPrePostFunctions += "extern void "+task.getName()+"_prego(void);\n";
				externalPrePostFunctions += "extern void "+task.getName()+"_postgo(void);\n";
				externalPrePostFunctions += "extern void "+task.getName()+"_preinit(void);\n";
				externalPrePostFunctions += "extern void "+task.getName()+"_postwrapup(void);\n";
				
				setPrePostFunction += "\ttasks[" + task.getIndex() + "].prego = " + task.getName() + "_prego;\n";
				setPrePostFunction += "\ttasks[" + task.getIndex() + "].postgo = " + task.getName() + "_postgo;\n";
				setPrePostFunction += "\ttasks[" + task.getIndex() + "].preinit = " + task.getName() + "_preinit;\n";
				setPrePostFunction += "\ttasks[" + task.getIndex() + "].postwrapup = " + task.getName() + "_postwrapup;\n\n";

			}
			index++;
		}
		setPrePostFunction += "}\n";
		
		// generate additional task info + adjust task runcount
		String gpuTaskInfoDataStructure = new String();
		String adjustTaskRuncountFunction = new String();
		
		gpuTaskInfoDataStructure += "GPUTASKINFO gpuTaskInfo[] = {\n";
		adjustTaskRuncountFunction += "void adjustTaskRuncountFunction(){\n";
		
		index = 0;
		while(index < mTask.size()) {
			Task task = null;
			for(Task t: mTask.values()){
				if(Integer.parseInt(t.getIndex()) == index){
					task = t;
					break;
				}
			}
			int globalPeriod = 0;
			if(mGlobalPeriodMetric.equalsIgnoreCase("h"))			globalPeriod = mGlobalPeriod * 3600 * 1000 * 1000;
			else if(mGlobalPeriodMetric.equalsIgnoreCase("m"))		globalPeriod = mGlobalPeriod * 60 * 1000 * 1000;
			else if(mGlobalPeriodMetric.equalsIgnoreCase("s"))		globalPeriod = mGlobalPeriod * 1000 * 1000;
			else if(mGlobalPeriodMetric.equalsIgnoreCase("ms"))		globalPeriod = mGlobalPeriod * 1000;
			else if(mGlobalPeriodMetric.equalsIgnoreCase("us"))		globalPeriod = mGlobalPeriod * 1;
			else{
				System.out.println("[makeTasks] Not supported metric of period");
				System.exit(-1);
			}
			int taskRunCount = globalPeriod / Integer.parseInt(task.getPeriod());
			int isGPU = 0;
			int gpuIndex = 0;
			for(GPUTaskType gpuTask: mGpusetup.getTasks().getTask()){
				if(gpuTask.getName().equals(task.getName())){
					if(gpuTask.getClustering().equals(gpuTask.getClustering().YES))	isGPU = 1;
					else															isGPU = 2;	
					if(gpuTask.getPipelining().equals(gpuTask.getPipelining().YES))
						taskRunCount += (gpuTask.getMaxStream().intValue()-1);
					int proc = task.getProc().get("Default").get("Default").get(0);	// Need to fix
					gpuIndex = proc - 1;
					gpuTaskInfoDataStructure += "\t{" + task.getIndex() + ", " + isGPU + ", " + gpuIndex + "}, \n";
					adjustTaskRuncountFunction += "\ttasks[" + task.getIndex() + "].run_count = " + taskRunCount + ";\n";
				}
			}
			if(isGPU == 0)	gpuTaskInfoDataStructure += "\t{" + task.getIndex() + ", -1, -1}, \n";
			index++;
		}
		gpuTaskInfoDataStructure += "};\n";
		adjustTaskRuncountFunction += "}\n";
		
		// generate additional channel info and adjust channel size function
		String gpuChannelInfoDataStructure = new String();
		String adjustChannelSizeFunction = new String();
		
		gpuChannelInfoDataStructure += "GPUCHANNELINFO gpuChannelInfo[] = {\n";
		adjustChannelSizeFunction += "void adjustChannelSizeFunction(){\n";
		index = 0;
		while(index < mQueue.size()) {
			Queue queue = mQueue.get(index);
			
			String queueSize = queue.getSize();
			for(GPUTaskType gpuTask: mGpusetup.getTasks().getTask()){
				if(gpuTask.getPipelining().equals(gpuTask.getPipelining().YES)){
					if(gpuTask.getName().equals(queue.getSrc()) || gpuTask.getName().equals(queue.getDst())){
						queueSize = queue.getSize() + "*" + queue.getSampleSize() + "*" + gpuTask.getMaxStream().intValue();
						break;
					}
				}
				else{
					queueSize = queue.getSize() + "*" + queue.getSampleSize();
				}
			}
			
			gpuChannelInfoDataStructure += "\t{" + index + ", NULL, NULL, 0, 0},\n";
			adjustChannelSizeFunction += "\tchannels[" + index + "].max_size = " + queueSize + ";\n";

			index++;
		}
		gpuChannelInfoDataStructure += "};\n";
		adjustChannelSizeFunction += "}\n";
		
		// generate cluster entries code
		String clusterEntriesDataStructure = new String();
		clusterEntriesDataStructure += "CLUSTER_BUFFER cluster_buffers[] = {\n";
		
		for(Task task: mTask.values()){
			int gpuFlag = 0;
			GPUTaskType gpuTask_1 = null;
			Task gpuTask_2 = null;
			for(GPUTaskType gpuTask: mGpusetup.getTasks().getTask()){
				if(gpuTask.getName().equals(task.getName()) && gpuTask.getClustering().equals(gpuTask.getClustering().YES)){
					gpuFlag = 1;
					gpuTask_1 = gpuTask;
					gpuTask_2 = task;
					break;
				}
			}
			if(gpuFlag == 1){
				int inChannelSize = 0;
				int outChannelSize = 0;
				index = 0;
				while(index < mQueue.size()) {
					Queue queue = mQueue.get(index);
					if(gpuTask_2.getName().equals(queue.getSrc()))
						outChannelSize = (Integer.parseInt(queue.getSize()) * Integer.parseInt(queue.getSampleSize()));
					else if(gpuTask_2.getName().equals(queue.getDst()))
						inChannelSize += (Integer.parseInt(queue.getSize()) * Integer.parseInt(queue.getSampleSize()));
					index++;
				}
				if(gpuTask_1.getPipelining().equals(gpuTask_1.getPipelining().YES)){
					clusterEntriesDataStructure += "\t{" + gpuTask_2.getIndex() + ", 0, 0, " + inChannelSize * gpuTask_1.getMaxStream()
					.intValue() + ", " + inChannelSize + ", 'r', NULL, NULL, NULL, MUTEX_INIT_INLINE},\n";
					clusterEntriesDataStructure += "\t{" + gpuTask_2.getIndex() + ", 0, 0, " + outChannelSize * gpuTask_1.getMaxStream()
					.intValue() + ", " + outChannelSize + ", 'w', NULL, NULL, NULL, MUTEX_INIT_INLINE},\n";
				}
				else{
					clusterEntriesDataStructure += "\t{" + gpuTask_2.getIndex() + ", 0, 0, " + inChannelSize
					                   + ", " + inChannelSize + ", \'r\', NULL, NULL, NULL, MUTEX_INIT_INLINE},\n";
					clusterEntriesDataStructure += "\t{" + gpuTask_2.getIndex() + ", 0, 0, " + outChannelSize 
					                   + ", " + outChannelSize + ", \'w\', NULL, NULL, NULL, MUTEX_INIT_INLINE},\n";
				}
			}
			else{
					clusterEntriesDataStructure += "\t{" + task.getIndex() + ", },\n";
					clusterEntriesDataStructure += "\t{" + task.getIndex() + ", },\n";
			}
		}
		clusterEntriesDataStructure += "};";
		
		int num_tasks = mTask.size() + mVTask.size();
		int num_channels = mQueue.size();
		int num_portmaps = mQueue.size() * 2;
		int num_clusters = mTask.size() * 2;
		
		String numDecl = "";
		numDecl += "int _num_tasks = " + num_tasks + ";\n";
		numDecl += "int _num_channels = " + num_channels + ";\n";
		numDecl += "int _num_portmaps = " + num_portmaps + ";\n";
		numDecl += "int _num_cluster_buffers = " + num_clusters + ";\n";
		
		String templateFile = mTranslatorPath + "templates/target/cuda/host_channel_manage/general_linux_cuda.template";
		String initWrapupClusters = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##INIT_WRAPUP_CLUSTERS");
		
		targetDependentImpl += "#include \"cic_gpuinfo.h\"\n";
		targetDependentImpl += numDecl +"\n";
		targetDependentImpl += externalPrePostFunctions +"\n";
		targetDependentImpl += clusterEntriesDataStructure +"\n";
		targetDependentImpl += gpuTaskInfoDataStructure +"\n";
		targetDependentImpl += gpuChannelInfoDataStructure +"\n";
		targetDependentImpl += setPrePostFunction +"\n";
		targetDependentImpl += adjustTaskRuncountFunction +"\n";
		targetDependentImpl += adjustChannelSizeFunction +"\n";
		targetDependentImpl += initWrapupClusters +"\n";
		
		code = code.replace("##TARGET_DEPENDENT_IMPLEMENTATION", targetDependentImpl);
		/////////////////////////////////////
		
		// LIB_INIT_WRAPUP //
		String libraryInitWrapup="";
		if(mLibrary != null){
			for(Library library: mLibrary.values()){
				libraryInitWrapup += "extern void l_"+library.getName()+"_init(void);\n";
				libraryInitWrapup += "extern void l_"+library.getName()+"_wrapup(void);\n\n";
			}
		}
	
		libraryInitWrapup += "static void init_libs() {\n";
		if(mLibrary != null){
			for(Library library: mLibrary.values())
				libraryInitWrapup += "\tl_" + library.getName() + "_init();\n";
		}
		libraryInitWrapup += "}\n\n";
		
		libraryInitWrapup += "static void wrapup_libs() {\n";
		if(mLibrary != null){
			for(Library library: mLibrary.values())
				libraryInitWrapup += "\tl_" + library.getName() + "_wrapup();\n";
		}
		libraryInitWrapup += "}\n\n";
		code = code.replace("##LIB_INIT_WRAPUP",  libraryInitWrapup);	
		/////////////////////
				
		// EXTERNAL_GLOBAL_HEADERS //
		String externalGlobalHeaders = "";
		if(mAlgorithm.getHeaders()!=null){
			for(String header: mAlgorithm.getHeaders().getHeaderFile())
				externalGlobalHeaders += "#include\"" + header +"\"\n";
		}
		code = code.replace("##EXTERNAL_GLOBAL_HEADERS", externalGlobalHeaders);
		////////////////////////////
		
		// SCHEDULE_CODE //
		String outPath = mOutputPath + "/convertedSDF3xml/";
		String staticScheduleCode = CommonLibraries.Schedule.generateSingleProcessorStaticScheduleCode(outPath, mTask, mVTask);
		code = code.replace("##SCHEDULE_CODE", staticScheduleCode);
		///////////////////
		
		// COMM_CODE //
		String commCode = "";
		code = code.replace("##COMM_CODE", commCode);
		///////////////////
		
		// COMM_SENDER_ROUTINE //
		String commSenderRoutine = "";
		code = code.replace("##COMM_SENDER_ROUTINE", commSenderRoutine);
		///////////////////
		
		// COMM_RECEIVER_ROUTINE //
		String commReceiverRoutine = "";
		code = code.replace("##COMM_RECEIVER_ROUTINE", commReceiverRoutine);
		///////////////////		
		
		// OS_DEPENDENT_INCLUDE_HEADERS //
		Util.copyFile(mOutputPath+"includes.h", mTranslatorPath + "templates/common/common_template/includes.h.linux");
		String os_dep_includeHeader = "";
		os_dep_includeHeader = "#include \"includes.h\"\n";
		code = code.replace("##OS_DEPENDENT_INCLUDE_HEADERS", os_dep_includeHeader);
		/////////////////////////////////
		
		// TARGET_DEPENDENT_HEADER_INCLUDE //
		String target_dep_includeHeader = "#include \"cuda_runtime.h\"\n\n#define CUDA_ERROR_CHECK checkCudaErrors\n#include \"helper_functions.h\"\n#include \"helper_cuda.h\"\n";
		target_dep_includeHeader += "\nextern \"C\" {\n";
		code = code.replace("##TARGET_DEPENDENT_INCLUDE_HEADERS", target_dep_includeHeader);
		/////////////////////////////////

		// TARGET_DEPENDENT_INIT_CALL //
		String targetDependentInit = "\tsetPrePostFunction();\n\tadjustTaskRuncountFunction();\n\tadjustChannelSizeFunction();\n\tinit_cluster();\n";
		code = code.replace("##TARGET_DEPENDENT_INIT_CALL", targetDependentInit);
		////////////////////////////////
				
		// TARGET_DEPENDENT_WRAPUP_CALL //
		String targetDependentWrapup = "\n\twrapup_cluster();\n\treturn EXIT_SUCCESS;\n";
		code = code.replace("##TARGET_DEPENDENT_WRAPUP_CALL", targetDependentWrapup);
		//////////////////////////////////
		
		templateFile = mTranslatorPath + "templates/target/cuda/host_channel_manage/general_linux_cuda.template";
		// INIT_WRAPUP_CHANNELS //
		String initWrapupChannels = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##INIT_WRAPUP_CHANNELS");
		code = code.replace("##INIT_WRAPUP_CHANNELS", initWrapupChannels);
		//////////////////////////
		
		// INIT_WRAPUP_CHANNELS //
		String initWrapupTaskChannels = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##INIT_WRAPUP_TASK_CHANNELS");
		code = code.replace("##INIT_WRAPUP_TASK_CHANNELS", initWrapupTaskChannels);
		//////////////////////////
		
		// READ_WRITE_PORT //
		String readWritePort = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##READ_WRITE_PORT");
		code = code.replace("##READ_WRITE_PORT", readWritePort);
		/////////////////////
		
		// READ_WRITE_BUF_PORT //
		String readWriteBufPort = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##READ_WRITE_BUF_PORT");
		code = code.replace("##READ_WRITE_BUF_PORT", readWriteBufPort);
		/////////////////////
		
		// READ_WRITE_AC_PORT //
		String readWriteACPort = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##READ_WRITE_AC_PORT");
		code = code.replace("##READ_WRITE_AC_PORT", readWriteACPort);
		////////////////////////
		
		templateFile = mTranslatorPath + "templates/common/task_execution/no_timed_top_pn_th_bottom_df_func.template";
		// DATA_TASK_ROUTINE //
		String dataTaskRoutine = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##DATA_TASK_ROUTINE");
		code = code.replace("##DATA_TASK_ROUTINE", dataTaskRoutine);
		//////////////////////////
		
		// TIME_TASK_ROUTINE //
		String timeTaskRoutine = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##TIME_TASK_ROUTINE");
		code = code.replace("##TIME_TASK_ROUTINE", timeTaskRoutine);
		//////////////////////////
		
		// EXECUTE_TASKS //
		String executeTasks = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##EXECUTE_TASKS");
		code = code.replace("##EXECUTE_TASKS", executeTasks);
		//////////////////////////
		
		// CONTROL_RUN_TASK //
		String controlRunTask = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##CONTROL_RUN_TASK");
		code = code.replace("##CONTROL_RUN_TASK", controlRunTask);
		//////////////////////////
		
		// CONTROL_STOP_TASK //
		String controlStopTask = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##CONTROL_STOP_TASK");
		code = code.replace("##CONTROL_STOP_TASK", controlStopTask);
		//////////////////////////
				
		// CONTROL_RESUME_TASK //
		String controlResumeTask = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##CONTROL_RESUME_TASK");
		code = code.replace("##CONTROL_RESUME_TASK", controlResumeTask);
		//////////////////////////
		
		// CONTROL_SUSPEND_TASK //
		String controlSuspendTask = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##CONTROL_SUSPEND_TASK");
		code = code.replace("##CONTROL_SUSPEND_TASK", controlSuspendTask);
		//////////////////////////
			
		// SET_PROC //
		String setProc= "";
		code = code.replace("##SET_PROC", setProc);
		//////////////
			
		// GET_CURRENT_TIME_BASE //
		templateFile = mTranslatorPath + "templates/common/time_code/general_linux.template";
		String getCurrentTimeBase = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##GET_CURRENT_TIME_BASE");
		code = code.replace("##GET_CURRENT_TIME_BASE", getCurrentTimeBase);
		///////////////////////////
		
		// CHANGE_TIME_UNIT //
		String changeTimeUnit = CommonLibraries.Util.getCodeFromTemplate(templateFile, "##CHANGE_TIME_UNIT");
		code = code.replace("##CHANGE_TIME_UNIT", changeTimeUnit);
		///////////////////////////
			
		// DEBUG_CODE //
		String debugCode = "";
		code = code.replace("##DEBUG_CODE_IMPLEMENTATION", debugCode);
		///////////////////////////
		
		// MAIN_FUNCTION //
		String mainFunc = "}\n\nint main(int argc, char *argv[])";
		code = code.replace("##MAIN_FUNCTION", mainFunc);
		//////////////
		
		// LIB_INCLUDE //
		String libInclude = "";
		code = code.replace("##LIB_INCLUDES", libInclude);
		//////////////
		
		// CONN_INCLUDES //
		String conInclude = "";
		code = code.replace("##CONN_INCLUDES", conInclude);
		//////////////
		
		// INIT_WRAPUP_LIB_CHANNELS //
		String initWrapupLibChannels = "";
		code = code.replace("##INIT_WRAPUP_LIBRARY_CHANNELS", initWrapupLibChannels);
		//////////////
		
		// INIT_WRAPUP_LIB_CHANNELS //
		String readWriteLibPort = "";
		code = code.replace("##READ_WRITE_LIB_PORT", readWriteLibPort);
		//////////////
		
		// LIB_INIT //
		String libInit = "\tinit_libs();\n";
		code = code.replace("##LIB_INIT", libInit);
		//////////////
		
		// LIB_WRAPUP //
		String libWrapup = "\twrapup_libs();";
		code = code.replace("##LIB_WRAPUP", libWrapup);
		//////////////
		
		// OUT_CONN_INIT //
		String outConnInit = "";				
		code = code.replace("##OUT_CONN_INIT", outConnInit);
		//////////////
		
		// OUT_CONN_WRAPUP //
		String outConnwrapup = "";				
		code = code.replace("##OUT_CONN_WRAPUP", outConnwrapup);
		//////////////
		
		return code;
	}

	public void generateMakefile(String mDestFile)
	{
		Map<String, String> extraSourceList = new HashMap<String, String>();
		Map<String, String> extraHeaderList = new HashMap<String, String>();
		Map<String, String> extraLibSourceList = new HashMap<String, String>();
		Map<String, String> extraLibHeaderList = new HashMap<String, String>();
		Map<String, String> ldFlagList = new HashMap<String, String>();
		
		try {
			FileOutputStream outstream = new FileOutputStream(mDestFile);
			
			for(Task task: mTask.values()){
				for(String extraSource: task.getExtraSource())
					extraSourceList.put(extraSource, extraSource.substring(0, extraSource.length()-2));
				for(String extraHeader: task.getExtraHeader())
					extraHeaderList.put(extraHeader, extraHeader.substring(0, extraHeader.length()-2));
			}
			
			if(mAlgorithm.getLibraries() != null){
				for(Library library: mLibrary.values()){
					for(String extraSource: library.getExtraSource())
						extraLibSourceList.put(extraSource, extraSource.substring(0, extraSource.length()-2));
					for(String extraHeader: library.getExtraHeader())
						extraLibHeaderList.put(extraHeader, extraHeader.substring(0, extraHeader.length()-2));
				}
			}
			
			for(Task task: mTask.values())
				if(!task.getLDflag().isEmpty())		ldFlagList.put(task.getLDflag(), task.getLDflag());


			outstream.write("CC=nvcc\n".getBytes());
			outstream.write("LD=nvcc\n".getBytes());

		    mRootPath = mRootPath.replace("\\","/");
		    mRootPath = mRootPath.replace("C:", "/cygdrive/C");
		    mRootPath = mRootPath.replace("D:", "/cygdrive/D");
		    mRootPath = mRootPath.replace("E:", "/cygdrive/E");
		    mRootPath = mRootPath.replace("F:", "/cygdrive/F");
		    mRootPath = mRootPath.replace("G:", "/cygdrive/G");
		    
		    String gpuarch = null;
		    for(Processor proc: mProcessor.values()){
		    	if(proc.getArchType() != null && !proc.getArchType().equals("default")){
		    		gpuarch = proc.getArchType();
		    		break;
		    	}
		    }
		    if(gpuarch == null){
		    	System.out.println("Non GPU Architecture!");
		    	System.exit(-1);
		    }
		    
		    if(mThreadVer.equals("m")){
		    	outstream.write(("#CFLAGS=-I$(CUDA_HOME) -arch=" +  gpuarch + " -O0 -g -DDISPLAY -DTHREAD_STYLE\n").getBytes());
		        outstream.write(("CFLAGS=-I$(CUDA_HOME) -arch=" + gpuarch + " -O2 -DDISPLAY -DTHREAD_STYLE\n").getBytes());
		    }
		    else if(mThreadVer.equals("s")){
		    	outstream.write(("#CFLAGS=-I$(CUDA_HOME) -arch=" + gpuarch + " -O0 -g -DDISPLAY\n").getBytes());
		        outstream.write(("CFLAGS=-I$(CUDA_HOME) -arch=" + gpuarch + " -O2 -DDISPLAY\n").getBytes());
		    }
		    
		    outstream.write("LDFLAGS=-lX11 -lpthread -lm -Xlinker --warn-common".getBytes());
		    for(String ldflag: ldFlagList.values())
		        outstream.write((" " + ldflag).getBytes());
		    outstream.write("\n\n".getBytes());
		    
		    outstream.write("all: proc\n\n".getBytes());
		    
		    outstream.write("proc:".getBytes());
		    for(Task task: mTask.values())
		    	outstream.write((" " + task.getName() + ".o").getBytes());
		    
		    for(String extraSource: extraSourceList.values())
		    	outstream.write((" " + extraSource + ".o").getBytes());

		    for(String extraLibSource: extraLibSourceList.values())
		    	outstream.write((" " + extraLibSource + ".o").getBytes());
		    
		    if(mAlgorithm.getLibraries() != null)
		    	for(Library library: mLibrary.values())
		    		outstream.write((" " + library.getName() + ".o").getBytes());
		    
		    outstream.write(" proc.o\n".getBytes());
		    outstream.write("\t$(LD) $^ -o proc $(LDFLAGS)\n\n".getBytes());
		    
		    outstream.write(("proc.o: proc.cu CIC_port.h ").getBytes());
		    
		    if(mAlgorithm.getHeaders() != null)
		    	for(String headerFile: mAlgorithm.getHeaders().getHeaderFile())
		    		outstream.write((" " + headerFile).getBytes());
		    
		    outstream.write("\n".getBytes());
		    outstream.write(("\t$(CC) $(CFLAGS) -c proc.cu -o proc.o\n\n").getBytes());
		    
		    for(Task task: mTask.values()){
		    	int gpuFlag = 0;
		    	for(GPUTaskType gTask: mGpusetup.getTasks().getTask()){
					if(gTask.getName().equals(task.getName())){
						gpuFlag = 1;
						break;
					}
				}
		    	if(gpuFlag == 0)
		    		outstream.write((task.getName() + ".o: " + task.getName() + ".c " + task.getCICFile() + " CIC_port.h ").getBytes());
		    	else
		    		outstream.write((task.getName() + ".o: " + task.getName() + ".cu " + task.getCICFile()).getBytes());
		    	for(String header: task.getExtraHeader())
		    		outstream.write((" " + header).getBytes());
		    	outstream.write("\n".getBytes());
		    	if(gpuFlag == 0)
		    		outstream.write(("\t$(CC) $(CFLAGS) " + task.getCflag() + " -c " + task.getName() + ".c -o " + task.getName() + ".o\n\n").getBytes());
		    	else 
		    		outstream.write(("\t$(CC) $(CFLAGS) " + task.getCflag() + " -c " + task.getName() + ".cu -o " + task.getName() + ".o\n\n").getBytes());
		    }
		    
		    for(String extraSource: extraSourceList.keySet()){
		    	outstream.write((extraSourceList.get(extraSource) + ".o: " + extraSourceList.get(extraSource) + ".c").getBytes());
		    	for(String extraHeader: extraHeaderList.keySet())
		    		outstream.write((" " + extraHeader).getBytes());
		    	outstream.write("\n".getBytes());
		    	outstream.write(("\t$(CC) $(CFLAGS) -c " + extraSourceList.get(extraSource) + ".c -o " + extraSourceList.get(extraSource) + ".o\n\n").getBytes());
		    }
		    
		    if(mAlgorithm.getLibraries() != null){
		    	for(Library library: mLibrary.values()){
		    		outstream.write((library.getName() + ".o: " + library.getName() + ".c").getBytes());
		    		for(String extraHeader: library.getExtraHeader())
		    			outstream.write((" " + extraHeader).getBytes());
		    		outstream.write("\n".getBytes());
		    		if(!library.getCflag().isEmpty())
		    			outstream.write(("\t$(CC) $(CFLAGS) " + library.getCflag()).getBytes());
		    		else
		    			outstream.write(("\t$(CC) $(CFLAGS)").getBytes());
		    		outstream.write((" -c " + library.getName() + ".c -o " + library.getName() + ".o\n").getBytes());
		    	}
		    }
		    outstream.write("\n".getBytes());
		    
		    for(String extraSource: extraLibSourceList.keySet()){
		    	outstream.write((extraLibSourceList.get(extraSource) + ".o: " + extraSource).getBytes());
		    	for(String extraHeader: extraLibHeaderList.values())
		    		outstream.write((" " + extraHeader).getBytes());
		    	outstream.write("\n".getBytes());
		    	outstream.write(("\t$(CC) $(CFLAGS) -c " + extraSource + " -o " + extraSourceList.get(extraSource) + ".o\n").getBytes());
		    }
		    outstream.write("\n".getBytes());
		    
		    outstream.write("test: all\n".getBytes());
		    outstream.write("\t./proc\n".getBytes());
		    outstream.write("\n".getBytes());
		    
		    outstream.write("graph:\n".getBytes());
		    outstream.write("\tdot -Tps portGraph.dot -o portGraph.ps\n".getBytes());
		    outstream.write("\n".getBytes());
		    
		    outstream.write("tags:\n".getBytes());
		    outstream.write("\tctags -R --c-types=cdefglmnstuv\n".getBytes());
		    outstream.write("\n".getBytes());
		    
		    outstream.write("clean:\n".getBytes());
		    outstream.write("\trm -f proc *.o\n".getBytes());
		    outstream.write("\n\n".getBytes());
		    
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}	
	}

	@Override
	public int generateCodeWithComm(String mTarget, String mTranslatorPath,
			String mOutputPath, String mRootPath,
			Map<Integer, Processor> mProcessor,
			List<Communication> mCommunication,
			Map<String, Task> mTask, Map<Integer, Queue> mQueue,
			Map<String, Library> mLibrary, Map<String, Library> mGlobalLibrary,
			int mGlobalPeriod, String mGlbalPeriodMetric, String mCICXMLFile,
			String language, String threadVer, CICAlgorithmType mAlgorithm,
			CICControlType mControl, CICScheduleType mSchedule,
			CICGPUSetupType mGpusetup, CICMappingType mMapping,
			Map<Integer, List<Task>> mConnectedTaskGraph,
			Map<Integer, List<List<Task>>> mConnectedSDFTaskSet,
			Map<String, Task> mVTask, Map<String, Task> mPVTask, String mCodeGenType) throws FileNotFoundException {
		// TODO Auto-generated method stub
		return 0;
	}
}

