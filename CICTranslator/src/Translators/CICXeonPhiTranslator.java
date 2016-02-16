package Translators;

import java.io.*;
import java.math.*;
import java.util.*;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import CommonLibraries.*;
import InnerDataStructures.*;
import InnerDataStructures.Library;
import InnerDataStructures.Queue;
import hopes.cic.exception.CICXMLException;
import hopes.cic.xml.*;
import mapss.dif.csdf.sdf.SDFEdgeWeight;
import mapss.dif.csdf.sdf.SDFGraph;
import mapss.dif.csdf.sdf.SDFNodeWeight;
import mapss.dif.csdf.sdf.sched.APGANStrategy;
import mapss.dif.csdf.sdf.sched.DLCStrategy;
import mapss.dif.csdf.sdf.sched.FlatStrategy;
import mapss.dif.csdf.sdf.sched.MinBufferStrategy;
import mapss.dif.csdf.sdf.sched.ProcedureStrategy;
import mapss.dif.csdf.sdf.sched.TwoNodeStrategy;
import mocgraph.Edge;
import mocgraph.sched.Firing;
import mocgraph.sched.Schedule;
import mocgraph.sched.ScheduleElement;

public class CICXeonPhiTranslator implements CICTargetCodeTranslator 
{	
	private String mTarget;
	private String mTranslatorPath;
	private String mOutputPath;
	private String mRootPath;
	private String mCICXMLFile;
	private int mGlobalPeriod;
	private String mGlobalPeriodMetric;
	private String mThreadVer;
	private String mLanguage;
	
	private Map<String, Task> mTask;
	private Map<Integer, Queue> mQueue;
	private Map<String, Library> mLibrary;
	private Map<Integer, Processor> mProcessor;
	
	private Map<String, Task> mVTask;
	
	HashMap<String, HashMap<Integer, String>> taskFileList;
	
	private CICAlgorithmType mAlgorithm;
	private CICControlType mControl;
	private CICScheduleType mSchedule;
	private CICGPUSetupType mGpusetup;
	private CICMappingType mMapping;

	private String strategy;

	@Override
	public int generateCode(String target, String translatorPath, String outputPath, String rootPath, Map<Integer, Processor> processor, Map<String, Task> task, Map<Integer, Queue> queue, Map<String, Library> library, Map<String, Library> globalLibrary, int globalPeriod, String globalPeriodMetric, String cicxmlfile, String language, String threadVer, CICAlgorithmType algorithm, CICControlType control, CICScheduleType schedule, CICGPUSetupType gpusetup, CICMappingType mapping, Map<Integer, List<Task>> connectedtaskgraph,  Map<Integer, List<List<Task>>> connectedsdftaskset, Map<String, Task> vtask, Map<String, Task> pvtask, String codegentype) throws FileNotFoundException
	{
		mTarget = target;
		mTranslatorPath = translatorPath;
		mOutputPath = outputPath;
		mRootPath = rootPath;
		mCICXMLFile = cicxmlfile;
		mGlobalPeriod = globalPeriod;
		mGlobalPeriodMetric = globalPeriodMetric;
		mThreadVer = threadVer;
		mLanguage = language;
		
		mTask = task;
		mQueue = queue;
		mLibrary = library;
		mProcessor = processor;
		
		mVTask = vtask;
		
		mAlgorithm = algorithm;
		mControl = control;
		mSchedule = schedule;
		mGpusetup = gpusetup;
		mMapping = mapping;
			
		Util.copyFile(mOutputPath+"CIC_port.h", mTranslatorPath + "templates/target/Xeonphi/CIC_port.h");
		
		// generate proc.c or proc.cpp
		String fileOut = null;	
		String templateFile = "";

		String srcExtension = "";
		if(mLanguage.equals("c++"))	srcExtension = ".cpp";
		else						srcExtension = ".c";
		
		// generate task_name.c (include task_name.cic)
		for(Task t: mTask.values()){
			if(t.getHasSubgraph().equalsIgnoreCase("Yes") && t.getHasMTM().equalsIgnoreCase("Yes")){
				fileOut = mOutputPath + t.getName() + srcExtension;
				templateFile = mTranslatorPath + "templates/common/common_template/task_mtm_code_template.c";
				CommonLibraries.CIC.generateTaskMTMCode(fileOut, templateFile, t, mAlgorithm);
			}
			else{
				fileOut = mOutputPath + t.getName() + srcExtension;
				if(mLanguage.equals("c++"))	templateFile = mTranslatorPath + "templates/target/Xeonphi/task_code_template.cpp";
				else						templateFile = mTranslatorPath + "templates/target/Xeonphi/task_code_template.c";
				this.generateTaskCode(fileOut, templateFile, t);
			}
		}
		
		if(mLibrary != null){
			for(Library l: mLibrary.values()) 
				CommonLibraries.Library.generateLibraryCode(mOutputPath, l, mAlgorithm);
		}
		
		fileOut = mOutputPath+"proc" + srcExtension;
		if(mThreadVer.equals("s"))
			templateFile = mTranslatorPath + "templates/target/Xeonphi/proc.c.template";	// NEED TO FIX
		else
			templateFile = mTranslatorPath + "templates/target/Xeonphi/proc.c.template";
		
		String schedPath = mOutputPath + "convertedSDF3xml";
		
		generateScheduleFile();
		
		generateProcCode(fileOut, templateFile);
	
	    // generate Makefile
	    fileOut = mOutputPath + "Makefile";
	    generateMakefile(fileOut);
	    return 0;
	}

	private String generateGraphInfo(String graphFileName, String schedFileName) {
		String content = "";
		File input = new File(graphFileName);

		Document doc = XmlParserUtil.parse(new File(graphFileName));
		if(doc==null) return null;

		Node appNode = XmlParserUtil.getChildNode(doc.getDocumentElement(), "applicationGraph");
		if(appNode == null) return null;
		
		Node sdfNode = XmlParserUtil.getChildNode(appNode, "sdf");
		if(sdfNode == null) return null;
		
		Vector actorVector = XmlParserUtil.getChildNodes(sdfNode, "actor");
		if(actorVector.size() == 0 ) return null;
		
		Vector channelVector = XmlParserUtil.getChildNodes(sdfNode, "channel");
		if(channelVector.size() == 0 ) return null;
		
		Node sdfPropertyNode = XmlParserUtil.getChildNode(appNode, "sdfProperties");
		if(sdfPropertyNode == null) return null;
		
		Vector actorPropertiesVector = XmlParserUtil.getChildNodes(sdfPropertyNode, "actorProperties");
		if(actorPropertiesVector == null) return null;
		
		CICScheduleType schedule = new CICScheduleType();
		CICScheduleTypeLoader scheduleLoader = new CICScheduleTypeLoader();
		File f = new File(schedFileName);
		if(f.exists()){
			try {
				schedule = scheduleLoader.loadResource(schedFileName);
			} catch (CICXMLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		String modename = (String)XmlParserUtil.getNodeAttribute(sdfNode, "type").toLowerCase();
		
		content += "num_task\n";
		content += actorVector.size() + "\n";
		content += "exec_times\n";
		for(int i=0; i<actorVector.size(); i++){
			Element actor = (Element)actorVector.get(i);
			String taskname = (String)actor.getAttribute("name");
			
			String procname = "";
			int exectime = 0;
			for(ScheduleGroupType sgt: schedule.getTaskGroups().getTaskGroup().get(0).getScheduleGroup()){
				for(ScheduleElementType set: sgt.getScheduleElement()){
					 String taskName = set.getTask().getName();
					 if(taskname.equals(taskName)){
						 procname = sgt.getPoolName().toLowerCase();
						 break;
					 }
				}
			}
			
			for(int j=0; j<actorPropertiesVector.size(); j++){
				Element actorProc = (Element)actorPropertiesVector.get(j);
				if(actorProc.getAttribute("actor").equals(taskname)){
					Vector processorVector = XmlParserUtil.getChildNodes(actorProc, "processor");
					if(processorVector == null) return null;
					for(int k=0; k<processorVector.size(); k++){
						Element proc = (Element)processorVector.get(k);
						if(proc.getAttribute("type").toLowerCase().equals(procname)){
							Element execTime = (Element)XmlParserUtil.getChildNode(proc, "executionTime");
							exectime = Integer.parseInt(execTime.getAttribute("time"));
							break;
						}
					}
					break;
				}
			}
			
			Task t = mTask.get(taskname);
			//content += t.getExecutionTimeValue().get(modename) + "\n";
			content += exectime + "\n";
		}
		for(int i=0; i<actorVector.size(); i++){
			Element actor = (Element)actorVector.get(i);
			String taskname = (String)actor.getAttribute("name");
			
			content += "Node " + i + "\n";
			int in_edge = 0;
			int out_edge = 0;
			int back_edge = 0;
			
			Vector portVector = XmlParserUtil.getChildNodes(actor, "port");
			for(int j=0; j<portVector.size(); j++){
				Element port = (Element)portVector.get(j);
				String portname = port.getAttribute("name");
				String porttype = port.getAttribute("type");
				if(porttype.equals("in")){
					for(int k=0; k<channelVector.size(); k++){
						Element channel = (Element)channelVector.get(k);
						String dstActor = channel.getAttribute("dstActor");
						String dstPort = channel.getAttribute("dstPort");
						
						if(dstActor.equals(taskname) && dstPort.equals(portname)){
							int initTokens = Integer.parseInt(channel.getAttribute("initialTokens"));
							if(initTokens == 0)	in_edge++;
							else				back_edge++;
						}
					}
					
				}
				else if(porttype.equals("out"))	out_edge++;
			}
			
			content += in_edge + "\n";
			content += out_edge + "\n";
			content += back_edge + "\n";
			
			content += "in_edge\n";
			for(int j=0; j<channelVector.size(); j++){
				Element channel = (Element)channelVector.get(j);
				String dstActor = channel.getAttribute("dstActor");
				int initTokens = Integer.parseInt(channel.getAttribute("initialTokens"));
				if(dstActor.equals(taskname) && initTokens == 0)	content += j + "\n";
			}
			content += "out_edge\n";
			for(int j=0; j<channelVector.size(); j++){
				Element channel = (Element)channelVector.get(j);
				String srcActor = channel.getAttribute("srcActor");
				if(srcActor.equals(taskname))	content += j + "\n";
			}
			content += "back_edge\n";
			for(int j=0; j<channelVector.size(); j++){
				Element channel = (Element)channelVector.get(j);
				String dstActor = channel.getAttribute("dstActor");
				int initTokens = Integer.parseInt(channel.getAttribute("initialTokens"));
				if(dstActor.equals(taskname) && initTokens > 0)	content += j + "\n";
			}
			content += "node_type\n";
			if(actor.getAttribute("dp") != "" && Integer.parseInt(actor.getAttribute("dp")) > 1)	content += "1\n";
			else	content += "0\n";
		}
		
		content += "num_edge\n";
		content += channelVector.size() + "\n";
		for(int i=0; i<channelVector.size(); i++){
			Element channel = (Element)channelVector.get(i);
			String srcActor = channel.getAttribute("srcActor");
			String dstActor = channel.getAttribute("dstActor");
			String srcPort = channel.getAttribute("srcPort");
			String dstPort = channel.getAttribute("dstPort");
			
			int src_sampleSize = 0, dst_sampleSize = 0;
			int src_sampleRate = 0, dst_sampleRate = 0;
			for(TaskType task: mAlgorithm.getTasks().getTask()){
				if(task.getName().equals(srcActor)){
					for(TaskPortType tp: task.getPort()){
						if(tp.getName().equals(srcPort)){
							src_sampleSize = tp.getSampleSize().intValue();
							for(TaskRateType tr: tp.getRate()){
								if(tr.getMode().toLowerCase().equals(modename)){
									src_sampleRate = tr.getRate().intValue();
									break;
								}
							}
						}
					}
				}
				if(task.getName().equals(dstActor)){
					for(TaskPortType tp: task.getPort()){
						if(tp.getName().equals(dstPort)){
							dst_sampleSize = tp.getSampleSize().intValue();
							for(TaskRateType tr: tp.getRate()){
								if(tr.getMode().toLowerCase().equals(modename)){
									dst_sampleRate = tr.getRate().intValue();
									break;
								}
							}
						}
					}
				}
			}
			
			int src_task_id = 0, dst_task_id = 0;
			for(int j=0; j<actorVector.size(); j++){
				Element actor = (Element)actorVector.get(j);
				String taskname = (String)actor.getAttribute("name");
				if(taskname.equals(srcActor))	src_task_id = j;
				if(taskname.equals(dstActor))	dst_task_id = j;
			}
			
			content += "Edge " + i + "\n";
			content += "in_rate\n";
			content += (src_sampleSize * src_sampleRate) + "\n";
			content += "out_rate\n";
			content += (dst_sampleSize * dst_sampleRate) + "\n";
			content += "src_node\n";
			content += src_task_id + "\n";
			content += "dst_node\n";
			content += dst_task_id + "\n";
		}
		//System.out.println(content);
		return content;
	}
	
	private String generateScheduleInfo(int index, int proc_num, int base_ii, String graphFileName, String schedFileName){
		String content = "";
		File input = new File(graphFileName);

		Document doc = XmlParserUtil.parse(new File(graphFileName));
		if(doc==null) return null;

		Node appNode = XmlParserUtil.getChildNode(doc.getDocumentElement(), "applicationGraph");
		if(appNode == null) return null;
		
		Node sdfNode = XmlParserUtil.getChildNode(appNode, "sdf");
		if(sdfNode == null) return null;
		
		Vector actorVector = XmlParserUtil.getChildNodes(sdfNode, "actor");
		if(actorVector.size() == 0 ) return null;
		
		Vector channelVector = XmlParserUtil.getChildNodes(sdfNode, "channel");
		if(channelVector.size() == 0 ) return null;
		
		Node sdfPropertyNode = XmlParserUtil.getChildNode(appNode, "sdfProperties");
		if(sdfPropertyNode == null) return null;
		
		Vector actorPropertiesVector = XmlParserUtil.getChildNodes(sdfPropertyNode, "actorProperties");
		if(actorPropertiesVector == null) return null;
		
		CICScheduleType schedule = new CICScheduleType();
		CICScheduleTypeLoader scheduleLoader = new CICScheduleTypeLoader();
		File f = new File(schedFileName);
		if(f.exists()){
			try {
				schedule = scheduleLoader.loadResource(schedFileName);
			} catch (CICXMLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		String modename = (String)XmlParserUtil.getNodeAttribute(sdfNode, "type").toLowerCase();
		
		content += "\nSchedule " + index + "\n";
		content += "mapped_proc\n";
		content += proc_num + "\n";
		content += "throughput\n";
		
		int max_ii = calcThroughput(schedFileName);
		double thr = (double)base_ii/(double)max_ii;
		content += thr + "\n";
		
		for(int i=0; i<proc_num; i++){
			content += "proc_id\n";
			content += i + "\n";
			
			for(ScheduleGroupType sgt: schedule.getTaskGroups().getTaskGroup().get(0).getScheduleGroup()){
				int p_id = sgt.getLocalId().intValue();
				if(p_id == i){
					content += "mapped_task\n";
					content += sgt.getScheduleElement().size() + "\n";
					content += "task_list\n";
					for(ScheduleElementType set: sgt.getScheduleElement()){
						String target_task_name = set.getTask().getName();
						for(int j=0; j<actorVector.size(); j++){
							Element actor = (Element)actorVector.get(j);
							String taskname = (String)actor.getAttribute("name");
							if(target_task_name.equals(taskname)){
								content += j + "\n";
								break;
							}
						}
					}
				}
			}
		}
		
		content += "Data Parallel\n";
		content += "Node\n";
		for(int j=0; j<actorVector.size(); j++){
			Element actor = (Element)actorVector.get(j);
			String sdp = actor.getAttribute("dp");
			if(sdp != "" && Integer.parseInt(sdp) > 1){
				content += j + "\n";
			}
		}
		content += "num_parallel\n";
		for(int j=0; j<actorVector.size(); j++){
			Element actor = (Element)actorVector.get(j);
			String sdp = actor.getAttribute("dp");
			if(sdp != "" && Integer.parseInt(sdp) > 1){
				content += Integer.parseInt(sdp) + "\n";
			}
		}
		
		//System.out.println(content);
		return content;
	}
	
	// calculation normalized throughput (assumption: throughput = 1/ii)
	// loopType is not supported (if need, must fix!)
	private int calcThroughput(String fileName){
		int max_ii = 0;
		CICScheduleType schedule = new CICScheduleType();
		CICScheduleTypeLoader scheduleLoader = new CICScheduleTypeLoader();
		File f = new File(fileName);
		if(f.exists()){
			try {
				schedule = scheduleLoader.loadResource(fileName);
			} catch (CICXMLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		for(ScheduleGroupType sgt: schedule.getTaskGroups().getTaskGroup().get(0).getScheduleGroup()){
			int min_start_time = Integer.MAX_VALUE;
			int max_end_time = 0;
			for(ScheduleElementType set: sgt.getScheduleElement()){
				if(min_start_time > set.getTask().getStartTime().intValue())	min_start_time = set.getTask().getStartTime().intValue();
				if(max_end_time < set.getTask().getEndTime().intValue())		max_end_time = set.getTask().getEndTime().intValue();
			}
			if(max_ii < max_end_time - min_start_time)	max_ii = max_end_time - min_start_time;
		}
		return max_ii;
	}

	private void generateScheduleFile() {
		// TODO Auto-generated method stub
		String infoPath = mOutputPath + "convertedSDF3xml/information.xml";
		taskFileList = new HashMap<String, HashMap<Integer, String>>();
		HashMap<String, Integer> minProcNum = new HashMap<String, Integer> ();
		HashMap<String, Integer> maxProcNum = new HashMap<String, Integer> ();
		File input = new File(infoPath);

		Document doc = XmlParserUtil.parse(new File(infoPath));
		if(doc==null) return;

		Node sdf3Node = XmlParserUtil.getChildNode(doc.getDocumentElement(), "sdf3files");
		if(sdf3Node == null) return;
		
		Vector sdf3Vector = XmlParserUtil.getChildNodes(sdf3Node, "sdf3file");
		if(sdf3Vector.size() == 0 ) return;

		for(Object sdf3: sdf3Vector){
			Element sdf3file = (Element)sdf3;
			String taskName = sdf3file.getAttribute("task_name") + "_" + sdf3file.getAttribute("mode_name");
			
			File dir = new File(mOutputPath + "convertedSDF3xml/");
			File[] overallFileList = dir.listFiles();
			HashMap<Integer, String> fileList = new HashMap<Integer, String>();
			int minprocnum = Integer.MAX_VALUE;
			int maxprocnum = 0;
			for(int i=0; i<overallFileList.length; i++){
				File file = overallFileList[i];
				if(file.isFile() && !file.getName().substring(0, file.getName().length()-4).equals(taskName) && file.getName().contains(taskName)){
					int l_id = file.getName().lastIndexOf("_schedule.xml");
					String tmp = file.getName().substring(0, l_id);
					int s_id = tmp.lastIndexOf("_")+1;
					int proc_num = Integer.parseInt(file.getName().subSequence(s_id, l_id).toString());
					if(proc_num > maxprocnum)	maxprocnum = proc_num;
					if(proc_num < minprocnum)	minprocnum = proc_num;;
					fileList.put(proc_num, file.getName());
				}
			}
			taskFileList.put(taskName, fileList);
			minProcNum.put(taskName, minprocnum);
			maxProcNum.put(taskName, maxprocnum);
		}
		
		// generate sophy schedule file
		for(String taskName: taskFileList.keySet()){
			HashMap<Integer, String> taskFiles = taskFileList.get(taskName);
			String graphInfo = "";
			String schedInfo = "";
			
			int history = 0;
			graphInfo = generateGraphInfo(mOutputPath + "convertedSDF3xml/" + taskName + ".xml", mOutputPath + "convertedSDF3xml/" + taskFiles.get(minProcNum.get(taskName)));
			
			int base_ii = calcThroughput(mOutputPath + "convertedSDF3xml/" + taskFiles.get(minProcNum.get(taskName)));
			
			int g_index = 0;
			for(int proc_index = minProcNum.get(taskName); proc_index <= maxProcNum.get(taskName); proc_index++){
				if(taskFiles.get(proc_index) != null){
					String fileName = taskFiles.get(proc_index);
					schedInfo += generateScheduleInfo(g_index, proc_index, base_ii, mOutputPath + "convertedSDF3xml/" + taskName + ".xml", mOutputPath + "convertedSDF3xml/" + fileName);
					history = proc_index;
				}
				else{
					String fileName = taskFiles.get(history);
					schedInfo += generateScheduleInfo(g_index, history, base_ii, mOutputPath + "convertedSDF3xml/" + taskName + ".xml", mOutputPath + "convertedSDF3xml/" + fileName);
				}
				g_index++;
			}
			
			int index = mCICXMLFile.lastIndexOf("\\");
		    String proj_name = mCICXMLFile.substring(index+1);
			
			File fileOut = new File(mOutputPath + proj_name + ".txt");
			try {
				FileOutputStream outstream = new FileOutputStream(fileOut);

				outstream.write((graphInfo + schedInfo).getBytes());	
				outstream.close();
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	public void generateTaskCode(String mDestFile, String mTemplateFile, Task task){
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
		String parameterDef="";
		String libraryDef="";
		String sysportDef="";
		String mtmDef="";
		String cicInclude="";

		if(mAlgorithm.getLibraries() != null){
			libraryDef += "#define LIBCALL(x, ...) LIBCALL_##x(__VA_ARGS__)\n";
			for(TaskLibraryConnectionType taskLibCon: mAlgorithm.getLibraryConnections().getTaskLibraryConnection()){
				if(taskLibCon.getMasterTask().equals(task.getName())){
					libraryDef += "\n#include \"" + taskLibCon.getSlaveLibrary()+".h\"\n";
					libraryDef += "#define LIBCALL_" + taskLibCon.getMasterPort() + "(f, ...) l_" + taskLibCon.getSlaveLibrary() + "_##f(__VA_ARGS__)\n";
				}
			}
		}
		
		sysportDef = "#define SYS_REQ(x, ...) SYS_REQ_##x(__VA_ARGS__)\n ";
		
		cicInclude += "#include \"" + task.getCICFile() + "\"\n";
		
		if(task.getHasMTM().equalsIgnoreCase("Yes")){
			mtmDef += "#define GET_CURRENT_MODE_NAME char* ##TASK_NAME_get_current_mode_name()\n" +
					  "#define GET_CURRENT_MODE_ID int ##TASK_NAME_get_current_mode_id()\n" +
					  "#define GET_MODE_NAME char* ##TASK_NAME_get_mode_name(int id)\n" +
					  "#define GET_VARIABLE_INT int ##TASK_NAME_get_variable_int(char* name)\n" +
					  "#define SET_VARIABLE_INT void ##TASK_NAME_set_variable_int(char* name, int value)\n" +
					  "#define GET_VARIABLE_STRING char* ##TASK_NAME_get_variable_string(char* name)\n" +
   					  "#define SET_VARIABLE_STRING void ##TASK_NAME_set_variable_string(char* name, char* value)\n" +
					  "#define TRANSITION void ##TASK_NAME_transition()\n";
			cicInclude += "#include \"" + task.getName() + ".mtm\"\n"; 
		}
		
		String includeHeaders = "#include <math.h>\n";
		includeHeaders += "#include \"CIC_port.h\"\n";
		
		code = code.replace("##INCLUDE_HEADERS", includeHeaders);
		code = code.replace("##LIBRARY_DEFINITION", libraryDef);
		code = code.replace("##TASK_INSTANCE_NAME", task.getName());	    
	    code = code.replace("##SYSPORT_DEFINITION", sysportDef);
	    code = code.replace("##CIC_INCLUDE", cicInclude);
	    code = code.replace("##MTM_DEFINITION", mtmDef);
	    code = code.replace("##TASK_NAME", task.getName());
		
		return code;
	}

	public void generateProcCode(String mDestFile, String mTemplateFile)
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
		String task_go_decl = "";
		String task_number = "";
		String task_go_addr = "";
		int task_count = 0;
		
		for(String taskName: taskFileList.keySet()){
			String graphFileName = mOutputPath + "convertedSDF3xml/" + taskName + ".xml";
			
			File input = new File(graphFileName);

			Document doc = XmlParserUtil.parse(new File(graphFileName));
			if(doc==null) return null;

			Node appNode = XmlParserUtil.getChildNode(doc.getDocumentElement(), "applicationGraph");
			if(appNode == null) return null;
			
			Node sdfNode = XmlParserUtil.getChildNode(appNode, "sdf");
			if(sdfNode == null) return null;
			
			Vector actorVector = XmlParserUtil.getChildNodes(sdfNode, "actor");
			if(actorVector.size() == 0 ) return null;
			
			for(int i=0; i<actorVector.size(); i++){
				Element actor = (Element)actorVector.get(i);
				String taskname = (String)actor.getAttribute("name");
				
				task_go_decl += "void " + taskname + "_go(volatile long * task_data_addr, volatile long * task_check_addr, int parallel_id, int max_parallel, char* argument);\n";
				task_go_addr += "\tfa.func_addr[" + task_count + "] = (long) " + taskname + "_go;\n";
				task_count++;
			}
		}
		task_number += "\tfa.func_number = " + task_count + ";\n";
	
		code = code.replace("##TASK_GO_DECLARATION", task_go_decl);
		code = code.replace("##TASK_NUMBER", task_number);
		code = code.replace("##TASK_GO_ADDR", task_go_addr);
		return code;
	}

	public void generateMakefile(String mDestFile)
	{
		Map<String, String> extraSourceList = new HashMap<String, String>();
		Map<String, String> extraHeaderList = new HashMap<String, String>();
		Map<String, String> extraLibSourceList = new HashMap<String, String>();
		Map<String, String> extraLibHeaderList = new HashMap<String, String>();
		Map<String, String> ldFlagList = new HashMap<String, String>();
		Map<String, String> cFlagList = new HashMap<String, String>();
		
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
			for(Task task: mTask.values())
				if(!task.getCflag().isEmpty())		cFlagList.put(task.getCflag(), task.getCflag());

			String srcExtension = null;
			if(mLanguage.equals("c++")){
				srcExtension = ".cpp";
				outstream.write("CC=g++\n".getBytes());
				outstream.write("LD=g++\n".getBytes());
			}
			else{
				srcExtension = ".c";
				outstream.write("CC=icc\n".getBytes());
				outstream.write("LD=icc\n".getBytes());
			}
			
		    mRootPath = mRootPath.replace("\\","/");
		    mRootPath = mRootPath.replace("C:", "/cygdrive/C");
		    mRootPath = mRootPath.replace("D:", "/cygdrive/D");
		    mRootPath = mRootPath.replace("E:", "/cygdrive/E");
		    mRootPath = mRootPath.replace("F:", "/cygdrive/F");
		    mRootPath = mRootPath.replace("G:", "/cygdrive/G");
		    
		    outstream.write("CFLAGS= -mmic -fPIC -shared -g".getBytes());
		    for(String cflag: cFlagList.values())
		        outstream.write((" " + cflag).getBytes());
		    outstream.write("\n\n".getBytes());
		    	    
		    outstream.write("LDFLAGS= -mmic  -g".getBytes());
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
		    
		    int index = mCICXMLFile.lastIndexOf("\\");
		    String proj_name = mCICXMLFile.substring(index+1);

		    outstream.write(" proc.o\n".getBytes());
		    outstream.write(("\t$(LD) $(LDFLAGS) -shared -fPIC  $^ -o " + proj_name +"_LIB.so\n\n").getBytes());
		    System.out.println(mCICXMLFile);
		    outstream.write(("proc.o: proc" + srcExtension + " CIC_port.h ").getBytes());
		    
		    if(mAlgorithm.getHeaders() != null)
		    	for(String headerFile: mAlgorithm.getHeaders().getHeaderFile())
		    		outstream.write((" " + headerFile).getBytes());
		    
		    outstream.write("\n".getBytes());
		    outstream.write(("\t$(CC) $(CFLAGS) -c proc" + srcExtension + " -o proc.o\n\n").getBytes());
		    
		    for(Task task: mTask.values()){
		    	if(task.getCICFile().endsWith(".cic"))
		    		outstream.write((task.getName() + ".o: " + task.getName() + srcExtension + " " + task.getCICFile() + " CIC_port.h ").getBytes());
		    	else if(task.getCICFile().endsWith(".xml"))
		    		outstream.write((task.getName() + ".o: " + task.getName() + srcExtension + " " + " CIC_port.h ").getBytes());
		    	for(String header: task.getExtraHeader())
		    		outstream.write((" " + header).getBytes());
		    	outstream.write("\n".getBytes());
		    	outstream.write(("\t$(CC) $(CFLAGS) -c $< -o $@\n\n").getBytes());
		    	 
		    }
		    
		    for(String extraSource: extraSourceList.keySet()){
		    	outstream.write((extraSourceList.get(extraSource) + ".o: " + extraSourceList.get(extraSource) + ".c").getBytes());
		    	for(String extraHeader: extraHeaderList.keySet())
		    		outstream.write((" " + extraHeader).getBytes());
		    	outstream.write("\n".getBytes());
		    	outstream.write(("\t$(CC) $(CFLAGS) -c $< -o $@\n\n").getBytes());
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
		    	outstream.write(("\t$(CC) $(CFLAGS) -c $< -o $@\n\n").getBytes());
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
		    outstream.write("\trm -f proc *.o *.so\n".getBytes());
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
			List<Communication> mCommunication, Map<String, Task> mTask,
			Map<Integer, Queue> mQueue, Map<String, Library> mLibrary,
			Map<String, Library> mGlobalLibrary, int mGlobalPeriod,
			String mGlbalPeriodMetric, String mCICXMLFile, String language,
			String threadVer, CICAlgorithmType mAlgorithm,
			CICControlType mControl, CICScheduleType mSchedule,
			CICGPUSetupType mGpusetup, CICMappingType mMapping,
			Map<Integer, List<Task>> mConnectedTaskGraph,
			Map<Integer, List<List<Task>>> mConnectedSDFTaskSet,
			Map<String, Task> mVTask, Map<String, Task> mPVTask, String mCodeGenType) throws FileNotFoundException {
		// TODO Auto-generated method stub
		return 0;
	}
}
