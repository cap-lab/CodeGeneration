package CommonLibraries;

import hopes.cic.exception.CICXMLException;
import hopes.cic.xml.CICAlgorithmType;
import hopes.cic.xml.CICScheduleType;
import hopes.cic.xml.CICScheduleTypeLoader;
import hopes.cic.xml.LoopType;
import hopes.cic.xml.ScheduleElementType;
import hopes.cic.xml.ScheduleGroupType;
import hopes.cic.xml.ScheduleType;
import hopes.cic.xml.TaskGroupForScheduleType;
import hopes.cic.xml.TaskGroupsType;
import hopes.cic.xml.TaskInstanceType;
import hopes.cic.xml.TaskType;

import java.io.File;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import javax.swing.JOptionPane;

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
import mocgraph.Node;
import mocgraph.sched.Firing;
import mocgraph.sched.ScheduleElement;
import InnerDataStructures.Queue;
import InnerDataStructures.Task;

public class Schedule {

	public static enum STRATEGY {
		APGAN, MINBUF, DLC, FLAT, PROCEDURE, TwoNode,
	}

	private static int loopId = 0;
	private static int sched_time;

	public static void generateSDFschedule(Map<String, Task> mTask, Map<Integer, Queue> mQueue, String mOutputPath,
			CICAlgorithmType mAlgorithm, Map<String, Task> mVTask) {
		STRATEGY strategy;

		Map<String, Task> mMergeTask = new HashMap<String, Task>();

		for (Task t : mTask.values())
			mMergeTask.put(t.getName(), t);
		for (Task t : mVTask.values())
			mMergeTask.put(t.getName(), t);

		for (Task t : mMergeTask.values()) {
			if (t.getHasSubgraph().equalsIgnoreCase("Yes")) {
				// Make SDFgraph
				ArrayList<Task> taskList = new ArrayList<Task>();
				ArrayList<Queue> channelList = new ArrayList<Queue>();

				for (Task i_t : mMergeTask.values()) {
					if (i_t.getParentTask().equals(t.getName()) && i_t.getHasSubgraph().equalsIgnoreCase("No")) {
						taskList.add(i_t);
					}
				}

				for (Queue i_q : mQueue.values()) {
					if (mMergeTask.get(i_q.getDst()).getParentTask().equals(t.getName())) {
						channelList.add(i_q);
					}
				}

				if (taskList.size() <= 2)
					strategy = STRATEGY.TwoNode;
				else
					strategy = STRATEGY.MINBUF;

				List<String> modeList = new ArrayList<String>();
				if (t.getMTM() != null)
					modeList = t.getMTM().getModes();
				else if (t.getMTM() == null)
					modeList.add("Default");

				for (String mode : modeList) {
					SDFGraph graph = new SDFGraph(taskList.size(), channelList.size());

					// Construct Node
					int instanceid = 0;
					for (int i = 0; i < taskList.size(); i++) {
						String nodeName = taskList.get(i).getName();
						SDFNodeWeight weight = new SDFNodeWeight(nodeName, instanceid++);
						Node node = new Node(weight);
						graph.addNode(node);
						graph.setName(node, nodeName);
					}

					int flag = 0;

					for (Queue i_i_q : channelList) {
						int srcRate = 0;
						int dstRate = 0;
						for (Task i_i_t : taskList) {
							// Src port
							if (i_i_q.getSrc().equals(i_i_t.getName())) {
								Map<String, Map<String, Integer>> portList = i_i_t.getPortList();
								for (String portName : portList.keySet()) {
									if (i_i_q.getSrcPortName().equals(portName)) {
										Map<String, Integer> port = portList.get(portName);
										if (port.size() != 0) {
											if (!port.containsKey(mode) && !port.containsKey("Default"))
												srcRate = -1;
											else if (port.containsKey("Default"))
												srcRate = port.get("Default");
											else
												srcRate = port.get(mode);
										}
										break;
									}
								}
							}
							// Dst port
							if (i_i_q.getDst().equals(i_i_t.getName())) {
								Map<String, Map<String, Integer>> portList = i_i_t.getPortList();
								for (String portName : portList.keySet()) {
									if (i_i_q.getDstPortName().equals(portName)) {
										Map<String, Integer> port = portList.get(portName);
										if (port.size() != 0) {
											if (!port.containsKey(mode) && !port.containsKey("Default"))
												dstRate = -1;
											else if (port.containsKey("Default"))
												dstRate = port.get("Default");
											else
												dstRate = port.get(mode);
										}
										break;
									}
								}
							}
						}

						if (srcRate == -1 || dstRate == -1) {
							flag = 1;
							break;
						}

						// Make Channel
						if (srcRate != 0 || dstRate != 0) {
							Node srcNode = null, dstNode = null;
							Iterator iter = graph.nodes().iterator();
							while (iter.hasNext()) {
								Node n = (Node) iter.next();
								if (graph.getName(n).compareTo(i_i_q.getSrc()) == 0)
									srcNode = n;
								if (graph.getName(n).compareTo(i_i_q.getDst()) == 0)
									dstNode = n;
							}
							SDFEdgeWeight weight = new SDFEdgeWeight(i_i_q.getSrcPortName(), i_i_q.getDstPortName(),
									srcRate, dstRate, Integer.parseInt(i_i_q.getInitData()));
							Edge edge = new Edge(srcNode, dstNode);
							edge.setWeight(weight);
							graph.addEdge(edge);
							graph.setName(edge, "" + "" + i_i_q.getSrcPortName() + "to" + "" + i_i_q.getDstPortName());
						}
					}

					Iterator iter = graph.nodes().iterator();
					ArrayList removeList = new ArrayList<Node>();
					while (iter.hasNext()) {
						Node n = (Node) iter.next();
						String nodeName = graph.getName(n);

						int zeroFlag = 0;
						for (int i = 0; i < taskList.size(); i++) {
							String taskName = taskList.get(i).getName();
							if (nodeName.equals(taskName)) {
								Map<String, Map<String, Integer>> portList = taskList.get(i).getPortList();
								for (String portName : portList.keySet()) {
									Map<String, Integer> port = portList.get(portName);
									int rate = 0;
									if (port.size() != 0) {
										if (port.containsKey("Default"))
											rate = port.get("Default");
										else
											rate = port.get(mode);
									}

									if (rate != 0) {
										zeroFlag = 1;
										break;
									}
								}
								break;
							}
						}
						if (zeroFlag == 0) {
							removeList.add(n);
						}
					}
					Iterator remove_iter = removeList.iterator();
					while (remove_iter.hasNext()) {
						Node node = (Node) remove_iter.next();
						graph.removeNode(node);
					}

					if (flag == 0) {
						System.out.println(graph);
						generateSchedule(t.getName(), mode, graph, strategy, mOutputPath, mTask);
						System.out.println(graph);
					} else
						System.out.println("Something Wrong!");
				}
			}
		}
	}

	public static void generateSchedule(String tg, String m, SDFGraph graph, STRATEGY strategy, String mOutputPath,
			Map<String, Task> mTask) {
		CICScheduleTypeLoader loaderSched;
		CICScheduleType schedule;

		TaskGroupsType taskGroups = new TaskGroupsType();
		TaskGroupForScheduleType tgst = new TaskGroupForScheduleType();
		ScheduleGroupType s_sgt = new ScheduleGroupType();
		mocgraph.sched.Schedule s = null;

		loaderSched = new CICScheduleTypeLoader();
		schedule = loaderSched.createResource(mOutputPath + tg + "_" + m + "_schedule.xml");

		if (strategy == STRATEGY.APGAN) {
			APGANStrategy st = new APGANStrategy(graph);
			s = st.schedule();
			schedule.setType("APGAN");
			s_sgt.setName("APGAN");
		} else if (strategy == STRATEGY.MINBUF) {
			MinBufferStrategy st = new MinBufferStrategy(graph);
			s = st.schedule();
			schedule.setType("MINBUF");
			s_sgt.setName("MINBUF");
		} else if (strategy == STRATEGY.DLC) {
			DLCStrategy st = new DLCStrategy(graph);
			s = st.schedule();
			schedule.setType("DLC");
			s_sgt.setName("DLC");
		} else if (strategy == STRATEGY.FLAT) {
			FlatStrategy st = new FlatStrategy(graph);
			s = st.schedule();
			schedule.setType("FLAT");
			s_sgt.setName("FLAT");
		} else if (strategy == STRATEGY.PROCEDURE) {
			ProcedureStrategy st = new ProcedureStrategy(graph);
			s = st.schedule();
			schedule.setType("PROCEDURE");
			s_sgt.setName("PROCEDURE");
		} else if (strategy == STRATEGY.TwoNode) {
			TwoNodeStrategy st = new TwoNodeStrategy(graph);
			s = st.schedule();
			schedule.setType("TwoNode");
			s_sgt.setName("TwoNode");
		}

		if (s == null)
			return;

		/* Update run-rate of tasks */

		for (Task task : mTask.values()) {
			Iterator iter = graph.nodes().iterator();
			while (iter.hasNext()) {
				Node n = (Node) iter.next();
				if (task.getName().compareTo(graph.getName(n)) == 0) {
					task.setRunRate(graph.getRepetitions(n));
				}
			}
		}

		tgst.setName(tg);

		Iterator iter = s.iterator();
		while (iter.hasNext()) {
			ScheduleElement se = (ScheduleElement) iter.next();
			if (se instanceof mocgraph.sched.Schedule) {
				mocgraph.sched.Schedule ss = (mocgraph.sched.Schedule) se;
				ScheduleElementType set = new ScheduleElementType();
				LoopType l = getLoopType(graph, ss);
				set.setLoop(l);
				s_sgt.getScheduleElement().add(set);
			} else if (se instanceof Firing) {
				Firing f = (Firing) se;
				TaskInstanceType ti = new TaskInstanceType();
				ti.setName(graph.getName((Node) f.getFiringElement()));
				ti.setRepetition(BigInteger.valueOf(f.getIterationCount()));
				ScheduleElementType set = new ScheduleElementType();
				set.setTask(ti);
				s_sgt.getScheduleElement().add(set);
			}
		}

		s_sgt.setScheduleType(ScheduleType.STATIC);
		s_sgt.setPoolName("HOSTPC");
		s_sgt.setLocalId(BigInteger.valueOf(0));
		tgst.getScheduleGroup().add(s_sgt);
		taskGroups.getTaskGroup().add(tgst);
		schedule.setTaskGroups(taskGroups);
		try {
			loaderSched.storeResource(schedule, new File(mOutputPath + tg + "_" + m + "_schedule.xml"));
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public static Map<String, Integer> generateIterationCount(Task t, String mode, Map<String, Task> mTask,
			Map<Integer, Queue> mQueue) {
		Map<String, Integer> result = new HashMap<String, Integer>();
		// Make SDFgraph
		ArrayList<Task> taskList = new ArrayList<Task>();
		ArrayList<Queue> channelList = new ArrayList<Queue>();

		if (t != null) {
			for (Task i_t : mTask.values()) {
				if (i_t.getParentTask().equals(t.getName()) && i_t.getHasSubgraph().equalsIgnoreCase("No")) {
					taskList.add(i_t);
				}
			}

			for (Queue i_q : mQueue.values()) {
				if (i_q.getDst().contains(t.getName())) {
					channelList.add(i_q);
				}
			}
		} else {
			for (Task i_t : mTask.values()) {
				taskList.add(i_t);
			}
			for (Queue i_q : mQueue.values()) {
				channelList.add(i_q);
			}
		}

		SDFGraph graph = new SDFGraph(taskList.size(), channelList.size());

		// Construct Node
		int instanceid = 0;
		for (int i = 0; i < taskList.size(); i++) {
			String nodeName = taskList.get(i).getName();
			SDFNodeWeight weight = new SDFNodeWeight(nodeName, instanceid++);
			Node node = new Node(weight);
			graph.addNode(node);
			graph.setName(node, nodeName);
		}

		int flag = 0;

		for (Queue i_i_q : channelList) {
			int srcRate = 0;
			int dstRate = 0;
			for (Task i_i_t : taskList) {
				// Src port
				if (i_i_q.getSrc().equals(i_i_t.getName())) {
					Map<String, Map<String, Integer>> portList = i_i_t.getPortList();
					for (String portName : portList.keySet()) {
						if (i_i_q.getSrcPortName().equals(portName)) {
							Map<String, Integer> port = portList.get(portName);
							if (port.size() != 0) {
								if (!port.containsKey(mode) && !port.containsKey("Default"))
									srcRate = -1;
								else if (port.containsKey("Default"))
									srcRate = port.get("Default");
								else
									srcRate = port.get(mode);
							}
							break;
						}
					}
				}
				// Dst port
				if (i_i_q.getDst().equals(i_i_t.getName())) {
					Map<String, Map<String, Integer>> portList = i_i_t.getPortList();
					for (String portName : portList.keySet()) {
						if (i_i_q.getDstPortName().equals(portName)) {
							Map<String, Integer> port = portList.get(portName);
							if (port.size() != 0) {
								if (!port.containsKey(mode) && !port.containsKey("Default"))
									dstRate = -1;
								else if (port.containsKey("Default"))
									dstRate = port.get("Default");
								else
									dstRate = port.get(mode);
							}
							break;
						}
					}
				}
			}

			if (srcRate == -1 || dstRate == -1) {
				flag = 1;
				break;
			}

			// Make Channel
			if (srcRate != 0 || dstRate != 0) {
				Node srcNode = null, dstNode = null;
				Iterator iter = graph.nodes().iterator();
				while (iter.hasNext()) {
					Node n = (Node) iter.next();
					if (graph.getName(n).compareTo(i_i_q.getSrc()) == 0)
						srcNode = n;
					if (graph.getName(n).compareTo(i_i_q.getDst()) == 0)
						dstNode = n;
				}
				SDFEdgeWeight weight = new SDFEdgeWeight(i_i_q.getSrcPortName(), i_i_q.getDstPortName(), srcRate,
						dstRate, Integer.parseInt(i_i_q.getInitData()));
				Edge edge = new Edge(srcNode, dstNode);
				edge.setWeight(weight);
				graph.addEdge(edge);
				graph.setName(edge, "" + "" + i_i_q.getSrcPortName() + "to" + "" + i_i_q.getDstPortName());
			}
		}

		Iterator iter = graph.nodes().iterator();
		ArrayList removeList = new ArrayList<Node>();
		while (iter.hasNext()) {
			Node n = (Node) iter.next();
			String nodeName = graph.getName(n);

			int zeroFlag = 0;
			for (int i = 0; i < taskList.size(); i++) {
				String taskName = taskList.get(i).getName();
				if (nodeName.equals(taskName)) {
					Map<String, Map<String, Integer>> portList = taskList.get(i).getPortList();
					for (String portName : portList.keySet()) {
						Map<String, Integer> port = portList.get(portName);
						int rate = 0;
						if (port.size() != 0) {
							if (port.containsKey("Default"))
								rate = port.get("Default");
							else
								rate = port.get(mode);
						}

						if (rate != 0) {
							zeroFlag = 1;
							break;
						}
					}
					break;
				}
			}
			if (zeroFlag == 0) {
				removeList.add(n);
			}
		}

		Iterator remove_iter = removeList.iterator();
		while (remove_iter.hasNext()) {
			Node node = (Node) remove_iter.next();
			graph.removeNode(node);
		}

		// System.out.println(graph);
		if (flag == 0) {
			mocgraph.sched.Schedule sched;
			if (taskList.size() <= 2) {
				TwoNodeStrategy st = new TwoNodeStrategy(graph);
				sched = st.schedule();
			} else {
				MinBufferStrategy st = new MinBufferStrategy(graph);
				sched = st.schedule();
			}

			Iterator sched_iter = sched.iterator();
			while (sched_iter.hasNext()) {
				ScheduleElement se = (ScheduleElement) sched_iter.next();

				if (se instanceof mocgraph.sched.Schedule) {
					mocgraph.sched.Schedule ss = (mocgraph.sched.Schedule) se;
					ScheduleElementType set = new ScheduleElementType();
					LoopType l = getLoopType(graph, ss, result);
				} else if (se instanceof Firing) {
					Firing f = (Firing) se;
					TaskInstanceType ti = new TaskInstanceType();
					String taskName = graph.getName((Node) f.getFiringElement());
					int taskRep = f.getIterationCount();
					if (result.get(taskName) != null)
						taskRep = result.get(taskName) + f.getIterationCount();
					result.put(taskName, taskRep);
				}
			}

		} else
			System.out.println("Something Wrong!");

		return result;
	}

	private static LoopType getLoopType(SDFGraph graph, mocgraph.sched.Schedule s, Map<String, Integer> result) {
		LoopType loop = new LoopType();
		loop.setName("loop" + (loopId++));
		loop.setRepetition(BigInteger.valueOf(s.getIterationCount()));

		Iterator iter = s.iterator();
		while (iter.hasNext()) {
			ScheduleElement se = (ScheduleElement) iter.next();
			if (se instanceof mocgraph.sched.Schedule) {
				mocgraph.sched.Schedule ss = (mocgraph.sched.Schedule) se;
				LoopType l = getLoopType(graph, ss, result);
			} else if (se instanceof Firing) {
				Firing f = (Firing) se;
				TaskInstanceType ti = new TaskInstanceType();
				String taskName = graph.getName((Node) f.getFiringElement());
				int taskRep = 1;
				if (result.get(taskName) != null)
					taskRep = result.get(taskName) + f.getIterationCount();
				result.put(taskName, taskRep);
			}
		}
		return loop;
	}

	private static LoopType getLoopType(SDFGraph graph, mocgraph.sched.Schedule s) {
		LoopType loop = new LoopType();
		loop.setName("loop" + (loopId++));
		loop.setRepetition(BigInteger.valueOf(s.getIterationCount()));

		Iterator iter = s.iterator();
		while (iter.hasNext()) {
			ScheduleElement se = (ScheduleElement) iter.next();
			if (se instanceof mocgraph.sched.Schedule) {
				mocgraph.sched.Schedule ss = (mocgraph.sched.Schedule) se;
				LoopType l = getLoopType(graph, ss);
				ScheduleElementType set = new ScheduleElementType();
				set.setLoop(l);
				loop.getScheduleElement().add(set);
			} else if (se instanceof Firing) {
				Firing f = (Firing) se;
				TaskInstanceType ti = new TaskInstanceType();
				ti.setName(graph.getName((Node) f.getFiringElement()));
				ti.setRepetition(BigInteger.valueOf(f.getIterationCount()));
				ScheduleElementType set = new ScheduleElementType();
				set.setTask(ti);
				loop.getScheduleElement().add(set);
			}
		}
		return loop;
	}

	static boolean go_skip = false;
		
	public static String scheduleParsingWithExecutionPolicy(ScheduleElementType sched, int resultFlag, int depth,
			String mRuntimeExecutionPolicy, Map<String, Task> mTask, String mode) {
		String code = "";
		String tab = "";
		for (int i = 0; i < depth; i++)
			tab += "\t";
		if (sched.getLoop() != null) {
			int count = sched.getLoop().getRepetition().intValue();
			code += tab + "\t\t{\n";
			code += tab + "\t\t\tCIC_T_INT i_" + depth + "=0;\n";
			code += tab + "\t\t\tfor(i_" + depth + "= 0; i_" + depth + "<" + count + "; i_" + depth + "++){\n";
			depth = depth + 2;
			{
				List<ScheduleElementType> scheds = sched.getLoop().getScheduleElement();
				for (int j = 0; j < scheds.size(); j++) {
					code += scheduleParsingWithExecutionPolicy(scheds.get(j), resultFlag, depth,
							mRuntimeExecutionPolicy, mTask, mode);
				}
			}
			code += tab + "\t\t\t}\n";
			code += tab + "\t\t}\n";
		} else { // if (getLoop == null) -> sdf
			tab += "\t";
			if (sched.getTask().getRepetition() == null)
				System.out.println(sched.getTask().getName());
			if (sched.getTask().getRepetition().intValue() == 1) {
				TaskInstanceType task = sched.getTask();
				if (go_skip == false) {
					if (mRuntimeExecutionPolicy.equals(HopesInterface.RuntimeExecutionPolicy_FullyStatic)) {
						if(sched.getTask().getStartTime().intValue() != sched_time){
							code += tab + "while(1){\n" 
									+ tab + "\tclock_gettime(CLOCK_MONOTONIC, &end);\n" 
									+ tab + "\tdiff = (end.tv_sec - start.tv_sec)*1000000 + ((end.tv_nsec - start.tv_nsec)/1000);\n"
									+ tab + "\tif(" + sched.getTask().getStartTime().intValue() + " <= diff)\n" 
									+ tab + "\t\tbreak;\n" 
									+ tab + "}\n";
						}				
					} else if (mRuntimeExecutionPolicy.equals(HopesInterface.RuntimeExecutionPolicy_StaticAssign)) {

						code += tab + "\tcase " + mTask.get(task.getName()).getIndex() + ":\n";

					}

					code += tab + task.getName() + "_Go();\n";
					if (mRuntimeExecutionPolicy.equals(HopesInterface.RuntimeExecutionPolicy_FullyStatic)) {
						// unit: us
						code += tab + "while(1){\n" 
								+ tab + "\tclock_gettime(CLOCK_MONOTONIC, &end);\n" 
								+ tab + "\tdiff = (end.tv_sec - start.tv_sec)*1000000 + ((end.tv_nsec - start.tv_nsec)/1000);\n"
								+ tab + "\tif(" + sched.getTask().getEndTime().intValue() + " <= diff)\n" 
								+ tab + "\t\tbreak;\n" 
								+ tab + "}\n\n";
						sched_time = sched.getTask().getEndTime().intValue();
					} else if (mRuntimeExecutionPolicy.equals(HopesInterface.RuntimeExecutionPolicy_StaticAssign)) {
						code += tab + "\ttasks[" + mTask.get(task.getName()).getIndex() + "].run_count++;\n";
						code += tab + "\tbreak;\n";
					}
				}
				go_skip = false;
			} else {	//that means, if (sched.getTask().getRepetition().intValue() != 1) {
				// fully-static : need to check
				// RuntimeExecutionPolicy_StaticAssign : need to check
				TaskInstanceType task = sched.getTask();
				int count = sched.getTask().getRepetition().intValue();
				code += tab + "\t{\n";
				code += tab + "\t\tCIC_T_INT i_" + depth + "=0;\n";
				code += tab + "\t\tfor(i_" + depth + "= 0; i_" + depth + "<" + count + "; i_" + depth + "++){\n";
				if (go_skip == true) {
					code += tab + "\t\t\tif(i_" + depth + "!= 0)";
					if (mRuntimeExecutionPolicy.equals(HopesInterface.RuntimeExecutionPolicy_FullyStatic)) {
						code += tab + "clock_gettime(CLOCK_MONOTONIC, &start);\n";
					}
					code += tab + "\t\t\t\t" + task.getName() + "_Go();\n";
					if (mRuntimeExecutionPolicy.equals(HopesInterface.RuntimeExecutionPolicy_FullyStatic)) {
						// unit: us
						code += tab + "while(1){\n" + tab + "\tclock_gettime(CLOCK_MONOTONIC, &end);\n" + tab
								+ "\tdiff = (end.tv_sec - start.tv_sec)*1000000 + ((end.tv_nsec - start.tv_nsec)/1000);\n"
								+ tab + "\tif(GetWorstCaseExecutionTimeFromTaskIdAndModeName("
								+ mTask.get(task.getName()).getIndex() + ", \"" + mode + "\") <= diff)\n" + tab
								+ "\t\tbreak;\n" + tab + "\t}\n" + tab + "}\n\n";
					}
				} else {
					if (mRuntimeExecutionPolicy.equals(HopesInterface.RuntimeExecutionPolicy_FullyStatic)) {
						code += tab + "clock_gettime(CLOCK_MONOTONIC, &start);\n";
					}
					code += tab + "\t\t\t" + task.getName() + "_Go();\n";
					if (mRuntimeExecutionPolicy.equals(HopesInterface.RuntimeExecutionPolicy_FullyStatic)) {
						// unit: us
						code += tab + "while(1){\n" + tab + "\tclock_gettime(CLOCK_MONOTONIC, &end);\n" + tab
								+ "\tdiff = (end.tv_sec - start.tv_sec)*1000000 + ((end.tv_nsec - start.tv_nsec)/1000);\n"
								+ tab + "\tif(GetWorstCaseExecutionTimeFromTaskIdAndModeName("
								+ mTask.get(task.getName()).getIndex() + ", \"" + mode + "\") <= diff)\n" + tab
								+ "\t\tbreak;\n" + tab + "\t}\n" + tab + "}\n\n";
					}
				}
				code += tab + "\t\t}\n";
				code += tab + "\t}\n";
				go_skip = false;
			}
		}
		return code;
	}

	public static String scheduleParsing(ScheduleElementType sched, int resultFlag, int depth) {
		String code = "";
		String tab = "";
		for (int i = 0; i < depth; i++)
			tab += "\t";
		if (sched.getLoop() != null) {
			int count = sched.getLoop().getRepetition().intValue();
			code += tab + "\t\t{\n";
			code += tab + "\t\t\tCIC_T_INT i_" + depth + "=0;\n";
			code += tab + "\t\t\tfor(i_" + depth + "= 0; i_" + depth + "<" + count + "; i_" + depth + "++){\n";
			depth = depth + 2;
			{
				List<ScheduleElementType> scheds = sched.getLoop().getScheduleElement();
				for (int j = 0; j < scheds.size(); j++) {
					code += scheduleParsing(scheds.get(j), resultFlag, depth);
				}
			}
			code += tab + "\t\t\t}\n";
			code += tab + "\t\t}\n";
		} else {  // if (getLoop == null) -> sdf
			tab += "\t";
			if (sched.getTask().getRepetition() == null)
				System.out.println(sched.getTask().getName());
			if (sched.getTask().getRepetition().intValue() == 1) {
				TaskInstanceType task = sched.getTask();
				if (go_skip == false) {
					code += tab + task.getName() + "_Go();\n";
				}
				go_skip = false;
			} else {
				// need to cover fully-static
				TaskInstanceType task = sched.getTask();
				int count = sched.getTask().getRepetition().intValue();
				code += tab + "\t{\n";
				code += tab + "\t\tCIC_T_INT i_" + depth + "=0;\n";
				code += tab + "\t\tfor(i_" + depth + "= 0; i_" + depth + "<" + count + "; i_" + depth + "++){\n";
				if (go_skip == true) {
					code += tab + "\t\t\tif(i_" + depth + "!= 0)";
					code += tab + "\t\t\t\t" + task.getName() + "_Go();\n";
				} else
					code += tab + "\t\t\t" + task.getName() + "_Go();\n";
				code += tab + "\t\t}\n";
				code += tab + "\t}\n";
				go_skip = false;
			}
		}
		return code;
	}

	public static String generateSingleProcessorStaticScheduleCode(String outputPath, Map<String, Task> mTask,
			Map<String, Task> mVTask) {
		String staticScheduleCode = "";
		CICScheduleType schedule;
		CICScheduleTypeLoader scheduleLoader = new CICScheduleTypeLoader();

		Map<String, Task> mMergeTask = new HashMap<String, Task>();

		for (Task t : mTask.values())
			mMergeTask.put(t.getName(), t);
		for (Task t : mVTask.values())
			mMergeTask.put(t.getName(), t);

		for (Task task : mMergeTask.values()) {
			if (task.getHasSubgraph().equalsIgnoreCase("Yes")) {
				// if(task.getHasMTM().equalsIgnoreCase("Yes")){
				staticScheduleCode += ("CIC_T_VOID " + task.getName() + "_Init(CIC_T_INT task_id){\n");

				for (Task t : mTask.values()) {
					if (t.getParentTask().equals(task.getName()) && t.getHasSubgraph().equalsIgnoreCase("No")) {
						staticScheduleCode += ("\t" + t.getName() + "_Init(" + t.getIndex() + ");\n");
					}
				}
				staticScheduleCode += ("\n}\n\n");

					System.out.println("$$$$$$$$$$$ task: " + task.getName());
				if (task.getHasMTM() == true) {
					staticScheduleCode += ("CIC_T_VOID " + task.getName() + "_Go(){\n");
					staticScheduleCode += ("\tCIC_T_INT i=0;\n\tCIC_T_INT mtm_id = 0;\n\tCIC_T_CHAR* mode = 0;\n\tCIC_T_INT task_id = GetTaskIdFromTaskName(\""
							+ task.getName() + "\");\n");
					staticScheduleCode += ("\tfor(i=0; i<CIC_UV_NUM_MTMS; i++){\n\t\tif(task_id == mtms[i].task_id){\n\t\t\tmtm_id = i;\n\t\t\tbreak;\n\t\t}\n\t}\n\n");
					staticScheduleCode += "\tmode = mtms[mtm_id].GetCurrentModeName();\n";

					int flag = 0;
					int resultFlag = 0;
					ScheduleElementType sched_1 = null;
					String prevModeFirstFunction = "";
					try {
						List<String> modeList = new ArrayList<String>();
						modeList = task.getMTM().getModes();

						for (String mode : modeList) {
							ArrayList<File> schedFileList = new ArrayList<File>();
							File file = new File(outputPath);
							File[] fileList = file.listFiles();
							for (File f : fileList) {
								if (f.getName().contains(task.getName() + "_" + mode)
										&& f.getName().endsWith("_schedule.xml")) {
									schedFileList.add(f);
								}
							}
							if (schedFileList.size() <= 0) {
								JOptionPane.showMessageDialog(null, "You should execute 'Analysis' before build!");
								System.exit(-1);
							}
							schedule = scheduleLoader.loadResource(schedFileList.get(0).getAbsolutePath());
							TaskGroupsType taskGroups = schedule.getTaskGroups();
							List<TaskGroupForScheduleType> taskGroupList = taskGroups.getTaskGroup();
							sched_1 = taskGroupList.get(0).getScheduleGroup().get(0).getScheduleElement().get(0);
							String currModeFirstFunction = "";
							while (true) {
								if (sched_1.getTask() != null) {
									currModeFirstFunction = sched_1.getTask().getName();
									break;
								} else if (sched_1.getLoop() != null) {
									sched_1 = sched_1.getLoop().getScheduleElement().get(0);
								}
							}
							if (flag == 0) {
								prevModeFirstFunction = currModeFirstFunction;
								flag++;
							} else {
								if (currModeFirstFunction.equals(prevModeFirstFunction))
									continue;
								else {
									resultFlag = 1;
									break;
								}
							}
						}
						if (resultFlag == 0) {
							staticScheduleCode += "\t" + sched_1.getTask().getName() + "_Go();\n";
							staticScheduleCode += "\tmtms[mtm_id].Transition();\n";
							staticScheduleCode += "\tmode = mtms[mtm_id].GetCurrentModeName();\n";
						}
					} catch (CICXMLException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}

					int index = 0;
					for (String mode : task.getMTM().getModes()) {
						try {
							if (resultFlag == 0)
								go_skip = true;
							else
								go_skip = false;
							if (index == 0)
								staticScheduleCode += "\tif(CIC_F_STRING_COMPARE(mode, \"" + mode + "\") == 0){\n";
							else if (index != 0)
								staticScheduleCode += "\telse if(CIC_F_STRING_COMPARE(mode, \"" + mode + "\") == 0){\n";

							ArrayList<File> schedFileList = new ArrayList<File>();
							File file = new File(outputPath);
							File[] fileList = file.listFiles();
							for (File f : fileList) {
								if (f.getName().contains(task.getName() + "_" + mode)
										&& f.getName().endsWith("_schedule.xml")) {
									schedFileList.add(f);
								}
							}
							if (schedFileList.size() <= 0) {
								JOptionPane.showMessageDialog(null, "You should execute 'Analysis' before build!");
								System.exit(-1);
							}

							schedule = scheduleLoader.loadResource(schedFileList.get(0).getAbsolutePath());
							TaskGroupsType taskGroups = schedule.getTaskGroups();
							List<TaskGroupForScheduleType> taskGroupList = taskGroups.getTaskGroup();
							for (int i = 0; i < taskGroupList.size(); i++) {
								List<ScheduleGroupType> schedGroup = taskGroupList.get(i).getScheduleGroup();
								for (int j = 0; j < schedGroup.size(); j++) {
									List<ScheduleElementType> scheds = schedGroup.get(j).getScheduleElement();
									for (int k = 0; k < scheds.size(); k++) {
										ScheduleElementType sched = scheds.get(k);
										staticScheduleCode += scheduleParsing(sched, resultFlag, 0);
									}
								}
							}
							staticScheduleCode += "\t}\n";
							index++;

						} catch (CICXMLException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
					}
					staticScheduleCode += "\n}\n\n";
				} else {
					try {
						go_skip = false;
						staticScheduleCode += ("CIC_T_VOID " + task.getName() + "_Go(){\n");
						
						String split = "";
						if (System.getProperty("os.name").contains("Windows"))	split = "\\";
						else													split = "/";
						
						if (!Util.fileIsLive(outputPath + split + task.getName() + "_" + "Default_1_schedule.xml")) {
							JOptionPane.showMessageDialog(null, "You should execute 'Analysis' before build!");
							System.exit(-1);
						}
						schedule = scheduleLoader.loadResource(outputPath + split + task.getName() + "_" + "Default_1_schedule.xml");
						TaskGroupsType taskGroups = schedule.getTaskGroups();
						List<TaskGroupForScheduleType> taskGroupList = taskGroups.getTaskGroup();
						for (int i = 0; i < taskGroupList.size(); i++) {
							List<ScheduleGroupType> schedGroup = taskGroupList.get(i).getScheduleGroup();
							for (int j = 0; j < schedGroup.size(); j++) {
								List<ScheduleElementType> scheds = schedGroup.get(j).getScheduleElement();
								for (int k = 0; k < scheds.size(); k++) {
									ScheduleElementType sched = scheds.get(k);
									staticScheduleCode += scheduleParsing(sched, 0, 0);
								}
							}
						}
						staticScheduleCode += "\n}\n\n";

					} catch (CICXMLException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}

				staticScheduleCode += ("CIC_T_VOID " + task.getName() + "_Wrapup(){\n");
				for (Task t : mTask.values()) {
					if (t.getParentTask().equals(task.getName()) && t.getHasSubgraph().equalsIgnoreCase("No")) {
						staticScheduleCode += ("\t" + t.getName() + "_Wrapup();\n");
					}
				}
				staticScheduleCode += ("\n}\n\n");
			} else
				staticScheduleCode += "";
		}
		return staticScheduleCode;
	}

	public static String generateMultiProcessorStaticScheduleCode(String outputPath, Map<String, Task> mTask,
			Map<String, Task> mVTask, Map<String, Task> mPVTask) {
		String staticScheduleCode = "";
		String goCode = "";
		String initCode = "";
		String wrapupCode = "";
		CICScheduleType schedule;
		CICScheduleTypeLoader scheduleLoader = new CICScheduleTypeLoader();
		for (Task task : mPVTask.values()) {
			Task parentTask = null;
			for (Task t : mTask.values()) {
				if (t.getName().equals(task.getParentTask())) {
					parentTask = t;
					break;
				}
			}
			for (Task t : mVTask.values()) {
				if (t.getName().equals(task.getParentTask())) {
					parentTask = t;
					break;
				}
			}
			List<String> modeList = new ArrayList<String>();

			if (parentTask.getMTM() != null)
				modeList = parentTask.getMTM().getModes();
			else
				modeList.add("Default");

			initCode += ("CIC_T_VOID " + task.getName() + "_Init(CIC_T_INT virtual_task_id){\n");
			goCode += ("CIC_T_VOID " + task.getName() + "_Go(){\n");
			wrapupCode += ("CIC_T_VOID " + task.getName() + "_Wrapup(){\n");

			if (modeList.size() > 1) {
				initCode += "\tCIC_T_INT i=0;\n\tCIC_T_INT mtm_index = 0;\n\tCIC_T_CHAR* mode = 0;\n";
				initCode += "\tCIC_T_INT task_id = GetTaskIdFromTaskName(\"" + task.getParentTask() + "\");\n";
				initCode += ("\tfor(i=0; i<CIC_UV_NUM_MTMS; i++){\n\t\tif(task_id == mtms[i].task_id){\n\t\t\tmtm_index = i;\n\t\t\tbreak;\n\t\t}\n\t}\n\n");
				initCode += "\tmode = mtms[mtm_index].GetCurrentModeName();\n";

				wrapupCode += "\tCIC_T_INT i=0;\n\tCIC_T_INT mtm_index = 0;\n\tCIC_T_CHAR* mode = 0;\n";
				wrapupCode += "\tCIC_T_INT task_id = GetTaskIdFromTaskName(\"" + task.getParentTask() + "\");\n";
				wrapupCode += ("\tfor(i=0; i<CIC_UV_NUM_MTMS; i++){\n\t\tif(task_id == mtms[i].task_id){\n\t\t\tmtm_index = i;\n\t\t\tbreak;\n\t\t}\n\t}\n\n");
				wrapupCode += "\tmode = mtms[mtm_index].GetCurrentModeName();\n";
			}

			int index = 0;
			for (String mode : modeList) {
				ArrayList<String> history = new ArrayList<String>();
				try {
					if (modeList.size() > 1) {
						if (index == 0) {
							initCode += "\tif(CIC_F_STRING_COMPARE(mode, \"" + mode + "\") == 0){\n";
							wrapupCode += "\tif(CIC_F_STRING_COMPARE(mode, \"" + mode + "\") == 0){\n";
						} else if (index != 0) {
							initCode += "\telse if(CIC_F_STRING_COMPARE(mode, \"" + mode + "\") == 0){\n";
							wrapupCode += "\telse if(CIC_F_STRING_COMPARE(mode, \"" + mode + "\") == 0){\n";
						}
					}
					ArrayList<File> schedFileList = new ArrayList<File>();
					File file = new File(outputPath);
					File[] fileList = file.listFiles();
					for (File f : fileList) {
						if (f.getName().contains(task.getParentTask() + "_" + mode)
								&& f.getName().endsWith("_schedule.xml")) {
							schedFileList.add(f);
						}
					}
					if (schedFileList.size() <= 0) {
						JOptionPane.showMessageDialog(null, "You should execute 'Analysis' before build!");
						System.exit(-1);
					}

					for (int f_i = 0; f_i < schedFileList.size(); f_i++) {
						schedule = scheduleLoader.loadResource(schedFileList.get(0).getAbsolutePath());
						int num_proc = schedule.getTaskGroups().getTaskGroup().get(0).getScheduleGroup().size();

						TaskGroupsType taskGroups = schedule.getTaskGroups();
						List<TaskGroupForScheduleType> taskGroupList = taskGroups.getTaskGroup();
						boolean srcFlag = false;
						for (int i = 0; i < taskGroupList.size(); i++) {
							List<ScheduleGroupType> schedGroup = taskGroupList.get(i).getScheduleGroup();
							for (int j = 0; j < schedGroup.size(); j++) {
								if (schedGroup.get(j).getLocalId().intValue() == task.getProc().get("Default")
										.get("Default").get(0)) // Need to fix
								{
									List<ScheduleElementType> scheds = schedGroup.get(j).getScheduleElement();

									for (int k = 0; k < scheds.size(); k++) {
										ScheduleElementType sched = scheds.get(k);
										String taskName = sched.getTask().getName();
										String taskId = "0";
										for (Task t : mTask.values()) {
											if (t.getName().equals(taskName)) {
												taskId = t.getIndex();
												break;
											}
										}
										if (!history.contains(taskName)) {
											initCode += "\t\t" + taskName + "_Init(" + taskId + ");\n";
											wrapupCode += "\t\t" + taskName + "_Wrapup();\n";
											history.add(taskName);
										}
									}
								}
							}
						}
						if (modeList.size() > 1) {
							initCode += "\t}\n";
							wrapupCode += "\t}\n";
						}
						index++;
					}
				} catch (CICXMLException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}

			if (modeList.size() > 1) {
				goCode += "\tCIC_T_INT i=0;\n\tCIC_T_INT mtm_index = 0;\n\tCIC_T_CHAR* mode = 0;\n";
				goCode += "\tCIC_T_INT task_id = GetTaskIdFromTaskName(\"" + task.getParentTask() + "\");\n";
				goCode += ("\tfor(i=0; i<CIC_UV_NUM_MTMS; i++){\n\t\tif(task_id == mtms[i].task_id){\n\t\t\tmtm_index = i;\n\t\t\tbreak;\n\t\t}\n\t}\n\n");
			}
			// Current assumption: A source task should be mapped on to the same
			// processor for all modes
			boolean isSrcTask = false;
			String srcGoCode = "";
			for (String mode : modeList) {
				try {
					ArrayList<File> schedFileList = new ArrayList<File>();
					File file = new File(outputPath);
					File[] fileList = file.listFiles();
					for (File f : fileList) {
						if (f.getName().contains(task.getParentTask() + "_" + mode)
								&& f.getName().endsWith("_schedule.xml")) {
							schedFileList.add(f);
						}
					}
					if (schedFileList.size() <= 0) {
						JOptionPane.showMessageDialog(null, "You should execute 'Analysis' before build!");
						System.exit(-1);
					}

					for (int f_i = 0; f_i < schedFileList.size(); f_i++) {
						schedule = scheduleLoader.loadResource(schedFileList.get(0).getAbsolutePath());
						int num_proc = schedule.getTaskGroups().getTaskGroup().get(0).getScheduleGroup().size();

						TaskGroupsType taskGroups = schedule.getTaskGroups();
						List<TaskGroupForScheduleType> taskGroupList = taskGroups.getTaskGroup();
						for (int i = 0; i < taskGroupList.size(); i++) {
							List<ScheduleGroupType> schedGroup = taskGroupList.get(i).getScheduleGroup();
							for (int j = 0; j < schedGroup.size(); j++) {
								if (schedGroup.get(j).getLocalId().intValue() == task.getProc().get("Default")
										.get("Default").get(0)) // Need to fix
								{
									List<ScheduleElementType> scheds = schedGroup.get(j).getScheduleElement();

									for (int k = 0; k < scheds.size(); k++) {
										ScheduleElementType sched = scheds.get(k);
										String firstTaskName = sched.getTask().getName();
										if (k == 0 && mTask.get(firstTaskName).getInPortList().size() == 0) {
											srcGoCode += scheduleParsing(sched, 0, 0);
											isSrcTask = true;
											break;
										}
									}
								}
								if (isSrcTask)
									break;
							}
							if (isSrcTask)
								break;
						}
						if (isSrcTask)
							break;
					}
					if (isSrcTask)
						break;

				} catch (CICXMLException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}

			if (isSrcTask) {
				goCode += srcGoCode;
				if (modeList.size() > 1)
					goCode += "\tmtms[mtm_index].Transition();\n";
				task.setIsSrcTask(true);
			}
			if (!isSrcTask && modeList.size() > 1)
				goCode += "\tmtms[mtm_index].UpdateCurrentMode(\"" + task.getName() + "\");\n";
			if (modeList.size() > 1)
				goCode += "\n\tmode = mtms[mtm_index].GetCurrentModeName(\"" + task.getName() + "\");\n";

			index = 0;
			for (String mode : modeList) {
				try {
					if (modeList.size() > 1) {
						if (index == 0)
							goCode += "\tif(CIC_F_STRING_COMPARE(mode, \"" + mode + "\") == 0){\n";
						else if (index != 0)
							goCode += "\telse if(CIC_F_STRING_COMPARE(mode, \"" + mode + "\") == 0){\n";
					}
					ArrayList<File> schedFileList = new ArrayList<File>();
					File file = new File(outputPath);
					File[] fileList = file.listFiles();
					for (File f : fileList) {
						if (f.getName().contains(task.getParentTask() + "_" + mode)
								&& f.getName().endsWith("_schedule.xml")) {
							schedFileList.add(f);
						}
					}
					if (schedFileList.size() <= 0) {
						JOptionPane.showMessageDialog(null, "You should execute 'Analysis' before build!");
						System.exit(-1);
					}

					schedule = scheduleLoader.loadResource(schedFileList.get(0).getAbsolutePath());
					TaskGroupsType taskGroups = schedule.getTaskGroups();
					List<TaskGroupForScheduleType> taskGroupList = taskGroups.getTaskGroup();
					for (int i = 0; i < taskGroupList.size(); i++) {
						List<ScheduleGroupType> schedGroup = taskGroupList.get(i).getScheduleGroup();
						for (int j = 0; j < schedGroup.size(); j++) {
							if (schedGroup.get(j).getLocalId().intValue() == task.getProc().get("Default")
									.get("Default").get(0)) // Need to fix
							{
								List<ScheduleElementType> scheds = schedGroup.get(j).getScheduleElement();

								for (int k = 0; k < scheds.size(); k++) {
									ScheduleElementType sched = scheds.get(k);
									String firstTaskName = sched.getTask().getName();
									if (mTask.get(firstTaskName).getInPortList().size() > 0) {
										goCode += scheduleParsing(sched, 0, 0);
									}
								}
							}
						}
					}
					if (modeList.size() > 1)
						goCode += "\t}\n";
					index++;

				} catch (CICXMLException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}

			initCode += ("}\n\n");
			goCode += ("}\n\n");
			wrapupCode += ("}\n\n");

			staticScheduleCode = initCode + goCode + wrapupCode;

		}

		return staticScheduleCode;
	}

	private static Task getParentTask(Task task, Map<String, Task> mTask, Map<String, Task> mVTask) {
		Task parentTask = null;
		for (Task t : mTask.values()) {
			if (t.getName().equals(task.getParentTask())) {
				parentTask = t;
				break;
			}
		}
		for (Task t : mVTask.values()) {
			if (t.getName().equals(task.getParentTask())) {
				parentTask = t;
				break;
			}
		}
		return parentTask;
	}

	private static String generateInitcode(String outputPath, Map<String, Task> mTask,
			CICScheduleTypeLoader scheduleLoader, List<String> modeList, Task task, String mRuntimeExecutionPolicy) {
		String code = "";
		CICScheduleType schedule;

		code += "CIC_T_VOID " + task.getName() + "_Init(CIC_T_INT virtual_task_id){\n";

		if (modeList.size() > 1) {
			code += "\tCIC_T_INT i=0;\n";
			code += "\tCIC_T_INT mtm_index = 0;\n";
			code += "\tCIC_T_CHAR* mode = 0;\n";
			code += "\tCIC_T_INT task_id = GetTaskIdFromTaskName(\"" + task.getParentTask() + "\");\n";
			code += "\tfor(i=0; i<CIC_UV_NUM_MTMS; i++){\n" + "\t\tif(task_id == mtms[i].task_id){\n"
					+ "\t\t\tmtm_index = i;\n\t\t\tbreak;\n" + "\t\t}\n" + "\t}\n\n";
			code += "\tmode = mtms[mtm_index].GetCurrentModeName();\n";
		}

		int index = 0;
		for (String mode : modeList) {
			ArrayList<String> history = new ArrayList<String>();
			try {
				if (modeList.size() > 1) {
					if (index == 0) {
						code += "\tif(CIC_F_STRING_COMPARE(mode, \"" + mode + "\") == 0){\n";
					} else if (index != 0) {
						code += "\telse if(CIC_F_STRING_COMPARE(mode, \"" + mode + "\") == 0){\n";
					}
				}
				ArrayList<File> schedFileList = new ArrayList<File>();
				File file = new File(outputPath);
				File[] fileList = file.listFiles();
				for (File f : fileList) {
					if (f.getName().contains(task.getParentTask() + "_" + mode)
							&& f.getName().endsWith("_schedule.xml")) {
						schedFileList.add(f);
					}
				}
				if (schedFileList.size() <= 0) {
					JOptionPane.showMessageDialog(null, "You should execute 'Analysis' before build!");
					System.exit(-1);
				}

				for (int f_i = 0; f_i < schedFileList.size(); f_i++) {
					// we assume that we save only the first schedule 
					schedule = scheduleLoader.loadResource(schedFileList.get(0).getAbsolutePath());

					TaskGroupsType taskGroups = schedule.getTaskGroups();
					List<TaskGroupForScheduleType> taskGroupList = taskGroups.getTaskGroup();
					for (int i = 0; i < taskGroupList.size(); i++) {
						List<ScheduleGroupType> schedGroup = taskGroupList.get(i).getScheduleGroup();
						for (int j = 0; j < schedGroup.size(); j++) {
							if (schedGroup.get(j).getLocalId().intValue() == task.getProc().get("Default")
									.get("Default").get(0)) // Need to fix
							{
								List<ScheduleElementType> scheds = schedGroup.get(j).getScheduleElement();

								for (int k = 0; k < scheds.size(); k++) {
									ScheduleElementType sched = scheds.get(k);
									String taskName = sched.getTask().getName();
									String taskId = "0";
									for (Task t : mTask.values()) {
										if (t.getName().equals(taskName)) {
											taskId = t.getIndex();
											break;
										}
									}
									if (!history.contains(taskName)) {
										code += "\t\t" + taskName + "_Init(" + taskId + ");\n";
										if (mRuntimeExecutionPolicy
												.equals(HopesInterface.RuntimeExecutionPolicy_StaticAssign)) {
											code += "\t\t" + "if(tasks[" + taskId + "].is_src_task == CIC_V_TRUE)\n"
													+ "\t\t\t" + "firable_task[" + taskId + "] = CIC_V_TRUE;\n";
										}
										history.add(taskName);
									}
								}
							}
						}
					}
					if (modeList.size() > 1) {
						code += "\t}\n";
					}
					index++;
				}
			} catch (CICXMLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		code += ("}\n\n");

		return code;
	}

	private static ArrayList<File> getSchedFileList(String outputPath, Task task, String mode) {
		ArrayList<File> schedFileList = new ArrayList<File>();

		File file = new File(outputPath);
		File[] fileList = file.listFiles();
		for (File f : fileList) {
			if (f.getName().contains(task.getParentTask() + "_" + mode) && f.getName().endsWith("_schedule.xml")) {
				schedFileList.add(f);
			}
		}
		if (schedFileList.size() <= 0) {
			JOptionPane.showMessageDialog(null, "You should execute 'Analysis' before build!");
			System.exit(-1);
		}

		return schedFileList;
	}

	private static String generateSADFGocode(String outputPath, Map<String, Task> mTask,
			CICScheduleTypeLoader scheduleLoader, List<String> modeList, Task task, String mRuntimeExecutionPolicy) {
		String code = "";
		CICScheduleType schedule;

		if (modeList.size() > 1) {
			code += "\tCIC_T_INT i=0;\n";
			code += "\tCIC_T_INT mtm_index = 0;\n";
			code += "\tCIC_T_CHAR* mode = 0;\n";
			code += "\tCIC_T_INT task_id = GetTaskIdFromTaskName(\"" + task.getParentTask() + "\");\n";
			code += "\tfor(i=0; i<CIC_UV_NUM_MTMS; i++){\n" + "\t\tif(task_id == mtms[i].task_id){\n"
					+ "\t\t\tmtm_index = i;\n" + "\t\t\tbreak;\n" + "\t\t}\n" + "\t}\n\n";
		}
		// Current assumption: A source task should be mapped on to the same
		// processor for all modes
		boolean isSrcTask = false; // in sadf, we assume there is only ONE! SrcTask!
		String srcGoCode = "";
		for (String mode : modeList) {
			try {
				ArrayList<File> schedFileList = getSchedFileList(outputPath, task, mode);

				for (int f_i = 0; f_i < schedFileList.size(); f_i++) {
					// we assume that we save only the first schedule 
					schedule = scheduleLoader.loadResource(schedFileList.get(0).getAbsolutePath());
					int num_proc = schedule.getTaskGroups().getTaskGroup().get(0).getScheduleGroup().size();

					TaskGroupsType taskGroups = schedule.getTaskGroups();
					List<TaskGroupForScheduleType> taskGroupList = taskGroups.getTaskGroup();
					for (int i = 0; i < taskGroupList.size(); i++) {
						List<ScheduleGroupType> schedGroup = taskGroupList.get(i).getScheduleGroup();
						for (int j = 0; j < schedGroup.size(); j++) {
							if (schedGroup.get(j).getLocalId().intValue() == task.getProc().get("Default")
									.get("Default").get(0)) // Need to fix
							{
								List<ScheduleElementType> scheds = schedGroup.get(j).getScheduleElement();

								for (int k = 0; k < scheds.size(); k++) {
									ScheduleElementType sched = scheds.get(k);
									String firstTaskName = sched.getTask().getName();

									if (k == 0 && mTask.get(firstTaskName).getInPortList().size() == 0) {
										srcGoCode += scheduleParsingWithExecutionPolicy(sched, 0, 0,
												mRuntimeExecutionPolicy, mTask, mode);
										isSrcTask = true;
										break;
									}
								}
							}
							if (isSrcTask)
								break;
						}
						if (isSrcTask)
							break;
					}
					if (isSrcTask)
						break;
				}
				if (isSrcTask)
					break;

			} catch (CICXMLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		if (isSrcTask) {
			code += srcGoCode;
			if (modeList.size() > 1)
				code += "\tmtms[mtm_index].Transition();\n";
			task.setIsSrcTask(true);
		}
		if (!isSrcTask && modeList.size() > 1)
			code += "\tmtms[mtm_index].UpdateCurrentMode(\"" + task.getName() + "\");\n";
		if (modeList.size() > 1)
			code += "\n\tmode = mtms[mtm_index].GetCurrentModeName(\"" + task.getName() + "\");\n";

		int index = 0;
		for (String mode : modeList) {
			try {
				if (modeList.size() > 1) {
					if (index == 0)
						code += "\tif(CIC_F_STRING_COMPARE(mode, \"" + mode + "\") == 0){\n";
					else if (index != 0)
						code += "\telse if(CIC_F_STRING_COMPARE(mode, \"" + mode + "\") == 0){\n";
				}

				ArrayList<File> schedFileList = getSchedFileList(outputPath, task, mode);

				// we assume that we save only the first schedule 
				schedule = scheduleLoader.loadResource(schedFileList.get(0).getAbsolutePath());
				if(!isSrcTask)
					sched_time = 0;	
				TaskGroupsType taskGroups = schedule.getTaskGroups();
				List<TaskGroupForScheduleType> taskGroupList = taskGroups.getTaskGroup();
				for (int i = 0; i < taskGroupList.size(); i++) {
					List<ScheduleGroupType> schedGroup = taskGroupList.get(i).getScheduleGroup();
					for (int j = 0; j < schedGroup.size(); j++) {
						if (schedGroup.get(j).getLocalId().intValue() == task.getProc().get("Default").get("Default")
								.get(0)) // Need to fix
						{
							List<ScheduleElementType> scheds = schedGroup.get(j).getScheduleElement();

							for (int k = 0; k < scheds.size(); k++) {
								ScheduleElementType sched = scheds.get(k);
								String firstTaskName = sched.getTask().getName();
								if (mTask.get(firstTaskName).getInPortList().size() > 0) {
									code += scheduleParsingWithExecutionPolicy(sched, 0, 0, mRuntimeExecutionPolicy,
											mTask, mode);

								}
							}
						}
					}
				}
				if (modeList.size() > 1)
					code += "\t}\n";
				index++;

			} catch (CICXMLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		return code;
	}
	
	private static String generateSDFGocode(String outputPath, Map<String, Task> mTask, CICScheduleTypeLoader scheduleLoader, String mode, Task task, String mRuntimeExecutionPolicy)
	{
		String code = "";
		CICScheduleType schedule;					
		
		try {				
			ArrayList<File> schedFileList = getSchedFileList(outputPath, task, mode); 
			
			// we assume that we save only the first schedule 
			schedule = scheduleLoader.loadResource(schedFileList.get(0).getAbsolutePath());
			TaskGroupsType taskGroups = schedule.getTaskGroups();
			List<TaskGroupForScheduleType> taskGroupList = taskGroups.getTaskGroup();
			for (int i = 0; i < taskGroupList.size(); i++) {
				List<ScheduleGroupType> schedGroup = taskGroupList.get(i).getScheduleGroup();
				for (int j = 0; j < schedGroup.size(); j++) {
					if (schedGroup.get(j).getLocalId().intValue() == task.getProc().get("Default")
							.get("Default").get(0)) // Need to fix
					{
						List<ScheduleElementType> scheds = schedGroup.get(j).getScheduleElement();
						if(mRuntimeExecutionPolicy.equals(HopesInterface.RuntimeExecutionPolicy_StaticAssign))
						{
//							code += "\tCIC_T_INT candidates[] = {";
//							for (int k = 0; k < scheds.size(); k++) {								
//								ScheduleElementType sched = scheds.get(k);
//								String taskName = sched.getTask().getName();
//								String taskId = "0";
//								
//								for(Task t: mTask.values())
//								{
//									if(t.getName().equals(taskName)){
//										taskId = t.getIndex();
//										break;
//									}
//								}
//								code += taskId + ", ";
//								
//							}
//							code += "};\n"
//									+ "\ttask_id = GetRunnableTaskFromProcessorId(processor_id, candidates, CIC_ARRAYLEN(candidates), mode_name);\n";							
							code += "\tswitch(task_id){\n";							
							
						}						
						
						ArrayList<String> history = new ArrayList<String>();
						for (int k = 0; k < scheds.size(); k++) {
							ScheduleElementType sched = scheds.get(k);							

							if(mRuntimeExecutionPolicy.equals(HopesInterface.RuntimeExecutionPolicy_StaticAssign))
							{
								if(!history.contains(sched.getTask().getName())){
									code += scheduleParsingWithExecutionPolicy(sched, 0, 0, mRuntimeExecutionPolicy, mTask, mode);
									history.add(sched.getTask().getName());
								}
							}
							else
							{
								code += scheduleParsingWithExecutionPolicy(sched, 0, 0, mRuntimeExecutionPolicy, mTask, mode);
							}
						}
						
						if(mRuntimeExecutionPolicy.equals(HopesInterface.RuntimeExecutionPolicy_StaticAssign))
						{
							code += "\t}\n";
						}
					}
				}
			}
		} catch (CICXMLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return code;
	}

	private static String generateGocode(String outputPath, Map<String, Task> mTask, Map<String, Task> mVTask,
			CICScheduleTypeLoader scheduleLoader, List<String> modeList, Task task, String mRuntimeExecutionPolicy) {
		String code = "";
		CICScheduleType schedule;

		code += "CIC_T_VOID " + task.getName() + "_Go(){\n";
		
		String parent_task_id = "-1";
		for(Task t: mTask.values())
		{
			if(t.getName().equals(task.getParentTask())){
				parent_task_id = t.getIndex();			
				break;
			}
		}
		for(Task vt: mVTask.values())
		{
			if(vt.getName().equals(task.getParentTask())){
				parent_task_id = vt.getIndex();			
				break;
			}
		}

		if (mRuntimeExecutionPolicy
				.equals(HopesInterface.RuntimeExecutionPolicy_FullyStatic)) 
		{
			code += "\tunsigned long diff;\n";
			code += "\tCIC_T_STRUCT timespec start, end;\n\n";
			code += "\tclock_gettime(CLOCK_MONOTONIC, &start);\n\n";
			
			sched_time = 0;
		} else if (mRuntimeExecutionPolicy.equals(HopesInterface.RuntimeExecutionPolicy_StaticAssign)) {
			String[] result = task.getName().split("_");
			String processor_id = result[result.length - 1];
			code += "\tCIC_T_INT processor_id = " + processor_id + ";\n";
			code += "\tCIC_T_INT task_id = -1;\n";
			code += "\tCIC_T_INT parent_task_id = " + parent_task_id + ";\n";
			code += "\tCIC_T_INT sched_index = GetScheduleIndexFromProcessorIdAndParentTaskId(processor_id, parent_task_id);\n";
			code += "\tCIC_T_INT i;\n";
			code += "\tCIC_T_BOOL executable = CIC_V_FALSE;\n";
			code += "\tCIC_T_INT num_task_stop = 0;\n\n";

			code += "\twhile(executable == CIC_V_FALSE){\n" 
					+ "\t\tnum_task_stop = 0;\n"
					+ "\t\tfor(i = 0; i < schedules[sched_index].max_schedule_length; i++){\n"
					+ "\t\t\tCIC_T_INT candidate_task_id = schedules[sched_index].task_execution_order[i];\n"
					+ "\t\t\tif(candidate_task_id == -1) continue;\n"
					+ "\t\t\tif(tasks[candidate_task_id].state == STATE_STOP){\n" 
					+ "\t\t\t\tnum_task_stop++;\n"
					+ "\t\t\t\tcontinue;\n" 
					+ "\t\t\t}\n"
					+ "\t\t\tif(firable_task[candidate_task_id] == CIC_V_TRUE){\n" 
					+ "\t\t\t\ttask_id = candidate_task_id;\n" 
					+ "\t\t\t\texecutable = CIC_V_TRUE;\n"
					+ "\t\t\t\tbreak;\n"
					+ "\t\t\t}\n" 
					+ "\t\t}\n" 
					+ "\t\tif(num_task_stop == schedules[sched_index].max_schedule_length)\n"
					+ "\t\t\texecutable = CIC_V_TRUE;\n" 
					+ "\t}\n\n";
		}

		if (modeList.size() > 1) // NOT support RuntimeExecutionPolicy_StaticAssign yet... 
			code += generateSADFGocode(outputPath, mTask, scheduleLoader, modeList, task, mRuntimeExecutionPolicy);
		else
			code += generateSDFGocode(outputPath, mTask, scheduleLoader,
					modeList.get(0)/* Default mode */, task, mRuntimeExecutionPolicy);

		if(mRuntimeExecutionPolicy.equals(HopesInterface.RuntimeExecutionPolicy_StaticAssign))
		{
			code += "\tif(task_id != -1)\n"
				+ "\t\tupdateRunQueueFromTaskId(task_id);\n";
		}

		code += ("}\n\n");
		return code;
	}

	private static String generateWrapupcode(String outputPath, CICScheduleTypeLoader scheduleLoader,
			List<String> modeList, Task task) {
		String code = "";
		CICScheduleType schedule;

		code += "CIC_T_VOID " + task.getName() + "_Wrapup(){\n";

		if (modeList.size() > 1) {
			code += "\tCIC_T_INT i=0;\n\tCIC_T_INT mtm_index = 0;\n\tCIC_T_CHAR* mode = 0;\n";
			code += "\tCIC_T_INT task_id = GetTaskIdFromTaskName(\"" + task.getParentTask() + "\");\n";
			code += "\tfor(i=0; i<CIC_UV_NUM_MTMS; i++){\n" + "\t\tif(task_id == mtms[i].task_id){\n"
					+ "\t\t\tmtm_index = i;\n" + "\t\t\tbreak;\n" + "\t\t}\n" + "\t}\n\n";
			code += "\tmode = mtms[mtm_index].GetCurrentModeName();\n";
		}

		int index = 0;
		for (String mode : modeList) {
			ArrayList<String> history = new ArrayList<String>();
			try {
				if (modeList.size() > 1) {
					if (index == 0) {
						code += "\tif(CIC_F_STRING_COMPARE(mode, \"" + mode + "\") == 0){\n";
					} else if (index != 0) {
						code += "\telse if(CIC_F_STRING_COMPARE(mode, \"" + mode + "\") == 0){\n";
					}
				}
				ArrayList<File> schedFileList = new ArrayList<File>();
				File file = new File(outputPath);
				File[] fileList = file.listFiles();
				for (File f : fileList) {
					if (f.getName().contains(task.getParentTask() + "_" + mode)
							&& f.getName().endsWith("_schedule.xml")) {
						schedFileList.add(f);
					}
				}
				if (schedFileList.size() <= 0) {
					JOptionPane.showMessageDialog(null, "You should execute 'Analysis' before build!");
					System.exit(-1);
				}

				for (int f_i = 0; f_i < schedFileList.size(); f_i++) {
					// we assume that we save only the first schedule 
					schedule = scheduleLoader.loadResource(schedFileList.get(0).getAbsolutePath());

					TaskGroupsType taskGroups = schedule.getTaskGroups();
					List<TaskGroupForScheduleType> taskGroupList = taskGroups.getTaskGroup();
					for (int i = 0; i < taskGroupList.size(); i++) {
						List<ScheduleGroupType> schedGroup = taskGroupList.get(i).getScheduleGroup();
						for (int j = 0; j < schedGroup.size(); j++) {
							if (schedGroup.get(j).getLocalId().intValue() == task.getProc().get("Default")
									.get("Default").get(0)) // Need to fix
							{
								List<ScheduleElementType> scheds = schedGroup.get(j).getScheduleElement();

								for (int k = 0; k < scheds.size(); k++) {
									ScheduleElementType sched = scheds.get(k);
									String taskName = sched.getTask().getName();
									if (!history.contains(taskName)) {
										code += "\t\t" + taskName + "_Wrapup();\n";
										history.add(taskName);
									}
								}
							}
						}
					}
					if (modeList.size() > 1) {
						code += "\t}\n";
					}
					index++;
				}
			} catch (CICXMLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		code += ("}\n\n");
		return code;
	}

	public static String generateMultiProcessorStaticScheduleCodeWithExecutionPolicy(String outputPath,
			Map<String, Task> mTask, Map<String, Task> mVTask, Map<String, Task> mPVTask,
			String mRuntimeExecutionPolicy) {
		String staticScheduleCode = "";
		String goCode = "";
		String initCode = "";
		String wrapupCode = "";
		CICScheduleTypeLoader scheduleLoader = new CICScheduleTypeLoader();
		for (Task task : mPVTask.values()) {
			Task parentTask = getParentTask(task, mTask, mVTask);

			List<String> modeList = new ArrayList<String>();

			if (parentTask.getMTM() != null)
				modeList = parentTask.getMTM().getModes();
			else
				modeList.add("Default");

			initCode += generateInitcode(outputPath, mTask, scheduleLoader, modeList, task, mRuntimeExecutionPolicy);
			goCode += generateGocode(outputPath, mTask, mVTask, scheduleLoader, modeList, task, mRuntimeExecutionPolicy);
			wrapupCode += generateWrapupcode(outputPath, scheduleLoader, modeList, task);

			staticScheduleCode = initCode + goCode + wrapupCode;
		}

		return staticScheduleCode;
	}

}
