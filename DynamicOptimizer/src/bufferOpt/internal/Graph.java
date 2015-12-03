package bufferOpt.internal;

//import hopes.cic.xml.*;
import java.io.BufferedReader;
import java.io.FileReader;
//import java.math.BigInteger;


import model.app.SDFGraph;
import model.architecture.GenericArch;

public class Graph {
	private SdfGraph tasks;
	private int total_num_of_nodes;
	private int total_num_of_edges;
	private int[][] commdelay;
	private int numProc;
	private int numPools;
	SDFGraph rawGraph;
	GenericArch arch;

	public Graph(String sdffilename, String archfilename) throws Exception {
		int j, k, l;
		BufferedReader reader;

		rawGraph = new SDFGraph();
		rawGraph.readSDF3Format(sdffilename);

		arch = new GenericArch();
		arch.readFromCICArchitecture(archfilename);
		
		this.total_num_of_nodes = 0;
		this.total_num_of_edges = 0;
		String line = null;
		// make and add tasks from file
		//boolean periodic;
		float period = 1;
		float deadline = -1;
		int repeat = 1;
		int priority = -1;
		int numn = 0;
		int nume = 0;
		int tmp;
		//periodic = true;
		//TODO : is it ok to fix only one application.
		int appIdx = 0;

		period = rawGraph.sdfPeriod.get(appIdx);
		deadline = rawGraph.sdfDeadline.get(appIdx);

		priority =rawGraph.sdfPrio.get(appIdx);
		numProc = arch.numProcs;
		numPools = arch.numPools;
		this.commdelay = new int [this.numProc+1][this.numProc+1];

		numn = rawGraph.actorName.get(appIdx).size();
		nume = rawGraph.chanName.get(appIdx).size();

		// add task to list
		this.tasks = new SdfGraph(this, period, deadline, repeat, priority, numn, nume, true);
		// make sdf graph from file
		int left, right;
		int arg1, arg2, arg3;
		int nodeIndex, edgeIndex;
		String substr, token[];

		SdfGraph curr_task = this.tasks;

		// make nodes and edges
		nodeIndex = 1;
		edgeIndex = 0;
		for( j = 0; j < curr_task.get_numNodes(); j++ ) {
			curr_task.add_node(new Node(curr_task, nodeIndex++));
			this.total_num_of_nodes++;
		}
		nodeIndex--;
		for( j = 0; j < curr_task.get_numEdges(); j++ )
			curr_task.add_edge(new Edge(null, null));
		// set information except priority

		//produce/consume rate, and initial delay.
		for (int chanIdx = 0 ; chanIdx < nume ; chanIdx++){

			int srcIdx = rawGraph.chanSrc.get(appIdx).get(chanIdx);
			int dstIdx = rawGraph.chanDst.get(appIdx).get(chanIdx);
			int srcPortIdx = rawGraph.chanSrcPort.get(appIdx).get(chanIdx);
			int dstPortIdx = rawGraph.chanDstPort.get(appIdx).get(chanIdx);
			int initDelay = rawGraph.chanInitDelay.get(appIdx).get(chanIdx);
			int srcRate = rawGraph.portRate.get(appIdx).get(srcIdx).get(srcPortIdx);
			int dstRate = rawGraph.portRate.get(appIdx).get(dstIdx).get(dstPortIdx);

			curr_task.get_edges().get(chanIdx).set_src(curr_task.get_nodes().get(srcIdx));
			curr_task.get_nodes().get(srcIdx).add_out_edge(curr_task.get_edges().get(chanIdx));

			curr_task.get_edges().get(chanIdx).set_dst(curr_task.get_nodes().get(dstIdx));
			curr_task.get_nodes().get(dstIdx).add_in_edge(curr_task.get_edges().get(chanIdx));

			curr_task.get_edges().get(chanIdx).set_inSize(srcRate);
			curr_task.get_edges().get(chanIdx).set_outSize(dstRate);
			curr_task.get_edges().get(chanIdx).set_initDelay(initDelay);

			this.total_num_of_edges++;
		}

		for (int nodeIdx = 0 ; nodeIdx < numn ; nodeIdx++){

			int tmpNumIns = rawGraph.actorRunRate.get(appIdx).get(nodeIdx);
			curr_task.get_nodes().get(nodeIdx).set_initNumberOfInstance(tmpNumIns);
			curr_task.get_nodes().get(nodeIdx).set_numberOfInstance(tmpNumIns * curr_task.get_repeat());
			//for( l = 0; l < tmpNumIns; l++ ) {
			for( l = 0; l < curr_task.get_nodes().get(nodeIdx).get_numberOfInstance(); l++ ) {
				curr_task.get_nodes().get(nodeIdx).add_instance(l);
				curr_task.get_nodes().get(nodeIdx).get_instanceByID(l).set_HighPriority(0);
			}
		}

		for (int procAIdx = 0 ; procAIdx < numProc ; procAIdx++){
			for (int procBIdx = 0 ; procBIdx < numProc ; procBIdx++){
				this.commdelay[procAIdx][procBIdx] = 0; 
			}
		}

		
		for (int nodeIdx = 0; nodeIdx < rawGraph.actorName.get(appIdx).size() ; nodeIdx++){

			for (int procIdx = 0 ; procIdx < numProc ; procIdx++){
				
				String proc_type = arch.PoolType.get(arch.getPoolIdx(procIdx));
				int time = rawGraph.execTime.get(appIdx).get(nodeIdx).get(proc_type);
				curr_task.get_nodes().get(nodeIdx).set_profExecTimemappedProc(arch.getPoolIdx(procIdx), time);	
			}
		}



		//profiling information



		this.print_sdf_graph(curr_task);
	}

	public void setPriorityGaNode(int [] startPriorityOrder)
	{
		/*
		 *   node1                       nodeN           node1                     nodeN          
		 *   *********************************           ###############################
		 *    last instance's priority                   offset value for next instance's prioity 
		 *    random int btw. node's 1 ~ exec.time       random int btw. 1 ~ exec.time
		 */
		int node_num = tasks.get_nodes().size();

		if(node_num *2 != startPriorityOrder.length)
		{
			System.err.println("ERORR node_num *2 != startPriorityOrder.length");
			System.exit(1);
		}


		for(int i = 0 ; i < node_num;i++)
		{		
			Node n = tasks.get_nodes().get(i);

			int instanceID = n.get_initNumberOfInstance()-1;
			int index =0;
			while(instanceID >= 0)
			{
				Instance ins = n.get_instanceByID(instanceID);
				int offset = startPriorityOrder[i+node_num]*index;
				ins.set_priority(startPriorityOrder[i] + offset);
				instanceID --;
				index++;
			}
		}

		if( tasks.get_repeat() > 1 ) {
			for( int i = 0; i < tasks.get_numNodes(); i++ ) {
				int currentInstanceCount = tasks.get_nodes().get(i).get_initNumberOfInstance();
				for( int j = 1; j < tasks.get_repeat(); j++ ) {
					for( int k = 0; k < tasks.get_nodes().get(i).get_initNumberOfInstance(); k++ ) {
						//task.get_nodes().get(i).add_instance(currentInstanceNumber);	
						int priority = tasks.get_nodes().get(i).get_instanceByID(k).get_priority();
						tasks.get_nodes().get(i).get_instanceByID(currentInstanceCount).set_HighPriority(j); // next repeatition
						tasks.get_nodes().get(i).get_instanceByID(currentInstanceCount).set_priority(priority);
						currentInstanceCount++;
					}
				}
				//task.get_nodes().get(i).set_numberOfInstance(task.get_nodes().get(i).get_initNumberOfInstance() * task.get_repeat());
			}
		}


		//		int repetitionCountSum = 0;
		//		for(int i=0;i<tasks.get_numNodes();i++)
		//		{
		//			Node n = tasks.get_nodes().get(i);
		//			repetitionCountSum += n.get_initNumberOfInstance();
		//		}
		////		for(int i=0;i<startPriorityOrder.length;i++)
		////		{
		////			System.out.print(startPriorityOrder[i] + " ");
		////		}
		////		System.out.println();
		//		
		//		
		//		for(int i=0;i<startPriorityOrder.length;i++)
		//		{			
		//			for(int j = 0 ; j< tasks.get_numNodes();j++)
		//			{
		//				Node n = tasks.get_nodes().get(j);	
		//				if(startPriorityOrder[i] == n.get_id() )
		//				{
		////					System.out.println("Node["+n.get_id()+"]'s initNumberOfInstacne["+n.get_initNumberOfInstance()+"]");
		//					n.setStartPriority(repetitionCountSum*n.get_execTime());
		////					n.setStartPriority(i);
		//					repetitionCountSum -= n.get_initNumberOfInstance();
		//					break;
		//				}
		//			}
		//		}
		////		for(int i=0;i<tasks.get_numNodes();i++)
		////		{
		////			System.out.println(tasks.get_nodes().get(i).getStartPriority());
		////		}
		//////		System.exit(1);
		//		for(int i = 0 ; i < tasks.get_numNodes(); i++)
		//		{
		//			Node n = tasks.get_nodes().get(i);
		//			int p=0;
		//			int hp=-1;
		//			for(int j = 0 ; j < n.get_numberOfInstance(); j++ )
		//			{
		//				if((j % n.get_initNumberOfInstance()) == 0)
		//				{
		//					p =  n.getStartPriority();
		//					hp++;
		//				}
		//				else
		//				{
		////					p--;
		//					p -= n.get_execTime();
		////					p++;
		//				}
		//				Instance ins = n.get_instanceByID(j);
		//				ins.set_priority(p);				
		//				ins.set_HighPriority(hp);
		//			}
		//		}
	}

	public void setPriorityGaInstance(int [] startPriorityOrder)
	{		
		int node_num = tasks.get_nodes().size();
		int totalInstanceCount=0;
		for(int i=0;i<node_num;i++)
		{
			totalInstanceCount+=tasks.get_nodes().get(i).get_initNumberOfInstance();
		}	

		//System.out.println("Total instance ["+totalInstanceCount+"] lengthPriroityGene["+startPriorityOrder.length+"]");

		int index= 0;
		for(int i = 0 ; i < node_num;i++)
		{		
			Node n = tasks.get_nodes().get(i);


			for(int j = 0 ; j < n.get_initNumberOfInstance(); j++)
			{
				int priority = startPriorityOrder[index++];
				Instance ins = n.get_instanceByID(j);
				ins.set_priority(priority);
			}			
		}

		if( tasks.get_repeat() > 1 ) {
			for( int i = 0; i < tasks.get_numNodes(); i++ ) {
				int currentInstanceCount = tasks.get_nodes().get(i).get_initNumberOfInstance();
				for( int j = 1; j < tasks.get_repeat(); j++ ) {
					for( int k = 0; k < tasks.get_nodes().get(i).get_initNumberOfInstance(); k++ ) {
						//task.get_nodes().get(i).add_instance(currentInstanceNumber);	
						int priority = tasks.get_nodes().get(i).get_instanceByID(k).get_priority();
						tasks.get_nodes().get(i).get_instanceByID(currentInstanceCount).set_HighPriority(j); // next repeatition
						tasks.get_nodes().get(i).get_instanceByID(currentInstanceCount).set_priority(priority);
						currentInstanceCount++;
					}
				}
				//task.get_nodes().get(i).set_numberOfInstance(task.get_nodes().get(i).get_initNumberOfInstance() * task.get_repeat());
			}
		}

	}

	public void setPriority(SdfGraph task) {
		// calculate and set priority		
		int i, j, k;
		for( i = 0; i < task.get_numNodes(); i++ ) {
			task.get_nodes().get(i).checkFlagForCalPriority = false;
		}
		for( i = 0; i < task.get_numNodes(); i++ ) {
			if( task.get_nodes().get(i).is_src() ) {
				task.get_nodes().get(i).get_instanceByID(0).set_priority(this.calculatePriority(task.get_nodes().get(i).get_instanceByID(0)));
			}
		}
		if( task.get_repeat() > 1 ) {
			for( i = 0; i < task.get_numNodes(); i++ ) {
				int currentInstanceNumber = task.get_nodes().get(i).get_initNumberOfInstance();
				for( j = 1; j < task.get_repeat(); j++ ) {
					for( k = 0; k < task.get_nodes().get(i).get_initNumberOfInstance(); k++ ) {
						//task.get_nodes().get(i).add_instance(currentInstanceNumber);
						task.get_nodes().get(i).get_instanceByID(currentInstanceNumber);
						task.get_nodes().get(i).get_instanceByID(currentInstanceNumber).set_HighPriority(j);
						task.get_nodes().get(i).get_instanceByID(currentInstanceNumber).set_priority(task.get_nodes().get(i).get_instanceByID(k).get_priority());
						currentInstanceNumber++;
					}
				}
				task.get_nodes().get(i).set_numberOfInstance(task.get_nodes().get(i).get_initNumberOfInstance() * task.get_repeat());
			}
		}
	}

	public int calculatePriority(Instance ins) {
		int i;
		int tmpPrior, tmpPrior2;
		if( ins.get_priority() == -1 ) {
			if( ins.get_id() < ins.get_node().get_initNumberOfInstance() - 1 ) // instance is not final
				/* tmpPrior = next instance priority + execTime */
				tmpPrior = this.calculatePriority(ins.get_node().get_instanceByID(ins.get_id() + 1)) + ins.get_node().get_execTime(); 
			else { // instance is final				
				tmpPrior = 0;

				for( i = 0; i < ins.get_node().get_out_edge().size(); i++ ) { // instance 의 node 의 outgoing edge iteration

					// next node 의 instance (0) 의 priority 를 tmpPrio2
					Edge outEdge = ins.get_node().get_out_edge().get(i);					
					Node dstNode = outEdge.get_dst();
					if(dstNode.checkFlagForCalPriority)
						continue;
					dstNode.checkFlagForCalPriority = true;
					//					System.out.println("my node id : " + ins.get_node().get_id());
					//					System.out.println("out edge connnected node id : " + ins.get_node().get_out_edge().get(i).get_dst().get_id());
					//					System.out.println("out edge init delay : " + outEdge.get_initDelay());
					tmpPrior2 = this.calculatePriority(dstNode.get_instanceByID(0));

					if( tmpPrior < tmpPrior2 )
						tmpPrior = tmpPrior2;
				}
				tmpPrior += ins.get_node().get_execTime();
			}
		}
		else
			tmpPrior = ins.get_priority();
		ins.set_priority(tmpPrior);
		return tmpPrior;
	}

	public SdfGraph get_tasks() { return this.tasks; }

	public int get_nodeNum() { return this.total_num_of_nodes; }
	public int get_edgeNum() { return this.total_num_of_edges; }
	public int get_procNum() { return this.numProc; }
	public int get_poolNum() { return this.numPools; }
	public int get_commDelay(int from, int to) { return this.commdelay[from][to]; }


	public void printTotalNumOfInstance()
	{
		int sum=0;
		int i,j;
		for( i = 0; i < tasks.get_nodes().size(); i++ ) {			
			sum += tasks.get_nodes().get(i).get_numberOfInstance();				
		}
		System.out.println("TotalSumOfInstance["+sum+"]");
	}
	public void print_sdf_graph( SdfGraph t ) {
		int i, j;
		System.out.println("### Nodes ###");
		for( i = 0; i < t.get_nodes().size(); i++ ) {
			System.out.println("#"+i+" NODE_ID: "+t.get_nodes().get(i).get_id()+" (numOfInstacne["
					+t.get_nodes().get(i).get_numberOfInstance()+"], mapping["
					//+t.get_nodes().get(i).get_profExecTime(0)+","
					//+t.get_nodes().get(i).get_profExecTime(1)+","
					//+t.get_nodes().get(i).get_profExecTime(2)+")");
					+t.get_nodes().get(i).get_mappedProc()+"], execTime["
					+t.get_nodes().get(i).get_execTime()+"])");
			System.out.print("HighPriority( ");
			for( j = 0; j < t.get_nodes().get(i).get_numberOfInstance(); j++ ) {
				System.out.print(t.get_nodes().get(i).get_instanceByID(j).get_HighPriority()+" ");
			}
			System.out.println(")");
			System.out.print("Priroty( ");
			for( j = 0; j < t.get_nodes().get(i).get_numberOfInstance(); j++ ) {
				System.out.print(t.get_nodes().get(i).get_instanceByID(j).get_priority()+" ");
			}
			System.out.println(")");
		}
		System.out.println("### edges ###");
		for( i = 0; i < t.get_edges().size(); i++ )
			System.out.println(t.get_edges().get(i).get_src().get_id()+" -> "
					+ t.get_edges().get(i).get_dst().get_id()+": size "
					+ t.get_edges().get(i).get_size()+" ("
					+ t.get_edges().get(i).get_initDelay()+")");
		return;
	}

	public int getRepeat() {
		return this.tasks.get_repeat();
	}
	
	
}