package bufferOpt.internal;


import static org.opt4j.core.Objective.Sign.MIN;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

import org.opt4j.core.Objective;
import org.opt4j.core.Objectives;
import org.opt4j.core.problem.Evaluator;

import com.google.inject.Inject;


public class BufferOptEvaluator implements Evaluator<BufferSizeNProcNum> {

	// MOEA
	protected List<Objective> objectives = new ArrayList<Objective>();
	protected Objective total_buffer_size = new Objective("total_buffer_size", MIN);
	protected Objective total_proc_num = new Objective("total_proc_num", MIN);
	protected Objective total_throughput = new Objective("total_throughput", MIN);
	
	protected final BufferOptProblem problem;
	
	private int latency = -1; // FIXME: to communicate between methods, not a good approach
	
	@Inject
	public BufferOptEvaluator(BufferOptProblem problem)
	{
		objectives.add(total_throughput);
		objectives.add(total_buffer_size);
		objectives.add(total_proc_num);
		
		this.problem = problem;	
	}
	
	public Objectives evaluate(BufferSizeNProcNum phenotype)
	{
		//System.out.println("KIN: " +phenotype);
		Objectives objs = new Objectives();
		int proc_cost = phenotype.total_proc_num();
		//objs.add(total_proc_num, proc_cost);
	
		// mapping info setting
		Graph gr = problem.gr;
		for( int j = 0; j < gr.get_tasks().get_numNodes(); j++ ) 
		{
			Node curr_node = gr.get_tasks().get_nodes().get(j);
			int mapped_proc = phenotype.proc_maps()[j];
			curr_node.set_execTime(curr_node.get_profExecTime(problem.arch.getPoolIdx(mapped_proc)));
			curr_node.set_mappedProc(mapped_proc);			
		}
		
		// priority setup
		if(Param.CYCLO_STATIC)
			gr.setPriority(gr.get_tasks());
		else if(Param.GA_PRIORITY_NODE)
			gr.setPriorityGaNode(phenotype.startPriorityOrder());
		else
			gr.setPriorityGaInstance(phenotype.startPriorityOrder());
		//gr.printTotalNumOfInstance();
		
//		gr.print_sdf_graph(gr.get_tasks());
//		System.exit(1);
		
		
		// try scheduling - dynamic
		//throughput = -1;		
		int buf_size;
		if(!Param.PREEMPTIVE)
			buf_size =  bufferSum(phenotype);
		else
			buf_size = bufferSumNP(phenotype);
		//System.exit(1);
		if ((buf_size) == -1 )
		{
			objs.add(total_throughput, Integer.MAX_VALUE);
			objs.add(total_buffer_size, Integer.MAX_VALUE);	
			objs.add(total_proc_num, Integer.MAX_VALUE);
			
			objs.setFeasible(false);
		}
		else 
		{
			phenotype.set_total_buffer_size(buf_size);
			phenotype.set_total_proc_num(proc_cost);
			phenotype.set_throughput(latency/problem.gr.getRepeat());
			
			objs.add(total_throughput, latency/problem.gr.getRepeat());
			objs.add(total_buffer_size, buf_size);
			objs.add(total_proc_num, proc_cost);
			
		}			
		System.out.println("objs:" + objs);
		//System.exit(1);
		return objs;
	}
	
	public Collection<Objective> getObjectives()
	{
		return objectives;
	} 
	
	/* non-preemptive scheduling */
	private int bufferSum(BufferSizeNProcNum phenotype) {
		Graph gr = problem.gr;
		int i, j, k;
		int tmpSize;
		final BigInteger L = BigInteger.valueOf(10000000).multiply(BigInteger.valueOf(10000000));
		BigInteger cost = L;
		ProcSchedule [] proc_sched = new ProcSchedule [gr.get_procNum()];
		for( i = 0; i < gr.get_procNum(); i++ ) proc_sched[i] = new ProcSchedule();
		// variables for cost value
		int buff_area = 0;
		boolean safe = true;
		
		// initialize scheduling information
		//System.out.println("*** Iteration Start ***");		
		for( j = 0; j < gr.get_tasks().get_nodes().size(); j++ ) {
			Node curr_node = gr.get_tasks().get_nodes().get(j);
			// schedule info initialize
			for( k = 0; k < curr_node.get_instances().size(); k++ ) {
				Instance temp_instance = curr_node.get_instances().get(k);
				// init shadow node schedule info: scheduled
				temp_instance.init_schedule_info();
			}
		}
		//System.out.print("( ");
		for( j = 0; j < gr.get_tasks().get_numEdges(); j++ ) {
			tmpSize = (Integer)phenotype.buffer_sizes()[j];
			gr.get_tasks().get_edges().get(j).resetToken();
			gr.get_tasks().get_edges().get(j).set_size(tmpSize);
			//System.out.print(gr.get_tasks().get(i).get_edges().get(j).get_size()+" ");
		}
		//gr.print_sdf_graph(gr.get_tasks());
//		System.exit(1);
		//System.out.println(")");
		/****************************
		 * build initial ready node 
		 ****************************/
		// task
		for( j = 0; j < gr.get_tasks().get_numNodes(); j++ ) {
			Node curr_node = gr.get_tasks().get_nodes().get(j);
			for( k = 0; k < curr_node.get_numberOfInstance(); k++ ) {
				//				System.out.println("MERONG1: " + curr_node.get_id());
				//				System.out.println("MERONG1: " + curr_node.get_numberOfInstance());
				//				System.out.println("MERONG2: " + curr_node.get_instances().size());
				//				System.exit(1);
				Instance curr_instance = curr_node.get_instances().get(k);
				/* initially runnable? then..*/
				if(curr_instance.is_ready()) {
					//int proc_index = curr_node.get_mappedProc() - 1;
					int proc_index = curr_node.get_mappedProc();
					proc_sched[proc_index].add_to_ready_node_list(curr_instance, BigInteger.valueOf(0));
				}
			}
		}
		
		//print_schedule(proc_sched);

		/********************
		 *  list schedule 
		 *******************/
		int ready_proc_id = -1;
		
		do {
			BigInteger min_ready_time=BigInteger.valueOf(0);
			/* check if there is ready node */
			ready_proc_id = -1;
			/* for every proc sched 
			 * find out the proc which has the least ready time */
			for( i = 0; i < gr.get_procNum(); i++ ) {
				/* if there is a node */
				if( proc_sched[i].is_there_ready_node() ) {
					
					BigInteger ready_time = proc_sched[i].get_the_least_ready_time();
					/* first found */
					if( ready_proc_id == -1 ) {
						ready_proc_id = i;
						min_ready_time = ready_time;
						/* minimum time */
					} else if ( min_ready_time.compareTo(ready_time) > 0 ) {
						ready_proc_id = i;
						min_ready_time = ready_time;
					}
				}
			}
			/* If there is a ready node on a proc, schedule the ready node */
			if( ready_proc_id != -1 ) {
				
				Instance readyInstance = proc_sched[ready_proc_id].pop_ins_from_ready_list();
				//System.out.println("Ready instance : procid [ "+ready_proc_id+" ] node[" + readyInstance.get_node().get_id() + "], instance["+readyInstance.get_id()+"]");
				/* if data parallel node -> deleted */
				/* if non data parallel node */
				{
					List<Instance> awaken_list;
					//System.out.println("Firing instance["+readyInstance.get_id()+"]");
					awaken_list = proc_sched[ready_proc_id].fire_a_ins(readyInstance, min_ready_time);
					
					//System.out.println("AWAKEN_LIST SIZE: "+ awaken_list.size());
					/* makespan calculation */
					//if( proc_sched[ready_proc_id].get_end_time().compareTo(makespan) > 0 )
					//	makespan = proc_sched[ready_proc_id].get_end_time();
					/* safe check */
					float tmpDoubleValue = proc_sched[ready_proc_id].get_end_time().floatValue();
					//if(tmpDoubleValue > latency)
					//	latency = proc_sched[ready_proc_id].get_end_time().intValue(); //KJW
					//System.out.println(tmpDoubleValue+" "+readyInstance.get_node().get_sdfg().get_deadline());
					//if( proc_sched[ready_proc_id].get_end_time().compareTo(BigInteger.valueOf(readyInstance.get_node().get_sdfg().get_deadline())) > 0 ) {
					if( tmpDoubleValue > readyInstance.get_node().get_sdfg().get_deadline() ) {
						safe = false;
						//latency = Integer.MAX_VALUE;
						System.out.println("schedule violate");
					}
//					proc_sched[ready_proc_id].printReadyInstances();
//					System.out.println("ready instance'node' " + readyInstance.get_node().get_id());
//					if(awaken_list.size() == 0)
//					{
//						System.out.println("ready instance'node' " + readyInstance.get_node().get_id()+" next instance is empty");
//					}
					/* add to ready list */
					
					for(j = 0; j < awaken_list.size(); j++ ) {
						
						//int ready_proc_ind = awaken_list.get(j).get_node().get_mappedProc() - 1;
						int ready_proc_ind = awaken_list.get(j).get_node().get_mappedProc();
						//System.out.println("Awaken instance : proc[ "+ready_proc_ind+" ] node[" + awaken_list.get(j).get_node().get_id() + "], instance["+awaken_list.get(j).get_id()+"]");
						proc_sched[ready_proc_ind].add_to_ready_node_list(awaken_list.get(j), awaken_list.get(j).get_readyTime());
						//System.out.println("("+awaken_list.get(j).get_node().get_id()+","+awaken_list.get(j).get_id()+") awaken with ready time "+awaken_list.get(j).get_readyTime());
					}
					//System.out.println("----------");
				}
			} /* end of if there is a ready node */
			//print_schedule(proc_sched);
		} while( ready_proc_id != -1 );
		//System.out.println("MERONG5");
		//print_schedule(proc_sched);
		/********************************
		 *  check all nodes are scheduled 
		 ********************************/
		for( j = 0; j < gr.get_tasks().get_numNodes(); j++ ) {
			Node curr_node = gr.get_tasks().get_nodes().get(j);
			if( curr_node.is_done() == false )
			{
				safe = false;
				System.out.println("Scheduling fail at node["+curr_node.get_id()+"], last scheduling instance id[");
			}
		}
//		//KJW
		if(safe)
		{
			int temp, max=0;
			for( i = 0; i < gr.get_procNum(); i++ )
			{
				temp = proc_sched[i].get_end_time().intValue();
				//System.out.println("end_time "+i+" : "+ temp);
				if(temp > max)
					max = temp;
			}		
			latency = max;
		}
//		latency = -1;
//		for( i = 0; i < gr.get_procNum(); i++ )
//		{
//			int temp = proc_sched[i].get_end_time().intValue();
//			if(temp>latency)
//				latency = temp;
//		}

		/*
		if( !safe )
			System.out.println("*** Schedule Fail ***");
		else
			System.out.println("*** Schedule Success ***");
		 */
		/**********************
		 *  cost calculation 
		 *********************/
		/* buff area */
		for( j = 0; j < gr.get_tasks().get_numEdges(); j++ )
			/* buffer size addition */
			buff_area += gr.get_tasks().get_edges().get(j).get_size();
		/* calculate cost value */
		if( safe )
			cost = BigInteger.valueOf(buff_area);
		/* not a safe schedule */
		else
		{
			//KJW
			cost = L;
			latency = Integer.MAX_VALUE;
			return -1;
			//END
		}
		/* cost */
		//System.out.println("Cost is "+cost.intValue());
		return cost.intValue();
	}
	
	
	/* preemptive scheduling */
	private int bufferSumNP(BufferSizeNProcNum phenotype) {
		
		Graph gr = problem.gr;
		int i, j, k;
		int tmpSize;
		final BigInteger L = BigInteger.valueOf(10000000).multiply(BigInteger.valueOf(10000000));
		BigInteger cost = L;
		
		ProcScheduleNP [] proc_sched = new ProcScheduleNP [gr.get_procNum()];
		
		/* Scheduling per every processor */
		for( i = 0; i < gr.get_procNum(); i++ ) proc_sched[i] = new ProcScheduleNP();
		// variables for cost value
		int buff_area = 0;
		boolean safe = true;
		
		// initialize scheduling information
		//System.out.println("*** Iteration Start ***");		
		for( j = 0; j < gr.get_tasks().get_nodes().size(); j++ ) {
			Node curr_node = gr.get_tasks().get_nodes().get(j);
			// schedule info initialize
			for( k = 0; k < curr_node.get_instances().size(); k++ ) {
				Instance temp_instance = curr_node.get_instances().get(k);
				// init shadow node schedule info: scheduled
				temp_instance.init_schedule_info();
			}
		}
		//System.out.print("( ");
		for( j = 0; j < gr.get_tasks().get_numEdges(); j++ ) {
			tmpSize = (Integer)phenotype.buffer_sizes()[j];
			gr.get_tasks().get_edges().get(j).resetToken();
			gr.get_tasks().get_edges().get(j).set_size(tmpSize);
			//System.out.print(gr.get_tasks().get(i).get_edges().get(j).get_size()+" ");
		}
		//System.out.println(")");
		/****************************
		 * build initial ready node 
		 ****************************/
		// task
		for( j = 0; j < gr.get_tasks().get_numNodes(); j++ ) {
			Node curr_node = gr.get_tasks().get_nodes().get(j);
			for( k = 0; k < curr_node.get_numberOfInstance(); k++ ) {
				//				System.out.println("MERONG1: " + curr_node.get_id());
				//				System.out.println("MERONG1: " + curr_node.get_numberOfInstance());
				//				System.out.println("MERONG2: " + curr_node.get_instances().size());
				//				System.exit(1);
				Instance curr_instance = curr_node.get_instances().get(k);
				/* initially runnable? then..*/
				if(curr_instance.is_ready()) {
					//int proc_index = curr_node.get_mappedProc() - 1;
					int proc_index = curr_node.get_mappedProc();
					proc_sched[proc_index].add_to_ready_node_list(curr_instance, BigInteger.valueOf(0));
				}
			}
		}
		
		//print_schedule(proc_sched);	
		

		/********************
		 *  list schedule 
		 *******************/
		int ready_proc_id = -1;		
		do {
			BigInteger min_ready_time=BigInteger.valueOf(0);
			BigInteger minSafeTime=BigInteger.valueOf(0);
			int minSafeTimeOffset = 0;
			
			/* check if there is ready node */
			ready_proc_id = -1;
			/* for every proc sched 
			 * find out the proc which has the least ready time */
//			System.out.println();
			for( i = 0; i < gr.get_procNum(); i++ ) {
				/* if there is a node */
				
//				System.out.println("Proc["+i+"] ready list size["+proc_sched[i].get_ready_ins().size()+"]");
				if( proc_sched[i].is_there_ready_node() ) {
					
					BigInteger ready_time = proc_sched[i].get_the_least_ready_time();
					int winnerId = proc_sched[i].getWinnerIdFromReadyList();
					BigInteger tempSafeTime = ready_time.add(BigInteger.valueOf(proc_sched[i].get_ready_ins().get(winnerId).getRemainExecTime()));					
//					System.out.println("Proc["+i+"]'s ready node is "+proc_sched[i].get_ready_ins().get(winnerId).get_node().get_id()+" with ready time "+ready_time + " safe time " +tempSafeTime);
					/* first found */
					if( ready_proc_id == -1 ) {
						ready_proc_id = i;
						min_ready_time = ready_time;
						minSafeTime = tempSafeTime;						
						/* minimum time */					} 
					else  
					{
						if( min_ready_time.compareTo(ready_time) > 0 )
						{
							ready_proc_id = i;
							min_ready_time = ready_time;
						}
						//if( tempSafeTime < minSafeTime)
						if( minSafeTime.compareTo(tempSafeTime) > 0 )
						{
							minSafeTime = tempSafeTime;
						}
					}										
				}
			}
			
			for( j = 0; j < gr.get_procNum(); j++ ) {
				/* if there is a node */
				if( proc_sched[j].is_there_ready_node() ) {					
					
					BigInteger readyT = proc_sched[j].get_the_least_ready_time();
					if(minSafeTime.compareTo(readyT) > 0 && !readyT.equals(min_ready_time))
						minSafeTime = readyT;
				}
			}

			minSafeTimeOffset = minSafeTime.subtract(min_ready_time).intValue();			
			
			/* If there is a ready node on a proc, schedule the ready node */
			if( ready_proc_id != -1 ) {
				
				Instance readyInstance = proc_sched[ready_proc_id].pop_ins_from_ready_list();
				//System.out.println("Ready instance : node[" + readyInstance.get_node().get_id() + "], instance["+readyInstance.get_id()+"]");
				/* if data parallel node -> deleted */
				/* if non data parallel node */
				{
					List<Instance> awaken_list;
					//System.out.println("Firing Node["+readyInstance.get_node().get_id()+"]");
					awaken_list = proc_sched[ready_proc_id].fire_a_ins(readyInstance, min_ready_time, minSafeTimeOffset);
					
					//System.out.println("AWAKEN_LIST SIZE: "+ awaken_list.size());
					/* makespan calculation */
					//if( proc_sched[ready_proc_id].get_end_time().compareTo(makespan) > 0 )
					//	makespan = proc_sched[ready_proc_id].get_end_time();
					/* safe check */
					float tmpDoubleValue = proc_sched[ready_proc_id].get_end_time().floatValue();
					//if(tmpDoubleValue > latency)
					//	latency = proc_sched[ready_proc_id].get_end_time().intValue(); //KJW
					//System.out.println(tmpDoubleValue+" "+readyInstance.get_node().get_sdfg().get_deadline());
					//if( proc_sched[ready_proc_id].get_end_time().compareTo(BigInteger.valueOf(readyInstance.get_node().get_sdfg().get_deadline())) > 0 ) {
					if( tmpDoubleValue > readyInstance.get_node().get_sdfg().get_deadline() ) {
						safe = false;
						//latency = Integer.MAX_VALUE;
						//System.out.println("schedule violate");
					}
					//System.out.println("Awaken list size: " + awaken_list.size());
					/* add to ready list */
					for(j = 0; j < awaken_list.size(); j++ ) {
						//System.out.println("Awaken instance : node[" + awaken_list.get(j).get_node().get_id() + "], instance["+awaken_list.get(j).get_id()+"]");
						//int ready_proc_ind = awaken_list.get(j).get_node().get_mappedProc() - 1;
						int ready_proc_ind = awaken_list.get(j).get_node().get_mappedProc();
						proc_sched[ready_proc_ind].add_to_ready_node_list(awaken_list.get(j), awaken_list.get(j).get_readyTime());
						//System.out.println("("+awaken_list.get(j).get_node().get_id()+","+awaken_list.get(j).get_id()+") awaken with ready time "+awaken_list.get(j).get_readyTime());
						
					}
				}
			} /* end of if there is a ready node */
			//print_schedule(proc_sched);
		} while( ready_proc_id != -1 );
		//System.out.println("MERONG5");
		//print_schedule(proc_sched);
		/********************************
		 *  check all nodes are scheduled 
		 ********************************/
		for( j = 0; j < gr.get_tasks().get_numNodes(); j++ ) {
			Node curr_node = gr.get_tasks().get_nodes().get(j);
			if( curr_node.is_done() == false )
			{
				safe = false;
				//System.out.println("Scheduling fail at node["+curr_node.get_id()+"], last scheduling instance id[");
			}
		}
//		//KJW
		if(safe)
		{
			int temp, max=0;
			for( i = 0; i < gr.get_procNum(); i++ )
			{
				temp = proc_sched[i].get_end_time().intValue();
				//System.out.println("end_time "+i+" : "+ temp);
				if(temp > max)
					max = temp;
			}		
			latency = max;
		}
//		latency = -1;
//		for( i = 0; i < gr.get_procNum(); i++ )
//		{
//			int temp = proc_sched[i].get_end_time().intValue();
//			if(temp>latency)
//				latency = temp;
//		}
		
		/*
		if( !safe )
			System.out.println("*** Schedule Fail ***");
		else
			System.out.println("*** Schedule Success ***");
		 */
		/**********************
		 *  cost calculation 
		 *********************/
		/* buff area */
		for( j = 0; j < gr.get_tasks().get_numEdges(); j++ )
			/* buffer size addition */
			buff_area += gr.get_tasks().get_edges().get(j).get_size();
		/* calculate cost value */
		if( safe )
			cost = BigInteger.valueOf(buff_area);
		/* not a safe schedule */
		else
		{
			//KJW
			cost = L;
			latency = Integer.MAX_VALUE;
			return -1;
			//END
		}
		/* cost */
		//System.out.println("Cost is "+cost.intValue());
		return cost.intValue();
	}
}
