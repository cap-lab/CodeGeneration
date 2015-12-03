package bufferOpt.internal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;


/* for Preemption */
public class ProcScheduleNP {
	enum schedule_policy {rm, edf};
	private LinkedList<Instance> ready_list;
	private LinkedList<Instance> scheduled_list;
	private BigInteger end_time;
	private schedule_policy policy;

	private Instance lastFiredInstance=null;
	
	public BigInteger get_end_time() { return this.end_time; }

	public boolean is_there_ready_node() {
		if(this.ready_list.size() != 0) return true;
		else return false;
	}
	
	public ProcScheduleNP() {
		this.ready_list = new LinkedList<Instance>();
		this.scheduled_list = new LinkedList<Instance>();
		this.end_time = BigInteger.valueOf(0);
		this.policy = schedule_policy.rm;
	}
	
	public void set_schedule_policy(String s) {
		if( s.compareToIgnoreCase("EDF") == 0 ) this.policy = schedule_policy.edf;
		else if( s.compareToIgnoreCase("RM") == 0 ) this.policy = schedule_policy.rm;
	}
	
	public void add_to_ready_node_list(Instance ins, BigInteger readyTime) {
		int i, index;
		BigInteger ready_time = this.end_time.max(readyTime);
		ins.set_readyTime(ready_time);
		// traverse list and add
		index = 0;
		for( i = 0; i < this.ready_list.size(); i++ ) {
			if( this.ready_list.get(i).get_readyTime().intValue() > readyTime.intValue()) 
				break;
			index++;
		}
		this.ready_list.add(index, ins);
		/* mark scheduled */
		//for( i = 0; i < ins.get_node().get_in_edge().size(); i++ )
			//ins.get_node().get_in_edge().get(i).consume4Schedule();
		//for( i = 0; i < ins.get_node().get_out_edge().size(); i++ )
			//ins.get_node().get_out_edge().get(i).produce4Schedule();
		ins.set_sched();
		//System.out.println("("+ins.get_node().get_id()+","+ins.get_id()+") is added to ready list on proc "+ins.get_node().get_mappedProc()+" at ready time "+ins.get_readyTime());
		return;
	}
	
	public BigInteger get_the_least_ready_time() {
		return this.ready_list.get(0).get_readyTime();
	}

	public int getWinnerIdFromReadyList()
	{
		int num_tied = 1;
		int i;
		BigInteger min_time=BigInteger.valueOf(0);

		for( i = 0; i < this.ready_list.size(); i++ ) {
			if( i == 0 )
				min_time = this.ready_list.get(0).get_readyTime();
			else if( min_time.equals(this.ready_list.get(i).get_readyTime()))
			{
				num_tied++;
			}
			else
				break;
		}
		
		int winner_id = 0;		
		int max_priority = -1;
		int min_hpriority = Integer.MAX_VALUE;
		
		for( i = 0; i < num_tied; i++ ) {
			int now_hpriority = this.ready_list.get(i).get_HighPriority();
			if( now_hpriority < min_hpriority )
				min_hpriority = now_hpriority;
		}
		for( i = 0; i < num_tied; i++ ) {
			if( this.policy == schedule_policy.rm ) {
				// RM: Priority Based
				if( this.ready_list.get(i).get_HighPriority() == min_hpriority ) {
					int now_priority = this.ready_list.get(i).get_priority();
					if( now_priority > max_priority ) {
						max_priority = now_priority;
						winner_id = i;
					}
				}
			}
			/* EDF is not supported yet
			else if(this.policy==schedule_policy.edf) {
				// EDF: Earliest Deadline First
				int now_to_deadline = this.ready_node_list.get(i).get_node().get_deadline().subtract(min_time).intValue();
				if(now_to_deadline < min_to_deadline) {
					winner_id = i;
					min_to_deadline = now_to_deadline;
				} else if (now_to_deadline == min_to_deadline ) {
					// tie breaking is done by priority
					if(now_priority<min_priority) {
						winner_id = i;
						min_priority = now_priority;
					}
				}
			}*/
		}
		//System.out.println("("+this.ready_list.get(winner_id).get_node().get_id()+","+this.ready_list.get(winner_id).get_id()+") is popped from ready list");
		return winner_id;
	}

	public Instance pop_ins_from_ready_list() {
		int num_tied = 1;
		int i;
		BigInteger min_time=BigInteger.valueOf(0);
		
		// get candidates
		//System.out.println("POPS ready list size["+ready_list.size()+"]");
		if(this.ready_list.size() > 1)
			for(i=0;i<this.ready_list.size();i++)
			{
//				System.out.println(this.ready_list.get(i).get_readyTime());
			}
		for( i = 0; i < this.ready_list.size(); i++ ) {
//			System.out.println(this.ready_list.get(i).get_readyTime());
			if( i == 0 )
				min_time = this.ready_list.get(0).get_readyTime();
			else if( min_time.equals(this.ready_list.get(i).get_readyTime()))
			{
//				System.out.println("MERONGONGO");
				num_tied++;
			}
			else
				break;
		}
		//num_tied = i;
		//System.out.println("TIED["+num_tied+"]");
		
		// find the final winner according to scheduling policy
		int winner_id = 0;
		//int max_priority = this.ready_list.get(0).get_priority();
		//int max_hpriority = this.ready_list.get(0).get_HighPriority();
		int max_priority = -1;
		int min_hpriority = Integer.MAX_VALUE;
		//int min_to_deadline = this.ready_list.get(0).get_node().getDeadline().subtract(min_time).intValue();
		for( i = 0; i < num_tied; i++ ) {
			int now_hpriority = this.ready_list.get(i).get_HighPriority();
			if( now_hpriority < min_hpriority )
				min_hpriority = now_hpriority;
		}
		for( i = 0; i < num_tied; i++ ) {
			if( this.policy == schedule_policy.rm ) {
				// RM: Priority Based
				if( this.ready_list.get(i).get_HighPriority() == min_hpriority ) {
					int now_priority = this.ready_list.get(i).get_priority();
					if( now_priority > max_priority ) {
						max_priority = now_priority;
						winner_id = i;
					}
				}
			}
			/* EDF is not supported yet
			else if(this.policy==schedule_policy.edf) {
				// EDF: Earliest Deadline First
				int now_to_deadline = this.ready_node_list.get(i).get_node().get_deadline().subtract(min_time).intValue();
				if(now_to_deadline < min_to_deadline) {
					winner_id = i;
					min_to_deadline = now_to_deadline;
				} else if (now_to_deadline == min_to_deadline ) {
					// tie breaking is done by priority
					if(now_priority<min_priority) {
						winner_id = i;
						min_priority = now_priority;
					}
				}
			}*/
		}
		//System.out.println("("+this.ready_list.get(winner_id).get_node().get_id()+","+this.ready_list.get(winner_id).get_id()+") is popped from ready list");
		return this.ready_list.remove(winner_id);
	}
	
	public List<Instance> fire_a_ins(Instance ready_ins, BigInteger time, int minSafeTime) {
		int i;
		//int tRand;
		List<Instance> awaken_ins;
		int candidate_ins_id;
		Instance candidate_ins;
		awaken_ins = new ArrayList<Instance>();
		
		//KJW
		boolean isPreempted=false;
		if(lastFiredInstance!=null)
		{
			//System.out.println("last inst. id["+lastFiredInstance.get_id()+"] , cur ready inst. id["+ready_ins.get_id()+"]");
			if(lastFiredInstance.get_node().get_id()!=ready_ins.get_node().get_id() && !lastFiredInstance.is_done())
			{
				lastFiredInstance.setPreemptedCount(lastFiredInstance.getPreemptedCount()+1);
				isPreempted = true;
//				System.out.println("MERONG");
			}
		}
		//System.out.println("BABO "+time.intValue());
		if(ready_ins.getRemainExecTime() == ready_ins.get_node().get_execTime()) // Start first
		{
			ready_ins.set_fireTime(time);
			lastFiredInstance = ready_ins;			
//			System.out.println("Node["+ready_ins.get_node().nodeName+"] is fired at fireTime["+ready_ins.get_fireTime()+"]");
		}
		else
		{
//			System.out.println("Node["+ready_ins.get_node().nodeName+"] is scheduled "+(ready_ins.get_node().get_execTime()-ready_ins.getRemainExecTime()+1)+" times at ["+time+"]");
		}
		ready_ins.decRemainExecTime(minSafeTime);
		//System.out.println("Remain Time"+ready_ins.getRemainExecTime());
		//System.out.println("KKK1 "+this.end_time.intValue());
		//this.end_time = time.add(BigInteger.valueOf(1));
		this.end_time = time.add(BigInteger.valueOf(minSafeTime));
		if(isPreempted)
		{
			//this.end_time = this.end_time.add(BigInteger.valueOf(Param.PREEMPTION_COST));
			ready_ins.incRemainExecTime(Param.PREEMPTION_COST);
		}
		//System.out.println("End Time"+this.end_time.intValue());
//		System.out.println("Remain Time"+ready_ins.getRemainExecTime());
		if(ready_ins.getRemainExecTime() == 0)
		{
			ready_ins.set_done();
			ready_ins.setDoneTime(this.end_time.intValue());
			this.scheduled_list.add(ready_ins);
//			System.out.println("Node["+ready_ins.get_node().nodeName+"] is done at["+ready_ins.getDoneTime()+"]");
		}
		else
		{
			//add_to_ready_node_list(ready_ins, this.end_time);
			ready_ins.set_readyTime(this.end_time);
			awaken_ins.add(ready_ins);
//			System.out.println("Node["+ready_ins.get_node().nodeName+"] is awaken as readyTime["+ready_ins.get_readyTime()+"]");
		}
		//tRand = (int)(Math.random() * 21) + 90; 
		//this.end_time = time.add(BigInteger.valueOf(ready_ins.get_node().get_execTime()));
		
		
		//this.end_time = time.add(BigInteger.valueOf((int)(ready_ins.get_node().get_execTime()*tRand/100)));
		//System.out.println("("+ready_ins.get_node().get_id()+","+ready_ins.get_id()+") is fired at "+ready_ins.get_fireTime()+" on proc "+ready_ins.get_node().get_mappedProc());
		//KJW
		if(ready_ins.is_done())
		{
			// consume input edges' tokens
			for( i = 0; i < ready_ins.get_node().get_in_edge().size(); i++ ) {
				Edge temp_edge = ready_ins.get_node().get_in_edge().get(i);
				temp_edge.consume();
				temp_edge.consume4Schedule();
				candidate_ins_id = temp_edge.get_src().get_lastSchedInsID() + 1;
				if( candidate_ins_id < temp_edge.get_src().get_numberOfInstance() ) {
					candidate_ins = temp_edge.get_src().get_instanceByID(candidate_ins_id);
					//System.out.println("("+candidate_ins.get_node().get_id()+","+candidate_ins.get_id()+") is candidate");
					if( candidate_ins.is_ready4Schedule() ) {
						//if( candidate_ins.is_ready() ) {
						// compute comm. overhead
						//int comm_overhead = temp_edge.get_comm_overhead();
						// new ready node is made
						//System.out.println("("+candidate_ins.get_node().get_id()+","+candidate_ins.get_id()+") is awaked with ready time "+this.end_time);
						candidate_ins.set_readyTime(this.end_time);
						candidate_ins.set_sched();
						temp_edge.produce4Schedule();
						awaken_ins.add(candidate_ins);				
					}
				}
			}
			// produce tokens on output edges, and find awaken nodes
			for( i = 0; i < ready_ins.get_node().get_out_edge().size(); i++ ) {
				Edge temp_edge = ready_ins.get_node().get_out_edge().get(i);
				// produce a token
				temp_edge.produce();
				temp_edge.produce4Schedule();
				// if awaken then
				candidate_ins_id = temp_edge.get_dst().get_lastSchedInsID() + 1;
				if( candidate_ins_id < temp_edge.get_dst().get_numberOfInstance() ) {
					candidate_ins = temp_edge.get_dst().get_instanceByID(candidate_ins_id);
					//System.out.println("("+candidate_ins.get_node().get_id()+","+candidate_ins.get_id()+") is candidate");
					if( candidate_ins.is_ready4Schedule() ) {
					//if( (candidate_ins.is_ready()) && (candidate_ins.is_sched() == false) ) {
						// compute comm. overhead
						//int comm_overhead = temp_edge.get_comm_overhead();
						// new ready node is made
						//System.out.println("("+candidate_ins.get_node().get_id()+","+candidate_ins.get_id()+") is awaked with ready time "+this.end_time);
						candidate_ins.set_readyTime(this.end_time);
						candidate_ins.set_sched();
						temp_edge.consume4Schedule();
						awaken_ins.add(candidate_ins);				
					}
				}
			}
			// if next instance in same node is ready, awake it.
			candidate_ins_id = ready_ins.get_id() + 1;
			//System.out.println("numberofInstance: "+ready_ins.get_node().get_numberOfInstance());
			if( ready_ins.get_node().get_numberOfInstance() > ready_ins.get_id() + 1 ) {
				candidate_ins = ready_ins.get_node().get_instanceByID(candidate_ins_id);
				//if( candidate_ins.is_ready4Schedule() ) {
				//System.out.println("Candidate instacne id : " + candidate_ins.get_id());
//				if( candidate_ins.is_ready())
//				{
//					System.out.println("Checking node["+candidate_ins.get_node().get_id()+"], instance["+candidate_ins.get_id()+"] is done?");				
//				}			
//				if((candidate_ins.is_sched() == false))
//					System.out.println("MERONG2");
				if( (candidate_ins.is_ready()) && (candidate_ins.is_sched() == false)) {
					//System.out.println("("+candidate_ins.get_node().get_id()+","+candidate_ins.get_id()+") is awaked with ready time "+this.end_time);
					
					candidate_ins.set_readyTime(this.end_time);
					candidate_ins.set_sched();
					awaken_ins.add(candidate_ins);
				}
			}
		} // end of if(ready_ins.is_done())
		
		
		// update ready list time
		for( i = 0; i < this.ready_list.size(); i++ ) {
			// if end time is bigger than ready time 
			if( this.end_time.subtract(this.ready_list.get(i).get_readyTime()).intValue() > 0)
				this.ready_list.get(i).set_readyTime(this.end_time);
			else
				break;
		}
//		System.out.println("Awakenins size: " + awaken_ins.size());
		// return the awaken node list
		return awaken_ins;
	}
		
	public LinkedList<Instance> get_sched_ins() { return this.scheduled_list; }
	public LinkedList<Instance> get_ready_ins() { return this.ready_list; }
}
