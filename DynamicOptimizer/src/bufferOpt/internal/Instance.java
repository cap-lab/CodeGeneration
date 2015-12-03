package bufferOpt.internal;

import java.math.BigInteger;
//import hopes.cic.xml.*;

public class Instance {
	private Node node;
	private int id;
	private int priority;
	private int HighPriority;
	private BigInteger readyTime;
	private BigInteger fireTime;
	private boolean isSched;
	private boolean isDone;
	
	//KJW
	private int remainExecTime;
	private int doneTime;
	private int preemptedCount = 0;
		
	public int getPreemptedCount() {
		return preemptedCount;
	}

	public void setPreemptedCount(int preemptedCount) {
		this.preemptedCount = preemptedCount;
	}

	public int getDoneTime() {
		return doneTime;
	}

	public void setDoneTime(int doneTime) {
		this.doneTime = doneTime;
	}

	public Instance(Node n, int i) {
		this.node = n;
		this.id = i;
		this.priority = -1;
		this.HighPriority = -1;
		this.readyTime = BigInteger.valueOf(-1);
		this.fireTime = BigInteger.valueOf(-1);
		this.isSched = false;
		this.isDone = false;
	}
	
	public Node get_node() { return this.node; }
	public int get_id() { return this.id; }
	
	public int get_priority() { return this.priority; }
	public void set_priority(int val) { this.priority = val; }

	public int get_HighPriority() { return this.HighPriority; }
	public void set_HighPriority(int val) { this.HighPriority = val; }

	public BigInteger get_readyTime() { return this.readyTime; }
	public void set_readyTime(BigInteger val) { this.readyTime = val; }

	public BigInteger get_fireTime() { return this.fireTime; }
	public void set_fireTime(BigInteger val) { this.fireTime = val; }
	
	public boolean is_done() { return this.isDone; }
	public void set_done() {
		this.isDone = true;
		this.node.set_lastFiredInsID(this.id);
//		System.out.println("node["+this.node.get_id()+"], inst.["+id+"] is set done");
		if( this.id == this.node.get_numberOfInstance()-1 )
		{
//			System.out.println("this.node.get_numberOfInstance:" + this.node.get_numberOfInstance());
			this.node.set_done();
		}
		//doneTime = fireTime.intValue() + get_node().get_execTime();
	}

	public boolean is_sched() { return this.isSched; }
	public void set_sched() {
		this.isSched = true;
		this.node.set_lastSchedInsID(this.id);
	}

	
	public void init_schedule_info() {
		this.readyTime = BigInteger.valueOf(-1);
		this.fireTime = BigInteger.valueOf(-1);
		this.isSched = false;
		this.isDone = false;
		this.node.set_lastSchedInsID(-1);
		this.node.set_lastFiredInsID(-1);
		this.node.reset_done();
		//KJW
		this.remainExecTime = get_node().get_execTime();
	}
	
	public void decRemainExecTime(int time)
	{
		remainExecTime-=time;
	}
	public void incRemainExecTime(int time)
	{
		remainExecTime+=time;
	}
	public int getRemainExecTime()
	{
		return remainExecTime;
	}
	
	public boolean is_ready() {
		int i;
		boolean priorInstance, token, area;
		// Check prior instance is done
		priorInstance = true;
		if( this.id > 0 ) {
//			System.out.println("this.id:" + this.id + " , this.node.get_instances().size(): " + this.node.get_instances().size());
//			System.out.println("this.id:" + this.id + " , this.node.numofInstance: " + this.node.get_numberOfInstance());
//			for( i = 0; i < this.node.get_instances().size(); i++ ) {
//				if( this.node.get_instances().get(i).get_id() == (this.id - 1) ) {
					if( this.node.is_src() )
					{						
						//priorInstance = this.node.get_instances().get(i).isDone;
						//priorInstance = this.node.get_instances().get(i).is_done();
						priorInstance = this.node.get_instanceByID(this.id-1).is_done();
//						if(priorInstance == false)
//							System.out.println("node["+node.get_id()+"] , instance["+(this.id-1)+"] is not done");
					}
					else
					{
						//priorInstance = this.node.get_instances().get(i).isSched;
						//priorInstance = this.node.get_instances().get(i).is_sched();
						priorInstance = this.node.get_instanceByID(this.id-1).is_sched();
//						if(priorInstance == false)
//							System.out.println("node["+node.get_id()+"] , instance["+(this.id-1)+"] is not sched");
					}
					
				//}
			//}			
		}
		// Check there are enough token in incoming edge
		token = true;
		for( i = 0; i < this.node.get_in_edge().size(); i++ ) {
			if( this.node.get_in_edge().get(i).get_usedSize() < this.node.get_in_edge().get(i).get_outSize() )
			{
				//System.out.println("BABO3");
				token = false;
			}
		}
		// Check there are enough area in outgoing edge
		area =  true;
		for( i = 0; i < this.node.get_out_edge().size(); i++ ) {
			if( this.node.get_out_edge().get(i).get_freeSize() < this.node.get_out_edge().get(i).get_inSize() )
			{
				//System.out.println("BABO4");
				area = false;
			}
		}
		
		return priorInstance && token && area;
	}
	
	public boolean is_ready4Schedule() {
		int i;
		boolean priorInstance, token, area;
		// Check prior instance is done
		priorInstance = true;
		if( this.id > 0 ) {
			for( i = 0; i < this.node.get_instances().size(); i++ ) {
				if( this.node.get_instances().get(i).get_id() == this.id - 1 ) {
					priorInstance = this.node.get_instances().get(i).isDone;
					/*
					if( this.node.is_src() )
						priorInstance = this.node.get_instances().get(i).isDone;
					else
						priorInstance = this.node.get_instances().get(i).isDone;
					*/
				}
			}			
		}
		// Check there are enough token in incoming edge
		token = true;
		for( i = 0; i < this.node.get_in_edge().size(); i++ ) {
			//if( this.node.get_in_edge().get(i).get_usedSize4Schedule() < this.node.get_in_edge().get(i).get_outSize() )
			if( this.node.get_in_edge().get(i).get_usedSize() < this.node.get_in_edge().get(i).get_outSize() )
				token = false;
		}
		// Check there are enough area in outgoing edge
		area =  true;
		for( i = 0; i < this.node.get_out_edge().size(); i++ ) {
			//if( this.node.get_out_edge().get(i).get_freeSize4Schedule() < this.node.get_out_edge().get(i).get_inSize() )
			if( this.node.get_out_edge().get(i).get_freeSize() < this.node.get_out_edge().get(i).get_inSize() )
				area = false;
		}
		return priorInstance && token && area;
	}
}