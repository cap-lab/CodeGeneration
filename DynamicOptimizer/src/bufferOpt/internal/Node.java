package bufferOpt.internal;
//import hopes.cic.xml.*;

import java.util.List;
import java.util.ArrayList;

public class Node {
	private SdfGraph sdfg;
	private int id;
	private int mappedProc;
	private int execTime;
	private int profExecTime[];
	private int numberOfInstance;
	private int initNumberOfInstance;
	private int lastSchedInsID;
	private int lastFiredInsID;
	private List<Instance> instance;
	private List<Edge> in;
	private List<Edge> out;
	private boolean isDone;
	String nodeName;
	
	private int startPriority = 0;
	
	boolean checkFlagForCalPriority = false; // KJW
	
	public int getStartPriority() {
		return startPriority;
	}

	public void setStartPriority(int startPriority) {
		this.startPriority = startPriority;
	}

	public Node(SdfGraph t, int i) {
		this.sdfg = t;
		this.id = i;
		this.mappedProc = -1;
		this.execTime = 0;
		this.profExecTime = new int [t.get_baseGr().get_poolNum()];
		this.numberOfInstance = 0;
		this.lastSchedInsID = -1;
		this.lastFiredInsID = -1;
		this.instance = new ArrayList<Instance>();
		this.in = new ArrayList<Edge>();
		this.out = new ArrayList<Edge>();
		this.isDone = false;
	}
	
	public SdfGraph get_sdfg() { return this.sdfg; }
	public int get_id() { return this.id; }
	
	public int get_profExecTime(int index) {
		if( index > this.sdfg.get_baseGr().get_poolNum() ) {
			System.out.println("Invalid pool index in profile");
			return -1;
		}
		else
			return this.profExecTime[index];
	}
	public void set_profExecTimemappedProc(int index, int val) {
		if( index > this.sdfg.get_baseGr().get_poolNum() )
			System.out.println("Invalid pool index in profile");
		else
			this.profExecTime[index] = val;
	}

	public int get_mappedProc() { return this.mappedProc; }
	public void set_mappedProc(int val) { this.mappedProc = val; }

	public int get_execTime() { return this.execTime; }
	public void set_execTime(int val) {	this.execTime = val; }

	public int get_numberOfInstance() { return this.numberOfInstance; }
	public void set_numberOfInstance(int val) {	this.numberOfInstance = val; }

	public int get_initNumberOfInstance() { return this.initNumberOfInstance; }
	public void set_initNumberOfInstance(int val) {	this.initNumberOfInstance = val; }

	public int get_lastSchedInsID() { return this.lastSchedInsID; }
	public void set_lastSchedInsID(int val) {	this.lastSchedInsID = val; }

	public int get_lastFiredInsID() { return this.lastFiredInsID; }
	public void set_lastFiredInsID(int val) {	this.lastFiredInsID = val; }

	public List<Instance> get_instances() { return this.instance; }
	public void add_instance(int id) {
		Instance ins = new Instance(this, id);
		this.instance.add(ins);
	}
	
	public void add_in_edge(Edge e) { this.in.add(e); }
	public void add_out_edge(Edge e) { this.out.add(e); }

	public List<Edge> get_in_edge() { return this.in; }
	public List<Edge> get_out_edge() { return this.out; }

	public boolean is_done() { return this.isDone; }
	public void set_done() { this.isDone = true; }
	public void reset_done() { this.isDone = false; }

	public Instance get_instanceByID(int id) {
		int i;
		
		for( i = 0; i < this.numberOfInstance; i++ ) {
			if( this.instance.get(i).get_id() == id )
				break;
		}
		//System.out.println("cannot found ID"+id);
		return this.instance.get(i);
	}
	
	public boolean is_src() {
		if( this.in.size() > 0 ) return false;
		else return true;
	}
}