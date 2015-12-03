package bufferOpt.internal;
//import hopes.cic.xml.*;

public class Edge {	
	private Node src;
	private Node dst;
	private int init_delay;
	private int maxSize;
	private int usedSize;
	private int usedSize4Schedule;
	private int inSize;
	private int outSize;
	
	public Edge(Node s, Node d) {
		this.src = s;
		this.dst = d;
		this.init_delay = 0;
		this.maxSize = 0;
		this.usedSize = 0;
		this.usedSize4Schedule = 0;
		this.inSize = 0;
		this.outSize = 0;
	}

	public Node get_src() { return this.src; }
	public void set_src(Node n) { this.src = n; }

	public Node get_dst() { return this.dst; }
	public void set_dst(Node n) { this.dst = n; }
	
	public int get_initDelay() { return this.init_delay; }
	public void set_initDelay(int val) { this.init_delay = val; }

	public int get_size() { return this.maxSize; }
	public void set_size(int val) { this.maxSize = val; }

	public int get_usedSize() { return this.usedSize; }
	public void set_usedSize(int val) { this.usedSize = val; }

	public int get_usedSize4Schedule() { return this.usedSize4Schedule; }
	public void set_usedSize4Schedule(int val) { this.usedSize4Schedule = val; }

	public int get_inSize() { return this.inSize; }
	public void set_inSize(int val) { this.inSize = val; }
	
	public int get_outSize() { return this.outSize; }
	public void set_outSize(int val) { this.outSize = val; }

	public int get_freeSize() { return this.maxSize-this.usedSize; }
	public int get_freeSize4Schedule() { return this.maxSize-this.usedSize4Schedule; }
	public int get_minSize() { return this.inSize + this.outSize - 1; }
	
	public void consume() {
		this.usedSize = this.usedSize - this.outSize;
		//System.out.println(this.get_dst().get_id()+" consume "+this.outSize+" "+this.get_size()+" "+this.get_freeSize());
	}
	public void produce() {
		this.usedSize = this.usedSize + this.inSize;
		//System.out.println(this.get_src().get_id()+" produce "+this.inSize+" "+this.get_size()+" "+this.get_freeSize());
	}
	
	public void consume4Schedule() {
		this.usedSize4Schedule = this.usedSize4Schedule - this.outSize;
		//System.out.println(this.get_dst().get_id()+" consume "+this.outSize+" "+this.get_size()+" "+this.get_freeSize());
	}
	public void produce4Schedule() {
		this.usedSize4Schedule = this.usedSize4Schedule + this.inSize;
		//System.out.println(this.get_src().get_id()+" produce "+this.inSize+" "+this.get_size()+" "+this.get_freeSize());
	}

	
	public void resetToken() {
		this.usedSize = this.init_delay;
		this.usedSize4Schedule = this.init_delay;
	}
}
