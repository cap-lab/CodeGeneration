package bufferOpt.internal;

//import hopes.cic.xml.SubtaskType;
//import hopes.cic.xml.TaskType;

import java.util.ArrayList;
import java.util.List;

public class SdfGraph {
	private Graph baseGr;
	private int numNodes;
	private List<Node> node;
	private int numEdges;
	private List<Edge> edge;
	private float period;
	private float deadline;
	private int repeat;
	private boolean periodic;
	private int priority;
	private List<Node> src_nodes;
	
	public SdfGraph(Graph gr, float p, float d, int r, int prior, int nodeNum, int edgeNum, boolean per) {
		baseGr = gr;
		period = p;
		deadline = d * r;
		repeat = r;
		periodic = per;
		priority = prior;
		numNodes = nodeNum;
		node = new ArrayList<Node>();
		numEdges = edgeNum;
		edge = new ArrayList<Edge>();
		src_nodes = new ArrayList<Node>();
	}

	public int get_priority() { return this.priority; }
	public Graph get_baseGr() { return this.baseGr; }
	public int get_repeat() { return this.repeat; }
	public boolean is_periodc() { return this.periodic; }
	
	public List<Node> get_nodes() { return this.node; }
	public List<Node> get_src_nodes() { return this.src_nodes; }
	public void add_node(Node n) {
		this.node.add(n);
		if( n.is_src() ) this.src_nodes.add(n);
	}
	
	public List<Edge> get_edges() { return this.edge; }
	public void add_edge(Edge e) { this.edge.add(e); }

	public int get_numNodes() { return this.numNodes; }
	//public void set_numNodes(int val) { this.numNodes = val; }

	public int get_numEdges() { return this.numEdges; }
	public void set_numEdges(int val) { this.numEdges = val; }
	
	public float get_period() { return this.period; }
	public float get_deadline() { return this.deadline; }
}