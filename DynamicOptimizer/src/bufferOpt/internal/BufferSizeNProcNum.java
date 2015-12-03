package bufferOpt.internal;

import model.app.SDFGraph;
import model.architecture.GenericArch;

import org.opt4j.core.problem.Phenotype;
import org.opt4j.genotype.PermutationGenotype;

public class BufferSizeNProcNum implements Phenotype{
	int total_buf_size=0;
	int total_proc_num=0;
	int throughput=0;
	
	int [] proc_maps;
	int [] buffer_sizes;
	int [] startPriorityOrder;
	int [] priorityInc;
	
	SDFGraph graph;
	GenericArch arch;
	
	public int total_buf_size() { return total_buf_size; }
	public int total_proc_num() { return total_proc_num; }
	public int throughput() { return throughput; }
	
	public void set_total_buffer_size(int size) { total_buf_size = size; }
	public void set_total_proc_num(int val)  { total_proc_num = val; }
	public void set_throughput(int val) {  throughput = val; }
	
	public int [] proc_maps()  { return proc_maps; }
	public int [] buffer_sizes() { return buffer_sizes; }
	public int [] startPriorityOrder() { return startPriorityOrder; }
	public int [] priorityInc() { return priorityInc; }
	
	public BufferSizeNProcNum(SDFGraph graph, GenericArch arch, int [] proc_maps, int[] buffer_sizes, int total_proc_num , 
			int total_buf_size, int [] startPriorityOrder/*, int [] priorityInc*/)
	{	
		this.graph = graph;
		this.arch = arch;
		this.proc_maps = new int [ proc_maps.length ];
		this.buffer_sizes = new int [ buffer_sizes.length ] ;
		this.startPriorityOrder = new int [ startPriorityOrder.length ] ;
//		this.priorityInc = new int [ priorityInc.length ] ;

		System.arraycopy(proc_maps, 0, this.proc_maps, 0, proc_maps.length);
		System.arraycopy(buffer_sizes, 0, this.buffer_sizes, 0, buffer_sizes.length);
		System.arraycopy(startPriorityOrder, 0, this.startPriorityOrder, 0, startPriorityOrder.length);
//		System.arraycopy(priorityInc, 0, this.priorityInc, 0, priorityInc.length);
		
		this.total_proc_num  = total_proc_num;
		this.total_buf_size = total_buf_size;
		//System.exit(0);
	}
	
	public String getMappingStr(){
		String re = "";
		int appIdx = 0;
		int numNode = graph.actorName.get(appIdx).size();
		
		
		re += "THROUGHPUT[1/";
		re += throughput();
		re += "] ";
		
		re += "BSIZE[";
		for(int i=0;i<buffer_sizes.length;i++)
		{
			re += "" + buffer_sizes[i] + " ";
		}
		re += "]\r\n";
				
		//re += "T_P : " + total_proc_num + "\n";
		
	
		//re += "T_B: " + total_buf_size + "\n";
		for (int i=0 ; i<proc_maps.length ; i++){
			
			String actorName = graph.actorName.get(appIdx).get(i);
			String priority = String.valueOf(startPriorityOrder[i]);
			String priorityOffset = String.valueOf(startPriorityOrder[i+numNode]);
			String mappedPool = arch.PoolName.get(arch.getPoolIdx(proc_maps[i]));
			String mappedProc = String.valueOf(arch.getLocalId(proc_maps[i]));
			
			
			re += actorName+"->"+mappedPool+"("+mappedProc+") with prioity ("+priority+","+priorityOffset+")\r\n";
			
		}
		System.out.println(toString());
		
	
		return re;
	}
	
	
	public String toString()
	{
		String re = "";
		//re += "T_P : " + total_proc_num + "\n";
		//re += "T_B: " + total_buf_size + "\n";
		re += "ProcMAP[";
		for(int i=0;i<proc_maps.length;i++)
		{
			re += ""+(proc_maps[i]) + " ";
		}
		//re += "\n ";
		re += "] BSIZE[";
		for(int i=0;i<buffer_sizes.length;i++)
		{
			re += "" + buffer_sizes[i] + " ";
		}
//		re += "\n";
		re += "] PrioORDER[";
		for(int i=0;i<startPriorityOrder.length;i++)
		{
			re += "" + startPriorityOrder[i] + " ";
		}
		re += "]";
		return re;
	}
}
