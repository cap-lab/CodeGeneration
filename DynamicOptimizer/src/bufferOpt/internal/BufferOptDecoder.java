package bufferOpt.internal;

import org.opt4j.core.problem.Decoder;
import org.opt4j.genotype.IntegerGenotype;
import org.opt4j.genotype.PermutationGenotype;

import com.google.inject.Inject;

public class BufferOptDecoder 
	implements Decoder<BufferOptGenotype, BufferSizeNProcNum >
{
	protected final BufferOptProblem problem;
	
	@Inject
	public BufferOptDecoder(BufferOptProblem problem)
	{
		this.problem = problem;
	}
	
	public BufferSizeNProcNum decode(BufferOptGenotype genotype)
	{
		int num_node = problem.gr.get_nodeNum();
		int num_proc = problem.gr.get_procNum();
		int num_edge = problem.gr.get_edgeNum();
		
		IntegerGenotype bufferSizeGene = genotype.getBufferSizeGene();
		IntegerGenotype procMappingGene = genotype.getProcMappingGene();
		//PermutationGenotype<Integer> priorityGene = genotype.getPriorityGene();
		IntegerGenotype priorityGene = genotype.getPriorityGene();
		//IntegerGenotype priorityIncGene = genotype.getPriorityIncGene();
		
		int [] proc_maps = new int[num_node];
		int [] buffer_sizes = new int[num_edge];
		
		int [] used_proc = new int[num_proc];
		int proc_id;
		for(int i=0; i< num_node;i++)
		{
			proc_id = procMappingGene.get(i).intValue();
			used_proc[ proc_id ]++;			
			proc_maps[i] = proc_id ;
		}

		int total_used_proc_num =0;
		for(int  i=0;i< num_proc;i++)
		{
			if(used_proc[i] != 0)
				total_used_proc_num++;
		}
		
		int total_buffer_size = 0;
		for(int i=0;i<num_edge;i++)
		{
			int buf_size = bufferSizeGene.get(i).intValue();
			
			if( (total_buffer_size + buf_size) < 0)
				total_buffer_size = Integer.MAX_VALUE;
			else
				total_buffer_size += buf_size;
			
			buffer_sizes[i] = buf_size;
		}
		
		int startPriorityOrder [] = new int[priorityGene.size()];		
		for(int i = 0 ; i < priorityGene.size();i++)
		{
			startPriorityOrder[i] = priorityGene.get(i).intValue();				
		}
		
//		int priorityInc [] = new int[num_node];		
//		for(int i = 0 ; i < num_node;i++)
//		{
//			priorityInc[i] = priorityIncGene.get(i).intValue();				
//		}
//		
		
		return new BufferSizeNProcNum(problem.rawGraph, problem.arch, proc_maps, buffer_sizes, total_used_proc_num, total_buffer_size, startPriorityOrder/*, priorityInc*/);
	}
}