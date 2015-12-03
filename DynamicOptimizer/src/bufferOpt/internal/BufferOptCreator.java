package bufferOpt.internal;

import org.opt4j.core.problem.Creator;
import org.opt4j.genotype.IntegerGenotype;
import org.opt4j.genotype.PermutationGenotype;
import org.opt4j.tutorial.City;

import com.google.inject.Inject;

import java.util.Collections;
import java.util.Random;

public class BufferOptCreator implements Creator<BufferOptGenotype> {

	protected final BufferOptProblem problem;
	Random random = new Random(0);
	@Inject
	public BufferOptCreator(BufferOptProblem problem)
	{
		this.problem = problem;
	}
	
	public BufferOptGenotype create()
	{
		int num_node = problem.gr.get_nodeNum();
		int num_proc = problem.gr.get_procNum();
		int num_edge = problem.gr.get_edgeNum();
		
		BufferOptGenotype geno = new BufferOptGenotype();
		
		IntegerGenotype procMappingGene = new IntegerGenotype(0, num_proc-1);
		procMappingGene.init(random, num_node);
	
		int [] lowbounds = new int[num_edge];
		int [] upperbouds = new int [num_edge];
		for(int i=0;i< num_edge;i++)
		{
			int min_size = problem.gr.get_tasks().get_edges().get(i).get_minSize();
			lowbounds[i] = min_size;
			
			//upperbouds[i] = Integer.MAX_VALUE;
			upperbouds[i] = min_size*100;
			//genotype.add( new Integer( min_size*5 + random.nextInt(min_size*5)) );
		}
		
		IntegerGenotype bufferSizeGene = new IntegerGenotype(lowbounds, upperbouds);
		bufferSizeGene.init(random, num_edge);
		
//		System.out.println("GENE1:" + procMappingGene);
//		System.out.println("GENE2:" + bufferSizeGene);
//		//System.exit(1);
		
//		PermutationGenotype<Integer> priorityGene = new PermutationGenotype<Integer>();
//		for (int i=0;i<num_node;i++) {
//			int nodeId = problem.gr.get_tasks().get_nodes().get(i).get_id();
//			priorityGene.add(new Integer(nodeId));
//		}
////		System.out.println(priorityGene);
//		Collections.shuffle(priorityGene);
		
		IntegerGenotype priorityGene;
		//PermutationGenotype<Integer> priorityGene = null;
		if(Param.GA_PRIORITY_NODE)
		{
			lowbounds = new int[num_node*2];
			upperbouds = new int[num_node*2];

			for(int i = 0 ; i < num_node ; i++)
			{
				/*
				 *        N                                         N          
				 *   ********************************* ###############################
				 *    last instance's priority         offset value for next instance's prioity 
				 *    random int btw. 1 ~ exec.time       random int btw. 1 ~ exec.time
				 */
				lowbounds[i] = lowbounds[i+num_node] = 1;
				upperbouds[i] = upperbouds[i+num_node] = problem.gr.get_tasks().get_nodes().get(i).get_profExecTime(0); // based on 0 processor profile time
				//System.out.println("Node["+i+"] lowbound["+lowbounds[i]+"] upperbound["+upperbouds[i]+"]");
			}		
			priorityGene = new IntegerGenotype(lowbounds, upperbouds);
			priorityGene.init(random, num_node*2);
		}
		else
		{
			int totalInstanceCount=0;
			for(int i=0;i<num_node;i++)
			{
				totalInstanceCount+=problem.gr.get_tasks().get_nodes().get(i).get_initNumberOfInstance();
			}			
			priorityGene = new IntegerGenotype(1, totalInstanceCount);
			priorityGene.init(random, totalInstanceCount);
//			priorityGene = new PermutationGenotype<Integer>();
//			for (int i=0;i<totalInstanceCount;i++) {				
//				priorityGene.add(new Integer(i+1));
//			}
//			//		System.out.println(priorityGene);
//			Collections.shuffle(priorityGene);
			/*
			 *        T : total number of instance                                                  
			 *   *******|*******|***********|******|****
			 *    Node1   Node2  ....     ....     NodeN
			 *    random int btw. 1 ~ T   
			 */
			
		}
//		
//		System.out.println(priorityGene);
//		System.exit(1);
		
//		int [] lowboundsPriority = new int[num_node];
//		int [] upperboudsPriority = new int [num_node];
//		for(int i=0;i< num_node;i++)
//		{
//			lowbounds[i] = 1;
//			//upperbouds[i] = Integer.MAX_VALUE;
//			upperbouds[i] = problem.gr.get_tasks().get_nodes().get(i).get_execTime();
//			//genotype.add( new Integer( min_size*5 + random.nextInt(min_size*5)) );
//		}
//		IntegerGenotype priorityIncGene = new IntegerGenotype(lowboundsPriority, upperboudsPriority);
//		priorityIncGene.init(random, num_node);
//		
//		System.out.println(priorityIncGene);
//		System.exit(1);
		
		geno.setProcMappingGene(procMappingGene);
		geno.setBufferSizeGene(bufferSizeGene);
		geno.setPriorityGene(priorityGene);
//		geno.setPriorityIncGene(priorityIncGene);
		
		return geno;
	}
}
