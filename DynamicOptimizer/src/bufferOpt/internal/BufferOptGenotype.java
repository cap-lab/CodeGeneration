package bufferOpt.internal;

import org.opt4j.core.problem.Genotype;
import org.opt4j.genotype.CompositeGenotype;
import org.opt4j.genotype.IntegerGenotype;
import org.opt4j.genotype.PermutationGenotype;

@SuppressWarnings("serial")
public class BufferOptGenotype extends CompositeGenotype<Integer, Genotype>
implements Genotype 
{	
	public void setProcMappingGene(IntegerGenotype genotype)
	{
		put(0, genotype);
	}
	
	public IntegerGenotype getProcMappingGene()
	{		
		return get(0); 
	}
	
	public void setBufferSizeGene(IntegerGenotype genotype)
	{
		put(1 , genotype);
	}
	
	public IntegerGenotype getBufferSizeGene()
	{
		return get(1);
	}
	
	//public void setPriorityGene(PermutationGenotype<Integer> genotype)
	public void setPriorityGene(IntegerGenotype genotype)
	{
		put(2, genotype);
	}
	
	//public PermutationGenotype<Integer> getPriorityGene()
	public IntegerGenotype getPriorityGene()
	{
		return get(2);
	}	
	
//	public void setPriorityIncGene(IntegerGenotype genotype)
//	{
//		put(3, genotype);
//	}
//	
//	public IntegerGenotype getPriorityIncGene()
//	{
//		return get(3);
//	}	
}
