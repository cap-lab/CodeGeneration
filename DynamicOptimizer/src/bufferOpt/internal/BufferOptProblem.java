package bufferOpt.internal;
import model.app.SDFGraph;
import model.architecture.GenericArch;

import org.opt4j.optimizer.sa.CoolingSchedule;
import org.opt4j.start.Constant;

import com.google.inject.Inject;


public class BufferOptProblem
{
	Graph gr;
	
	
	String inputfile;
	String archfile;
	
	String priorityType;
	
	SDFGraph rawGraph;
	GenericArch arch;
	
	
	
	@Inject
	public BufferOptProblem(@Constant(value = "inputfile", namespace = BufferOptProblem.class) String inputfile,
			@Constant(value = "archfile", namespace = BufferOptProblem.class) String archfile,
			@Constant(value = "priorityType", namespace = BufferOptProblem.class) String priorityType) throws Exception
	{
		//this.gr = new Graph(file);
		this.inputfile = inputfile;
		this.archfile = archfile;
		if(priorityType.compareTo("cyclo_static") == 0)
		{
			Param.CYCLO_STATIC = true;
		}
		else if(priorityType.compareTo("ga_node") == 0)
		{
			Param.CYCLO_STATIC = false;
			Param.GA_PRIORITY_NODE = true;
		}
		else if(priorityType.compareTo("ga_instance") == 0)
		{
			Param.CYCLO_STATIC = false;
			Param.GA_PRIORITY_NODE = false;
		}
		//this.gr = new Graph(new String("30m0"));
		System.out.println(this.gr);
		this.gr = new Graph(this.inputfile, this.archfile);
		this.rawGraph = this.gr.rawGraph;
		this.arch = this.gr.arch;
		
		if(Param.PREEMPTIVE)
		{
			System.out.println("Task preemptive, preemptive cost["+Param.PREEMPTION_COST+"]");
		}
		else
			System.out.println("Task non-preemptive");
		//this.gr = new Graph(new String("yangs_ex"));
		//this.gr = new Graph(new String("yangs_ex_proc8"));
	}
}