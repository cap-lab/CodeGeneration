package bufferOpt.internal;

import org.opt4j.config.annotations.Parent;
import org.opt4j.core.problem.ProblemModule;

import org.opt4j.start.Constant;
import org.opt4j.tutorial.TutorialModule;

@Parent(TutorialModule.class)
public class BufferOptModule extends ProblemModule {
	
	@Constant(value = "inputfile", namespace = BufferOptProblem.class)
	protected String inputfile="yangs_ex_proc8";
	
	@Constant(value = "archfile", namespace = BufferOptProblem.class)
	protected String archfile;
	
	@Constant(value = "priorityType", namespace = BufferOptProblem.class)
	protected String priorityType="cyclo_static";
	//protected int inputfile=3;
	//protected String inputfile="yangs_ex_proc8";
	
	
	public String getPriorityType() {
		return priorityType;
	}

	public void setPriorityType(String priorityType) {
		this.priorityType = priorityType;
	}

	public void setInputfile(String inputfile) {
		this.inputfile = inputfile;
	}
	
	public String getInputfile() {
		return inputfile;
	}
	
	public void setArchfile(String archfile) {
		this.archfile = archfile;
		
	}
	
	public String getArchfile() {
		return archfile;
	}

	@Override
	protected void config() {
		bindProblem(BufferOptCreator.class, BufferOptDecoder.class,
				BufferOptEvaluator.class);
	}
}
