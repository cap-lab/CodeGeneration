package InnerDataStructures;

import java.util.List;

public class MTM {
	List <String> mModes;
	List <Variable> mVariables;
	List <Transition> mTransitions;
	
	public MTM(List <String> modes, List <Variable> variables, List <Transition> transitions){
		mModes = modes;
		mVariables = variables;
		mTransitions = transitions;
	}

	public MTM() {
		
	}

	public List<String> getModes()				{return mModes;}
	public List<Variable> getVariables()		{return mVariables;}
	public List<Transition> getTransitions()	{return mTransitions;}
}
