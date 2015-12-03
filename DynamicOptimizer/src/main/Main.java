package main;

import org.opt4j.core.Archive;
import org.opt4j.core.Individual;
import org.opt4j.optimizer.ea.EvolutionaryAlgorithmModule;
import org.opt4j.start.Opt4JTask;
import org.opt4j.viewer.ViewerModule;

import bufferOpt.internal.BufferOptModule;




public class Main {
	public static void main(String[] args){
		//MutateOptimizerModule ea = new MutateOptimizerModule();
		EvolutionaryAlgorithmModule ea = new EvolutionaryAlgorithmModule();
		ea.setGenerations(100);
		ea.setAlpha(100);
		
		BufferOptModule dtlz = new BufferOptModule();
		
		dtlz.setInputfile("benchmarks/simplest.xml");
		dtlz.setArchfile("benchmarks/arch1.xml");
		//dtlz.setFunction(DTLZModule.Function.DTLZ1);
		ViewerModule viewer = new ViewerModule();
		viewer.setCloseOnStop(true);
	
		Opt4JTask task = new Opt4JTask(false);
		task.init(ea,dtlz,viewer);
		try {
		        task.execute();
		        Archive archive = task.getInstance(Archive.class);
		        for (Individual individual : archive) {
		        	System.out.println(individual.getPhenotype());
		                // obtain the phenotype and objective, etc. of each individual
		        }
		} catch (Exception e) {
		        e.printStackTrace();
		} finally {
		        task.close();
		} 
	}
}
