package bufferOpt.scheduler;

import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.io.FileReader;
import java.io.StreamTokenizer;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Iterator;

import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;

import org.opt4j.core.Archive;
import org.opt4j.core.Individual;
import org.opt4j.core.Objective;
import org.opt4j.optimizer.ea.EvolutionaryAlgorithmModule;
import org.opt4j.start.Opt4JTask;
import org.opt4j.viewer.ViewerModule;

import bufferOpt.internal.BufferOptModule;
import bufferOpt.internal.BufferSizeNProcNum;





public class DSEPanel extends JPanel{
	public TaehoDynamicScheduler mo;
	public int idx;
	
	JTextArea statusText = new JTextArea();
	JScrollPane scrollText;
	JPanel ButtonArea = new JPanel();
	JButton runButton = new JButton();
	PopulationPlot plot = new PopulationPlot();

	
	public DSEPanel(TaehoDynamicScheduler mo, int idx){
		this.mo = mo;
		this.idx = idx;
		plot = new PopulationPlot(0, 0, "Current Population", mo.objectiveList);
		plot.setPreferredSize(new Dimension(500, 400));
		this.setLayout(new FlowLayout());
		
		runButton.setText("Run");
		runButton.addActionListener(new Frame_runButton_actionAdapter(this));
	
		//just two.
		this.add(runButton);
		this.add(plot);
		
		
	}
	
	
	void runButton_actionPerformed(ActionEvent e) {
		//MutateOptimizerModule ea = new MutateOptimizerModule();
		EvolutionaryAlgorithmModule ea = new EvolutionaryAlgorithmModule();
		ea.setGenerations(100);
		ea.setAlpha(100);
		
		BufferOptModule dtlz = new BufferOptModule();
		
		dtlz.setInputfile(mo.specFileList.get(idx));
		dtlz.setArchfile(mo.archFile);
		//dtlz.setFunction(DTLZModule.Function.DTLZ1);
		ViewerModule viewer = new ViewerModule();
		viewer.setCloseOnStop(true);
	
		Opt4JTask task = new Opt4JTask(false);
		task.init(ea,dtlz,viewer);
		
		try {
		        task.execute();
		        Archive archive = task.getInstance(Archive.class);
		        mo.archiveList.add(archive);
		        drawPopulation(archive, mo.objectiveList, false);
		        
		} catch (Exception exc) {
		        exc.printStackTrace();
		} finally {
		        task.close();
		} 
	
		
		
	}
	
	
	
	
	
	
	public String getName(){
		return mo.specFileList.get(idx);
	}
	


	public   void drawPopulation(Archive archive, ArrayList<String> objectiveList, boolean update) {
		
		if (objectiveList!=null){
			plot.pt.setXLabel(objectiveList.get(0));
			plot.pt.setYLabel(objectiveList.get(1));
		}

		
		// read data of current population
		FileReader reader = null;
		StreamTokenizer tokenStream = null;

		int dim = 2;

		double[][] fitness = new double[dim][];
		double[] xWeight = new double[dim];
		double[] yWeight = new double[dim];
		
		StringReader stringReader;
		double[] x;
		double[] y;
		int[] geneID;
		
		ArrayList<BufferSizeNProcNum> tempList = new ArrayList<BufferSizeNProcNum>();

		int numberOfGenes = archive.size();
		for (int j = 0; j < dim; j++) {
			fitness[j] = new double[numberOfGenes];
		}

		int idx = 0;
		for (Individual ind : archive) {
			BufferSizeNProcNum gene = (BufferSizeNProcNum) ind.getPhenotype();
			tempList.add(gene);
			fitness[0][idx] = gene.throughput();
			fitness[1][idx] = gene.total_buf_size();
			idx++;
		}
		plot.setGeneList(tempList);
		

		// make weight vectors from user input
		for (int i = 0; i < dim; i++) {
			xWeight[0] =1;
			xWeight[1] =0;
			
			yWeight[0] =0;
			yWeight[1] =1;
		}



		// calculate vectors x and y which contain the coordinates
		// if Weight Values are negative -> invert
		x = new double[numberOfGenes];
		y = new double[numberOfGenes];
		geneID = new int[numberOfGenes];

		for (int i = 0; i < numberOfGenes; i++) {
			x[i] = 0.0;
			y[i] = 0.0;
			
			geneID[i] = i;

			//shkang.
			for (int j = 0; j < dim; j++) {
				x[i] += xWeight[j] * fitness[j][i];
				y[i] += yWeight[j] * fitness[j][i];
			}
			
			
/*			
	        for (int j = 0; j < dim; j++) {
	          if (j > 0) {
	            x[i] += xWeight[j] / fitness[j][i];
	          }
	          else {
	            x[i] += xWeight[j] * fitness[j][i];
	          }
	          if (j > 0) {
	            y[i] += yWeight[j] / fitness[j][i];
	          }
	          else {
	            y[i] += yWeight[j] * fitness[j][i];
	          }

	        }*/
			 
		}
		if (update) {
			plot.pt.clear(0);
		}
	
		
		
		plot.drawPoints(x, y, geneID);



	}
	
	
}


class Frame_runButton_actionAdapter implements java.awt.event.ActionListener {
	DSEPanel adaptee;

	Frame_runButton_actionAdapter(DSEPanel adaptee) {
		this.adaptee = adaptee;
	}
	public void actionPerformed(ActionEvent e) {
		adaptee.runButton_actionPerformed(e);
		

	}
}
