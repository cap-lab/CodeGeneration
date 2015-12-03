package bufferOpt.scheduler;


import java.awt.Color;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.TextArea;
import java.awt.event.ActionEvent;
import java.util.ArrayList;
import java.util.Random;

import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JTabbedPane;




public class AutomaticDSEPanel2 extends JPanel{

	Container selectionContainer;
	JTabbedPane tabContainer;
	ArrayList<DSEPanel> optPanelList;

	JProgressBar progressBar;
	TaehoDynamicScheduler mo;

	public boolean ready=false;

	public AutomaticDSEPanel2(TaehoDynamicScheduler mo){
		this.mo = mo;
		tabContainer = new JTabbedPane();
		progressBar = new JProgressBar();

	}


	//get ready here.
	public JTabbedPane generateTabContainer(){


		tabContainer.removeAll();
		tabContainer.setPreferredSize(new Dimension(400, 700));
		optPanelList = new ArrayList<DSEPanel>();

		progressBar.setMaximum(mo.specFileList.size());
		progressBar.setValue(0);
		progressBar.setStringPainted(true);
		ready=false;
	
		int bar = 0;
		int idx=0;
		for (String specFile : mo.specFileList){

			model.app.SDFGraph rawGraph = new model.app.SDFGraph();
			rawGraph.readSDF3Format(specFile);

			boolean consistency = rawGraph.isConsistentAll();

			if (!consistency){
				System.out.println("THis graph is not consistent");
				JOptionPane.showMessageDialog(getParent(),
						specFile+" is not consistent");
			}
			else {

				boolean deadlockfree = rawGraph.isDeadlockFree();
				if (!deadlockfree){
					System.out.println("THis graph is deadlocked");
					JOptionPane.showMessageDialog(getParent(),
							specFile+" is deadlocked");
				}


				else {
					mo.rawGraphList.add(rawGraph);
					progressBar.setValue(++bar);
					optPanelList.add(new DSEPanel(mo, idx++));

				}
			}
		}

		for (int i=0 ; i<mo.rawGraphList.size() ; i++){
			tabContainer.add(optPanelList.get(i), i);
			tabContainer.setTitleAt(i, optPanelList.get(i).getName());
		}
		return tabContainer;
	}

	public void addEXPO(DSEPanel mainPanel) {
		optPanelList.add(mainPanel);

	}



}


