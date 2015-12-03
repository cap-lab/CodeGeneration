package bufferOpt.scheduler;

import javax.swing.JFrame;
import javax.swing.JPanel;

import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.graphics.Rectangle;
import org.eclipse.swt.layout.FormLayout;
import org.eclipse.swt.widgets.Button;
import org.eclipse.swt.widgets.Display;
import org.eclipse.swt.widgets.Event;
import org.eclipse.swt.widgets.Label;
import org.eclipse.swt.widgets.Listener;
import org.eclipse.swt.widgets.Shell;
import org.eclipse.swt.widgets.Text;
import org.opt4j.core.Archive;
import org.opt4j.core.Individual;

import bufferOpt.internal.BufferSizeNProcNum;
import ptolemy.plot.Plot;

import java.awt.Dimension;
import java.awt.event.*;
import java.util.ArrayList;


/**
 * Title:        EXPO
 * Description:  Design Space Exploration for Packet Processors
 * Copyright:    Copyright (c) 2001
 * Company:      ETH Zurich<p><p>
 *
 * The class PopulationPlot shows a new frame with a Ptolemy plot
 * when initialized. Its purpose is to show points which
 * represent the current population.
 *
 * @author Lothar Thiele
 * @version 1.0
 */

public class PopulationPlot extends JPanel {

	/**
	 * A new Ptolemy plot.
	 */
	public Plot pt = new Plot();

	protected int[] _geneID;
	protected double[] _x;
	protected double[] _y;
	ArrayList<BufferSizeNProcNum> genes;

	/**
	 * Initializes the frame and a default Ptolemy plot.
	 * @param xLoc x-coordinate of the plot on the screen. May be multiple of 400.
	 * @param yLoc y-coordinate of the plot on the screen. May be multiple of 300.
	 */

	public PopulationPlot(int xLoc, int yLoc, String title, ArrayList<String> axisName) {
		this();
	
		pt.setSize(400, 300);
		pt.setPreferredSize(new Dimension(400, 300));
		pt.setButtons(true);
		pt.setTitle(title);
		if (axisName != null){
			pt.setXLabel(axisName.get(0));
			pt.setYLabel(axisName.get(1));
		}
		else {
			pt.setXLabel("");
			pt.setYLabel("");
		}
		pt.setMarksStyle("none");
		pt.setConnected(true);
		pt.setMarksStyle("dots");

		pt.addMouseListener(new MyListener());

		setSize(400, 300);
		setLocation(xLoc, yLoc);
		this.add(pt);
	
	}

	/**
	 * Plots points.
	 * @param x x-coordinates of the points.
	 * @param y y-coordinates of the points.
	 */
	public void drawPoints(double[] x, double[] y, int[] geneID) {

		_geneID = new int[geneID.length];
		_x = new double[x.length];
		_y = new double[y.length];

		for (int i = 0; i < x.length; i++) {
			_geneID[i] = geneID[i];
			_x[i] = x[i];
			_y[i] = y[i];
			pt.addPoint(0, x[i], y[i], false);
		}

		pt.fillPlot();
	}

	public PopulationPlot() {

	}
	
	


	/**
	 * This class is used as a MouseListener. We make use
	 * of the mouseClicked-method only.
	 */
	public class MyListener implements MouseListener {

		/**
		 * This method is used to check, whether a mouse click event
		 * occured inside the plotting area of the population plot
		 * window. If so, the ID corresponding to the clicked point
		 * is written to a dialog.
		 */
		public void mouseClicked(MouseEvent event) {

			int id = -1;
			java.awt.Rectangle rec = pt.getPlotRectangle();
			
			int _ulx = (int) rec.getMinX(); 
			int _uly = (int) rec.getMinY();
			int _lrx = (int) rec.getMaxX();
			int _lry = (int) rec.getMaxY();

						
			
			double _padding = 0.05;


			int graphX = event.getX();
			int graphY = event.getY();

			// check whether click happened inside the plot window.
			double[] xAxis = pt.getXRange();
			double[] yAxis = pt.getYRange();
			if ( (graphX > _ulx) && (graphX < _lrx)
					&& (graphY > _uly) && (graphY < _lry)) {
				double scaledX = (double) (graphX - _ulx) / (_lrx - _ulx);
				double scaledY = (double) (graphY - _lry) / (_uly - _lry);
				xAxis[0] = xAxis[0] - (xAxis[1] - xAxis[0]) * _padding;
				xAxis[1] = xAxis[1] + (xAxis[1] - xAxis[0]) * _padding;
				yAxis[0] = yAxis[0] - (yAxis[1] - yAxis[0]) * _padding;
				yAxis[1] = yAxis[1] + (yAxis[1] - yAxis[0]) * _padding;
				double xEpsilon = Math.abs(0.02 * (xAxis[1] - xAxis[0] + 1E-10));
				double yEpsilon = Math.abs(0.02 * (yAxis[1] - yAxis[0] + 1E-10));
				double actualX = scaledX * (xAxis[1] - xAxis[0]) + xAxis[0];
				double actualY = scaledY * (yAxis[1] - yAxis[0]) + yAxis[0];
				
					for (int i = 0; i < _geneID.length; i++) {
						if ( (Math.abs(_x[i] - actualX) < xEpsilon) &&
								(Math.abs(_y[i] - actualY) < yEpsilon)) {
							id = _geneID[i];
						}
					}
					if (id > -1) {
						
							if (event.getButton() == MouseEvent.BUTTON1 || event.getButton() == MouseEvent.BUTTON2) {
								// left button clicked
								/*
                javax.swing.JOptionPane.showMessageDialog(pt,
                    new String("Gene number: " + id),
                    "Population Plot",
                    javax.swing.JOptionPane.INFORMATION_MESSAGE);
								 */
								Object[] options = {"OK"};
								System.out.println(genes);
								BufferSizeNProcNum curGene = genes.get(id);
					
								
								int n = javax.swing.JOptionPane.showOptionDialog(pt,
										curGene.getMappingStr(),
										"Solution information",
										javax.swing.JOptionPane.YES_NO_CANCEL_OPTION,
										javax.swing.JOptionPane.QUESTION_MESSAGE,
										null,
										options,
										options[0]);
								
							}
							


					
						
					
				}
				
			}
			else {
				System.out.println("Point outside plotting area");
			}
		}
		
		public void createPopupSaveComplete(){
			Display display = Display.getDefault();
			final Shell dialog = new Shell(display, SWT.APPLICATION_MODAL | SWT.DIALOG_TRIM);
			dialog.setText("Alarm");
			Point pt = display.getCursorLocation();
		    dialog.setLocation(pt.x, pt.y);
			dialog.setSize(250,120);
			final Button buttonOK = new Button(dialog, SWT.PUSH);
			buttonOK.setText("OK");
		    buttonOK.setBounds(20, 55, 80, 25);
		    final Label label = new Label(dialog, SWT.NONE);
		    label.setText("Saves is complete");
		    label.setBounds(20, 15, 100, 20);
		    
			dialog.open();
			Listener listener = new Listener() {
				public void handleEvent(Event event) {
					if (event.widget == buttonOK) {
						dialog.close();
					}
				}
			};
			buttonOK.addListener(SWT.Selection, listener);
		}

		/**
		 * not implemented
		 */
		public void mouseMoved(MouseEvent event) {
		}

		/**
		 * not implemented
		 */
		public void mouseExited(MouseEvent event) {
		}

		/**
		 * not implemented
		 */
		public void mouseEntered(MouseEvent event) {
		}

		/**
		 * not implemented
		 */
		public void mousePressed(MouseEvent event) {
		}

		/**
		 * not implemented
		 */
		public void mouseReleased(MouseEvent event) {
		}


	}




	public void setGeneList(ArrayList<BufferSizeNProcNum> tempList) {

		genes = tempList;
		
	}




}