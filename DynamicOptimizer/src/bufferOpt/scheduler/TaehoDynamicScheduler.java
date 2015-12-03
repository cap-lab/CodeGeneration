package bufferOpt.scheduler;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;



import java.util.List;

import javax.swing.JPanel;

import org.jdom2.Document;
import org.jdom2.Element;
import org.jdom2.JDOMException;
import org.jdom2.input.SAXBuilder;
import org.opt4j.core.Archive;

import model.app.SDFGraph;
import model.architecture.GenericArch;

public class TaehoDynamicScheduler {
	public ArrayList<String> specFileList;
	public ArrayList<SDFGraph> rawGraphList;
	public ArrayList<Archive> archiveList;
	public String path;
	public String archFile;
	GenericArch arch;
	public ArrayList<String> objectiveList;


	public void setArchFile(String archFile){
		this.archFile = archFile;
		this.arch = new GenericArch();
		this.arch.readFromCICArchitecture(archFile);
	}

	public void setSourcePath(String path){
		this.path = path;
	}

	public TaehoDynamicScheduler(){
		
		archiveList = new  ArrayList<Archive>();
		objectiveList = new ArrayList<String>();
		objectiveList.add("1/Throughput");
		objectiveList.add("Buffer Size");
		
		specFileList = new ArrayList<String>();
		rawGraphList = new ArrayList();
		
	}
	
	public AutomaticDSEPanel2 getDSEPanel(){
		AutomaticDSEPanel2 panel = new AutomaticDSEPanel2(this);
		return panel;
	}


	//prepare SDFFileList
	public void readFilesFromHopes(String relativePath, String target, String appName, String fileName){

		try {
			File input = new File(fileName);
			SAXBuilder saxBuilder = new SAXBuilder();  
			Document document  = saxBuilder.build(input);
			specFileList = new ArrayList<String>();
			// get root node from xml  
			Element rootNode = document.getRootElement();  
			Element sdf3filesElement = rootNode.getChild("sdf3files");
			List<Element> sdf3fileElements = sdf3filesElement.getChildren("sdf3file");
			for (Element sdf3file : sdf3fileElements){
				String filename = sdf3file.getAttributeValue("file_name");
				specFileList.add(relativePath + target + "/convertedSDF3xml/" + filename);				
			}
		}
		catch (JDOMException | IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
	}

}
