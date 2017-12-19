package org.snu.cse.cap.translator.structure.library;

enum LibraryMasterType {
	LIBRARY,
	TASK,
}

public class LibraryConnection {
	private String masterName; 
	private String portName;
	private LibraryMasterType masterType;
	
	public LibraryConnection(String masterName, String portName, boolean isLibrary)
	{
		this.masterName = masterName;
		this.portName = portName;
		if(isLibrary == true)
		{
			this.masterType = LibraryMasterType.LIBRARY;
		}
		else
		{
			this.masterType = LibraryMasterType.TASK;
		}
	}

	public String getMasterName() {
		return masterName;
	}

	public String getPortName() {
		return portName;
	}
	
	public boolean isMasterLibrary() {
		if(this.masterType == LibraryMasterType.LIBRARY)
		{
			return true;
		}
		else
		{
			return false;
		}
	}
}
