package org.snu.cse.cap.translator.structure.library;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

public class Library {
	private String name;
	private String type;
	private String file;
	private String header;
	private ArrayList<Function> functionList;
	private ArrayList<LibraryConnection> libraryConnectionList; 
	private String headerGuard;
	private HashSet<String> extraHeaderSet;
	private HashSet<String> extraSourceSet;
	private HashMap<String, Library> masterPortToLibraryMap;
	private String ldFlags;
	private String cFlags;
	
	// // Master can be a task or a library
	public Library(String name, String type, String file, String header)
	{
		this.name = name;
		this.type = type;
		this.file = file;
		this.header = header;
		this.functionList = new ArrayList<Function>();
		this.libraryConnectionList = new ArrayList<LibraryConnection>();
		this.headerGuard = header.toUpperCase().replace(".", "_");
		this.extraHeaderSet = new HashSet<String>();
		this.extraSourceSet = new HashSet<String>();
		this.masterPortToLibraryMap = new HashMap<String, Library>();
		this.ldFlags = null;
		this.cFlags = null;
	}
	
	public void setExtraHeaderSet(List<String> extraHeaderList)
	{
		for(String extraHeaderFile: extraHeaderList)
		{
			this.extraHeaderSet.add(extraHeaderFile);
		}
	}
	
	public void setExtraSourceSet(List<String> extraSourceList)
	{
		for(String extraSourceFile: extraSourceList)
		{
			this.extraSourceSet.add(extraSourceFile);
		}	
	}

	public String getName() {
		return name;
	}

	public String getType() {
		return type;
	}

	public String getFile() {
		return file;
	}

	public String getHeader() {
		return header;
	}

	public ArrayList<Function> getFunctionList() {
		return functionList;
	}

	public ArrayList<LibraryConnection> getLibraryConnectionList() {
		return libraryConnectionList;
	}

	public String getHeaderGuard() {
		return headerGuard;
	}

	public HashSet<String> getExtraHeaderSet() {
		return extraHeaderSet;
	}

	public String getLdFlags() {
		return ldFlags;
	}

	public void setLdFlags(String ldFlags) {
		this.ldFlags = ldFlags;
	}

	public HashSet<String> getExtraSourceSet() {
		return extraSourceSet;
	}

	public HashMap<String, Library> getMasterPortToLibraryMap() {
		return masterPortToLibraryMap;
	}

	public String getcFlags() {
		return cFlags;
	}

	public void setcFlags(String cFlags) {
		this.cFlags = cFlags;
	}
}
