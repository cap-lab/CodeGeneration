package org.snu.cse.cap.translator.structure.library;

import java.util.ArrayList;

public class Library {
	private String name;
	private String type;
	private String file;
	private String header;
	private ArrayList<Function> functionList;
	private ArrayList<LibraryConnection> libraryConnectionList; 
	private String headerGuard;
	
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
}
