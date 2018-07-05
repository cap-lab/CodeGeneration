package org.snu.cse.cap.translator.structure.module;

import java.util.ArrayList;

public class Module {
	private String name;
	private String cflags;
	private String ldflags;
	private String initializer;
	private String finalizer;
	private ArrayList<String> sourceList;
	private ArrayList<String> headerList;
	
	public Module(String name, String cflags, String ldflags, String initializer, String finalizer)
	{
		this.name = name;
		this.cflags = cflags;
		this.ldflags = ldflags;
		this.initializer = initializer;
		this.finalizer = finalizer;
		this.sourceList = new ArrayList<String>();
		this.headerList = new ArrayList<String>();
	}
	
	public void putSourceFile(String fileName)
	{
		this.sourceList.add(fileName);
	}
	
	public void putHeaderFile(String fileName)
	{
		this.headerList.add(fileName);
	}
	
	public String getName() {
		return name;
	}
	
	public String getCflags() {
		return cflags;
	}
	
	public String getLdflags() {
		return ldflags;
	}
	
	public String getInitializer() {
		return initializer;
	}
	
	public String getFinalizer() {
		return finalizer;
	}
	
	public ArrayList<String> getSourceList() {
		return sourceList;
	}
	
	public ArrayList<String> getHeaderList() {
		return headerList;
	}
}
