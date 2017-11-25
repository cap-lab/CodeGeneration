package org.snu.cse.cap.translator.structure.device;


public abstract class HWElementType {
	protected String name;
	protected HWCategory category;
	
	public HWElementType(String name, HWCategory category) 
	{
		this.name = name;
		this.category = category;
	}
	
	public String getName() 
	{
		return name;
	}
	
	public void setName(String name) 
	{
		this.name = name;
	}
}
