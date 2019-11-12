package hae.peace.container.cic.mapping.xml;

import java.io.ByteArrayInputStream;
import java.io.FileOutputStream;
import java.io.StringWriter;

import hopes.cic.exception.CICXMLException;
import hopes.cic.xml.CICGPUSetupType;
import hopes.cic.xml.CICGPUSetupTypeLoader;
import hopes.cic.xml.GPUTaskListType;
import hopes.cic.xml.GPUTaskType;


public class CICGPUSetupXMLHandler extends CICXMLHandler{
	private CICGPUSetupTypeLoader loader;
	private CICGPUSetupType gpuSetup;
	
	public CICGPUSetupXMLHandler() {
		loader = new CICGPUSetupTypeLoader();
	}
	
	protected void storeResource(StringWriter writer) throws CICXMLException {
		loader.storeResource(gpuSetup, writer);
	}
	
	protected void loadResource(ByteArrayInputStream is) throws CICXMLException {
		gpuSetup = loader.loadResource(is);
	}
	
	public void makeGpuSetup() {
		gpuSetup = new CICGPUSetupType();
	}
		
	public void clearDevice() throws CICXMLException
	{
		for (Object obj : gpuSetup.getTasks().getTask()) {
			GPUTaskType gpuTask = (GPUTaskType) obj;
			gpuTask.getDevice().clear();
		}
	}
	
	public boolean containTask(String taskName)
	{
		for (Object obj : gpuSetup.getTasks().getTask()) {
			GPUTaskType gpuTask = (GPUTaskType) obj;
			if (taskName.equalsIgnoreCase(gpuTask.getName())) {
				return true;
			}
		}
		return false;
	}
	
	public void setTasks(GPUTaskListType taskList) 
	{
		gpuSetup.setTasks(taskList);
	}
	
	public int getTaskSize() {
		return gpuSetup.getTasks().getTask().size();
	}
}
