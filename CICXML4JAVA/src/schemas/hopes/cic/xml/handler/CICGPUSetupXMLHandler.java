package hopes.cic.xml.handler;

import java.io.ByteArrayInputStream;
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
		gpuSetup = new CICGPUSetupType();
	}
	
	public void init() {
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
		if(findTask(taskName) == null) {
			return false;
		} else {
			return true;
		}
	}
	
	public GPUTaskType findTask(String taskName)
	{
		if (gpuSetup.getTasks() != null) {
			for (Object obj : gpuSetup.getTasks().getTask()) {
				GPUTaskType gpuTask = (GPUTaskType) obj;
				if (taskName.equalsIgnoreCase(gpuTask.getName())) {
					return gpuTask;
				}
			}
		}
		return null;
	}
	
	public void setTasks(GPUTaskListType taskList) 
	{
		gpuSetup.setTasks(taskList);
	}
	
	public int getTaskSize() {
		return gpuSetup.getTasks().getTask().size();
	}
}
