package hae.peace.container.cic.mapping.xml;

import java.io.ByteArrayInputStream;
import java.io.FileOutputStream;
import java.io.StringWriter;

import hae.peace.container.cic.mapping.CICDSEPanel;
import hae.peace.container.cic.mapping.CICManualDSEPanel;
import hopes.cic.exception.CICXMLException;
import hopes.cic.xml.CICGPUSetupType;
import hopes.cic.xml.CICGPUSetupTypeLoader;


public class CICGPUSetupXMLHandler{
	private CICGPUSetupTypeLoader loader;
	private CICGPUSetupType gpuSetup;
	
	public CICGPUSetupXMLHandler() {
		loader = new CICGPUSetupTypeLoader();
	}
	
	public void storeXMLString(String fileName, CICGPUSetupType gpuSetup) throws CICXMLException {
		StringWriter writer = new StringWriter();
		loader.storeResource(gpuSetup, writer);
		writer.flush();
		
		FileOutputStream os;
		try {
			os = new FileOutputStream(fileName);
			byte[] abContents = writer.toString().getBytes();
			os.write(abContents, 0, abContents.length);
			os.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public CICGPUSetupType getGPUSetup( String xmlString )
	{
		ByteArrayInputStream is = new ByteArrayInputStream(xmlString.getBytes());
		try {
			gpuSetup = loader.loadResource(is);
		} catch (CICXMLException e) {
			e.printStackTrace();
		}
			
		return gpuSetup;		
	}
}
