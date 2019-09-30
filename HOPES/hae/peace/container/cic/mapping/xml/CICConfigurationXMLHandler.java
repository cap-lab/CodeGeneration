package hae.peace.container.cic.mapping.xml;

import java.io.ByteArrayInputStream;
import java.io.StringWriter;

import hopes.cic.exception.CICXMLException;
import hopes.cic.xml.CICConfigurationType;
import hopes.cic.xml.CICConfigurationTypeLoader;


public class CICConfigurationXMLHandler extends CICXMLHandler {
	private CICConfigurationTypeLoader loader;
	private CICConfigurationType configuration;
	
	public CICConfigurationXMLHandler() {
		loader = new CICConfigurationTypeLoader();
	}
	
	public String getXMLString() throws CICXMLException {
		StringWriter writer = new StringWriter();
		loader.storeResource(configuration, writer);
		writer.flush();
		return writer.toString();		
	}
	
	public void setXMLString(String xmlString) throws CICXMLException
	{
		ByteArrayInputStream is = new ByteArrayInputStream(xmlString.getBytes());
		configuration = loader.loadResource(is);
	}
	
	public CICConfigurationType getConfiguration()
	{
		return configuration;
	}
}
