package hopes.cic.xml.handler;

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
		configuration = new CICConfigurationType();
	}
	
	public void init() {
	}

	protected void storeResource(StringWriter writer) throws CICXMLException {
		loader.storeResource(configuration, writer);
	}
	
	protected void loadResource(ByteArrayInputStream is) throws CICXMLException {
		configuration = loader.loadResource(is);
	}
	
	public CICConfigurationType getConfiguration()
	{
		return configuration;
	}
}
