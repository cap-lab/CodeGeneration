package hopes.cic.xml;

import javax.xml.bind.JAXBElement;

public class CICGPUSetupTypeLoader extends ResourceLoader<CICGPUSetupType> {

	@Override
	public CICGPUSetupType createResource(String name) {
		return OBJECT_FACTORY.createCICGPUSetupType();
	}

	@Override
	protected JAXBElement<CICGPUSetupType> getJAXBElement(
			CICGPUSetupType resource) {
		return OBJECT_FACTORY.createCICGPUSetup(resource);
	}

	@Override
	protected String getSchemaFile() {
		return "CICGPUSetup.xsd";
	}
}
