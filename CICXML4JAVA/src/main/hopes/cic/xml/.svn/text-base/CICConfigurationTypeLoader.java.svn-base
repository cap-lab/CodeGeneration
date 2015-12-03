package hopes.cic.xml;

import javax.xml.bind.JAXBElement;

public class CICConfigurationTypeLoader extends ResourceLoader<CICConfigurationType> {

	@Override
	public CICConfigurationType createResource(String name) {
		return OBJECT_FACTORY.createCICConfigurationType();
	}

	@Override
	protected JAXBElement<CICConfigurationType> getJAXBElement(
			CICConfigurationType resource) {
		return OBJECT_FACTORY.createCICConfiguration(resource);
	}

	@Override
	protected String getSchemaFile() {
		return "CICConfiguration.xsd";
	}
}
