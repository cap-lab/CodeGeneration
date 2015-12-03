package hopes.cic.xml;

import javax.xml.bind.JAXBElement;

public class CICMappingTypeLoader extends ResourceLoader<CICMappingType> {

	@Override
	public CICMappingType createResource(String name) {
		return OBJECT_FACTORY.createCICMappingType();
	}

	@Override
	protected JAXBElement<CICMappingType> getJAXBElement(
			CICMappingType resource) {
		return OBJECT_FACTORY.createCICMapping(resource);
	}

	@Override
	protected String getSchemaFile() {
		return "CICMapping.xsd";
	}
}
