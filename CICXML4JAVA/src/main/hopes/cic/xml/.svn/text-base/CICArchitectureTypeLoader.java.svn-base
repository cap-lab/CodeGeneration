package hopes.cic.xml;

import javax.xml.bind.JAXBElement;

public class CICArchitectureTypeLoader extends ResourceLoader<CICArchitectureType> {

	@Override
	public CICArchitectureType createResource(String name) {
		return OBJECT_FACTORY.createCICArchitectureType();
	}

	@Override
	protected JAXBElement<CICArchitectureType> getJAXBElement(
			CICArchitectureType resource) {
		return OBJECT_FACTORY.createCICArchitecture(resource);
	}

	@Override
	protected String getSchemaFile() {
		return "CICArchitecture.xsd";
	}

}
