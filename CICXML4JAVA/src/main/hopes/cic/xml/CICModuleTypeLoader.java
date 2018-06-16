package hopes.cic.xml;

import javax.xml.bind.JAXBElement;

public class CICModuleTypeLoader extends ResourceLoader<CICModuleType> {
	@Override
	public CICModuleType createResource(String name) {
		return OBJECT_FACTORY.createCICModuleType();
	}

	@Override
	protected JAXBElement<CICModuleType> getJAXBElement(
			CICModuleType resource) {
		return OBJECT_FACTORY.createCICModule(resource);
	}

	@Override
	protected String getSchemaFile() {
		return "CICModule.xsd";
	}
}
