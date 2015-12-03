package hopes.cic.xml;

import javax.xml.bind.JAXBElement;

public class CICControlTypeLoader extends ResourceLoader<CICControlType> {

	@Override
	public CICControlType createResource(String name) {
		return OBJECT_FACTORY.createCICControlType();
	}

	@Override
	protected JAXBElement<CICControlType> getJAXBElement(
			CICControlType resource) {
		return OBJECT_FACTORY.createCICControl(resource);
	}

	@Override
	protected String getSchemaFile() {
		return "CICControl.xsd";
	}
}
