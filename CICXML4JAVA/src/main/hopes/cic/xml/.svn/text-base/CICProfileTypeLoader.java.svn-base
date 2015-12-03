package hopes.cic.xml;

import javax.xml.bind.JAXBElement;

public class CICProfileTypeLoader extends ResourceLoader<CICProfileType> {

	@Override
	public CICProfileType createResource(String name) {
		return OBJECT_FACTORY.createCICProfileType();
	}

	@Override
	protected JAXBElement<CICProfileType> getJAXBElement(
			CICProfileType resource) {
		return OBJECT_FACTORY.createCICProfile(resource);
	}

	@Override
	protected String getSchemaFile() {
		return "CICProfile.xsd";
	}
}
