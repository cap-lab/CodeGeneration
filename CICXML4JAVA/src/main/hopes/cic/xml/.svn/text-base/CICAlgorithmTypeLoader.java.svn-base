package hopes.cic.xml;

import javax.xml.bind.JAXBElement;

public class CICAlgorithmTypeLoader extends ResourceLoader<CICAlgorithmType> {

	@Override
	public CICAlgorithmType createResource(String name) {
		return OBJECT_FACTORY.createCICAlgorithmType();
	}

	@Override
	protected JAXBElement<CICAlgorithmType> getJAXBElement(
			CICAlgorithmType resource) {
		return OBJECT_FACTORY.createCICAlgorithm(resource);
	}

	@Override
	protected String getSchemaFile() {
		return "CICAlgorithm.xsd";
	}
}
