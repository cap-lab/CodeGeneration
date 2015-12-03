package hopes.cic.xml;

import javax.xml.bind.JAXBElement;

public class CICScheduleTypeLoader extends ResourceLoader<CICScheduleType> {

	@Override
	public CICScheduleType createResource(String name) {
		return OBJECT_FACTORY.createCICScheduleType();
	}

	@Override
	protected JAXBElement<CICScheduleType> getJAXBElement(
			CICScheduleType resource) {
		return OBJECT_FACTORY.createCICSchedule(resource);
	}

	@Override
	protected String getSchemaFile() {
		return "CICSchedule.xsd";
	}
}
