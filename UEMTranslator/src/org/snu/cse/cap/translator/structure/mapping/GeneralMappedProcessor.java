package org.snu.cse.cap.translator.structure.mapping;

public class GeneralMappedProcessor extends MappedProcessor {
	protected String mappingSet;

	public GeneralMappedProcessor(int processorId, int processorLocalId, String mappingSet)
	{
		super(processorId, processorLocalId);
		this.mappingSet = mappingSet;
	}

	public String getMappingSet() {
		return mappingSet;
	}

	public void setMappingSet(String mappingName) {
		this.mappingSet = mappingName;
	}
}
