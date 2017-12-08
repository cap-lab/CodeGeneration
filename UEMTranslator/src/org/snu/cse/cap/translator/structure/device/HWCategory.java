package org.snu.cse.cap.translator.structure.device;

public enum HWCategory {
	PROCESSOR("processor"),
	MEMORY("memory"),
	DMA("dma"),
	HWIP("hwip"),
	;

	private final String value;
	
	private HWCategory(final String value) {
		this.value = value;
	}
	
	@Override
	public String toString() {
		return value;
	}

	public String getValue() {
		return value;
	}
}