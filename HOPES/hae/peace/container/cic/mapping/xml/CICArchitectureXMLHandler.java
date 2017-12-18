// hs: need to delete before release

package hae.peace.container.cic.mapping.xml;

import java.io.ByteArrayInputStream;
import java.io.StringWriter;
import java.math.BigInteger;
import java.util.HashMap;
import java.util.Map;

import hae.kernel.util.ObjectList;
import hae.peace.container.cic.mapping.MemoryRegion;
import hae.peace.container.cic.mapping.Processor;
import hopes.cic.exception.CICXMLException;
import hopes.cic.xml.ArchitectureDeviceType;
import hopes.cic.xml.ArchitectureElementCategoryType;
import hopes.cic.xml.ArchitectureElementSlavePortType;
import hopes.cic.xml.ArchitectureElementType;
import hopes.cic.xml.ArchitectureElementTypeType;
import hopes.cic.xml.CICArchitectureType;
import hopes.cic.xml.CICArchitectureTypeLoader;

public class CICArchitectureXMLHandler {
	private CICArchitectureTypeLoader loader;
	private CICArchitectureType architecture;
//	private CICManualDSEPanel panel;
	public CICArchitectureXMLHandler(/*CICManualDSEPanel cicManualDSEPanel*/) {
		loader = new CICArchitectureTypeLoader();
//		this.panel = cicManualDSEPanel;
	}
	
	public CICArchitectureType getArchitecture() {
		return architecture;
	}
	
	public void setArchitecture(CICArchitectureType architecture) {
		this.architecture = architecture;
	}
	
	public void setXMLString(String xmlString) throws CICXMLException {
		ByteArrayInputStream is = new ByteArrayInputStream(xmlString.getBytes());
		architecture = loader.loadResource(is);
		processed = false;
		getProcessorList();
	}
	
	public String getXMLString() throws CICXMLException {
		update();
		StringWriter writer = new StringWriter();
		loader.storeResource(architecture, writer);
		writer.flush();
		return writer.toString();
	}
	
	private ObjectList procList = new ObjectList();
	private boolean processed = false;
	public ObjectList getProcessorList() {
		if (!processed && architecture != null) {
			procList.clear();
			makeProcessorList();
			makeMemoryRegionMap();
			// * [hshong, 2014/06/27] : 임시로 잠시 안함! (xml)setMaps 때문
			// * [hshong, 2014/07/01]: deleted: architecture.xml 변경에 따른 map 삭제
			//updateMemoryRegionMap();
			processed = true;
			// * [hshong, 2014/06/27] : 임시로 잠시 안함! (xml)setMaps 때문
			// * [hshong, 2014/07/01]: deleted: architecture.xml 변경에 따른 map 삭제
			//addMemoryRegionListToProcessor();
		}
		
		return procList;
	}
	
	private ArchitectureElementTypeType getElementType(ArchitectureElementCategoryType category, String typeName) {
		for (ArchitectureElementTypeType elementType : architecture.getElementTypes().getElementType()) {
			if (category == elementType.getCategory() &&
					typeName.equals(elementType.getName()))
				return elementType;
		}
		return null;
	}
	
	private void makeProcessorList() {
		for (ArchitectureDeviceType device : architecture.getDevices().getDevice()) {
			for (ArchitectureElementType element : device.getElements().getElement()) {
				ArchitectureElementTypeType elementType = getElementType(ArchitectureElementCategoryType.PROCESSOR,
						element.getType());
				if (elementType == null) // processor가 아니거나 invalid한 경우임
					continue;

				int poolSize = element.getPoolSize() != null ? element.getPoolSize().intValue() : 1;
				String os = elementType.getOS();
				if (os == null)
					os = "NONE";

				// deleted for release (2015/12) - 더이상 processor 에서는 allow Data
				// ParallelMapping이 쓰이지 않음
				// boolean bParallel = element.isAllowDataParallelMapping();
				for (int i = 0; i < poolSize; i++) {
					// deleted for release (2015/12) - 더이상 processor 에서는 allow
					// Data ParallelMapping이 쓰이지 않음
					Processor proc = new Processor(i, element.getName(), /* bParallel, */os, element.getType(), device.getName());
					procList.add(proc);
				}
			}
		}
	}
	
	Map<String, MemoryRegion> memoryRegionMap = new HashMap<String, MemoryRegion>();

	private void makeMemoryRegionMap() {
		for (ArchitectureDeviceType device : architecture.getDevices().getDevice()) {
			for (ArchitectureElementType element : device.getElements().getElement()) {
				ArchitectureElementTypeType elementType = getElementType(ArchitectureElementCategoryType.MEMORY,
						element.getType());
				if (elementType == null) // memory가 아니거나 invalid한 경우임
					continue;

				ArchitectureElementSlavePortType slavePort = elementType.getSlavePort().get(0);
				BigInteger memorySize = getMemorySize(slavePort);
				MemoryRegion memoryRegion = new MemoryRegion("0x" + memorySize.toString(16));
				memoryRegionMap.put(element.getName(), memoryRegion);
			}
		}
	}
	
	private BigInteger getMemorySize(ArchitectureElementSlavePortType slavePort) {
		BigInteger size = slavePort.getSize();
		switch (slavePort.getMetric()) {
		case B:
			return size;
		case KI_B:
			return size.shiftLeft(10);
		case MI_B:
			return size.shiftLeft(20);
		case GI_B:
			return size.shiftLeft(30);
		case TI_B:
			return size.shiftLeft(40);
		}
		
		return BigInteger.ZERO;
	}
	
	// * [hshong, 2014/07/01]: deleted: architecture.xml 변경에 따른 map 삭제
//	private void updateMemoryRegionMap() {
//		for (ArchitectureMapType master : architecture.getMaps().getMaster()) {
//			for (ArchitectureMapSlaveType slave : master.getSlave()) {
//				MemoryRegion memoryRegion = memoryRegionMap.get(slave.getName());
//				if (memoryRegion != null)
//					memoryRegion.addShareCount();
//			}
//		}
//	}
	
// * [hshong, 2014/07/01]: deleted: architecture.xml 변경에 따른 map 삭제
//	private void addMemoryRegionListToProcessor() {
//		for (ArchitectureMapType master : architecture.getMaps().getMaster()) {
//			for (ArchitectureMapSlaveType slave : master.getSlave()) {
//				MemoryRegion memoryRegion = memoryRegionMap.get(slave.getName());
//				if (memoryRegion == null)
//					continue;
//				
//				memoryRegion = memoryRegion.clone();
//				memoryRegion.setBaseAddress(slave.getBaseAddress());
//				
//				// TODO localId가 optional인데 없을 경우 어떻게 처리해야 하나?
//				Processor proc = getProcessor(master.getName(), master.getLocalId());
//				if (proc == null)
//					continue;
//				
//				proc.getMemoryRegionList().add(memoryRegion);
//			}
//		}
//	}

	private void update() {
		for (ArchitectureDeviceType device : architecture.getDevices().getDevice()) {
			for (ArchitectureElementType element : device.getElements().getElement()) {
				ArchitectureElementTypeType elementType = getElementType(ArchitectureElementCategoryType.PROCESSOR,
						element.getType());
				if (elementType == null) // processor가 아니거나 invalid한 경우임
					continue;

				Processor processor = getProcessor(element.getName(), BigInteger.ZERO);
			}
		}
	}
	
	public Processor getProcessor(String name, BigInteger localId) {
		for (Object obj : getProcessorList()) {
			Processor proc = (Processor)obj;
			if (proc.getName().equals(name) &&
					(localId == null || proc.getIndex() == localId.intValue()))
				return proc;
		}
		
		return null;
	}
	
	public String getTarget() {
		return architecture.getTarget();
	}
	
	public void setTarget(String target) {
		architecture.setTarget(target);
	}
	
//	public String getFileName() {
//		return panel.getParentPanel().getNickName() + "_architecture.xml";
//	}
}
