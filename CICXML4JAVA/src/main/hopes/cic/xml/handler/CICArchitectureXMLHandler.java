// hs: need to delete before release

package hopes.cic.xml.handler;

import java.io.ByteArrayInputStream;
import java.io.StringWriter;
import java.math.BigInteger;
import java.util.List;

import hopes.cic.exception.CICXMLException;
import hopes.cic.xml.ArchitectureConnectType;
import hopes.cic.xml.ArchitectureDeviceListType;
import hopes.cic.xml.ArchitectureDeviceType;
import hopes.cic.xml.ArchitectureElementCategoryType;
import hopes.cic.xml.ArchitectureElementSlavePortType;
import hopes.cic.xml.ArchitectureElementTypeType;
import hopes.cic.xml.CICArchitectureType;
import hopes.cic.xml.CICArchitectureTypeLoader;

public class CICArchitectureXMLHandler extends CICXMLHandler {
	private CICArchitectureTypeLoader loader;
	private CICArchitectureType architecture;
	public CICArchitectureXMLHandler() {
		loader = new CICArchitectureTypeLoader();
		architecture = new CICArchitectureType();
	}
	
	protected void storeResource(StringWriter writer) throws CICXMLException {
		loader.storeResource(architecture, writer);
	}
	protected void loadResource(ByteArrayInputStream is) throws CICXMLException {
		architecture = loader.loadResource(is);
	}
	
	public void init() {
	}
	
	public CICArchitectureType getArchitecture() {
		return architecture;
	}
	
	public void setArchitecture(CICArchitectureType architecture) {
		this.architecture = architecture;
	}

	public ArchitectureElementTypeType getElementType(ArchitectureElementCategoryType category, String typeName) {
		for (ArchitectureElementTypeType elementType : architecture.getElementTypes().getElementType()) {
			if (category == elementType.getCategory() &&
					typeName.equals(elementType.getName()))
				return elementType;
		}
		return null;
	}
	
	public BigInteger getMemorySize(ArchitectureElementSlavePortType slavePort) {
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
		default:
			break;
		}
		
		return BigInteger.ZERO;
	}

	
	public String getTarget() {
		return architecture.getTarget();
	}
	
	public void setTarget(String target) {
		architecture.setTarget(target);
	}

	public CICArchitectureTypeLoader getLoader() {
		return loader;
	}

	public List<ArchitectureElementTypeType> getElementTypeList() {
		return architecture.getElementTypes().getElementType();
	}

	public List<ArchitectureDeviceType> getDeviceList() {
		return architecture.getDevices().getDevice();
	}

	public List<ArchitectureConnectType> getConnectionList() {
		return architecture.getConnections().getConnection();
	}

}
