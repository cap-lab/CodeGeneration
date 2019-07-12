package hae.peace.container.cic.mapping.xml;

import hae.kernel.util.ObjectList;
import hopes.cic.exception.CICXMLException;

public abstract class CICXMLHandler {
	public abstract void setXMLString(String xmlString) throws CICXMLException;
}