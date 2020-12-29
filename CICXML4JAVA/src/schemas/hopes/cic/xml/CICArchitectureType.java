
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for CICArchitectureType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="CICArchitectureType"&gt;
 *   &lt;complexContent&gt;
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType"&gt;
 *       &lt;sequence&gt;
 *         &lt;element name="elementTypes" type="{http://peace.snu.ac.kr/CICXMLSchema}ArchitectureElementTypeListType"/&gt;
 *         &lt;element name="elements" type="{http://peace.snu.ac.kr/CICXMLSchema}ArchitectureElementListType" minOccurs="0"/&gt;
 *         &lt;element name="devices" type="{http://peace.snu.ac.kr/CICXMLSchema}ArchitectureDeviceListType" minOccurs="0"/&gt;
 *         &lt;element name="connections" type="{http://peace.snu.ac.kr/CICXMLSchema}ArchitectureConnectionListType"/&gt;
 *       &lt;/sequence&gt;
 *       &lt;attribute name="target" use="required" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *     &lt;/restriction&gt;
 *   &lt;/complexContent&gt;
 * &lt;/complexType&gt;
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "CICArchitectureType", propOrder = {
    "elementTypes",
    "elements",
    "devices",
    "connections"
})
public class CICArchitectureType {

    @XmlElement(required = true)
    protected ArchitectureElementTypeListType elementTypes;
    protected ArchitectureElementListType elements;
    protected ArchitectureDeviceListType devices;
    @XmlElement(required = true)
    protected ArchitectureConnectionListType connections;
    @XmlAttribute(name = "target", required = true)
    protected String target;

    /**
     * Gets the value of the elementTypes property.
     * 
     * @return
     *     possible object is
     *     {@link ArchitectureElementTypeListType }
     *     
     */
    public ArchitectureElementTypeListType getElementTypes() {
        return elementTypes;
    }

    /**
     * Sets the value of the elementTypes property.
     * 
     * @param value
     *     allowed object is
     *     {@link ArchitectureElementTypeListType }
     *     
     */
    public void setElementTypes(ArchitectureElementTypeListType value) {
        this.elementTypes = value;
    }

    /**
     * Gets the value of the elements property.
     * 
     * @return
     *     possible object is
     *     {@link ArchitectureElementListType }
     *     
     */
    public ArchitectureElementListType getElements() {
        return elements;
    }

    /**
     * Sets the value of the elements property.
     * 
     * @param value
     *     allowed object is
     *     {@link ArchitectureElementListType }
     *     
     */
    public void setElements(ArchitectureElementListType value) {
        this.elements = value;
    }

    /**
     * Gets the value of the devices property.
     * 
     * @return
     *     possible object is
     *     {@link ArchitectureDeviceListType }
     *     
     */
    public ArchitectureDeviceListType getDevices() {
        return devices;
    }

    /**
     * Sets the value of the devices property.
     * 
     * @param value
     *     allowed object is
     *     {@link ArchitectureDeviceListType }
     *     
     */
    public void setDevices(ArchitectureDeviceListType value) {
        this.devices = value;
    }

    /**
     * Gets the value of the connections property.
     * 
     * @return
     *     possible object is
     *     {@link ArchitectureConnectionListType }
     *     
     */
    public ArchitectureConnectionListType getConnections() {
        return connections;
    }

    /**
     * Sets the value of the connections property.
     * 
     * @param value
     *     allowed object is
     *     {@link ArchitectureConnectionListType }
     *     
     */
    public void setConnections(ArchitectureConnectionListType value) {
        this.connections = value;
    }

    /**
     * Gets the value of the target property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getTarget() {
        return target;
    }

    /**
     * Sets the value of the target property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setTarget(String value) {
        this.target = value;
    }

}
