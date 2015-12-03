
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
 * &lt;complexType name="CICArchitectureType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="elementTypes" type="{http://peace.snu.ac.kr/CICXMLSchema}ArchitectureElementTypeListType"/>
 *         &lt;element name="elements" type="{http://peace.snu.ac.kr/CICXMLSchema}ArchitectureElementListType"/>
 *         &lt;element name="connections" type="{http://peace.snu.ac.kr/CICXMLSchema}ArchitectureConnectionListType"/>
 *       &lt;/sequence>
 *       &lt;attribute name="target" use="required" type="{http://www.w3.org/2001/XMLSchema}string" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "CICArchitectureType", propOrder = {
    "elementTypes",
    "elements",
    "connections"
})
public class CICArchitectureType {

    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema", required = true)
    protected ArchitectureElementTypeListType elementTypes;
    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema", required = true)
    protected ArchitectureElementListType elements;
    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema", required = true)
    protected ArchitectureConnectionListType connections;
    @XmlAttribute(required = true)
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
