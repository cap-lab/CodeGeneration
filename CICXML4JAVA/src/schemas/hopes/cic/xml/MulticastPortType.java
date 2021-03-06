
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for MulticastPortType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="MulticastPortType"&gt;
 *   &lt;complexContent&gt;
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType"&gt;
 *       &lt;attribute name="name" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" /&gt;
 *       &lt;attribute name="group" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" /&gt;
 *       &lt;attribute name="direction" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}PortDirectionType" /&gt;
 *     &lt;/restriction&gt;
 *   &lt;/complexContent&gt;
 * &lt;/complexType&gt;
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "MulticastPortType")
public class MulticastPortType {

    @XmlAttribute(name = "name", required = true)
    protected String name;
    @XmlAttribute(name = "group", required = true)
    protected String group;
    @XmlAttribute(name = "direction", required = true)
    protected PortDirectionType direction;

    /**
     * Gets the value of the name property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getName() {
        return name;
    }

    /**
     * Sets the value of the name property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setName(String value) {
        this.name = value;
    }

    /**
     * Gets the value of the group property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getGroup() {
        return group;
    }

    /**
     * Sets the value of the group property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setGroup(String value) {
        this.group = value;
    }

    /**
     * Gets the value of the direction property.
     * 
     * @return
     *     possible object is
     *     {@link PortDirectionType }
     *     
     */
    public PortDirectionType getDirection() {
        return direction;
    }

    /**
     * Sets the value of the direction property.
     * 
     * @param value
     *     allowed object is
     *     {@link PortDirectionType }
     *     
     */
    public void setDirection(PortDirectionType value) {
        this.direction = value;
    }

}
