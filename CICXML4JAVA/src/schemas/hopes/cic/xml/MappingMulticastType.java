
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for MappingMulticastType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="MappingMulticastType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="connectionType" type="{http://peace.snu.ac.kr/CICXMLSchema}MappingMulticastConnectionType" minOccurs="0"/>
 *       &lt;/sequence>
 *       &lt;attribute name="groupName" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "MappingMulticastType", propOrder = {
    "connectionType"
})
public class MappingMulticastType {

    protected MappingMulticastConnectionType connectionType;
    @XmlAttribute(required = true)
    protected String groupName;

    /**
     * Gets the value of the connectionType property.
     * 
     * @return
     *     possible object is
     *     {@link MappingMulticastConnectionType }
     *     
     */
    public MappingMulticastConnectionType getConnectionType() {
        return connectionType;
    }

    /**
     * Sets the value of the connectionType property.
     * 
     * @param value
     *     allowed object is
     *     {@link MappingMulticastConnectionType }
     *     
     */
    public void setConnectionType(MappingMulticastConnectionType value) {
        this.connectionType = value;
    }

    /**
     * Gets the value of the groupName property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getGroupName() {
        return groupName;
    }

    /**
     * Sets the value of the groupName property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setGroupName(String value) {
        this.groupName = value;
    }

}
