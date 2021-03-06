
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for MappingMulticastConnectionType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="MappingMulticastConnectionType"&gt;
 *   &lt;complexContent&gt;
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType"&gt;
 *       &lt;sequence&gt;
 *         &lt;element name="UDP" type="{http://peace.snu.ac.kr/CICXMLSchema}MappingMulticastUDPType" minOccurs="0"/&gt;
 *       &lt;/sequence&gt;
 *     &lt;/restriction&gt;
 *   &lt;/complexContent&gt;
 * &lt;/complexType&gt;
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "MappingMulticastConnectionType", propOrder = {
    "udp"
})
public class MappingMulticastConnectionType {

    @XmlElement(name = "UDP")
    protected MappingMulticastUDPType udp;

    /**
     * Gets the value of the udp property.
     * 
     * @return
     *     possible object is
     *     {@link MappingMulticastUDPType }
     *     
     */
    public MappingMulticastUDPType getUDP() {
        return udp;
    }

    /**
     * Sets the value of the udp property.
     * 
     * @param value
     *     allowed object is
     *     {@link MappingMulticastUDPType }
     *     
     */
    public void setUDP(MappingMulticastUDPType value) {
        this.udp = value;
    }

}
