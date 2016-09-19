
package hopes.cic.xml;

import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for ArchitectureConnectionWIFIType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="ArchitectureConnectionWIFIType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="server" type="{http://peace.snu.ac.kr/CICXMLSchema}ArchitectureConnectionWIFIserverType"/>
 *         &lt;element name="client" type="{http://peace.snu.ac.kr/CICXMLSchema}ArchitectureConnectionWIFIclientType" maxOccurs="unbounded"/>
 *       &lt;/sequence>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "ArchitectureConnectionWIFIType", propOrder = {
    "server",
    "client"
})
public class ArchitectureConnectionWIFIType {

    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema", required = true)
    protected ArchitectureConnectionWIFIserverType server;
    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema", required = true)
    protected List<ArchitectureConnectionWIFIclientType> client;

    /**
     * Gets the value of the server property.
     * 
     * @return
     *     possible object is
     *     {@link ArchitectureConnectionWIFIserverType }
     *     
     */
    public ArchitectureConnectionWIFIserverType getServer() {
        return server;
    }

    /**
     * Sets the value of the server property.
     * 
     * @param value
     *     allowed object is
     *     {@link ArchitectureConnectionWIFIserverType }
     *     
     */
    public void setServer(ArchitectureConnectionWIFIserverType value) {
        this.server = value;
    }

    /**
     * Gets the value of the client property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the client property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getClient().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link ArchitectureConnectionWIFIclientType }
     * 
     * 
     */
    public List<ArchitectureConnectionWIFIclientType> getClient() {
        if (client == null) {
            client = new ArrayList<ArchitectureConnectionWIFIclientType>();
        }
        return this.client;
    }

}
