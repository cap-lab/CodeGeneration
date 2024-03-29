
package hopes.cic.xml;

import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for ArchitectureConnectType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="ArchitectureConnectType"&gt;
 *   &lt;complexContent&gt;
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType"&gt;
 *       &lt;sequence&gt;
 *         &lt;element name="slave" type="{http://peace.snu.ac.kr/CICXMLSchema}ArchitectureConnectionSlaveType" maxOccurs="unbounded"/&gt;
 *       &lt;/sequence&gt;
 *       &lt;attribute name="master" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" /&gt;
 *       &lt;attribute name="connection" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" /&gt;
 *       &lt;attribute name="encryption" type="{http://peace.snu.ac.kr/CICXMLSchema}EncryptionType" /&gt;
 *       &lt;attribute name="userkey" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *     &lt;/restriction&gt;
 *   &lt;/complexContent&gt;
 * &lt;/complexType&gt;
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "ArchitectureConnectType", propOrder = {
    "slave"
})
public class ArchitectureConnectType {

    @XmlElement(required = true)
    protected List<ArchitectureConnectionSlaveType> slave;
    @XmlAttribute(name = "master", required = true)
    protected String master;
    @XmlAttribute(name = "connection", required = true)
    protected String connection;
    @XmlAttribute(name = "encryption")
    protected EncryptionType encryption;
    @XmlAttribute(name = "userkey")
    protected String userkey;

    /**
     * Gets the value of the slave property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the slave property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getSlave().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link ArchitectureConnectionSlaveType }
     * 
     * 
     */
    public List<ArchitectureConnectionSlaveType> getSlave() {
        if (slave == null) {
            slave = new ArrayList<ArchitectureConnectionSlaveType>();
        }
        return this.slave;
    }

    /**
     * Gets the value of the master property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getMaster() {
        return master;
    }

    /**
     * Sets the value of the master property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setMaster(String value) {
        this.master = value;
    }

    /**
     * Gets the value of the connection property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getConnection() {
        return connection;
    }

    /**
     * Sets the value of the connection property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setConnection(String value) {
        this.connection = value;
    }

    /**
     * Gets the value of the encryption property.
     * 
     * @return
     *     possible object is
     *     {@link EncryptionType }
     *     
     */
    public EncryptionType getEncryption() {
        return encryption;
    }

    /**
     * Sets the value of the encryption property.
     * 
     * @param value
     *     allowed object is
     *     {@link EncryptionType }
     *     
     */
    public void setEncryption(EncryptionType value) {
        this.encryption = value;
    }

    /**
     * Gets the value of the userkey property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getUserkey() {
        return userkey;
    }

    /**
     * Sets the value of the userkey property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setUserkey(String value) {
        this.userkey = value;
    }

}
