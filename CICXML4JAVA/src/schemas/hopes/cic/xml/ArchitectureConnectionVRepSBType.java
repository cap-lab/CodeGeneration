
package hopes.cic.xml;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for ArchitectureConnectionVRepSBType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="ArchitectureConnectionVRepSBType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="master" type="{http://peace.snu.ac.kr/CICXMLSchema}ArchitectureConnectionVRepSBMemberType"/>
 *         &lt;element name="slave" type="{http://peace.snu.ac.kr/CICXMLSchema}ArchitectureConnectionVRepSBMemberType" maxOccurs="unbounded"/>
 *       &lt;/sequence>
 *       &lt;attribute name="key" use="required" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "ArchitectureConnectionVRepSBType", propOrder = {
    "master",
    "slave"
})
public class ArchitectureConnectionVRepSBType {

    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema", required = true)
    protected ArchitectureConnectionVRepSBMemberType master;
    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema", required = true)
    protected List<ArchitectureConnectionVRepSBMemberType> slave;
    @XmlAttribute(required = true)
    protected BigInteger key;

    /**
     * Gets the value of the master property.
     * 
     * @return
     *     possible object is
     *     {@link ArchitectureConnectionVRepSBMemberType }
     *     
     */
    public ArchitectureConnectionVRepSBMemberType getMaster() {
        return master;
    }

    /**
     * Sets the value of the master property.
     * 
     * @param value
     *     allowed object is
     *     {@link ArchitectureConnectionVRepSBMemberType }
     *     
     */
    public void setMaster(ArchitectureConnectionVRepSBMemberType value) {
        this.master = value;
    }

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
     * {@link ArchitectureConnectionVRepSBMemberType }
     * 
     * 
     */
    public List<ArchitectureConnectionVRepSBMemberType> getSlave() {
        if (slave == null) {
            slave = new ArrayList<ArchitectureConnectionVRepSBMemberType>();
        }
        return this.slave;
    }

    /**
     * Gets the value of the key property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getKey() {
        return key;
    }

    /**
     * Sets the value of the key property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setKey(BigInteger value) {
        this.key = value;
    }

}
