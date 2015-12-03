
package hopes.cic.xml;

import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for ArchitectureConnectionI2CType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="ArchitectureConnectionI2CType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="master" type="{http://peace.snu.ac.kr/CICXMLSchema}ArchitectureConnectionI2CMasterType"/>
 *         &lt;element name="slave" type="{http://peace.snu.ac.kr/CICXMLSchema}ArchitectureConnectionI2CSlaveType" maxOccurs="unbounded"/>
 *       &lt;/sequence>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "ArchitectureConnectionI2CType", propOrder = {
    "master",
    "slave"
})
public class ArchitectureConnectionI2CType {

    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema", required = true)
    protected ArchitectureConnectionI2CMasterType master;
    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema", required = true)
    protected List<ArchitectureConnectionI2CSlaveType> slave;

    /**
     * Gets the value of the master property.
     * 
     * @return
     *     possible object is
     *     {@link ArchitectureConnectionI2CMasterType }
     *     
     */
    public ArchitectureConnectionI2CMasterType getMaster() {
        return master;
    }

    /**
     * Sets the value of the master property.
     * 
     * @param value
     *     allowed object is
     *     {@link ArchitectureConnectionI2CMasterType }
     *     
     */
    public void setMaster(ArchitectureConnectionI2CMasterType value) {
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
     * {@link ArchitectureConnectionI2CSlaveType }
     * 
     * 
     */
    public List<ArchitectureConnectionI2CSlaveType> getSlave() {
        if (slave == null) {
            slave = new ArrayList<ArchitectureConnectionI2CSlaveType>();
        }
        return this.slave;
    }

}
