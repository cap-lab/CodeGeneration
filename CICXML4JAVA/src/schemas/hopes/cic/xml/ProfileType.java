
package hopes.cic.xml;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlSchemaType;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for ProfileType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="ProfileType"&gt;
 *   &lt;complexContent&gt;
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType"&gt;
 *       &lt;sequence&gt;
 *         &lt;element name="bound" type="{http://peace.snu.ac.kr/CICXMLSchema}ProfileExecutionBoundType" maxOccurs="2" minOccurs="0"/&gt;
 *       &lt;/sequence&gt;
 *       &lt;attribute name="processorType" use="required" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *       &lt;attribute name="value" use="required" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" /&gt;
 *       &lt;attribute name="unit" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}TimeMetricType" /&gt;
 *       &lt;attribute name="memoryAccessCount" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" /&gt;
 *     &lt;/restriction&gt;
 *   &lt;/complexContent&gt;
 * &lt;/complexType&gt;
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "ProfileType", propOrder = {
    "bound"
})
public class ProfileType {

    protected List<ProfileExecutionBoundType> bound;
    @XmlAttribute(name = "processorType", required = true)
    protected String processorType;
    @XmlAttribute(name = "value", required = true)
    @XmlSchemaType(name = "nonNegativeInteger")
    protected BigInteger value;
    @XmlAttribute(name = "unit", required = true)
    protected TimeMetricType unit;
    @XmlAttribute(name = "memoryAccessCount")
    @XmlSchemaType(name = "nonNegativeInteger")
    protected BigInteger memoryAccessCount;

    /**
     * Gets the value of the bound property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the bound property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getBound().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link ProfileExecutionBoundType }
     * 
     * 
     */
    public List<ProfileExecutionBoundType> getBound() {
        if (bound == null) {
            bound = new ArrayList<ProfileExecutionBoundType>();
        }
        return this.bound;
    }

    /**
     * Gets the value of the processorType property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getProcessorType() {
        return processorType;
    }

    /**
     * Sets the value of the processorType property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setProcessorType(String value) {
        this.processorType = value;
    }

    /**
     * Gets the value of the value property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getValue() {
        return value;
    }

    /**
     * Sets the value of the value property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setValue(BigInteger value) {
        this.value = value;
    }

    /**
     * Gets the value of the unit property.
     * 
     * @return
     *     possible object is
     *     {@link TimeMetricType }
     *     
     */
    public TimeMetricType getUnit() {
        return unit;
    }

    /**
     * Sets the value of the unit property.
     * 
     * @param value
     *     allowed object is
     *     {@link TimeMetricType }
     *     
     */
    public void setUnit(TimeMetricType value) {
        this.unit = value;
    }

    /**
     * Gets the value of the memoryAccessCount property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getMemoryAccessCount() {
        return memoryAccessCount;
    }

    /**
     * Sets the value of the memoryAccessCount property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setMemoryAccessCount(BigInteger value) {
        this.memoryAccessCount = value;
    }

}
