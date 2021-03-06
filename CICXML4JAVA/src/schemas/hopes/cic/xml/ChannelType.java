
package hopes.cic.xml;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlSchemaType;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for ChannelType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="ChannelType"&gt;
 *   &lt;complexContent&gt;
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType"&gt;
 *       &lt;sequence&gt;
 *         &lt;element name="src" type="{http://peace.snu.ac.kr/CICXMLSchema}ChannelPortType" maxOccurs="unbounded"/&gt;
 *         &lt;element name="dst" type="{http://peace.snu.ac.kr/CICXMLSchema}ChannelPortType" maxOccurs="unbounded"/&gt;
 *       &lt;/sequence&gt;
 *       &lt;attribute name="type" type="{http://peace.snu.ac.kr/CICXMLSchema}ChannelTypeType" /&gt;
 *       &lt;attribute name="size" use="required" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" /&gt;
 *       &lt;attribute name="sampleSize" use="required" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" /&gt;
 *       &lt;attribute name="sampleType" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *       &lt;attribute name="initialDataSize" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" /&gt;
 *     &lt;/restriction&gt;
 *   &lt;/complexContent&gt;
 * &lt;/complexType&gt;
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "ChannelType", propOrder = {
    "src",
    "dst"
})
public class ChannelType {

    @XmlElement(required = true)
    protected List<ChannelPortType> src;
    @XmlElement(required = true)
    protected List<ChannelPortType> dst;
    @XmlAttribute(name = "type")
    protected ChannelTypeType type;
    @XmlAttribute(name = "size", required = true)
    @XmlSchemaType(name = "nonNegativeInteger")
    protected BigInteger size;
    @XmlAttribute(name = "sampleSize", required = true)
    @XmlSchemaType(name = "nonNegativeInteger")
    protected BigInteger sampleSize;
    @XmlAttribute(name = "sampleType")
    protected String sampleType;
    @XmlAttribute(name = "initialDataSize")
    @XmlSchemaType(name = "nonNegativeInteger")
    protected BigInteger initialDataSize;

    /**
     * Gets the value of the src property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the src property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getSrc().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link ChannelPortType }
     * 
     * 
     */
    public List<ChannelPortType> getSrc() {
        if (src == null) {
            src = new ArrayList<ChannelPortType>();
        }
        return this.src;
    }

    /**
     * Gets the value of the dst property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the dst property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getDst().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link ChannelPortType }
     * 
     * 
     */
    public List<ChannelPortType> getDst() {
        if (dst == null) {
            dst = new ArrayList<ChannelPortType>();
        }
        return this.dst;
    }

    /**
     * Gets the value of the type property.
     * 
     * @return
     *     possible object is
     *     {@link ChannelTypeType }
     *     
     */
    public ChannelTypeType getType() {
        return type;
    }

    /**
     * Sets the value of the type property.
     * 
     * @param value
     *     allowed object is
     *     {@link ChannelTypeType }
     *     
     */
    public void setType(ChannelTypeType value) {
        this.type = value;
    }

    /**
     * Gets the value of the size property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getSize() {
        return size;
    }

    /**
     * Sets the value of the size property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setSize(BigInteger value) {
        this.size = value;
    }

    /**
     * Gets the value of the sampleSize property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getSampleSize() {
        return sampleSize;
    }

    /**
     * Sets the value of the sampleSize property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setSampleSize(BigInteger value) {
        this.sampleSize = value;
    }

    /**
     * Gets the value of the sampleType property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getSampleType() {
        return sampleType;
    }

    /**
     * Sets the value of the sampleType property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setSampleType(String value) {
        this.sampleType = value;
    }

    /**
     * Gets the value of the initialDataSize property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getInitialDataSize() {
        return initialDataSize;
    }

    /**
     * Sets the value of the initialDataSize property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setInitialDataSize(BigInteger value) {
        this.initialDataSize = value;
    }

}
