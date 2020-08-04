
package hopes.cic.xml;

import java.math.BigInteger;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for ProfileMigrationType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="ProfileMigrationType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;attribute name="cost" use="required" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" />
 *       &lt;attribute name="dst" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" />
 *       &lt;attribute name="settingcost" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" />
 *       &lt;attribute name="size" use="required" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" />
 *       &lt;attribute name="sizeunit" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}SizeMetricType" />
 *       &lt;attribute name="src" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" />
 *       &lt;attribute name="timeunit" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}TimeMetricType" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "ProfileMigrationType")
public class ProfileMigrationType {

    @XmlAttribute(required = true)
    protected BigInteger cost;
    @XmlAttribute(required = true)
    protected String dst;
    @XmlAttribute
    protected BigInteger settingcost;
    @XmlAttribute(required = true)
    protected BigInteger size;
    @XmlAttribute(required = true)
    protected SizeMetricType sizeunit;
    @XmlAttribute(required = true)
    protected String src;
    @XmlAttribute(required = true)
    protected TimeMetricType timeunit;

    /**
     * Gets the value of the cost property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getCost() {
        return cost;
    }

    /**
     * Sets the value of the cost property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setCost(BigInteger value) {
        this.cost = value;
    }

    /**
     * Gets the value of the dst property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getDst() {
        return dst;
    }

    /**
     * Sets the value of the dst property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setDst(String value) {
        this.dst = value;
    }

    /**
     * Gets the value of the settingcost property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getSettingcost() {
        return settingcost;
    }

    /**
     * Sets the value of the settingcost property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setSettingcost(BigInteger value) {
        this.settingcost = value;
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
     * Gets the value of the sizeunit property.
     * 
     * @return
     *     possible object is
     *     {@link SizeMetricType }
     *     
     */
    public SizeMetricType getSizeunit() {
        return sizeunit;
    }

    /**
     * Sets the value of the sizeunit property.
     * 
     * @param value
     *     allowed object is
     *     {@link SizeMetricType }
     *     
     */
    public void setSizeunit(SizeMetricType value) {
        this.sizeunit = value;
    }

    /**
     * Gets the value of the src property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getSrc() {
        return src;
    }

    /**
     * Sets the value of the src property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setSrc(String value) {
        this.src = value;
    }

    /**
     * Gets the value of the timeunit property.
     * 
     * @return
     *     possible object is
     *     {@link TimeMetricType }
     *     
     */
    public TimeMetricType getTimeunit() {
        return timeunit;
    }

    /**
     * Sets the value of the timeunit property.
     * 
     * @param value
     *     allowed object is
     *     {@link TimeMetricType }
     *     
     */
    public void setTimeunit(TimeMetricType value) {
        this.timeunit = value;
    }

}
