
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for ProfileCommType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="ProfileCommType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;attribute name="constant" use="required" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="dst" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" />
 *       &lt;attribute name="firstPowerCoef" use="required" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="secondPowerCoef" use="required" type="{http://www.w3.org/2001/XMLSchema}string" />
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
@XmlType(name = "ProfileCommType")
public class ProfileCommType {

    @XmlAttribute(required = true)
    protected String constant;
    @XmlAttribute(required = true)
    protected String dst;
    @XmlAttribute(required = true)
    protected String firstPowerCoef;
    @XmlAttribute(required = true)
    protected String secondPowerCoef;
    @XmlAttribute(required = true)
    protected String src;
    @XmlAttribute(required = true)
    protected TimeMetricType timeunit;

    /**
     * Gets the value of the constant property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getConstant() {
        return constant;
    }

    /**
     * Sets the value of the constant property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setConstant(String value) {
        this.constant = value;
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
     * Gets the value of the firstPowerCoef property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getFirstPowerCoef() {
        return firstPowerCoef;
    }

    /**
     * Sets the value of the firstPowerCoef property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setFirstPowerCoef(String value) {
        this.firstPowerCoef = value;
    }

    /**
     * Gets the value of the secondPowerCoef property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getSecondPowerCoef() {
        return secondPowerCoef;
    }

    /**
     * Sets the value of the secondPowerCoef property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setSecondPowerCoef(String value) {
        this.secondPowerCoef = value;
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
