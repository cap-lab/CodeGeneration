
package hopes.cic.xml;

import java.math.BigDecimal;
import java.math.BigInteger;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlSchemaType;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for ArchitectureElementClockType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="ArchitectureElementClockType"&gt;
 *   &lt;complexContent&gt;
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType"&gt;
 *       &lt;attribute name="clock" use="required" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" /&gt;
 *       &lt;attribute name="activePower" use="required" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" /&gt;
 *       &lt;attribute name="sleepPower" use="required" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" /&gt;
 *       &lt;attribute name="relativeCost" use="required" type="{http://www.w3.org/2001/XMLSchema}decimal" /&gt;
 *     &lt;/restriction&gt;
 *   &lt;/complexContent&gt;
 * &lt;/complexType&gt;
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "ArchitectureElementClockType")
public class ArchitectureElementClockType {

    @XmlAttribute(name = "clock", required = true)
    @XmlSchemaType(name = "nonNegativeInteger")
    protected BigInteger clock;
    @XmlAttribute(name = "activePower", required = true)
    @XmlSchemaType(name = "nonNegativeInteger")
    protected BigInteger activePower;
    @XmlAttribute(name = "sleepPower", required = true)
    @XmlSchemaType(name = "nonNegativeInteger")
    protected BigInteger sleepPower;
    @XmlAttribute(name = "relativeCost", required = true)
    protected BigDecimal relativeCost;

    /**
     * Gets the value of the clock property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getClock() {
        return clock;
    }

    /**
     * Sets the value of the clock property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setClock(BigInteger value) {
        this.clock = value;
    }

    /**
     * Gets the value of the activePower property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getActivePower() {
        return activePower;
    }

    /**
     * Sets the value of the activePower property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setActivePower(BigInteger value) {
        this.activePower = value;
    }

    /**
     * Gets the value of the sleepPower property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getSleepPower() {
        return sleepPower;
    }

    /**
     * Sets the value of the sleepPower property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setSleepPower(BigInteger value) {
        this.sleepPower = value;
    }

    /**
     * Gets the value of the relativeCost property.
     * 
     * @return
     *     possible object is
     *     {@link BigDecimal }
     *     
     */
    public BigDecimal getRelativeCost() {
        return relativeCost;
    }

    /**
     * Sets the value of the relativeCost property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigDecimal }
     *     
     */
    public void setRelativeCost(BigDecimal value) {
        this.relativeCost = value;
    }

}
