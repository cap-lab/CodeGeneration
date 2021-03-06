
package hopes.cic.xml;

import java.math.BigInteger;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlSchemaType;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for ModeTaskGroupType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="ModeTaskGroupType"&gt;
 *   &lt;complexContent&gt;
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType"&gt;
 *       &lt;sequence&gt;
 *         &lt;element name="period" type="{http://peace.snu.ac.kr/CICXMLSchema}TimeType" minOccurs="0"/&gt;
 *         &lt;element name="deadline" type="{http://peace.snu.ac.kr/CICXMLSchema}TimeType" minOccurs="0"/&gt;
 *         &lt;element name="maxInitialInterval" type="{http://peace.snu.ac.kr/CICXMLSchema}TimeType" minOccurs="0"/&gt;
 *       &lt;/sequence&gt;
 *       &lt;attribute name="name" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" /&gt;
 *       &lt;attribute name="priority" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" /&gt;
 *       &lt;attribute name="preemptionType" type="{http://peace.snu.ac.kr/CICXMLSchema}preemptionTypeType" default="nonPreemptive" /&gt;
 *     &lt;/restriction&gt;
 *   &lt;/complexContent&gt;
 * &lt;/complexType&gt;
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "ModeTaskGroupType", propOrder = {
    "period",
    "deadline",
    "maxInitialInterval"
})
public class ModeTaskGroupType {

    protected TimeType period;
    protected TimeType deadline;
    protected TimeType maxInitialInterval;
    @XmlAttribute(name = "name", required = true)
    protected String name;
    @XmlAttribute(name = "priority")
    @XmlSchemaType(name = "nonNegativeInteger")
    protected BigInteger priority;
    @XmlAttribute(name = "preemptionType")
    protected PreemptionTypeType preemptionType;

    /**
     * Gets the value of the period property.
     * 
     * @return
     *     possible object is
     *     {@link TimeType }
     *     
     */
    public TimeType getPeriod() {
        return period;
    }

    /**
     * Sets the value of the period property.
     * 
     * @param value
     *     allowed object is
     *     {@link TimeType }
     *     
     */
    public void setPeriod(TimeType value) {
        this.period = value;
    }

    /**
     * Gets the value of the deadline property.
     * 
     * @return
     *     possible object is
     *     {@link TimeType }
     *     
     */
    public TimeType getDeadline() {
        return deadline;
    }

    /**
     * Sets the value of the deadline property.
     * 
     * @param value
     *     allowed object is
     *     {@link TimeType }
     *     
     */
    public void setDeadline(TimeType value) {
        this.deadline = value;
    }

    /**
     * Gets the value of the maxInitialInterval property.
     * 
     * @return
     *     possible object is
     *     {@link TimeType }
     *     
     */
    public TimeType getMaxInitialInterval() {
        return maxInitialInterval;
    }

    /**
     * Sets the value of the maxInitialInterval property.
     * 
     * @param value
     *     allowed object is
     *     {@link TimeType }
     *     
     */
    public void setMaxInitialInterval(TimeType value) {
        this.maxInitialInterval = value;
    }

    /**
     * Gets the value of the name property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getName() {
        return name;
    }

    /**
     * Sets the value of the name property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setName(String value) {
        this.name = value;
    }

    /**
     * Gets the value of the priority property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getPriority() {
        return priority;
    }

    /**
     * Sets the value of the priority property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setPriority(BigInteger value) {
        this.priority = value;
    }

    /**
     * Gets the value of the preemptionType property.
     * 
     * @return
     *     possible object is
     *     {@link PreemptionTypeType }
     *     
     */
    public PreemptionTypeType getPreemptionType() {
        if (preemptionType == null) {
            return PreemptionTypeType.NON_PREEMPTIVE;
        } else {
            return preemptionType;
        }
    }

    /**
     * Sets the value of the preemptionType property.
     * 
     * @param value
     *     allowed object is
     *     {@link PreemptionTypeType }
     *     
     */
    public void setPreemptionType(PreemptionTypeType value) {
        this.preemptionType = value;
    }

}
