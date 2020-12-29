
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
 * <p>Java class for TaskGroupForScheduleType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="TaskGroupForScheduleType"&gt;
 *   &lt;complexContent&gt;
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType"&gt;
 *       &lt;sequence&gt;
 *         &lt;element name="scheduleGroup" type="{http://peace.snu.ac.kr/CICXMLSchema}ScheduleGroupType" maxOccurs="unbounded" minOccurs="0"/&gt;
 *       &lt;/sequence&gt;
 *       &lt;attribute name="name" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" /&gt;
 *       &lt;attribute name="modeName" use="required" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *       &lt;attribute name="initiationInterval" use="required" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" /&gt;
 *       &lt;attribute name="throughput" use="required" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *       &lt;attribute name="latency" use="required" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" /&gt;
 *       &lt;attribute name="modeTransitionDelay" use="required" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" /&gt;
 *     &lt;/restriction&gt;
 *   &lt;/complexContent&gt;
 * &lt;/complexType&gt;
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "TaskGroupForScheduleType", propOrder = {
    "scheduleGroup"
})
public class TaskGroupForScheduleType {

    protected List<ScheduleGroupType> scheduleGroup;
    @XmlAttribute(name = "name", required = true)
    protected String name;
    @XmlAttribute(name = "modeName", required = true)
    protected String modeName;
    @XmlAttribute(name = "initiationInterval", required = true)
    @XmlSchemaType(name = "nonNegativeInteger")
    protected BigInteger initiationInterval;
    @XmlAttribute(name = "throughput", required = true)
    protected String throughput;
    @XmlAttribute(name = "latency", required = true)
    @XmlSchemaType(name = "nonNegativeInteger")
    protected BigInteger latency;
    @XmlAttribute(name = "modeTransitionDelay", required = true)
    @XmlSchemaType(name = "nonNegativeInteger")
    protected BigInteger modeTransitionDelay;

    /**
     * Gets the value of the scheduleGroup property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the scheduleGroup property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getScheduleGroup().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link ScheduleGroupType }
     * 
     * 
     */
    public List<ScheduleGroupType> getScheduleGroup() {
        if (scheduleGroup == null) {
            scheduleGroup = new ArrayList<ScheduleGroupType>();
        }
        return this.scheduleGroup;
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
     * Gets the value of the modeName property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getModeName() {
        return modeName;
    }

    /**
     * Sets the value of the modeName property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setModeName(String value) {
        this.modeName = value;
    }

    /**
     * Gets the value of the initiationInterval property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getInitiationInterval() {
        return initiationInterval;
    }

    /**
     * Sets the value of the initiationInterval property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setInitiationInterval(BigInteger value) {
        this.initiationInterval = value;
    }

    /**
     * Gets the value of the throughput property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getThroughput() {
        return throughput;
    }

    /**
     * Sets the value of the throughput property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setThroughput(String value) {
        this.throughput = value;
    }

    /**
     * Gets the value of the latency property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getLatency() {
        return latency;
    }

    /**
     * Sets the value of the latency property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setLatency(BigInteger value) {
        this.latency = value;
    }

    /**
     * Gets the value of the modeTransitionDelay property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getModeTransitionDelay() {
        return modeTransitionDelay;
    }

    /**
     * Sets the value of the modeTransitionDelay property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setModeTransitionDelay(BigInteger value) {
        this.modeTransitionDelay = value;
    }

}
