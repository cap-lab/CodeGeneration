
package hopes.cic.xml;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlSchemaType;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for ArchitectureElementTypeType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="ArchitectureElementTypeType"&gt;
 *   &lt;complexContent&gt;
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType"&gt;
 *       &lt;sequence&gt;
 *         &lt;element name="slavePort" type="{http://peace.snu.ac.kr/CICXMLSchema}ArchitectureElementSlavePortType" maxOccurs="unbounded" minOccurs="0"/&gt;
 *         &lt;element name="clockType" type="{http://peace.snu.ac.kr/CICXMLSchema}ArchitectureElementClockType" maxOccurs="unbounded" minOccurs="0"/&gt;
 *       &lt;/sequence&gt;
 *       &lt;attribute name="name" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" /&gt;
 *       &lt;attribute name="category" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}ArchitectureElementCategoryType" /&gt;
 *       &lt;attribute name="subcategory" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *       &lt;attribute name="model" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *       &lt;attribute name="OS" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *       &lt;attribute name="scheduler" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *       &lt;attribute name="clock" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" /&gt;
 *       &lt;attribute name="relativeCost" type="{http://www.w3.org/2001/XMLSchema}decimal" /&gt;
 *       &lt;attribute name="archiType" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *       &lt;attribute name="activePower" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" /&gt;
 *       &lt;attribute name="sleepPower" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" /&gt;
 *       &lt;attribute name="nMasterPorts" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" /&gt;
 *       &lt;attribute name="nInterruptPorts" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" /&gt;
 *       &lt;attribute name="memorySize" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" /&gt;
 *     &lt;/restriction&gt;
 *   &lt;/complexContent&gt;
 * &lt;/complexType&gt;
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "ArchitectureElementTypeType", propOrder = {
    "slavePort",
    "clockType"
})
public class ArchitectureElementTypeType {

    protected List<ArchitectureElementSlavePortType> slavePort;
    protected List<ArchitectureElementClockType> clockType;
    @XmlAttribute(name = "name", required = true)
    protected String name;
    @XmlAttribute(name = "category", required = true)
    protected ArchitectureElementCategoryType category;
    @XmlAttribute(name = "subcategory")
    protected String subcategory;
    @XmlAttribute(name = "model")
    protected String model;
    @XmlAttribute(name = "OS")
    protected String os;
    @XmlAttribute(name = "scheduler")
    protected String scheduler;
    @XmlAttribute(name = "clock")
    @XmlSchemaType(name = "nonNegativeInteger")
    protected BigInteger clock;
    @XmlAttribute(name = "relativeCost")
    protected BigDecimal relativeCost;
    @XmlAttribute(name = "archiType")
    protected String archiType;
    @XmlAttribute(name = "activePower")
    @XmlSchemaType(name = "nonNegativeInteger")
    protected BigInteger activePower;
    @XmlAttribute(name = "sleepPower")
    @XmlSchemaType(name = "nonNegativeInteger")
    protected BigInteger sleepPower;
    @XmlAttribute(name = "nMasterPorts")
    @XmlSchemaType(name = "nonNegativeInteger")
    protected BigInteger nMasterPorts;
    @XmlAttribute(name = "nInterruptPorts")
    @XmlSchemaType(name = "nonNegativeInteger")
    protected BigInteger nInterruptPorts;
    @XmlAttribute(name = "memorySize")
    @XmlSchemaType(name = "nonNegativeInteger")
    protected BigInteger memorySize;

    /**
     * Gets the value of the slavePort property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the slavePort property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getSlavePort().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link ArchitectureElementSlavePortType }
     * 
     * 
     */
    public List<ArchitectureElementSlavePortType> getSlavePort() {
        if (slavePort == null) {
            slavePort = new ArrayList<ArchitectureElementSlavePortType>();
        }
        return this.slavePort;
    }

    /**
     * Gets the value of the clockType property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the clockType property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getClockType().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link ArchitectureElementClockType }
     * 
     * 
     */
    public List<ArchitectureElementClockType> getClockType() {
        if (clockType == null) {
            clockType = new ArrayList<ArchitectureElementClockType>();
        }
        return this.clockType;
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
     * Gets the value of the category property.
     * 
     * @return
     *     possible object is
     *     {@link ArchitectureElementCategoryType }
     *     
     */
    public ArchitectureElementCategoryType getCategory() {
        return category;
    }

    /**
     * Sets the value of the category property.
     * 
     * @param value
     *     allowed object is
     *     {@link ArchitectureElementCategoryType }
     *     
     */
    public void setCategory(ArchitectureElementCategoryType value) {
        this.category = value;
    }

    /**
     * Gets the value of the subcategory property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getSubcategory() {
        return subcategory;
    }

    /**
     * Sets the value of the subcategory property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setSubcategory(String value) {
        this.subcategory = value;
    }

    /**
     * Gets the value of the model property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getModel() {
        return model;
    }

    /**
     * Sets the value of the model property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setModel(String value) {
        this.model = value;
    }

    /**
     * Gets the value of the os property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getOS() {
        return os;
    }

    /**
     * Sets the value of the os property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setOS(String value) {
        this.os = value;
    }

    /**
     * Gets the value of the scheduler property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getScheduler() {
        return scheduler;
    }

    /**
     * Sets the value of the scheduler property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setScheduler(String value) {
        this.scheduler = value;
    }

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

    /**
     * Gets the value of the archiType property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getArchiType() {
        return archiType;
    }

    /**
     * Sets the value of the archiType property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setArchiType(String value) {
        this.archiType = value;
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
     * Gets the value of the nMasterPorts property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getNMasterPorts() {
        return nMasterPorts;
    }

    /**
     * Sets the value of the nMasterPorts property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setNMasterPorts(BigInteger value) {
        this.nMasterPorts = value;
    }

    /**
     * Gets the value of the nInterruptPorts property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getNInterruptPorts() {
        return nInterruptPorts;
    }

    /**
     * Sets the value of the nInterruptPorts property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setNInterruptPorts(BigInteger value) {
        this.nInterruptPorts = value;
    }

    /**
     * Gets the value of the memorySize property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getMemorySize() {
        return memorySize;
    }

    /**
     * Sets the value of the memorySize property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setMemorySize(BigInteger value) {
        this.memorySize = value;
    }

}
