
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
 * <p>Java class for ArchitectureElementTypeType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="ArchitectureElementTypeType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="slavePort" type="{http://peace.snu.ac.kr/CICXMLSchema}ArchitectureElementSlavePortType" maxOccurs="unbounded" minOccurs="0"/>
 *       &lt;/sequence>
 *       &lt;attribute name="OS" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="archiType" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="category" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}ArchitectureElementCategoryType" />
 *       &lt;attribute name="clock" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" />
 *       &lt;attribute name="memorySize" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" />
 *       &lt;attribute name="model" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="nInterruptPorts" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" />
 *       &lt;attribute name="nMasterPorts" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" />
 *       &lt;attribute name="name" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" />
 *       &lt;attribute name="power" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" />
 *       &lt;attribute name="scheduler" type="{http://peace.snu.ac.kr/CICXMLSchema}ArchitectureSchedulerType" />
 *       &lt;attribute name="subcategory" type="{http://www.w3.org/2001/XMLSchema}string" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "ArchitectureElementTypeType", propOrder = {
    "slavePort"
})
public class ArchitectureElementTypeType {

    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema", required = true)
    protected List<ArchitectureElementSlavePortType> slavePort;
    @XmlAttribute(name = "OS")
    protected String os;
    @XmlAttribute
    protected String archiType;
    @XmlAttribute(required = true)
    protected ArchitectureElementCategoryType category;
    @XmlAttribute
    protected BigInteger clock;
    @XmlAttribute
    protected BigInteger memorySize;
    @XmlAttribute
    protected String model;
    @XmlAttribute
    protected BigInteger nInterruptPorts;
    @XmlAttribute
    protected BigInteger nMasterPorts;
    @XmlAttribute(required = true)
    protected String name;
    @XmlAttribute
    protected BigInteger power;
    @XmlAttribute
    protected ArchitectureSchedulerType scheduler;
    @XmlAttribute
    protected String subcategory;

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
     * Gets the value of the power property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getPower() {
        return power;
    }

    /**
     * Sets the value of the power property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setPower(BigInteger value) {
        this.power = value;
    }

    /**
     * Gets the value of the scheduler property.
     * 
     * @return
     *     possible object is
     *     {@link ArchitectureSchedulerType }
     *     
     */
    public ArchitectureSchedulerType getScheduler() {
        return scheduler;
    }

    /**
     * Sets the value of the scheduler property.
     * 
     * @param value
     *     allowed object is
     *     {@link ArchitectureSchedulerType }
     *     
     */
    public void setScheduler(ArchitectureSchedulerType value) {
        this.scheduler = value;
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

}
