
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for ArchitectureDeviceType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="ArchitectureDeviceType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="elements" type="{http://peace.snu.ac.kr/CICXMLSchema}ArchitectureElementListType"/>
 *         &lt;element name="connections" type="{http://peace.snu.ac.kr/CICXMLSchema}DeviceConnectionListType" minOccurs="0"/>
 *         &lt;element name="modules" type="{http://peace.snu.ac.kr/CICXMLSchema}ModuleListType" minOccurs="0"/>
 *         &lt;element name="environmentVariables" type="{http://peace.snu.ac.kr/CICXMLSchema}EnvironmentVariableListType" minOccurs="0"/>
 *       &lt;/sequence>
 *       &lt;attribute name="architecture" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" />
 *       &lt;attribute name="name" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" />
 *       &lt;attribute name="platform" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" />
 *       &lt;attribute name="runtime" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "ArchitectureDeviceType", propOrder = {
    "elements",
    "connections",
    "modules",
    "environmentVariables"
})
public class ArchitectureDeviceType {

    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema", required = true)
    protected ArchitectureElementListType elements;
    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema")
    protected DeviceConnectionListType connections;
    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema")
    protected ModuleListType modules;
    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema")
    protected EnvironmentVariableListType environmentVariables;
    @XmlAttribute(required = true)
    protected String architecture;
    @XmlAttribute(required = true)
    protected String name;
    @XmlAttribute(required = true)
    protected String platform;
    @XmlAttribute(required = true)
    protected String runtime;

    /**
     * Gets the value of the elements property.
     * 
     * @return
     *     possible object is
     *     {@link ArchitectureElementListType }
     *     
     */
    public ArchitectureElementListType getElements() {
        return elements;
    }

    /**
     * Sets the value of the elements property.
     * 
     * @param value
     *     allowed object is
     *     {@link ArchitectureElementListType }
     *     
     */
    public void setElements(ArchitectureElementListType value) {
        this.elements = value;
    }

    /**
     * Gets the value of the connections property.
     * 
     * @return
     *     possible object is
     *     {@link DeviceConnectionListType }
     *     
     */
    public DeviceConnectionListType getConnections() {
        return connections;
    }

    /**
     * Sets the value of the connections property.
     * 
     * @param value
     *     allowed object is
     *     {@link DeviceConnectionListType }
     *     
     */
    public void setConnections(DeviceConnectionListType value) {
        this.connections = value;
    }

    /**
     * Gets the value of the modules property.
     * 
     * @return
     *     possible object is
     *     {@link ModuleListType }
     *     
     */
    public ModuleListType getModules() {
        return modules;
    }

    /**
     * Sets the value of the modules property.
     * 
     * @param value
     *     allowed object is
     *     {@link ModuleListType }
     *     
     */
    public void setModules(ModuleListType value) {
        this.modules = value;
    }

    /**
     * Gets the value of the environmentVariables property.
     * 
     * @return
     *     possible object is
     *     {@link EnvironmentVariableListType }
     *     
     */
    public EnvironmentVariableListType getEnvironmentVariables() {
        return environmentVariables;
    }

    /**
     * Sets the value of the environmentVariables property.
     * 
     * @param value
     *     allowed object is
     *     {@link EnvironmentVariableListType }
     *     
     */
    public void setEnvironmentVariables(EnvironmentVariableListType value) {
        this.environmentVariables = value;
    }

    /**
     * Gets the value of the architecture property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getArchitecture() {
        return architecture;
    }

    /**
     * Sets the value of the architecture property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setArchitecture(String value) {
        this.architecture = value;
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
     * Gets the value of the platform property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getPlatform() {
        return platform;
    }

    /**
     * Sets the value of the platform property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setPlatform(String value) {
        this.platform = value;
    }

    /**
     * Gets the value of the runtime property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getRuntime() {
        return runtime;
    }

    /**
     * Sets the value of the runtime property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setRuntime(String value) {
        this.runtime = value;
    }

}
