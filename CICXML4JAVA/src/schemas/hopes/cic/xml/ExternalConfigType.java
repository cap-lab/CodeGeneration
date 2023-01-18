
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for ExternalConfigType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="ExternalConfigType"&gt;
 *   &lt;complexContent&gt;
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType"&gt;
 *       &lt;attribute name="networkFile" use="required" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *       &lt;attribute name="architectureFile" use="required" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *       &lt;attribute name="profileFile" use="required" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *       &lt;attribute name="mappingFile" use="required" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *       &lt;attribute name="useCICFile" use="required" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *     &lt;/restriction&gt;
 *   &lt;/complexContent&gt;
 * &lt;/complexType&gt;
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "ExternalConfigType")
public class ExternalConfigType {

    @XmlAttribute(name = "networkFile", required = true)
    protected String networkFile;
    @XmlAttribute(name = "architectureFile", required = true)
    protected String architectureFile;
    @XmlAttribute(name = "profileFile", required = true)
    protected String profileFile;
    @XmlAttribute(name = "mappingFile", required = true)
    protected String mappingFile;
    @XmlAttribute(name = "useCICFile", required = true)
    protected String useCICFile;

    /**
     * Gets the value of the networkFile property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getNetworkFile() {
        return networkFile;
    }

    /**
     * Sets the value of the networkFile property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setNetworkFile(String value) {
        this.networkFile = value;
    }

    /**
     * Gets the value of the architectureFile property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getArchitectureFile() {
        return architectureFile;
    }

    /**
     * Sets the value of the architectureFile property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setArchitectureFile(String value) {
        this.architectureFile = value;
    }

    /**
     * Gets the value of the profileFile property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getProfileFile() {
        return profileFile;
    }

    /**
     * Sets the value of the profileFile property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setProfileFile(String value) {
        this.profileFile = value;
    }

    /**
     * Gets the value of the mappingFile property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getMappingFile() {
        return mappingFile;
    }

    /**
     * Sets the value of the mappingFile property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setMappingFile(String value) {
        this.mappingFile = value;
    }

    /**
     * Gets the value of the useCICFile property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getUseCICFile() {
        return useCICFile;
    }

    /**
     * Sets the value of the useCICFile property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setUseCICFile(String value) {
        this.useCICFile = value;
    }

}
