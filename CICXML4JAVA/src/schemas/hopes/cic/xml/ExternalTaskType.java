
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for ExternalTaskType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="ExternalTaskType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;attribute name="ParentTask" use="required" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="architectureFile" use="required" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="description" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="mappingFile" use="required" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="name" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" />
 *       &lt;attribute name="networkFile" use="required" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="profileFile" use="required" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="taskType" use="required" type="{http://www.w3.org/2001/XMLSchema}string" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "ExternalTaskType")
public class ExternalTaskType {

    @XmlAttribute(name = "ParentTask", required = true)
    protected String parentTask;
    @XmlAttribute(required = true)
    protected String architectureFile;
    @XmlAttribute
    protected String description;
    @XmlAttribute(required = true)
    protected String mappingFile;
    @XmlAttribute(required = true)
    protected String name;
    @XmlAttribute(required = true)
    protected String networkFile;
    @XmlAttribute(required = true)
    protected String profileFile;
    @XmlAttribute(required = true)
    protected String taskType;

    /**
     * Gets the value of the parentTask property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getParentTask() {
        return parentTask;
    }

    /**
     * Sets the value of the parentTask property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setParentTask(String value) {
        this.parentTask = value;
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
     * Gets the value of the description property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getDescription() {
        return description;
    }

    /**
     * Sets the value of the description property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setDescription(String value) {
        this.description = value;
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
     * Gets the value of the taskType property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getTaskType() {
        return taskType;
    }

    /**
     * Sets the value of the taskType property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setTaskType(String value) {
        this.taskType = value;
    }

}
