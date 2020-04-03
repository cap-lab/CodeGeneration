
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for CICScheduleType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="CICScheduleType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="taskGroups" type="{http://peace.snu.ac.kr/CICXMLSchema}TaskGroupsType"/>
 *       &lt;/sequence>
 *       &lt;attribute name="type" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "CICScheduleType", propOrder = {
    "taskGroups"
})
public class CICScheduleType {

    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema", required = true)
    protected TaskGroupsType taskGroups;
    @XmlAttribute(required = true)
    protected String type;

    /**
     * Gets the value of the taskGroups property.
     * 
     * @return
     *     possible object is
     *     {@link TaskGroupsType }
     *     
     */
    public TaskGroupsType getTaskGroups() {
        return taskGroups;
    }

    /**
     * Sets the value of the taskGroups property.
     * 
     * @param value
     *     allowed object is
     *     {@link TaskGroupsType }
     *     
     */
    public void setTaskGroups(TaskGroupsType value) {
        this.taskGroups = value;
    }

    /**
     * Gets the value of the type property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getType() {
        return type;
    }

    /**
     * Sets the value of the type property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setType(String value) {
        this.type = value;
    }

}
