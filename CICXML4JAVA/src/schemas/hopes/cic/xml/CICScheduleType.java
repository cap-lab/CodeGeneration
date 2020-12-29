
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
 * &lt;complexType name="CICScheduleType"&gt;
 *   &lt;complexContent&gt;
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType"&gt;
 *       &lt;sequence&gt;
 *         &lt;element name="taskGroups" type="{http://peace.snu.ac.kr/CICXMLSchema}TaskGroupsType"/&gt;
 *       &lt;/sequence&gt;
 *       &lt;attribute name="type" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" /&gt;
 *     &lt;/restriction&gt;
 *   &lt;/complexContent&gt;
 * &lt;/complexType&gt;
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "CICScheduleType", propOrder = {
    "taskGroups"
})
public class CICScheduleType {

    @XmlElement(required = true)
    protected TaskGroupsType taskGroups;
    @XmlAttribute(name = "type", required = true)
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
