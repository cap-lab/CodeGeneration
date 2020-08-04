
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for CICControlType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="CICControlType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="controlTasks" type="{http://peace.snu.ac.kr/CICXMLSchema}ControlTaskListType" minOccurs="0"/>
 *         &lt;element name="exclusiveControlTasksList" type="{http://peace.snu.ac.kr/CICXMLSchema}ExclusiveControlTasksListType" minOccurs="0"/>
 *       &lt;/sequence>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "CICControlType", propOrder = {
    "controlTasks",
    "exclusiveControlTasksList"
})
public class CICControlType {

    protected ControlTaskListType controlTasks;
    protected ExclusiveControlTasksListType exclusiveControlTasksList;

    /**
     * Gets the value of the controlTasks property.
     * 
     * @return
     *     possible object is
     *     {@link ControlTaskListType }
     *     
     */
    public ControlTaskListType getControlTasks() {
        return controlTasks;
    }

    /**
     * Sets the value of the controlTasks property.
     * 
     * @param value
     *     allowed object is
     *     {@link ControlTaskListType }
     *     
     */
    public void setControlTasks(ControlTaskListType value) {
        this.controlTasks = value;
    }

    /**
     * Gets the value of the exclusiveControlTasksList property.
     * 
     * @return
     *     possible object is
     *     {@link ExclusiveControlTasksListType }
     *     
     */
    public ExclusiveControlTasksListType getExclusiveControlTasksList() {
        return exclusiveControlTasksList;
    }

    /**
     * Sets the value of the exclusiveControlTasksList property.
     * 
     * @param value
     *     allowed object is
     *     {@link ExclusiveControlTasksListType }
     *     
     */
    public void setExclusiveControlTasksList(ExclusiveControlTasksListType value) {
        this.exclusiveControlTasksList = value;
    }

}
