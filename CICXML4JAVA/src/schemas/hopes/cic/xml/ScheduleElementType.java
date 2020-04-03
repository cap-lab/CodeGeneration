
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for ScheduleElementType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="ScheduleElementType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;choice>
 *         &lt;element name="loop" type="{http://peace.snu.ac.kr/CICXMLSchema}LoopType" minOccurs="0"/>
 *         &lt;element name="task" type="{http://peace.snu.ac.kr/CICXMLSchema}TaskInstanceType" minOccurs="0"/>
 *       &lt;/choice>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "ScheduleElementType", propOrder = {
    "loop",
    "task"
})
public class ScheduleElementType {

    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema")
    protected LoopType loop;
    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema")
    protected TaskInstanceType task;

    /**
     * Gets the value of the loop property.
     * 
     * @return
     *     possible object is
     *     {@link LoopType }
     *     
     */
    public LoopType getLoop() {
        return loop;
    }

    /**
     * Sets the value of the loop property.
     * 
     * @param value
     *     allowed object is
     *     {@link LoopType }
     *     
     */
    public void setLoop(LoopType value) {
        this.loop = value;
    }

    /**
     * Gets the value of the task property.
     * 
     * @return
     *     possible object is
     *     {@link TaskInstanceType }
     *     
     */
    public TaskInstanceType getTask() {
        return task;
    }

    /**
     * Sets the value of the task property.
     * 
     * @param value
     *     allowed object is
     *     {@link TaskInstanceType }
     *     
     */
    public void setTask(TaskInstanceType value) {
        this.task = value;
    }

}