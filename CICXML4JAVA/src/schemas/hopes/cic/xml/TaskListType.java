
package hopes.cic.xml;

import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for TaskListType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="TaskListType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="task" type="{http://peace.snu.ac.kr/CICXMLSchema}TaskType" maxOccurs="unbounded"/>
 *         &lt;element name="externalTask" type="{http://peace.snu.ac.kr/CICXMLSchema}ExternalTaskType" maxOccurs="unbounded" minOccurs="0"/>
 *       &lt;/sequence>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "TaskListType", propOrder = {
    "task",
    "externalTask"
})
public class TaskListType {

    @XmlElement(required = true)
    protected List<TaskType> task;
    protected List<ExternalTaskType> externalTask;

    /**
     * Gets the value of the task property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the task property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getTask().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link TaskType }
     * 
     * 
     */
    public List<TaskType> getTask() {
        if (task == null) {
            task = new ArrayList<TaskType>();
        }
        return this.task;
    }

    /**
     * Gets the value of the externalTask property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the externalTask property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getExternalTask().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link ExternalTaskType }
     * 
     * 
     */
    public List<ExternalTaskType> getExternalTask() {
        if (externalTask == null) {
            externalTask = new ArrayList<ExternalTaskType>();
        }
        return this.externalTask;
    }

}
