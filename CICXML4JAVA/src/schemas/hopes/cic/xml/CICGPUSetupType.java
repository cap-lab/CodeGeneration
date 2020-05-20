
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for CICGPUSetupType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="CICGPUSetupType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="tasks" type="{http://peace.snu.ac.kr/CICXMLSchema}GPUTaskListType" minOccurs="0"/>
 *       &lt;/sequence>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "CICGPUSetupType", propOrder = {
    "tasks"
})
public class CICGPUSetupType {

    protected GPUTaskListType tasks;

    /**
     * Gets the value of the tasks property.
     * 
     * @return
     *     possible object is
     *     {@link GPUTaskListType }
     *     
     */
    public GPUTaskListType getTasks() {
        return tasks;
    }

    /**
     * Sets the value of the tasks property.
     * 
     * @param value
     *     allowed object is
     *     {@link GPUTaskListType }
     *     
     */
    public void setTasks(GPUTaskListType value) {
        this.tasks = value;
    }

}
