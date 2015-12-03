
package hopes.cic.xml;

import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for CICProfileType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="CICProfileType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="task" type="{http://peace.snu.ac.kr/CICXMLSchema}ProfileTaskType" maxOccurs="unbounded"/>
 *         &lt;element name="comm" type="{http://peace.snu.ac.kr/CICXMLSchema}ProfileCommType" maxOccurs="unbounded" minOccurs="0"/>
 *       &lt;/sequence>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "CICProfileType", propOrder = {
    "task",
    "comm"
})
public class CICProfileType {

    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema", required = true)
    protected List<ProfileTaskType> task;
    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema", required = true)
    protected List<ProfileCommType> comm;

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
     * {@link ProfileTaskType }
     * 
     * 
     */
    public List<ProfileTaskType> getTask() {
        if (task == null) {
            task = new ArrayList<ProfileTaskType>();
        }
        return this.task;
    }

    /**
     * Gets the value of the comm property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the comm property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getComm().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link ProfileCommType }
     * 
     * 
     */
    public List<ProfileCommType> getComm() {
        if (comm == null) {
            comm = new ArrayList<ProfileCommType>();
        }
        return this.comm;
    }

}
