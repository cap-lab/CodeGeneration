
package hopes.cic.xml;

import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for ExclusiveControlTasksListType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="ExclusiveControlTasksListType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="exclusiveControlTasks" type="{http://peace.snu.ac.kr/CICXMLSchema}ExclusiveControlTasksType" maxOccurs="unbounded"/>
 *       &lt;/sequence>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "ExclusiveControlTasksListType", propOrder = {
    "exclusiveControlTasks"
})
public class ExclusiveControlTasksListType {

    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema", required = true)
    protected List<ExclusiveControlTasksType> exclusiveControlTasks;

    /**
     * Gets the value of the exclusiveControlTasks property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the exclusiveControlTasks property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getExclusiveControlTasks().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link ExclusiveControlTasksType }
     * 
     * 
     */
    public List<ExclusiveControlTasksType> getExclusiveControlTasks() {
        if (exclusiveControlTasks == null) {
            exclusiveControlTasks = new ArrayList<ExclusiveControlTasksType>();
        }
        return this.exclusiveControlTasks;
    }

}
