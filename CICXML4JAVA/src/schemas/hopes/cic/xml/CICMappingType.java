
package hopes.cic.xml;

import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for CICMappingType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="CICMappingType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="task" type="{http://peace.snu.ac.kr/CICXMLSchema}MappingTaskType" maxOccurs="unbounded"/>
 *         &lt;element name="externalTask" type="{http://peace.snu.ac.kr/CICXMLSchema}MappingExternalTaskType" maxOccurs="unbounded" minOccurs="0"/>
 *         &lt;element name="library" type="{http://peace.snu.ac.kr/CICXMLSchema}MappingLibraryType" maxOccurs="unbounded" minOccurs="0"/>
 *         &lt;element name="multicast" type="{http://peace.snu.ac.kr/CICXMLSchema}MappingMulticastType" maxOccurs="unbounded" minOccurs="0"/>
 *       &lt;/sequence>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "CICMappingType", propOrder = {
    "task",
    "externalTask",
    "library",
    "multicast"
})
public class CICMappingType {

    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema", required = true)
    protected List<MappingTaskType> task;
    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema", required = true)
    protected List<MappingExternalTaskType> externalTask;
    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema", required = true)
    protected List<MappingLibraryType> library;
    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema", required = true)
    protected List<MappingMulticastType> multicast;

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
     * {@link MappingTaskType }
     * 
     * 
     */
    public List<MappingTaskType> getTask() {
        if (task == null) {
            task = new ArrayList<MappingTaskType>();
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
     * {@link MappingExternalTaskType }
     * 
     * 
     */
    public List<MappingExternalTaskType> getExternalTask() {
        if (externalTask == null) {
            externalTask = new ArrayList<MappingExternalTaskType>();
        }
        return this.externalTask;
    }

    /**
     * Gets the value of the library property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the library property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getLibrary().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link MappingLibraryType }
     * 
     * 
     */
    public List<MappingLibraryType> getLibrary() {
        if (library == null) {
            library = new ArrayList<MappingLibraryType>();
        }
        return this.library;
    }

    /**
     * Gets the value of the multicast property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the multicast property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getMulticast().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link MappingMulticastType }
     * 
     * 
     */
    public List<MappingMulticastType> getMulticast() {
        if (multicast == null) {
            multicast = new ArrayList<MappingMulticastType>();
        }
        return this.multicast;
    }

}
