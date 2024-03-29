
package hopes.cic.xml;

import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for MappingLibraryType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="MappingLibraryType"&gt;
 *   &lt;complexContent&gt;
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType"&gt;
 *       &lt;sequence&gt;
 *         &lt;element name="mappingSet" type="{http://peace.snu.ac.kr/CICXMLSchema}MappingSetType" minOccurs="0"/&gt;
 *         &lt;element name="device" type="{http://peace.snu.ac.kr/CICXMLSchema}MappingDeviceType" maxOccurs="unbounded" minOccurs="0"/&gt;
 *         &lt;element name="task" type="{http://peace.snu.ac.kr/CICXMLSchema}LibraryAccessItemType" maxOccurs="unbounded" minOccurs="0"/&gt;
 *         &lt;element name="library" type="{http://peace.snu.ac.kr/CICXMLSchema}LibraryAccessItemType" maxOccurs="unbounded" minOccurs="0"/&gt;
 *       &lt;/sequence&gt;
 *       &lt;attribute name="name" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" /&gt;
 *     &lt;/restriction&gt;
 *   &lt;/complexContent&gt;
 * &lt;/complexType&gt;
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "MappingLibraryType", propOrder = {
    "mappingSet",
    "device",
    "task",
    "library"
})
public class MappingLibraryType {

    protected MappingSetType mappingSet;
    protected List<MappingDeviceType> device;
    protected List<LibraryAccessItemType> task;
    protected List<LibraryAccessItemType> library;
    @XmlAttribute(name = "name", required = true)
    protected String name;

    /**
     * Gets the value of the mappingSet property.
     * 
     * @return
     *     possible object is
     *     {@link MappingSetType }
     *     
     */
    public MappingSetType getMappingSet() {
        return mappingSet;
    }

    /**
     * Sets the value of the mappingSet property.
     * 
     * @param value
     *     allowed object is
     *     {@link MappingSetType }
     *     
     */
    public void setMappingSet(MappingSetType value) {
        this.mappingSet = value;
    }

    /**
     * Gets the value of the device property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the device property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getDevice().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link MappingDeviceType }
     * 
     * 
     */
    public List<MappingDeviceType> getDevice() {
        if (device == null) {
            device = new ArrayList<MappingDeviceType>();
        }
        return this.device;
    }

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
     * {@link LibraryAccessItemType }
     * 
     * 
     */
    public List<LibraryAccessItemType> getTask() {
        if (task == null) {
            task = new ArrayList<LibraryAccessItemType>();
        }
        return this.task;
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
     * {@link LibraryAccessItemType }
     * 
     * 
     */
    public List<LibraryAccessItemType> getLibrary() {
        if (library == null) {
            library = new ArrayList<LibraryAccessItemType>();
        }
        return this.library;
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

}
