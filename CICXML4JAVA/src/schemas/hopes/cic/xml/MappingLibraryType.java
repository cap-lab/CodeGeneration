
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
 * &lt;complexType name="MappingLibraryType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="processor" type="{http://peace.snu.ac.kr/CICXMLSchema}MappingProcessorIdType" minOccurs="0"/>
 *         &lt;element name="device" type="{http://peace.snu.ac.kr/CICXMLSchema}MappingDeviceType" maxOccurs="unbounded" minOccurs="0"/>
 *         &lt;element name="task" type="{http://peace.snu.ac.kr/CICXMLSchema}LibraryAccessItemType" maxOccurs="unbounded" minOccurs="0"/>
 *         &lt;element name="library" type="{http://peace.snu.ac.kr/CICXMLSchema}LibraryAccessItemType" maxOccurs="unbounded" minOccurs="0"/>
 *       &lt;/sequence>
 *       &lt;attribute name="name" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "MappingLibraryType", propOrder = {
    "processor",
    "device",
    "task",
    "library"
})
public class MappingLibraryType {

    protected MappingProcessorIdType processor;
    protected List<MappingDeviceType> device;
    protected List<LibraryAccessItemType> task;
    protected List<LibraryAccessItemType> library;
    @XmlAttribute(required = true)
    protected String name;

    /**
     * Gets the value of the processor property.
     * 
     * @return
     *     possible object is
     *     {@link MappingProcessorIdType }
     *     
     */
    public MappingProcessorIdType getProcessor() {
        return processor;
    }

    /**
     * Sets the value of the processor property.
     * 
     * @param value
     *     allowed object is
     *     {@link MappingProcessorIdType }
     *     
     */
    public void setProcessor(MappingProcessorIdType value) {
        this.processor = value;
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
