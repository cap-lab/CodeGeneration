
package hopes.cic.xml;

import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for MappingTaskType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="MappingTaskType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="processor" type="{http://peace.snu.ac.kr/CICXMLSchema}MappingProcessorIdType" maxOccurs="unbounded" minOccurs="0"/>
 *         &lt;element name="device" type="{http://peace.snu.ac.kr/CICXMLSchema}MappingDeviceType" maxOccurs="unbounded" minOccurs="0"/>
 *       &lt;/sequence>
 *       &lt;attribute name="dataParallel" type="{http://peace.snu.ac.kr/CICXMLSchema}DataParallelType" />
 *       &lt;attribute name="name" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" />
 *       &lt;attribute name="remappable" type="{http://www.w3.org/2001/XMLSchema}boolean" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "MappingTaskType", propOrder = {
    "processor",
    "device"
})
public class MappingTaskType {

    protected List<MappingProcessorIdType> processor;
    protected List<MappingDeviceType> device;
    @XmlAttribute
    protected DataParallelType dataParallel;
    @XmlAttribute(required = true)
    protected String name;
    @XmlAttribute
    protected Boolean remappable;

    /**
     * Gets the value of the processor property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the processor property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getProcessor().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link MappingProcessorIdType }
     * 
     * 
     */
    public List<MappingProcessorIdType> getProcessor() {
        if (processor == null) {
            processor = new ArrayList<MappingProcessorIdType>();
        }
        return this.processor;
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
     * Gets the value of the dataParallel property.
     * 
     * @return
     *     possible object is
     *     {@link DataParallelType }
     *     
     */
    public DataParallelType getDataParallel() {
        return dataParallel;
    }

    /**
	 * Sets the value of the dataParallel property.
	 * 
	 * @param value allowed object is
	 * 
	 *              {@link DataParallelType }
	 * 
	 */
    public void setDataParallel(DataParallelType value) {
        this.dataParallel = value;
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

    /**
     * Gets the value of the remappable property.
     * 
     * @return
     *     possible object is
     *     {@link Boolean }
     *     
     */
    public Boolean isRemappable() {
        return remappable;
    }

    /**
     * Sets the value of the remappable property.
     * 
     * @param value
     *     allowed object is
     *     {@link Boolean }
     *     
     */
    public void setRemappable(Boolean value) {
        this.remappable = value;
    }

}
