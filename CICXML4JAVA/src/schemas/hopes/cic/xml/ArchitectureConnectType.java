
package hopes.cic.xml;

import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for ArchitectureConnectType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="ArchitectureConnectType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="Bluetoothconnection" type="{http://peace.snu.ac.kr/CICXMLSchema}ArchitectureConnectionBTType" maxOccurs="unbounded" minOccurs="0"/>
 *         &lt;element name="WIFIconnection" type="{http://peace.snu.ac.kr/CICXMLSchema}ArchitectureConnectionWIFIType" maxOccurs="unbounded" minOccurs="0"/>
 *         &lt;element name="I2Cconnection" type="{http://peace.snu.ac.kr/CICXMLSchema}ArchitectureConnectionI2CType" maxOccurs="unbounded" minOccurs="0"/>
 *         &lt;element name="VRep_SharedBusconnection" type="{http://peace.snu.ac.kr/CICXMLSchema}ArchitectureConnectionVRepSBType" maxOccurs="unbounded" minOccurs="0"/>
 *       &lt;/sequence>
 *       &lt;attribute name="type" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}ArchitectureConnectionCategoryType" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "ArchitectureConnectType", propOrder = {
    "bluetoothconnection",
    "wifIconnection",
    "i2Cconnection",
    "vRepSharedBusconnection"
})
public class ArchitectureConnectType {

    @XmlElement(name = "Bluetoothconnection", namespace = "http://peace.snu.ac.kr/CICXMLSchema", required = true)
    protected List<ArchitectureConnectionBTType> bluetoothconnection;
    @XmlElement(name = "WIFIconnection", namespace = "http://peace.snu.ac.kr/CICXMLSchema", required = true)
    protected List<ArchitectureConnectionWIFIType> wifIconnection;
    @XmlElement(name = "I2Cconnection", namespace = "http://peace.snu.ac.kr/CICXMLSchema", required = true)
    protected List<ArchitectureConnectionI2CType> i2Cconnection;
    @XmlElement(name = "VRep_SharedBusconnection", namespace = "http://peace.snu.ac.kr/CICXMLSchema", required = true)
    protected List<ArchitectureConnectionVRepSBType> vRepSharedBusconnection;
    @XmlAttribute(required = true)
    protected ArchitectureConnectionCategoryType type;

    /**
     * Gets the value of the bluetoothconnection property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the bluetoothconnection property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getBluetoothconnection().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link ArchitectureConnectionBTType }
     * 
     * 
     */
    public List<ArchitectureConnectionBTType> getBluetoothconnection() {
        if (bluetoothconnection == null) {
            bluetoothconnection = new ArrayList<ArchitectureConnectionBTType>();
        }
        return this.bluetoothconnection;
    }

    /**
     * Gets the value of the wifIconnection property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the wifIconnection property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getWIFIconnection().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link ArchitectureConnectionWIFIType }
     * 
     * 
     */
    public List<ArchitectureConnectionWIFIType> getWIFIconnection() {
        if (wifIconnection == null) {
            wifIconnection = new ArrayList<ArchitectureConnectionWIFIType>();
        }
        return this.wifIconnection;
    }

    /**
     * Gets the value of the i2Cconnection property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the i2Cconnection property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getI2Cconnection().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link ArchitectureConnectionI2CType }
     * 
     * 
     */
    public List<ArchitectureConnectionI2CType> getI2Cconnection() {
        if (i2Cconnection == null) {
            i2Cconnection = new ArrayList<ArchitectureConnectionI2CType>();
        }
        return this.i2Cconnection;
    }

    /**
     * Gets the value of the vRepSharedBusconnection property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the vRepSharedBusconnection property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getVRepSharedBusconnection().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link ArchitectureConnectionVRepSBType }
     * 
     * 
     */
    public List<ArchitectureConnectionVRepSBType> getVRepSharedBusconnection() {
        if (vRepSharedBusconnection == null) {
            vRepSharedBusconnection = new ArrayList<ArchitectureConnectionVRepSBType>();
        }
        return this.vRepSharedBusconnection;
    }

    /**
     * Gets the value of the type property.
     * 
     * @return
     *     possible object is
     *     {@link ArchitectureConnectionCategoryType }
     *     
     */
    public ArchitectureConnectionCategoryType getType() {
        return type;
    }

    /**
     * Sets the value of the type property.
     * 
     * @param value
     *     allowed object is
     *     {@link ArchitectureConnectionCategoryType }
     *     
     */
    public void setType(ArchitectureConnectionCategoryType value) {
        this.type = value;
    }

}
