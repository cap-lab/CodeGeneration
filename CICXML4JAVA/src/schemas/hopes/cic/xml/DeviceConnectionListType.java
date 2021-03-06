
package hopes.cic.xml;

import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for DeviceConnectionListType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="DeviceConnectionListType"&gt;
 *   &lt;complexContent&gt;
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType"&gt;
 *       &lt;sequence&gt;
 *         &lt;element name="TCPConnection" type="{http://peace.snu.ac.kr/CICXMLSchema}TCPConnectionType" maxOccurs="unbounded" minOccurs="0"/&gt;
 *         &lt;element name="UDPConnection" type="{http://peace.snu.ac.kr/CICXMLSchema}UDPConnectionType" maxOccurs="unbounded" minOccurs="0"/&gt;
 *         &lt;element name="SerialConnection" type="{http://peace.snu.ac.kr/CICXMLSchema}SerialConnectionType" maxOccurs="unbounded" minOccurs="0"/&gt;
 *       &lt;/sequence&gt;
 *     &lt;/restriction&gt;
 *   &lt;/complexContent&gt;
 * &lt;/complexType&gt;
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "DeviceConnectionListType", propOrder = {
    "tcpConnection",
    "udpConnection",
    "serialConnection"
})
public class DeviceConnectionListType {

    @XmlElement(name = "TCPConnection")
    protected List<TCPConnectionType> tcpConnection;
    @XmlElement(name = "UDPConnection")
    protected List<UDPConnectionType> udpConnection;
    @XmlElement(name = "SerialConnection")
    protected List<SerialConnectionType> serialConnection;

    /**
     * Gets the value of the tcpConnection property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the tcpConnection property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getTCPConnection().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link TCPConnectionType }
     * 
     * 
     */
    public List<TCPConnectionType> getTCPConnection() {
        if (tcpConnection == null) {
            tcpConnection = new ArrayList<TCPConnectionType>();
        }
        return this.tcpConnection;
    }

    /**
     * Gets the value of the udpConnection property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the udpConnection property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getUDPConnection().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link UDPConnectionType }
     * 
     * 
     */
    public List<UDPConnectionType> getUDPConnection() {
        if (udpConnection == null) {
            udpConnection = new ArrayList<UDPConnectionType>();
        }
        return this.udpConnection;
    }

    /**
     * Gets the value of the serialConnection property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the serialConnection property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getSerialConnection().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link SerialConnectionType }
     * 
     * 
     */
    public List<SerialConnectionType> getSerialConnection() {
        if (serialConnection == null) {
            serialConnection = new ArrayList<SerialConnectionType>();
        }
        return this.serialConnection;
    }

}
