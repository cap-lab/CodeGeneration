
package hopes.cic.xml;

import java.math.BigInteger;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for SerialConnectionType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="SerialConnectionType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;attribute name="boardRXPinNumber" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" />
 *       &lt;attribute name="boardTXPinNumber" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" />
 *       &lt;attribute name="multicast" use="required" type="{http://www.w3.org/2001/XMLSchema}boolean" />
 *       &lt;attribute name="name" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" />
 *       &lt;attribute name="network" type="{http://peace.snu.ac.kr/CICXMLSchema}NetworkType" />
 *       &lt;attribute name="portAddress" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="role" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}MasterSlaveRoleType" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "SerialConnectionType")
public class SerialConnectionType {

    @XmlAttribute
    protected BigInteger boardRXPinNumber;
    @XmlAttribute
    protected BigInteger boardTXPinNumber;
    @XmlAttribute(required = true)
    protected boolean multicast;
    @XmlAttribute(required = true)
    protected String name;
    @XmlAttribute
    protected NetworkType network;
    @XmlAttribute
    protected String portAddress;
    @XmlAttribute(required = true)
    protected MasterSlaveRoleType role;

    /**
     * Gets the value of the boardRXPinNumber property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getBoardRXPinNumber() {
        return boardRXPinNumber;
    }

    /**
     * Sets the value of the boardRXPinNumber property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setBoardRXPinNumber(BigInteger value) {
        this.boardRXPinNumber = value;
    }

    /**
     * Gets the value of the boardTXPinNumber property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getBoardTXPinNumber() {
        return boardTXPinNumber;
    }

    /**
     * Sets the value of the boardTXPinNumber property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setBoardTXPinNumber(BigInteger value) {
        this.boardTXPinNumber = value;
    }

    /**
     * Gets the value of the multicast property.
     * 
     */
    public boolean isMulticast() {
        return multicast;
    }

    /**
     * Sets the value of the multicast property.
     * 
     */
    public void setMulticast(boolean value) {
        this.multicast = value;
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
     * Gets the value of the network property.
     * 
     * @return
     *     possible object is
     *     {@link NetworkType }
     *     
     */
    public NetworkType getNetwork() {
        return network;
    }

    /**
     * Sets the value of the network property.
     * 
     * @param value
     *     allowed object is
     *     {@link NetworkType }
     *     
     */
    public void setNetwork(NetworkType value) {
        this.network = value;
    }

    /**
     * Gets the value of the portAddress property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getPortAddress() {
        return portAddress;
    }

    /**
     * Sets the value of the portAddress property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setPortAddress(String value) {
        this.portAddress = value;
    }

    /**
     * Gets the value of the role property.
     * 
     * @return
     *     possible object is
     *     {@link MasterSlaveRoleType }
     *     
     */
    public MasterSlaveRoleType getRole() {
        return role;
    }

    /**
     * Sets the value of the role property.
     * 
     * @param value
     *     allowed object is
     *     {@link MasterSlaveRoleType }
     *     
     */
    public void setRole(MasterSlaveRoleType value) {
        this.role = value;
    }

}
