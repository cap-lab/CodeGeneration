
package hopes.cic.xml;

import java.math.BigInteger;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlSchemaType;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for SerialConnectionType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="SerialConnectionType"&gt;
 *   &lt;complexContent&gt;
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType"&gt;
 *       &lt;attribute name="name" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" /&gt;
 *       &lt;attribute name="network" type="{http://peace.snu.ac.kr/CICXMLSchema}NetworkType" /&gt;
 *       &lt;attribute name="role" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}MasterSlaveRoleType" /&gt;
 *       &lt;attribute name="boardTXPinNumber" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" /&gt;
 *       &lt;attribute name="boardRXPinNumber" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" /&gt;
 *       &lt;attribute name="portAddress" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *     &lt;/restriction&gt;
 *   &lt;/complexContent&gt;
 * &lt;/complexType&gt;
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "SerialConnectionType")
public class SerialConnectionType {

    @XmlAttribute(name = "name", required = true)
    protected String name;
    @XmlAttribute(name = "network")
    protected NetworkType network;
    @XmlAttribute(name = "role", required = true)
    protected MasterSlaveRoleType role;
    @XmlAttribute(name = "boardTXPinNumber")
    @XmlSchemaType(name = "nonNegativeInteger")
    protected BigInteger boardTXPinNumber;
    @XmlAttribute(name = "boardRXPinNumber")
    @XmlSchemaType(name = "nonNegativeInteger")
    protected BigInteger boardRXPinNumber;
    @XmlAttribute(name = "portAddress")
    protected String portAddress;

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

}
