
package hopes.cic.xml;

import java.math.BigInteger;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for TCPConnectionType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="TCPConnectionType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;attribute name="caPublicKey" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="ip" use="required" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="multicast" use="required" type="{http://www.w3.org/2001/XMLSchema}boolean" />
 *       &lt;attribute name="name" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" />
 *       &lt;attribute name="network" type="{http://peace.snu.ac.kr/CICXMLSchema}NetworkType" />
 *       &lt;attribute name="port" use="required" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" />
 *       &lt;attribute name="privateKey" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="publicKey" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="role" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}ServerClientRoleType" />
 *       &lt;attribute name="secure" use="required" type="{http://www.w3.org/2001/XMLSchema}boolean" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "TCPConnectionType")
public class TCPConnectionType {

    @XmlAttribute
    protected String caPublicKey;
    @XmlAttribute(required = true)
    protected String ip;
    @XmlAttribute(required = true)
    protected boolean multicast;
    @XmlAttribute(required = true)
    protected String name;
    @XmlAttribute
    protected NetworkType network;
    @XmlAttribute(required = true)
    protected BigInteger port;
    @XmlAttribute
    protected String privateKey;
    @XmlAttribute
    protected String publicKey;
    @XmlAttribute(required = true)
    protected ServerClientRoleType role;
    @XmlAttribute(required = true)
    protected boolean secure;

    /**
     * Gets the value of the caPublicKey property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getCaPublicKey() {
        return caPublicKey;
    }

    /**
     * Sets the value of the caPublicKey property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setCaPublicKey(String value) {
        this.caPublicKey = value;
    }

    /**
     * Gets the value of the ip property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getIp() {
        return ip;
    }

    /**
     * Sets the value of the ip property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setIp(String value) {
        this.ip = value;
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
     * Gets the value of the port property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getPort() {
        return port;
    }

    /**
     * Sets the value of the port property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setPort(BigInteger value) {
        this.port = value;
    }

    /**
     * Gets the value of the privateKey property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getPrivateKey() {
        return privateKey;
    }

    /**
     * Sets the value of the privateKey property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setPrivateKey(String value) {
        this.privateKey = value;
    }

    /**
     * Gets the value of the publicKey property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getPublicKey() {
        return publicKey;
    }

    /**
     * Sets the value of the publicKey property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setPublicKey(String value) {
        this.publicKey = value;
    }

    /**
     * Gets the value of the role property.
     * 
     * @return
     *     possible object is
     *     {@link ServerClientRoleType }
     *     
     */
    public ServerClientRoleType getRole() {
        return role;
    }

    /**
     * Sets the value of the role property.
     * 
     * @param value
     *     allowed object is
     *     {@link ServerClientRoleType }
     *     
     */
    public void setRole(ServerClientRoleType value) {
        this.role = value;
    }

    /**
     * Gets the value of the secure property.
     * 
     */
    public boolean isSecure() {
        return secure;
    }

    /**
     * Sets the value of the secure property.
     * 
     */
    public void setSecure(boolean value) {
        this.secure = value;
    }

}
