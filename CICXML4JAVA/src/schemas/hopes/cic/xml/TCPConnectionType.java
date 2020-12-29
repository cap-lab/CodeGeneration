
package hopes.cic.xml;

import java.math.BigInteger;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlSchemaType;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for TCPConnectionType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="TCPConnectionType"&gt;
 *   &lt;complexContent&gt;
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType"&gt;
 *       &lt;attribute name="name" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" /&gt;
 *       &lt;attribute name="network" type="{http://peace.snu.ac.kr/CICXMLSchema}NetworkType" /&gt;
 *       &lt;attribute name="role" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}ServerClientRoleType" /&gt;
 *       &lt;attribute name="ip" use="required" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *       &lt;attribute name="port" use="required" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" /&gt;
 *       &lt;attribute name="secure" use="required" type="{http://www.w3.org/2001/XMLSchema}boolean" /&gt;
 *       &lt;attribute name="caPublicKey" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *       &lt;attribute name="publicKey" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *       &lt;attribute name="privateKey" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *     &lt;/restriction&gt;
 *   &lt;/complexContent&gt;
 * &lt;/complexType&gt;
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "TCPConnectionType")
public class TCPConnectionType {

    @XmlAttribute(name = "name", required = true)
    protected String name;
    @XmlAttribute(name = "network")
    protected NetworkType network;
    @XmlAttribute(name = "role", required = true)
    protected ServerClientRoleType role;
    @XmlAttribute(name = "ip", required = true)
    protected String ip;
    @XmlAttribute(name = "port", required = true)
    @XmlSchemaType(name = "nonNegativeInteger")
    protected BigInteger port;
    @XmlAttribute(name = "secure", required = true)
    protected boolean secure;
    @XmlAttribute(name = "caPublicKey")
    protected String caPublicKey;
    @XmlAttribute(name = "publicKey")
    protected String publicKey;
    @XmlAttribute(name = "privateKey")
    protected String privateKey;

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

}
