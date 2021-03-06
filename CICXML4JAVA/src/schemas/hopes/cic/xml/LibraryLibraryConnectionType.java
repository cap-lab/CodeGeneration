
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for LibraryLibraryConnectionType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="LibraryLibraryConnectionType"&gt;
 *   &lt;complexContent&gt;
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType"&gt;
 *       &lt;attribute name="masterLibrary" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" /&gt;
 *       &lt;attribute name="masterPort" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" /&gt;
 *       &lt;attribute name="slaveLibrary" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" /&gt;
 *     &lt;/restriction&gt;
 *   &lt;/complexContent&gt;
 * &lt;/complexType&gt;
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "LibraryLibraryConnectionType")
public class LibraryLibraryConnectionType {

    @XmlAttribute(name = "masterLibrary", required = true)
    protected String masterLibrary;
    @XmlAttribute(name = "masterPort", required = true)
    protected String masterPort;
    @XmlAttribute(name = "slaveLibrary", required = true)
    protected String slaveLibrary;

    /**
     * Gets the value of the masterLibrary property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getMasterLibrary() {
        return masterLibrary;
    }

    /**
     * Sets the value of the masterLibrary property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setMasterLibrary(String value) {
        this.masterLibrary = value;
    }

    /**
     * Gets the value of the masterPort property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getMasterPort() {
        return masterPort;
    }

    /**
     * Sets the value of the masterPort property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setMasterPort(String value) {
        this.masterPort = value;
    }

    /**
     * Gets the value of the slaveLibrary property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getSlaveLibrary() {
        return slaveLibrary;
    }

    /**
     * Sets the value of the slaveLibrary property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setSlaveLibrary(String value) {
        this.slaveLibrary = value;
    }

}
