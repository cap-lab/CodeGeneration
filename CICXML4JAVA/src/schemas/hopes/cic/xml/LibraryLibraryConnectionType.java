
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
 * &lt;complexType name="LibraryLibraryConnectionType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;attribute name="masterLibrary" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" />
 *       &lt;attribute name="masterPort" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" />
 *       &lt;attribute name="slaveLibrary" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "LibraryLibraryConnectionType")
public class LibraryLibraryConnectionType {

    @XmlAttribute(required = true)
    protected String masterLibrary;
    @XmlAttribute(required = true)
    protected String masterPort;
    @XmlAttribute(required = true)
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
