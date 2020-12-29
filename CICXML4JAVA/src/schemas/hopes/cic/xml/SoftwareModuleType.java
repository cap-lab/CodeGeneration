
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for SoftwareModuleType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="SoftwareModuleType"&gt;
 *   &lt;complexContent&gt;
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType"&gt;
 *       &lt;sequence&gt;
 *         &lt;element name="sources" type="{http://peace.snu.ac.kr/CICXMLSchema}FileSourceListType"/&gt;
 *         &lt;element name="headers" type="{http://peace.snu.ac.kr/CICXMLSchema}FileSourceListType"/&gt;
 *       &lt;/sequence&gt;
 *       &lt;attribute name="name" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" /&gt;
 *       &lt;attribute name="cflags" use="required" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *       &lt;attribute name="ldflags" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *       &lt;attribute name="initializer" use="required" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *       &lt;attribute name="finalizer" use="required" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *     &lt;/restriction&gt;
 *   &lt;/complexContent&gt;
 * &lt;/complexType&gt;
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "SoftwareModuleType", propOrder = {
    "sources",
    "headers"
})
public class SoftwareModuleType {

    @XmlElement(required = true)
    protected FileSourceListType sources;
    @XmlElement(required = true)
    protected FileSourceListType headers;
    @XmlAttribute(name = "name", required = true)
    protected String name;
    @XmlAttribute(name = "cflags", required = true)
    protected String cflags;
    @XmlAttribute(name = "ldflags")
    protected String ldflags;
    @XmlAttribute(name = "initializer", required = true)
    protected String initializer;
    @XmlAttribute(name = "finalizer", required = true)
    protected String finalizer;

    /**
     * Gets the value of the sources property.
     * 
     * @return
     *     possible object is
     *     {@link FileSourceListType }
     *     
     */
    public FileSourceListType getSources() {
        return sources;
    }

    /**
     * Sets the value of the sources property.
     * 
     * @param value
     *     allowed object is
     *     {@link FileSourceListType }
     *     
     */
    public void setSources(FileSourceListType value) {
        this.sources = value;
    }

    /**
     * Gets the value of the headers property.
     * 
     * @return
     *     possible object is
     *     {@link FileSourceListType }
     *     
     */
    public FileSourceListType getHeaders() {
        return headers;
    }

    /**
     * Sets the value of the headers property.
     * 
     * @param value
     *     allowed object is
     *     {@link FileSourceListType }
     *     
     */
    public void setHeaders(FileSourceListType value) {
        this.headers = value;
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
     * Gets the value of the cflags property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getCflags() {
        return cflags;
    }

    /**
     * Sets the value of the cflags property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setCflags(String value) {
        this.cflags = value;
    }

    /**
     * Gets the value of the ldflags property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getLdflags() {
        return ldflags;
    }

    /**
     * Sets the value of the ldflags property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setLdflags(String value) {
        this.ldflags = value;
    }

    /**
     * Gets the value of the initializer property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getInitializer() {
        return initializer;
    }

    /**
     * Sets the value of the initializer property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setInitializer(String value) {
        this.initializer = value;
    }

    /**
     * Gets the value of the finalizer property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getFinalizer() {
        return finalizer;
    }

    /**
     * Sets the value of the finalizer property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setFinalizer(String value) {
        this.finalizer = value;
    }

}
