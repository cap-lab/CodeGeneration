
package hopes.cic.xml;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlSchemaType;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for LibraryType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="LibraryType"&gt;
 *   &lt;complexContent&gt;
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType"&gt;
 *       &lt;sequence&gt;
 *         &lt;element name="libraryMasterPort" type="{http://peace.snu.ac.kr/CICXMLSchema}LibraryMasterPortType" maxOccurs="unbounded" minOccurs="0"/&gt;
 *         &lt;element name="extraHeader" type="{http://www.w3.org/2001/XMLSchema}string" maxOccurs="unbounded" minOccurs="0"/&gt;
 *         &lt;element name="extraSource" type="{http://www.w3.org/2001/XMLSchema}string" maxOccurs="unbounded" minOccurs="0"/&gt;
 *         &lt;element name="extraCIC" type="{http://www.w3.org/2001/XMLSchema}string" maxOccurs="unbounded" minOccurs="0"/&gt;
 *         &lt;element name="extraFile" type="{http://www.w3.org/2001/XMLSchema}string" maxOccurs="unbounded" minOccurs="0"/&gt;
 *         &lt;element name="function" type="{http://peace.snu.ac.kr/CICXMLSchema}LibraryFunctionType" maxOccurs="unbounded" minOccurs="0"/&gt;
 *         &lt;element name="hardwareDependency" type="{http://peace.snu.ac.kr/CICXMLSchema}HardwareDependencyType" minOccurs="0"/&gt;
 *       &lt;/sequence&gt;
 *       &lt;attribute name="name" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" /&gt;
 *       &lt;attribute name="id" use="required" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" /&gt;
 *       &lt;attribute name="type" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" /&gt;
 *       &lt;attribute name="hasInternalStates" type="{http://peace.snu.ac.kr/CICXMLSchema}YesNoType" /&gt;
 *       &lt;attribute name="description" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *       &lt;attribute name="file" use="required" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *       &lt;attribute name="header" use="required" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *       &lt;attribute name="cflags" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *       &lt;attribute name="ldflags" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *       &lt;attribute name="isHardwareDependent" type="{http://peace.snu.ac.kr/CICXMLSchema}YesNoType" /&gt;
 *       &lt;attribute name="language" type="{http://peace.snu.ac.kr/CICXMLSchema}LanguageType" /&gt;
 *     &lt;/restriction&gt;
 *   &lt;/complexContent&gt;
 * &lt;/complexType&gt;
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "LibraryType", propOrder = {
    "libraryMasterPort",
    "extraHeader",
    "extraSource",
    "extraCIC",
    "extraFile",
    "function",
    "hardwareDependency"
})
public class LibraryType {

    protected List<LibraryMasterPortType> libraryMasterPort;
    protected List<String> extraHeader;
    protected List<String> extraSource;
    protected List<String> extraCIC;
    protected List<String> extraFile;
    protected List<LibraryFunctionType> function;
    protected HardwareDependencyType hardwareDependency;
    @XmlAttribute(name = "name", required = true)
    protected String name;
    @XmlAttribute(name = "id", required = true)
    @XmlSchemaType(name = "nonNegativeInteger")
    protected BigInteger id;
    @XmlAttribute(name = "type", required = true)
    protected String type;
    @XmlAttribute(name = "hasInternalStates")
    protected YesNoType hasInternalStates;
    @XmlAttribute(name = "description")
    protected String description;
    @XmlAttribute(name = "file", required = true)
    protected String file;
    @XmlAttribute(name = "header", required = true)
    protected String header;
    @XmlAttribute(name = "cflags")
    protected String cflags;
    @XmlAttribute(name = "ldflags")
    protected String ldflags;
    @XmlAttribute(name = "isHardwareDependent")
    protected YesNoType isHardwareDependent;
    @XmlAttribute(name = "language")
    protected String language;

    /**
     * Gets the value of the libraryMasterPort property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the libraryMasterPort property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getLibraryMasterPort().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link LibraryMasterPortType }
     * 
     * 
     */
    public List<LibraryMasterPortType> getLibraryMasterPort() {
        if (libraryMasterPort == null) {
            libraryMasterPort = new ArrayList<LibraryMasterPortType>();
        }
        return this.libraryMasterPort;
    }

    /**
     * Gets the value of the extraHeader property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the extraHeader property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getExtraHeader().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link String }
     * 
     * 
     */
    public List<String> getExtraHeader() {
        if (extraHeader == null) {
            extraHeader = new ArrayList<String>();
        }
        return this.extraHeader;
    }

    /**
     * Gets the value of the extraSource property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the extraSource property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getExtraSource().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link String }
     * 
     * 
     */
    public List<String> getExtraSource() {
        if (extraSource == null) {
            extraSource = new ArrayList<String>();
        }
        return this.extraSource;
    }

    /**
     * Gets the value of the extraCIC property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the extraCIC property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getExtraCIC().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link String }
     * 
     * 
     */
    public List<String> getExtraCIC() {
        if (extraCIC == null) {
            extraCIC = new ArrayList<String>();
        }
        return this.extraCIC;
    }

    /**
     * Gets the value of the extraFile property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the extraFile property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getExtraFile().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link String }
     * 
     * 
     */
    public List<String> getExtraFile() {
        if (extraFile == null) {
            extraFile = new ArrayList<String>();
        }
        return this.extraFile;
    }

    /**
     * Gets the value of the function property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the function property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getFunction().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link LibraryFunctionType }
     * 
     * 
     */
    public List<LibraryFunctionType> getFunction() {
        if (function == null) {
            function = new ArrayList<LibraryFunctionType>();
        }
        return this.function;
    }

    /**
     * Gets the value of the hardwareDependency property.
     * 
     * @return
     *     possible object is
     *     {@link HardwareDependencyType }
     *     
     */
    public HardwareDependencyType getHardwareDependency() {
        return hardwareDependency;
    }

    /**
     * Sets the value of the hardwareDependency property.
     * 
     * @param value
     *     allowed object is
     *     {@link HardwareDependencyType }
     *     
     */
    public void setHardwareDependency(HardwareDependencyType value) {
        this.hardwareDependency = value;
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
     * Gets the value of the id property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getId() {
        return id;
    }

    /**
     * Sets the value of the id property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setId(BigInteger value) {
        this.id = value;
    }

    /**
     * Gets the value of the type property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getType() {
        return type;
    }

    /**
     * Sets the value of the type property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setType(String value) {
        this.type = value;
    }

    /**
     * Gets the value of the hasInternalStates property.
     * 
     * @return
     *     possible object is
     *     {@link YesNoType }
     *     
     */
    public YesNoType getHasInternalStates() {
        return hasInternalStates;
    }

    /**
     * Sets the value of the hasInternalStates property.
     * 
     * @param value
     *     allowed object is
     *     {@link YesNoType }
     *     
     */
    public void setHasInternalStates(YesNoType value) {
        this.hasInternalStates = value;
    }

    /**
     * Gets the value of the description property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getDescription() {
        return description;
    }

    /**
     * Sets the value of the description property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setDescription(String value) {
        this.description = value;
    }

    /**
     * Gets the value of the file property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getFile() {
        return file;
    }

    /**
     * Sets the value of the file property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setFile(String value) {
        this.file = value;
    }

    /**
     * Gets the value of the header property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getHeader() {
        return header;
    }

    /**
     * Sets the value of the header property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setHeader(String value) {
        this.header = value;
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
     * Gets the value of the isHardwareDependent property.
     * 
     * @return
     *     possible object is
     *     {@link YesNoType }
     *     
     */
    public YesNoType getIsHardwareDependent() {
        return isHardwareDependent;
    }

    /**
     * Sets the value of the isHardwareDependent property.
     * 
     * @param value
     *     allowed object is
     *     {@link YesNoType }
     *     
     */
    public void setIsHardwareDependent(YesNoType value) {
        this.isHardwareDependent = value;
    }

    /**
     * Gets the value of the language property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getLanguage() {
        return language;
    }

    /**
     * Sets the value of the language property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setLanguage(String value) {
        this.language = value;
    }

}
