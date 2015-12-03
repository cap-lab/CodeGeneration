
package hopes.cic.xml;

import java.math.BigInteger;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for GPUTaskType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="GPUTaskType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="globalWorkSize" type="{http://peace.snu.ac.kr/CICXMLSchema}VectorType" minOccurs="0"/>
 *         &lt;element name="localWorkSize" type="{http://peace.snu.ac.kr/CICXMLSchema}VectorType" minOccurs="0"/>
 *       &lt;/sequence>
 *       &lt;attribute name="clustering" type="{http://peace.snu.ac.kr/CICXMLSchema}YesNoType" />
 *       &lt;attribute name="maxStream" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" />
 *       &lt;attribute name="name" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" />
 *       &lt;attribute name="pipelining" type="{http://peace.snu.ac.kr/CICXMLSchema}YesNoType" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "GPUTaskType", propOrder = {
    "globalWorkSize",
    "localWorkSize"
})
public class GPUTaskType {

    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema")
    protected VectorType globalWorkSize;
    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema")
    protected VectorType localWorkSize;
    @XmlAttribute
    protected YesNoType clustering;
    @XmlAttribute
    protected BigInteger maxStream;
    @XmlAttribute(required = true)
    protected String name;
    @XmlAttribute
    protected YesNoType pipelining;

    /**
     * Gets the value of the globalWorkSize property.
     * 
     * @return
     *     possible object is
     *     {@link VectorType }
     *     
     */
    public VectorType getGlobalWorkSize() {
        return globalWorkSize;
    }

    /**
     * Sets the value of the globalWorkSize property.
     * 
     * @param value
     *     allowed object is
     *     {@link VectorType }
     *     
     */
    public void setGlobalWorkSize(VectorType value) {
        this.globalWorkSize = value;
    }

    /**
     * Gets the value of the localWorkSize property.
     * 
     * @return
     *     possible object is
     *     {@link VectorType }
     *     
     */
    public VectorType getLocalWorkSize() {
        return localWorkSize;
    }

    /**
     * Sets the value of the localWorkSize property.
     * 
     * @param value
     *     allowed object is
     *     {@link VectorType }
     *     
     */
    public void setLocalWorkSize(VectorType value) {
        this.localWorkSize = value;
    }

    /**
     * Gets the value of the clustering property.
     * 
     * @return
     *     possible object is
     *     {@link YesNoType }
     *     
     */
    public YesNoType getClustering() {
        return clustering;
    }

    /**
     * Sets the value of the clustering property.
     * 
     * @param value
     *     allowed object is
     *     {@link YesNoType }
     *     
     */
    public void setClustering(YesNoType value) {
        this.clustering = value;
    }

    /**
     * Gets the value of the maxStream property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getMaxStream() {
        return maxStream;
    }

    /**
     * Sets the value of the maxStream property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setMaxStream(BigInteger value) {
        this.maxStream = value;
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
     * Gets the value of the pipelining property.
     * 
     * @return
     *     possible object is
     *     {@link YesNoType }
     *     
     */
    public YesNoType getPipelining() {
        return pipelining;
    }

    /**
     * Sets the value of the pipelining property.
     * 
     * @param value
     *     allowed object is
     *     {@link YesNoType }
     *     
     */
    public void setPipelining(YesNoType value) {
        this.pipelining = value;
    }

}
