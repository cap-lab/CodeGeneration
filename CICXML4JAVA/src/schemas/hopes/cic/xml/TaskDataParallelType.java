
package hopes.cic.xml;

import java.math.BigInteger;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlSchemaType;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for TaskDataParallelType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="TaskDataParallelType"&gt;
 *   &lt;complexContent&gt;
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType"&gt;
 *       &lt;sequence&gt;
 *         &lt;element name="volume" type="{http://peace.snu.ac.kr/CICXMLSchema}VectorType" minOccurs="0"/&gt;
 *         &lt;element name="dependencyVector" type="{http://peace.snu.ac.kr/CICXMLSchema}VectorListType" minOccurs="0"/&gt;
 *       &lt;/sequence&gt;
 *       &lt;attribute name="type" type="{http://peace.snu.ac.kr/CICXMLSchema}DataParallelType" /&gt;
 *       &lt;attribute name="maxParallel" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" /&gt;
 *       &lt;attribute name="loopCount" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" /&gt;
 *     &lt;/restriction&gt;
 *   &lt;/complexContent&gt;
 * &lt;/complexType&gt;
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "TaskDataParallelType", propOrder = {
    "volume",
    "dependencyVector"
})
public class TaskDataParallelType {

    protected VectorType volume;
    protected VectorListType dependencyVector;
    @XmlAttribute(name = "type")
    protected DataParallelType type;
    @XmlAttribute(name = "maxParallel")
    @XmlSchemaType(name = "nonNegativeInteger")
    protected BigInteger maxParallel;
    @XmlAttribute(name = "loopCount")
    @XmlSchemaType(name = "nonNegativeInteger")
    protected BigInteger loopCount;

    /**
     * Gets the value of the volume property.
     * 
     * @return
     *     possible object is
     *     {@link VectorType }
     *     
     */
    public VectorType getVolume() {
        return volume;
    }

    /**
     * Sets the value of the volume property.
     * 
     * @param value
     *     allowed object is
     *     {@link VectorType }
     *     
     */
    public void setVolume(VectorType value) {
        this.volume = value;
    }

    /**
     * Gets the value of the dependencyVector property.
     * 
     * @return
     *     possible object is
     *     {@link VectorListType }
     *     
     */
    public VectorListType getDependencyVector() {
        return dependencyVector;
    }

    /**
     * Sets the value of the dependencyVector property.
     * 
     * @param value
     *     allowed object is
     *     {@link VectorListType }
     *     
     */
    public void setDependencyVector(VectorListType value) {
        this.dependencyVector = value;
    }

    /**
     * Gets the value of the type property.
     * 
     * @return
     *     possible object is
     *     {@link DataParallelType }
     *     
     */
    public DataParallelType getType() {
        return type;
    }

    /**
     * Sets the value of the type property.
     * 
     * @param value
     *     allowed object is
     *     {@link DataParallelType }
     *     
     */
    public void setType(DataParallelType value) {
        this.type = value;
    }

    /**
     * Gets the value of the maxParallel property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getMaxParallel() {
        return maxParallel;
    }

    /**
     * Sets the value of the maxParallel property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setMaxParallel(BigInteger value) {
        this.maxParallel = value;
    }

    /**
     * Gets the value of the loopCount property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getLoopCount() {
        return loopCount;
    }

    /**
     * Sets the value of the loopCount property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setLoopCount(BigInteger value) {
        this.loopCount = value;
    }

}
