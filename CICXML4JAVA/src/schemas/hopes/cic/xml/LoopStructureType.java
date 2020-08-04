
package hopes.cic.xml;

import java.math.BigInteger;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for LoopStructureType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="LoopStructureType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;attribute name="designatedTask" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="loopCount" use="required" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" />
 *       &lt;attribute name="type" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}LoopStructureTypeType" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "LoopStructureType")
public class LoopStructureType {

    @XmlAttribute
    protected String designatedTask;
    @XmlAttribute(required = true)
    protected BigInteger loopCount;
    @XmlAttribute(required = true)
    protected LoopStructureTypeType type;

    /**
     * Gets the value of the designatedTask property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getDesignatedTask() {
        return designatedTask;
    }

    /**
     * Sets the value of the designatedTask property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setDesignatedTask(String value) {
        this.designatedTask = value;
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

    /**
     * Gets the value of the type property.
     * 
     * @return
     *     possible object is
     *     {@link LoopStructureTypeType }
     *     
     */
    public LoopStructureTypeType getType() {
        return type;
    }

    /**
     * Sets the value of the type property.
     * 
     * @param value
     *     allowed object is
     *     {@link LoopStructureTypeType }
     *     
     */
    public void setType(LoopStructureTypeType value) {
        this.type = value;
    }

}
