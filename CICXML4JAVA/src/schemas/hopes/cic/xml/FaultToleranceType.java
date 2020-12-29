
package hopes.cic.xml;

import java.math.BigInteger;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlSchemaType;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for FaultToleranceType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="FaultToleranceType"&gt;
 *   &lt;complexContent&gt;
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType"&gt;
 *       &lt;sequence&gt;
 *         &lt;element name="validationTaskCIC" type="{http://www.w3.org/2001/XMLSchema}string" minOccurs="0"/&gt;
 *       &lt;/sequence&gt;
 *       &lt;attribute name="validationTaskType" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}ValidationTaskType" /&gt;
 *       &lt;attribute name="type" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}FaultToleranceTypeType" /&gt;
 *       &lt;attribute name="number" use="required" type="{http://www.w3.org/2001/XMLSchema}positiveInteger" /&gt;
 *     &lt;/restriction&gt;
 *   &lt;/complexContent&gt;
 * &lt;/complexType&gt;
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "FaultToleranceType", propOrder = {
    "validationTaskCIC"
})
public class FaultToleranceType {

    protected String validationTaskCIC;
    @XmlAttribute(name = "validationTaskType", required = true)
    protected ValidationTaskType validationTaskType;
    @XmlAttribute(name = "type", required = true)
    protected FaultToleranceTypeType type;
    @XmlAttribute(name = "number", required = true)
    @XmlSchemaType(name = "positiveInteger")
    protected BigInteger number;

    /**
     * Gets the value of the validationTaskCIC property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getValidationTaskCIC() {
        return validationTaskCIC;
    }

    /**
     * Sets the value of the validationTaskCIC property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setValidationTaskCIC(String value) {
        this.validationTaskCIC = value;
    }

    /**
     * Gets the value of the validationTaskType property.
     * 
     * @return
     *     possible object is
     *     {@link ValidationTaskType }
     *     
     */
    public ValidationTaskType getValidationTaskType() {
        return validationTaskType;
    }

    /**
     * Sets the value of the validationTaskType property.
     * 
     * @param value
     *     allowed object is
     *     {@link ValidationTaskType }
     *     
     */
    public void setValidationTaskType(ValidationTaskType value) {
        this.validationTaskType = value;
    }

    /**
     * Gets the value of the type property.
     * 
     * @return
     *     possible object is
     *     {@link FaultToleranceTypeType }
     *     
     */
    public FaultToleranceTypeType getType() {
        return type;
    }

    /**
     * Sets the value of the type property.
     * 
     * @param value
     *     allowed object is
     *     {@link FaultToleranceTypeType }
     *     
     */
    public void setType(FaultToleranceTypeType value) {
        this.type = value;
    }

    /**
     * Gets the value of the number property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getNumber() {
        return number;
    }

    /**
     * Sets the value of the number property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setNumber(BigInteger value) {
        this.number = value;
    }

}
