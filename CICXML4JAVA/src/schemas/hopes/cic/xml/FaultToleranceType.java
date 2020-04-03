
package hopes.cic.xml;

import java.math.BigInteger;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for FaultToleranceType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="FaultToleranceType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="validationTaskCIC" type="{http://www.w3.org/2001/XMLSchema}string" minOccurs="0"/>
 *       &lt;/sequence>
 *       &lt;attribute name="number" use="required" type="{http://www.w3.org/2001/XMLSchema}positiveInteger" />
 *       &lt;attribute name="type" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}FaultToleranceTypeType" />
 *       &lt;attribute name="validationTaskType" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}ValidationTaskType" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "FaultToleranceType", propOrder = {
    "validationTaskCIC"
})
public class FaultToleranceType {

    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema")
    protected String validationTaskCIC;
    @XmlAttribute(required = true)
    protected BigInteger number;
    @XmlAttribute(required = true)
    protected FaultToleranceTypeType type;
    @XmlAttribute(required = true)
    protected ValidationTaskType validationTaskType;

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

}