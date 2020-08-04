
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for CodeGenerationType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="CodeGenerationType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;attribute name="runtimeExecutionPolicy" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" />
 *       &lt;attribute name="threadOrFunctioncall" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "CodeGenerationType")
public class CodeGenerationType {

    @XmlAttribute(required = true)
    protected String runtimeExecutionPolicy;
    @XmlAttribute(required = true)
    protected String threadOrFunctioncall;

    /**
     * Gets the value of the runtimeExecutionPolicy property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getRuntimeExecutionPolicy() {
        return runtimeExecutionPolicy;
    }

    /**
     * Sets the value of the runtimeExecutionPolicy property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setRuntimeExecutionPolicy(String value) {
        this.runtimeExecutionPolicy = value;
    }

    /**
     * Gets the value of the threadOrFunctioncall property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getThreadOrFunctioncall() {
        return threadOrFunctioncall;
    }

    /**
     * Sets the value of the threadOrFunctioncall property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setThreadOrFunctioncall(String value) {
        this.threadOrFunctioncall = value;
    }

}
