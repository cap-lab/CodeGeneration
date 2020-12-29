
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
 * &lt;complexType name="CodeGenerationType"&gt;
 *   &lt;complexContent&gt;
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType"&gt;
 *       &lt;attribute name="runtimeExecutionPolicy" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" /&gt;
 *       &lt;attribute name="threadOrFunctioncall" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" /&gt;
 *     &lt;/restriction&gt;
 *   &lt;/complexContent&gt;
 * &lt;/complexType&gt;
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "CodeGenerationType")
public class CodeGenerationType {

    @XmlAttribute(name = "runtimeExecutionPolicy", required = true)
    protected String runtimeExecutionPolicy;
    @XmlAttribute(name = "threadOrFunctioncall", required = true)
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
