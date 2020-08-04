
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for CICConfigurationType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="CICConfigurationType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="simulation" type="{http://peace.snu.ac.kr/CICXMLSchema}SimulationType" minOccurs="0"/>
 *         &lt;element name="codeGeneration" type="{http://peace.snu.ac.kr/CICXMLSchema}CodeGenerationType" minOccurs="0"/>
 *       &lt;/sequence>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "CICConfigurationType", propOrder = {
    "simulation",
    "codeGeneration"
})
public class CICConfigurationType {

    protected SimulationType simulation;
    protected CodeGenerationType codeGeneration;

    /**
     * Gets the value of the simulation property.
     * 
     * @return
     *     possible object is
     *     {@link SimulationType }
     *     
     */
    public SimulationType getSimulation() {
        return simulation;
    }

    /**
     * Sets the value of the simulation property.
     * 
     * @param value
     *     allowed object is
     *     {@link SimulationType }
     *     
     */
    public void setSimulation(SimulationType value) {
        this.simulation = value;
    }

    /**
     * Gets the value of the codeGeneration property.
     * 
     * @return
     *     possible object is
     *     {@link CodeGenerationType }
     *     
     */
    public CodeGenerationType getCodeGeneration() {
        return codeGeneration;
    }

    /**
     * Sets the value of the codeGeneration property.
     * 
     * @param value
     *     allowed object is
     *     {@link CodeGenerationType }
     *     
     */
    public void setCodeGeneration(CodeGenerationType value) {
        this.codeGeneration = value;
    }

}
