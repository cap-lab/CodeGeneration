//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.3.2 
// See <a href="https://javaee.github.io/jaxb-v2/">https://javaee.github.io/jaxb-v2/</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2020.12.29 at 01:56:08 PM KST 
//


package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for LoopStructureTypeType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="LoopStructureTypeType"&gt;
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string"&gt;
 *     &lt;enumeration value="data"/&gt;
 *     &lt;enumeration value="convergent"/&gt;
 *   &lt;/restriction&gt;
 * &lt;/simpleType&gt;
 * </pre>
 * 
 */
@XmlType(name = "LoopStructureTypeType")
@XmlEnum
public enum LoopStructureTypeType {

    @XmlEnumValue("data")
    DATA("data"),
    @XmlEnumValue("convergent")
    CONVERGENT("convergent");
    private final String value;

    LoopStructureTypeType(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static LoopStructureTypeType fromValue(String v) {
        for (LoopStructureTypeType c: LoopStructureTypeType.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v);
    }

}
