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
 * <p>Java class for FaultToleranceTypeType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="FaultToleranceTypeType"&gt;
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string"&gt;
 *     &lt;enumeration value="reexecution"/&gt;
 *     &lt;enumeration value="activeReplication"/&gt;
 *   &lt;/restriction&gt;
 * &lt;/simpleType&gt;
 * </pre>
 * 
 */
@XmlType(name = "FaultToleranceTypeType")
@XmlEnum
public enum FaultToleranceTypeType {

    @XmlEnumValue("reexecution")
    REEXECUTION("reexecution"),
    @XmlEnumValue("activeReplication")
    ACTIVE_REPLICATION("activeReplication");
    private final String value;

    FaultToleranceTypeType(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static FaultToleranceTypeType fromValue(String v) {
        for (FaultToleranceTypeType c: FaultToleranceTypeType.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v);
    }

}
