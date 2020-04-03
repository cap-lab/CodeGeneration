
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;


/**
 * <p>Java class for FaultToleranceTypeType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="FaultToleranceTypeType">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="reexecution"/>
 *     &lt;enumeration value="activeReplication"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlEnum
public enum FaultToleranceTypeType {

    @XmlEnumValue("activeReplication")
    ACTIVE_REPLICATION("activeReplication"),
    @XmlEnumValue("reexecution")
    REEXECUTION("reexecution");
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
        throw new IllegalArgumentException(v.toString());
    }

}
