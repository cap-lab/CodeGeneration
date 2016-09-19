
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;


/**
 * <p>Java class for preemptionTypeType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="preemptionTypeType">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="preemptive"/>
 *     &lt;enumeration value="nonPreemptive"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlEnum
public enum PreemptionTypeType {

    @XmlEnumValue("nonPreemptive")
    NON_PREEMPTIVE("nonPreemptive"),
    @XmlEnumValue("preemptive")
    PREEMPTIVE("preemptive");
    private final String value;

    PreemptionTypeType(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static PreemptionTypeType fromValue(String v) {
        for (PreemptionTypeType c: PreemptionTypeType.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v.toString());
    }

}
