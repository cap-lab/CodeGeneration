
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for preemptionTypeType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="preemptionTypeType"&gt;
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string"&gt;
 *     &lt;enumeration value="preemptive"/&gt;
 *     &lt;enumeration value="nonPreemptive"/&gt;
 *   &lt;/restriction&gt;
 * &lt;/simpleType&gt;
 * </pre>
 * 
 */
@XmlType(name = "preemptionTypeType")
@XmlEnum
public enum PreemptionTypeType {

    @XmlEnumValue("preemptive")
    PREEMPTIVE("preemptive"),
    @XmlEnumValue("nonPreemptive")
    NON_PREEMPTIVE("nonPreemptive");
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
        throw new IllegalArgumentException(v);
    }

}
