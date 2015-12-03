
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;


/**
 * <p>Java class for RunConditionType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="RunConditionType">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="time-driven"/>
 *     &lt;enumeration value="data-driven"/>
 *     &lt;enumeration value="control-driven"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlEnum
public enum RunConditionType {

    @XmlEnumValue("control-driven")
    CONTROL_DRIVEN("control-driven"),
    @XmlEnumValue("data-driven")
    DATA_DRIVEN("data-driven"),
    @XmlEnumValue("time-driven")
    TIME_DRIVEN("time-driven");
    private final String value;

    RunConditionType(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static RunConditionType fromValue(String v) {
        for (RunConditionType c: RunConditionType.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v.toString());
    }

}
