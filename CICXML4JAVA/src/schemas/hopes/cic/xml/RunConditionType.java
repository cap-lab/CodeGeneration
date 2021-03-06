
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for RunConditionType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="RunConditionType"&gt;
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string"&gt;
 *     &lt;enumeration value="time-driven"/&gt;
 *     &lt;enumeration value="data-driven"/&gt;
 *     &lt;enumeration value="control-driven"/&gt;
 *   &lt;/restriction&gt;
 * &lt;/simpleType&gt;
 * </pre>
 * 
 */
@XmlType(name = "RunConditionType")
@XmlEnum
public enum RunConditionType {

    @XmlEnumValue("time-driven")
    TIME_DRIVEN("time-driven"),
    @XmlEnumValue("data-driven")
    DATA_DRIVEN("data-driven"),
    @XmlEnumValue("control-driven")
    CONTROL_DRIVEN("control-driven");
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
        throw new IllegalArgumentException(v);
    }

}
