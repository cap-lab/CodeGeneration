
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;


/**
 * <p>Java class for ScheduleType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="ScheduleType">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="static"/>
 *     &lt;enumeration value="dynamic"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlEnum
public enum ScheduleType {

    @XmlEnumValue("static")
    STATIC("static"),
    @XmlEnumValue("dynamic")
    DYNAMIC("dynamic");
    private final String value;

    ScheduleType(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static ScheduleType fromValue(String v) {
        for (ScheduleType c: ScheduleType.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v.toString());
    }

}
