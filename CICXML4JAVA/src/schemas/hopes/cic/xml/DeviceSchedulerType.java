
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for DeviceSchedulerType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="DeviceSchedulerType"&gt;
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string"&gt;
 *     &lt;enumeration value="other"/&gt;
 *     &lt;enumeration value="fifo"/&gt;
 *     &lt;enumeration value="rr"/&gt;
 *     &lt;enumeration value="high"/&gt;
 *     &lt;enumeration value="realtime"/&gt;
 *   &lt;/restriction&gt;
 * &lt;/simpleType&gt;
 * </pre>
 * 
 */
@XmlType(name = "DeviceSchedulerType")
@XmlEnum
public enum DeviceSchedulerType {

    @XmlEnumValue("other")
    OTHER("other"),
    @XmlEnumValue("fifo")
    FIFO("fifo"),
    @XmlEnumValue("rr")
    RR("rr"),
    @XmlEnumValue("high")
    HIGH("high"),
    @XmlEnumValue("realtime")
    REALTIME("realtime");
    private final String value;

    DeviceSchedulerType(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static DeviceSchedulerType fromValue(String v) {
        for (DeviceSchedulerType c: DeviceSchedulerType.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v);
    }

}
