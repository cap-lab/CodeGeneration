
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for TimeMetricType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="TimeMetricType"&gt;
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string"&gt;
 *     &lt;enumeration value="h"/&gt;
 *     &lt;enumeration value="m"/&gt;
 *     &lt;enumeration value="s"/&gt;
 *     &lt;enumeration value="ms"/&gt;
 *     &lt;enumeration value="us"/&gt;
 *     &lt;enumeration value="ns"/&gt;
 *     &lt;enumeration value="ps"/&gt;
 *     &lt;enumeration value="fs"/&gt;
 *     &lt;enumeration value="cycle"/&gt;
 *     &lt;enumeration value="count"/&gt;
 *   &lt;/restriction&gt;
 * &lt;/simpleType&gt;
 * </pre>
 * 
 */
@XmlType(name = "TimeMetricType")
@XmlEnum
public enum TimeMetricType {

    @XmlEnumValue("h")
    H("h"),
    @XmlEnumValue("m")
    M("m"),
    @XmlEnumValue("s")
    S("s"),
    @XmlEnumValue("ms")
    MS("ms"),
    @XmlEnumValue("us")
    US("us"),
    @XmlEnumValue("ns")
    NS("ns"),
    @XmlEnumValue("ps")
    PS("ps"),
    @XmlEnumValue("fs")
    FS("fs"),
    @XmlEnumValue("cycle")
    CYCLE("cycle"),
    @XmlEnumValue("count")
    COUNT("count");
    private final String value;

    TimeMetricType(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static TimeMetricType fromValue(String v) {
        for (TimeMetricType c: TimeMetricType.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v);
    }

}
