
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;


/**
 * <p>Java class for TimeMetricType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="TimeMetricType">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="h"/>
 *     &lt;enumeration value="m"/>
 *     &lt;enumeration value="s"/>
 *     &lt;enumeration value="ms"/>
 *     &lt;enumeration value="us"/>
 *     &lt;enumeration value="ns"/>
 *     &lt;enumeration value="ps"/>
 *     &lt;enumeration value="fs"/>
 *     &lt;enumeration value="cycle"/>
 *     &lt;enumeration value="count"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
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
        throw new IllegalArgumentException(v.toString());
    }

}
