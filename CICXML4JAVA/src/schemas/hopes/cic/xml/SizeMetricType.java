
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for SizeMetricType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="SizeMetricType"&gt;
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string"&gt;
 *     &lt;enumeration value="YiB"/&gt;
 *     &lt;enumeration value="ZiB"/&gt;
 *     &lt;enumeration value="EiB"/&gt;
 *     &lt;enumeration value="PiB"/&gt;
 *     &lt;enumeration value="TiB"/&gt;
 *     &lt;enumeration value="GiB"/&gt;
 *     &lt;enumeration value="MiB"/&gt;
 *     &lt;enumeration value="KiB"/&gt;
 *     &lt;enumeration value="B"/&gt;
 *   &lt;/restriction&gt;
 * &lt;/simpleType&gt;
 * </pre>
 * 
 */
@XmlType(name = "SizeMetricType")
@XmlEnum
public enum SizeMetricType {

    @XmlEnumValue("YiB")
    YI_B("YiB"),
    @XmlEnumValue("ZiB")
    ZI_B("ZiB"),
    @XmlEnumValue("EiB")
    EI_B("EiB"),
    @XmlEnumValue("PiB")
    PI_B("PiB"),
    @XmlEnumValue("TiB")
    TI_B("TiB"),
    @XmlEnumValue("GiB")
    GI_B("GiB"),
    @XmlEnumValue("MiB")
    MI_B("MiB"),
    @XmlEnumValue("KiB")
    KI_B("KiB"),
    B("B");
    private final String value;

    SizeMetricType(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static SizeMetricType fromValue(String v) {
        for (SizeMetricType c: SizeMetricType.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v);
    }

}
