
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;


/**
 * <p>Java class for SizeMetricType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="SizeMetricType">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="YiB"/>
 *     &lt;enumeration value="ZiB"/>
 *     &lt;enumeration value="EiB"/>
 *     &lt;enumeration value="PiB"/>
 *     &lt;enumeration value="TiB"/>
 *     &lt;enumeration value="GiB"/>
 *     &lt;enumeration value="MiB"/>
 *     &lt;enumeration value="KiB"/>
 *     &lt;enumeration value="B"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
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
        throw new IllegalArgumentException(v.toString());
    }

}
