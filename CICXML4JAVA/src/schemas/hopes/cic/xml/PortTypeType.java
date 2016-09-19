
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;


/**
 * <p>Java class for PortTypeType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="PortTypeType">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="fifo"/>
 *     &lt;enumeration value="array"/>
 *     &lt;enumeration value="overwritable"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlEnum
public enum PortTypeType {

    @XmlEnumValue("array")
    ARRAY("array"),
    @XmlEnumValue("fifo")
    FIFO("fifo"),
    @XmlEnumValue("overwritable")
    OVERWRITABLE("overwritable");
    private final String value;

    PortTypeType(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static PortTypeType fromValue(String v) {
        for (PortTypeType c: PortTypeType.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v.toString());
    }

}
