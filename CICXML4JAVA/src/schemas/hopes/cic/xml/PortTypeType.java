
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for PortTypeType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="PortTypeType"&gt;
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string"&gt;
 *     &lt;enumeration value="fifo"/&gt;
 *     &lt;enumeration value="array"/&gt;
 *     &lt;enumeration value="overwritable"/&gt;
 *   &lt;/restriction&gt;
 * &lt;/simpleType&gt;
 * </pre>
 * 
 */
@XmlType(name = "PortTypeType")
@XmlEnum
public enum PortTypeType {

    @XmlEnumValue("fifo")
    FIFO("fifo"),
    @XmlEnumValue("array")
    ARRAY("array"),
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
        throw new IllegalArgumentException(v);
    }

}
