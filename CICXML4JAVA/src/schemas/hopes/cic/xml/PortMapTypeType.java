
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;


/**
 * <p>Java class for PortMapTypeType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="PortMapTypeType">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="normal"/>
 *     &lt;enumeration value="distributing"/>
 *     &lt;enumeration value="broadcasting"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlEnum
public enum PortMapTypeType {

    @XmlEnumValue("normal")
    NORMAL("normal"),
    @XmlEnumValue("distributing")
    DISTRIBUTING("distributing"),
    @XmlEnumValue("broadcasting")
    BROADCASTING("broadcasting");
    private final String value;

    PortMapTypeType(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static PortMapTypeType fromValue(String v) {
        for (PortMapTypeType c: PortMapTypeType.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v.toString());
    }

}
