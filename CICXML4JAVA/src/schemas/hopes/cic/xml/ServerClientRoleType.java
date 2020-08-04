
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;


/**
 * <p>Java class for ServerClientRoleType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="ServerClientRoleType">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="server"/>
 *     &lt;enumeration value="client"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlEnum
public enum ServerClientRoleType {

    @XmlEnumValue("server")
    SERVER("server"),
    @XmlEnumValue("client")
    CLIENT("client");
    private final String value;

    ServerClientRoleType(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static ServerClientRoleType fromValue(String v) {
        for (ServerClientRoleType c: ServerClientRoleType.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v.toString());
    }

}
