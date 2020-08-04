
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;


/**
 * <p>Java class for MasterSlaveRoleType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="MasterSlaveRoleType">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="master"/>
 *     &lt;enumeration value="slave"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlEnum
public enum MasterSlaveRoleType {

    @XmlEnumValue("master")
    MASTER("master"),
    @XmlEnumValue("slave")
    SLAVE("slave");
    private final String value;

    MasterSlaveRoleType(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static MasterSlaveRoleType fromValue(String v) {
        for (MasterSlaveRoleType c: MasterSlaveRoleType.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v.toString());
    }

}
