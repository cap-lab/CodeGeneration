
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for MasterSlaveRoleType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="MasterSlaveRoleType"&gt;
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string"&gt;
 *     &lt;enumeration value="master"/&gt;
 *     &lt;enumeration value="slave"/&gt;
 *   &lt;/restriction&gt;
 * &lt;/simpleType&gt;
 * </pre>
 * 
 */
@XmlType(name = "MasterSlaveRoleType")
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
        throw new IllegalArgumentException(v);
    }

}
