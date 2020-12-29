
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for groupTypeType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="groupTypeType"&gt;
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string"&gt;
 *     &lt;enumeration value="StaticRate"/&gt;
 *   &lt;/restriction&gt;
 * &lt;/simpleType&gt;
 * </pre>
 * 
 */
@XmlType(name = "groupTypeType")
@XmlEnum
public enum GroupTypeType {

    @XmlEnumValue("StaticRate")
    STATIC_RATE("StaticRate");
    private final String value;

    GroupTypeType(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static GroupTypeType fromValue(String v) {
        for (GroupTypeType c: GroupTypeType.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v);
    }

}
