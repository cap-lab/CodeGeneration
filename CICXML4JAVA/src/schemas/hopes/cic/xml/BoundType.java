
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;


/**
 * <p>Java class for BoundType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="BoundType">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="Maximum"/>
 *     &lt;enumeration value="Minimum"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlEnum
public enum BoundType {

    @XmlEnumValue("Maximum")
    MAXIMUM("Maximum"),
    @XmlEnumValue("Minimum")
    MINIMUM("Minimum");
    private final String value;

    BoundType(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static BoundType fromValue(String v) {
        for (BoundType c: BoundType.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v.toString());
    }

}
