
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;


/**
 * <p>Java class for LibraryAccessType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="LibraryAccessType">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="duplicate"/>
 *     &lt;enumeration value="mapping"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlEnum
public enum LibraryAccessType {

    @XmlEnumValue("duplicate")
    DUPLICATE("duplicate"),
    @XmlEnumValue("mapping")
    MAPPING("mapping");
    private final String value;

    LibraryAccessType(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static LibraryAccessType fromValue(String v) {
        for (LibraryAccessType c: LibraryAccessType.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v.toString());
    }

}
