
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for LibraryAccessType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="LibraryAccessType"&gt;
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string"&gt;
 *     &lt;enumeration value="duplicate"/&gt;
 *     &lt;enumeration value="mapping"/&gt;
 *   &lt;/restriction&gt;
 * &lt;/simpleType&gt;
 * </pre>
 * 
 */
@XmlType(name = "LibraryAccessType")
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
        throw new IllegalArgumentException(v);
    }

}
