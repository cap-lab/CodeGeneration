
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for DataParallelType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="DataParallelType"&gt;
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string"&gt;
 *     &lt;enumeration value="none"/&gt;
 *     &lt;enumeration value="loop"/&gt;
 *     &lt;enumeration value="wavefront"/&gt;
 *   &lt;/restriction&gt;
 * &lt;/simpleType&gt;
 * </pre>
 * 
 */
@XmlType(name = "DataParallelType")
@XmlEnum
public enum DataParallelType {

    @XmlEnumValue("none")
    NONE("none"),
    @XmlEnumValue("loop")
    LOOP("loop"),
    @XmlEnumValue("wavefront")
    WAVEFRONT("wavefront");
    private final String value;

    DataParallelType(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static DataParallelType fromValue(String v) {
        for (DataParallelType c: DataParallelType.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v);
    }

}
