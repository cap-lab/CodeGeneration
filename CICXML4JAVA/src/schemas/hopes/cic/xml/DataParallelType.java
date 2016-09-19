
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;


/**
 * <p>Java class for DataParallelType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="DataParallelType">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="none"/>
 *     &lt;enumeration value="loop"/>
 *     &lt;enumeration value="wavefront"/>
 *     &lt;enumeration value="loopStructure"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlEnum
public enum DataParallelType {

    @XmlEnumValue("loop")
    LOOP("loop"),
    @XmlEnumValue("loopStructure")
    LOOP_STRUCTURE("loopStructure"),
    @XmlEnumValue("none")
    NONE("none"),
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
        throw new IllegalArgumentException(v.toString());
    }

}
