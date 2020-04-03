
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;


/**
 * <p>Java class for LoopStructureTypeType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="LoopStructureTypeType">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="data"/>
 *     &lt;enumeration value="convergent"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlEnum
public enum LoopStructureTypeType {

    @XmlEnumValue("convergent")
    CONVERGENT("convergent"),
    @XmlEnumValue("data")
    DATA("data");
    private final String value;

    LoopStructureTypeType(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static LoopStructureTypeType fromValue(String v) {
        for (LoopStructureTypeType c: LoopStructureTypeType.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v.toString());
    }

}
