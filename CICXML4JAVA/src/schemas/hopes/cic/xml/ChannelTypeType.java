
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;


/**
 * <p>Java class for ChannelTypeType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="ChannelTypeType">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="fifo"/>
 *     &lt;enumeration value="array"/>
 *     &lt;enumeration value="overwritable"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlEnum
public enum ChannelTypeType {

    @XmlEnumValue("fifo")
    FIFO("fifo"),
    @XmlEnumValue("array")
    ARRAY("array"),
    @XmlEnumValue("overwritable")
    OVERWRITABLE("overwritable");
    private final String value;

    ChannelTypeType(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static ChannelTypeType fromValue(String v) {
        for (ChannelTypeType c: ChannelTypeType.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v.toString());
    }

}
